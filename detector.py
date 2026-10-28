"""The scan pipeline from detector.worker.ts (`runScan`, `scoreWindows`, `scoreGroups`).

Left out on purpose, because they only make sense in a browser: the WebGPU
tier upgrade, progress/partial messages and scan cancellation.
"""

import threading
from collections import OrderedDict
from typing import Literal, Protocol

from .aggregate import (
    compose_bands,
    confidence_for,
    text_stats,
    to_group_scores,
    to_passage_scores,
    verdict_for,
    word_weighted_score,
)
from .constants import DEFAULT_DTYPE, HARD_WORD_LIMIT, MAX_WINDOW_TOKENS, ScanSettings
from .results import DetectionResult, Segment, Window
from .segment import count_words, segment_text
from .windows import build_sentence_groups, build_windows, normalize_for_model

ScanDepth = Literal['fast', 'full']

MAX_WINDOW_CACHE = 2000
MAX_TOKEN_CACHE = 10000
# q8 is dynamically quantised: activations are scaled over the whole batch
# tensor, so a window's score depends on what it is batched with (up to ~0.18
# apart on the sample texts). Batching q8 in twos, in the same order as the
# browser's WASM tier, reproduces its scores to ~1e-8. fp16/fp32 do not move.
BATCH_SIZES = {'q8': 2}
DEFAULT_BATCH_SIZE = 8


class Classifier(Protocol):
    dtype: str

    def count_tokens(self, text: str) -> int: ...

    def score(self, texts: list[str]) -> list[float]: ...


class TextTooLong(ValueError):
    pass


class EmptyText(ValueError):
    pass


def _remember(cache: OrderedDict, key, value, limit: int, lock: threading.Lock) -> None:
    with lock:
        cache[key] = value
        while len(cache) > limit:
            cache.popitem(last=False)


class AiDetector:
    """Keep one instance per process: it holds the model and the score caches."""

    def __init__(self, classifier: Classifier | None = None, dtype: str = DEFAULT_DTYPE):
        if classifier is None:
            from .model import OnnxClassifier

            classifier = OnnxClassifier(dtype=dtype)
        self.classifier = classifier
        self._window_cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def _token_count(self, segment: Segment) -> int:
        cached = self._token_cache.get(segment.text)
        if cached is not None:
            return cached
        try:
            count = self.classifier.count_tokens(normalize_for_model(segment.text))
        except Exception:
            count = -(-segment.words * 135 // 100)
        _remember(self._token_cache, segment.text, count, MAX_TOKEN_CACHE, self._lock)
        return count

    def _score_windows(self, windows: list[Window]) -> tuple[list[float], int]:
        dtype = self.classifier.dtype
        scores = [float('nan')] * len(windows)
        pending: list[int] = []
        for index, window in enumerate(windows):
            cached = self._window_cache.get((dtype, window.text))
            if cached is None:
                pending.append(index)
            else:
                scores[index] = cached
        cached_count = len(windows) - len(pending)

        batch_size = BATCH_SIZES.get(dtype, DEFAULT_BATCH_SIZE)
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            for index, score in zip(batch, self.classifier.score([windows[i].text for i in batch]), strict=True):
                scores[index] = score
                _remember(self._window_cache, (dtype, windows[index].text), score, MAX_WINDOW_CACHE, self._lock)

        return scores, cached_count

    def scan(
        self,
        text: str,
        settings: ScanSettings | None = None,
        depth: ScanDepth = 'full',
    ) -> DetectionResult:
        settings = settings or ScanSettings()

        words = count_words(text)
        if words > HARD_WORD_LIMIT:
            raise TextTooLong(f'{words} words; scan up to {HARD_WORD_LIMIT} at a time.')

        segments, paragraphs = segment_text(text)
        if not segments:
            raise EmptyText('There is no text to scan yet.')

        tokens = [self._token_count(segment) for segment in segments]
        windows = build_windows(segments, lambda s: tokens[s.index], max_tokens=MAX_WINDOW_TOKENS, stride=1, balance=True)

        window_scores, cached_count = self._score_windows(windows)
        passages = to_passage_scores(windows, window_scores, settings)

        document_score = word_weighted_score(passages)
        peak_score = max([document_score, *(p.score for p in passages)])
        stats = text_stats(text, segments, len(paragraphs), sum(tokens))

        groups = []
        if depth == 'full':
            sentence_groups = build_sentence_groups(segments, windows, tokens)
            group_scores, _ = self._score_windows(sentence_groups)
            groups = to_group_scores(sentence_groups, group_scores, passages, settings)

        return DetectionResult(
            depth=depth,
            document_score=document_score,
            peak_score=peak_score,
            verdict=verdict_for(document_score, stats.words),
            confidence=confidence_for(document_score, window_scores, stats.words),
            composition=compose_bands(passages),
            passages=[] if depth == 'fast' else passages,
            groups=groups,
            stats=stats,
            window_count=len(windows),
            cached_window_count=cached_count,
            settings=settings,
            dtype=self.classifier.dtype,
        )
