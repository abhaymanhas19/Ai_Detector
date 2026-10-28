"""Token-bounded windows and sentence groups, ported from lib/windows.ts."""

import math
import re
from collections.abc import Callable

from .constants import MAX_WINDOW_TOKENS, MIN_GROUP_TOKENS
from .results import Segment, SentenceGroup, Window

_WHITESPACE = re.compile(r'\s+')


def normalize_for_model(text: str) -> str:
    return _WHITESPACE.sub(' ', text).strip()


def _make_window(segments: list[Segment], start: int, stop: int, tokens: list[int], index: int) -> Window:
    included = list(range(start, stop))
    return Window(
        index=index,
        segments=included,
        text=normalize_for_model(''.join(s.text for s in segments[start:stop])),
        tokens=sum(tokens[i] for i in included),
        words=sum(segments[i].words for i in included),
        start=segments[start].start,
        end=segments[stop - 1].end,
    )


def _build_balanced_windows(segments: list[Segment], tokens: list[int], max_tokens: int) -> list[Window]:
    total = sum(tokens)
    count = max(1, math.ceil(total / max_tokens))
    target = total / count

    windows: list[Window] = []
    cursor = 0
    accumulated = 0

    for i in range(len(segments)):
        nxt = tokens[i]
        if accumulated > 0 and accumulated + nxt > max_tokens:
            windows.append(_make_window(segments, cursor, i, tokens, len(windows)))
            cursor = i
            accumulated = 0
        accumulated += nxt

        is_last_window = len(windows) >= count - 1
        enough_segments_left = len(segments) - (i + 1) >= count - len(windows) - 1
        if not is_last_window and accumulated >= target and enough_segments_left:
            windows.append(_make_window(segments, cursor, i + 1, tokens, len(windows)))
            cursor = i + 1
            accumulated = 0

    if cursor < len(segments):
        windows.append(_make_window(segments, cursor, len(segments), tokens, len(windows)))

    return windows


def build_windows(
    segments: list[Segment],
    token_count_of: Callable[[Segment], int],
    max_tokens: int = MAX_WINDOW_TOKENS,
    stride: float = 0.5,
    balance: bool = False,
) -> list[Window]:
    if not segments:
        return []

    tokens = [token_count_of(segment) for segment in segments]

    if balance:
        return _build_balanced_windows(segments, tokens, max_tokens)

    windows: list[Window] = []
    cursor = 0
    while cursor < len(segments):
        total = 0
        end = cursor
        while end < len(segments):
            nxt = tokens[end]
            if end > cursor and total + nxt > max_tokens:
                break
            total += nxt
            end += 1

        windows.append(_make_window(segments, cursor, end, tokens, len(windows)))

        if end >= len(segments):
            break
        consumed = end - cursor
        cursor += max(1, math.floor(consumed * stride))

    return windows


def _make_group(
    segments: list[Segment],
    included: list[int],
    tokens: list[int],
    index: int,
    passage: int,
) -> SentenceGroup:
    return SentenceGroup(
        index=index,
        passage=passage,
        segments=list(included),
        sentences=len(included),
        text=normalize_for_model(''.join(segments[i].text for i in included)),
        tokens=sum(tokens[i] for i in included),
        words=sum(segments[i].words for i in included),
        start=segments[included[0]].start,
        end=segments[included[-1]].end,
    )


def build_sentence_groups(
    segments: list[Segment],
    windows: list[Window],
    tokens: list[int],
    min_tokens: int = MIN_GROUP_TOKENS,
) -> list[SentenceGroup]:
    """Packs each window's sentences into groups that clear `min_tokens`.

    Groups never straddle a window and together partition its segments exactly,
    which is what lets a window's score be redistributed across its groups.
    """
    groups: list[SentenceGroup] = []

    for window in windows:
        buckets: list[list[int]] = []
        current: list[int] = []
        accumulated = 0

        for segment_index in window.segments:
            current.append(segment_index)
            accumulated += tokens[segment_index]
            if accumulated >= min_tokens:
                buckets.append(current)
                current = []
                accumulated = 0

        if current:
            if buckets:
                buckets[-1].extend(current)
            else:
                buckets.append(current)

        for bucket in buckets:
            groups.append(_make_group(segments, bucket, tokens, len(groups), window.index))

    return groups
