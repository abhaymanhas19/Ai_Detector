"""Pure-logic tests for the AI-detection port. No model download: the
classifier is faked. Expected segmentations were taken from Node's
`Intl.Segmenter`, which is what the browser detector uses."""

import math

import pytest

from apps.helpersapps.ai_detection import AiDetector, EmptyText, ScanSettings, TextTooLong
from apps.helpersapps.ai_detection.aggregate import (
    compose_bands,
    confidence_for,
    verdict_for,
    word_weighted_score,
)
from apps.helpersapps.ai_detection.align import align_to_weighted_mean
from apps.helpersapps.ai_detection.results import PassageScore
from apps.helpersapps.ai_detection.segment import _sentence_spans, count_words, segment_text
from apps.helpersapps.ai_detection.windows import build_sentence_groups, build_windows


class FakeClassifier:
    """Scores a text by its share of words starting with 'a'; counts a token per word."""

    def __init__(self, dtype='q8'):
        self.dtype = dtype
        self.batches: list[int] = []

    def count_tokens(self, text):
        return count_words(text)

    def score(self, texts):
        self.batches.append(len(texts))
        out = []
        for text in texts:
            words = text.split()
            out.append(sum(w.lower().startswith('a') for w in words) / max(1, len(words)))
        return out


@pytest.mark.parametrize('text, words', [
    ('liver-pill and housemaid’s knee', ['liver', 'pill', 'and', 'housemaid’s', 'knee']),
    ('1,200 acre — 3.14 pi', ['1,200', 'acre', '3.14', 'pi']),
    ('U.S. case, e.g. v2.1 — £1.5m', ['U.S', 'case', 'e.g', 'v2.1', '1.5m']),
    ("don't co-op’s a:b foo_bar —", ["don't", 'co', 'op’s', 'a:b', 'foo_bar']),
    ('Zürich naïve x—y 50% #tag', ['Zürich', 'naïve', 'x', 'y', '50', 'tag']),
])
def test_non_latin_text_counts_words_like_icu(text, words):
    assert count_words(text) == len(words)


def test_latin_only_text_uses_the_simple_regex():
    # Every char <= U+058F: the browser keeps hyphenated words whole here.
    assert count_words("liver-pill and housemaid's knee") == 4


@pytest.mark.parametrize('text, sentences', [
    ('Wait… Then he left.', ['Wait… Then he left.']),
    ('Stop!… Go.', ['Stop!', '… Go.']),
    ('Hi.(Yes) No.', ['Hi.(', 'Yes) No.']),
    ('He said. “Yes.” Then', ['He said. ', '“Yes.” ', 'Then']),
    ('Dr. Smith went to Washington. He arrived at 3.5 p.m. on the U.S.A. tour! Did it work? yes.',
     ['Dr. ', 'Smith went to Washington. ', 'He arrived at 3.5 p.m. on the U.S.A. tour! ', 'Did it work? ', 'yes.']),
    ('Title line\nBody starts here. And e.g. this.', ['Title line\n', 'Body starts here. ', 'And e.g. this.']),
    ('x... 2 y', ['x... 2 y']),
    ('中文句子。第二句！English after.', ['中文句子。', '第二句！', 'English after.']),
])
def test_sentence_spans_follow_icu(text, sentences):
    assert [text[a:b] for a, b in _sentence_spans(text)] == sentences


def test_segment_text_folds_blank_lines_into_paragraphs():
    text = 'First one. Second one.\n\n\n   \nThird one.'
    segments, paragraphs = segment_text(text)
    assert [s.text for s in segments] == ['First one. ', 'Second one.\n\n\n   \n', 'Third one.']
    assert [p.segments for p in paragraphs] == [[0, 1], [2]]
    assert segment_text('  \n ') == ([], [])


def test_balanced_windows_partition_segments_under_the_token_cap():
    text = ' '.join(f'Sentence number {i} has exactly seven words.' for i in range(200))
    segments, _ = segment_text(text)
    windows = build_windows(segments, lambda s: s.words, max_tokens=500, stride=1, balance=True)
    assert [i for w in windows for i in w.segments] == list(range(len(segments)))
    assert all(w.tokens <= 500 for w in windows)
    assert len(windows) == math.ceil(sum(s.words for s in segments) / 500)


def test_sentence_groups_partition_each_window_and_clear_the_floor():
    text = ' '.join(f'Short line {i} here.' for i in range(120))
    segments, _ = segment_text(text)
    tokens = [s.words for s in segments]
    windows = build_windows(segments, lambda s: tokens[s.index], max_tokens=100, stride=1, balance=True)
    groups = build_sentence_groups(segments, windows, tokens, min_tokens=20)
    for window in windows:
        members = [g for g in groups if g.passage == window.index]
        assert [i for g in members for i in g.segments] == window.segments
        assert all(g.tokens >= 20 for g in members)


def test_alignment_hits_the_target_mean_without_reordering():
    scores, weights = [0.9, 0.2, 0.6, 0.99], [10, 40, 25, 5]
    aligned = align_to_weighted_mean(scores, weights, 0.3)
    mean = sum(a * w for a, w in zip(aligned, weights)) / sum(weights)
    assert mean == pytest.approx(0.3, abs=1e-9)
    assert sorted(range(4), key=aligned.__getitem__) == sorted(range(4), key=scores.__getitem__)
    assert align_to_weighted_mean([0.4, 0.8], [1, 1], 1.0) == [1.0, 1.0]


@pytest.mark.parametrize('score, words, verdict', [
    (0.9, 59, 'insufficient-text'),
    (0.19, 60, 'very-likely-human'),
    (0.2, 60, 'probably-human'),
    (0.5, 60, 'inconclusive'),
    (0.55, 60, 'probably-ai'),
    (0.8, 60, 'very-likely-ai'),
])
def test_verdict_thresholds(score, words, verdict):
    assert verdict_for(score, words) == verdict


def test_confidence_levels():
    assert confidence_for(0.95, [0.9, 0.95, 0.97, 0.99], 500).level == 'high'
    assert confidence_for(0.6, [0.6], 100).level == 'low'
    assert confidence_for(0.95, [0.95], 30).level == 'low'


def test_composition_always_sums_to_100():
    spans = [PassageScore(i, 0, 0, s, b, w, w, False)
             for i, (s, b, w) in enumerate([(0.1, 'human', 1), (0.5, 'mixed', 1), (0.9, 'ai', 1)])]
    composition = compose_bands(spans)
    assert (composition.human, composition.mixed, composition.ai) == (34, 33, 33)
    assert word_weighted_score(spans) == pytest.approx(0.5)


def test_scan_end_to_end_with_a_fake_model():
    human = 'The quick brown fox jumps over the lazy dog near the river bank today. ' * 30
    ai_like = 'An agile algorithm always analyses abundant archives across any area. ' * 30
    detector = AiDetector(classifier=FakeClassifier())

    result = detector.scan(human + '\n\n' + ai_like)

    assert result.window_count == len(result.passages) >= 2
    assert result.passages[0].band == 'human' and result.passages[-1].band == 'ai'
    assert word_weighted_score(result.groups) == pytest.approx(result.document_score, abs=1e-6)
    assert result.to_dict()['documentScore'] == result.document_score
    assert 'lowConfidence' in result.to_dict()['passages'][0]


def test_rescan_reuses_cached_window_scores_and_q8_batches_in_twos():
    fake = FakeClassifier('q8')
    detector = AiDetector(classifier=fake)
    text = 'Another line about apples and pears in autumn. ' * 300

    first = detector.scan(text, depth='fast')
    assert first.passages == [] and first.groups == []
    assert set(fake.batches) <= {1, 2}

    second = detector.scan(text, depth='fast')
    assert second.cached_window_count == second.window_count == first.window_count


def test_settings_change_bands_not_scores():
    detector = AiDetector(classifier=FakeClassifier())
    text = 'Apples are always around. Bananas belong elsewhere. ' * 30
    strict = detector.scan(text, ScanSettings(human_threshold=0.05, ai_threshold=0.95))
    loose = detector.scan(text, ScanSettings(human_threshold=0.45, ai_threshold=0.46))
    assert strict.document_score == loose.document_score
    assert strict.passages[0].band != loose.passages[0].band


def test_rejects_empty_and_oversized_text():
    detector = AiDetector(classifier=FakeClassifier())
    with pytest.raises(EmptyText):
        detector.scan(' \n ')
    with pytest.raises(TextTooLong):
        detector.scan('word ' * 20001)
