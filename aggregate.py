"""Scores to bands, verdict, confidence and stats, ported from lib/aggregate.ts."""

import math

from .align import align_to_weighted_mean
from .constants import MIN_PASSAGES_FOR_SPREAD, MIN_WORDS_FOR_VERDICT, ScanSettings
from .results import (
    Composition,
    Confidence,
    PassageScore,
    Segment,
    SentenceGroup,
    SentenceGroupScore,
    TextStats,
    Window,
)
from .segment import count_words


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _usable(value: float | None) -> float:
    return 0.5 if value is None or math.isnan(value) else _clamp01(value)


def band_for(score: float, settings: ScanSettings) -> str:
    if score <= settings.human_threshold:
        return 'human'
    if score >= settings.ai_threshold:
        return 'ai'
    return 'mixed'


def to_passage_scores(windows: list[Window], scores: list[float], settings: ScanSettings) -> list[PassageScore]:
    passages = []
    for index, window in enumerate(windows):
        score = _usable(scores[index] if index < len(scores) else None)
        passages.append(PassageScore(
            index=window.index,
            start=window.start,
            end=window.end,
            score=score,
            band=band_for(score, settings),
            words=window.words,
            tokens=window.tokens,
            low_confidence=window.words < settings.low_confidence_words,
        ))
    return passages


def to_group_scores(
    groups: list[SentenceGroup],
    scores: list[float],
    passages: list[PassageScore],
    settings: ScanSettings,
) -> list[SentenceGroupScore]:
    """Levels each passage's isolated group scores onto that passage's score.

    Groups partition their passage's words, so the word-weighted mean of all
    groups equals `word_weighted_score(passages)` by construction.
    """
    by_passage: dict[int, list[int]] = {}
    for index, group in enumerate(groups):
        by_passage.setdefault(group.passage, []).append(index)

    passage_score = {passage.index: passage.score for passage in passages}
    aligned = [0.5] * len(groups)

    for passage_index, members in by_passage.items():
        raw = [_usable(scores[i] if i < len(scores) else None) for i in members]
        target = passage_score.get(passage_index)
        if target is None:
            shifted = raw
        else:
            shifted = align_to_weighted_mean(raw, [groups[i].words for i in members], target)
        for i, value in zip(members, shifted):
            aligned[i] = value

    return [
        SentenceGroupScore(
            index=group.index,
            passage=group.passage,
            start=group.start,
            end=group.end,
            score=aligned[index],
            raw_score=_usable(scores[index] if index < len(scores) else None),
            band=band_for(aligned[index], settings),
            words=group.words,
            tokens=group.tokens,
            sentences=group.sentences,
        )
        for index, group in enumerate(groups)
    ]


def _largest_remainder(counts: list[int], total: int) -> list[int]:
    exact = [count / total * 100 for count in counts]
    floors = [math.floor(value) for value in exact]
    remainder = 100 - sum(floors)
    order = sorted(range(len(exact)), key=lambda i: -(exact[i] - math.floor(exact[i])))
    result = list(floors)
    for index in order:
        if remainder <= 0:
            break
        result[index] += 1
        remainder -= 1
    return result


def compose_bands(spans: list[PassageScore]) -> Composition:
    words = {'human': 0, 'mixed': 0, 'ai': 0}
    for span in spans:
        words[span.band if span.band in words else 'mixed'] += span.words

    total = sum(words.values())
    if total == 0:
        return Composition(0, 0, 0, 0, 0, 0, 0)

    human, mixed, ai = _largest_remainder([words['human'], words['mixed'], words['ai']], total)
    return Composition(human, mixed, ai, words['human'], words['mixed'], words['ai'], total)


def word_weighted_score(spans: list[PassageScore]) -> float:
    weighted = 0.0
    weight = 0
    for span in spans:
        words = max(1, span.words)
        weighted += span.score * words
        weight += words
    return _clamp01(weighted / weight) if weight > 0 else 0.5


def verdict_for(score: float, total_words: int) -> str:
    if total_words < MIN_WORDS_FOR_VERDICT:
        return 'insufficient-text'
    if score < 0.2:
        return 'very-likely-human'
    if score < 0.45:
        return 'probably-human'
    if score < 0.55:
        return 'inconclusive'
    if score < 0.8:
        return 'probably-ai'
    return 'very-likely-ai'


def confidence_for(score: float, window_scores: list[float], total_words: int) -> Confidence:
    reasons: list[str] = []
    points = 0

    if total_words >= 400:
        points += 2
        reasons.append(f'{total_words} words analysed — a comfortable sample')
    elif total_words >= MIN_WORDS_FOR_VERDICT:
        points += 1
        reasons.append(f'{total_words} words analysed — a short sample')
    else:
        reasons.append(f'Only {total_words} words — too short to be reliable')

    usable = [value for value in window_scores if not math.isnan(value)]
    if len(usable) >= MIN_PASSAGES_FOR_SPREAD:
        mean = sum(usable) / len(usable)
        spread = math.sqrt(sum((value - mean) ** 2 for value in usable) / len(usable))
        if spread < 0.15:
            points += 2
            reasons.append('Passages agree closely with each other')
        elif spread < 0.3:
            points += 1
            reasons.append('Passages mostly agree with each other')
        else:
            reasons.append('Passages disagree — the text may mix sources')
    else:
        noun = 'passage' if len(usable) == 1 else 'passages'
        reasons.append(f'Only {len(usable)} {noun} — too few to tell whether they agree')

    decisiveness = abs(score - 0.5) * 2
    if decisiveness > 0.6:
        points += 2
        reasons.append('The overall score is far from the 50/50 line')
    elif decisiveness > 0.3:
        points += 1
        reasons.append('The overall score leans clearly one way')
    else:
        reasons.append('The overall score sits close to the 50/50 line')

    if total_words < MIN_WORDS_FOR_VERDICT:
        level = 'low'
    else:
        level = 'high' if points >= 5 else 'medium' if points >= 3 else 'low'
    return Confidence(level=level, reasons=reasons)


def text_stats(text: str, segments: list[Segment], paragraph_count: int, tokens: int) -> TextStats:
    words = count_words(text)
    return TextStats(
        # JS `text.length` counts UTF-16 code units.
        characters=len(text.encode('utf-16-le')) // 2,
        words=words,
        sentences=len(segments),
        paragraphs=paragraph_count,
        tokens=tokens,
        reading_minutes=words / 238,
    )
