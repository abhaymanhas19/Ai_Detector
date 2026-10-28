"""Sentence and paragraph segmentation, ported from lib/segment.ts.

The browser splits sentences with `Intl.Segmenter` (ICU's sentence rules), and
counts words with a regex only when every character is at or below U+058F;
anything else (an em dash, a curly quote) switches it to ICU's word segmenter.
Python has no ICU without PyICU, so both are ported here as the subset of the
Unicode rules (UAX #29) that ordinary prose exercises. Chinese, Japanese and
Thai are the exception: ICU segments them with a dictionary, which is not
reproduced, so word counts there are approximate.
"""

import re
import unicodedata

import regex

from .results import Paragraph, Segment

WORD_RE = regex.compile(r"[\p{L}\p{N}][\p{L}\p{N}'’-]*")

PARAGRAPH_BREAK = re.compile(r'\r?\n[ \t]*\r?\n')
_PARAGRAPH_BREAK_SPLIT = re.compile(r'\r?\n[ \t]*\r?\n\s*')


# UAX #29 word boundaries, the rules that decide `isWordLike` segments:
# letters and digits run together (WB5, WB8-10), "_" joins anything (WB13a/b),
# a MidLetter or MidNumLet mark joins two letters ("don't", "U.S", "a:b"),
# a MidNum or MidNumLet mark joins two digits ("1,200", "3.14"), combining
# marks attach to what precedes them (WB4), and Katakana runs stay whole
# (WB13). Han and Hiragana are dictionary-segmented by ICU; one word per
# character is the closest rule-based stand-in.
_EXT = r"[\p{M}\p{Cf}]*"
_LETTER = rf"(?:(?![\p{{Han}}\p{{Hiragana}}\p{{Katakana}}])\p{{L}}{_EXT})"
_DIGIT = rf"(?:\p{{Nd}}{_EXT})"
_CONNECT = rf"(?:\p{{Pc}}{_EXT})"
_MID_LETTER = r"[:·‧︓﹕：.’'﹒．＇‘]"
_MID_NUM = r"[,;;؛،߸⁄﹐﹔，；.’'﹒．＇‘]"
_ALNUM_WORD = (
    rf"{_CONNECT}*(?:{_LETTER}|{_DIGIT})"
    rf"(?:{_LETTER}|{_DIGIT}|{_CONNECT}"
    rf"|(?<=\p{{L}}\p{{M}}*){_MID_LETTER}{_EXT}(?={_LETTER})"
    rf"|(?<=\p{{Nd}}\p{{M}}*){_MID_NUM}{_EXT}(?={_DIGIT}))*"
)
_ICU_WORD_RE = regex.compile(
    rf"{_ALNUM_WORD}|{_CONNECT}{{2,}}|(?:\p{{Katakana}}{_EXT}|[ーｰ])(?:\p{{Katakana}}{_EXT}|[ーｰ]|{_CONNECT})*"
    rf"|[\p{{Han}}\p{{Hiragana}}]{_EXT}"
)


def _is_latin_only(text: str) -> bool:
    return all(ord(char) <= 0x058F for char in text)


def count_words(text: str) -> int:
    if _is_latin_only(text):
        return len(WORD_RE.findall(text))
    return len(_ICU_WORD_RE.findall(text))


# UAX #29 Sentence_Break classes, as ICU assigns them.
_SEP = '\n\r\u0085\u2028\u2029'
_ATERM = '.\u2024\ufe52\uff0e'
_STERM = (
    '!?\u203c\u203d\u2047\u2048\u2049\u3002\uff01\uff1f\uff61'
    '\u0964\u0965\u061f\u06d4\u0589\u1362\u1367\u1368\u0700\u0701\u0702\u07f9'
)
_SCONTINUE = (
    ',-:;\u2013\u2014\u055d\u060c\u060d\u07f8\u1802\u1808\u3001'
    '\ufe10\ufe11\ufe13\ufe14\ufe31\ufe32\ufe50\ufe51\ufe54\ufe55\ufe58\ufe63\uff0c\uff0d\uff1a\uff1b\uff64'
)
_CLOSE = regex.compile(r'[\p{Ps}\p{Pe}\p{Pi}\p{Pf}"\']')
_EXTEND = regex.compile(r'[\p{M}\p{Cf}]')


def _sentence_class(char: str) -> str:
    if char in _SEP:
        return 'sep'
    if char in _ATERM:
        return 'aterm'
    if char in _STERM:
        return 'sterm'
    if char in _SCONTINUE:
        return 'scontinue'
    if char.isspace():
        return 'sp'
    if _CLOSE.match(char):
        return 'close'
    if char.isupper() or unicodedata.category(char) == 'Lt':
        return 'upper'
    if char.islower():
        return 'lower'
    if char.isalpha():
        return 'oletter'
    if unicodedata.category(char) == 'Nd':
        return 'numeric'
    return 'other'


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries by the UAX #29 rules ICU (and so `Intl.Segmenter`) applies."""
    # SB5: combining marks and format characters belong to what precedes them.
    starts: list[int] = []
    classes: list[str] = []
    for index, char in enumerate(text):
        if classes and classes[-1] != 'sep' and _EXTEND.match(char):
            continue
        if char == '\n' and index > 0 and text[index - 1] == '\r':
            continue  # SB3
        starts.append(index)
        classes.append(_sentence_class(char))
    starts.append(len(text))
    count = len(classes)

    breaks: list[int] = []
    i = 0
    while i < count:
        kind = classes[i]
        if kind == 'sep':
            breaks.append(i + 1)  # SB4
            i += 1
            continue
        if kind not in ('aterm', 'sterm'):
            i += 1
            continue

        j = i + 1
        while j < count and classes[j] == 'close':
            j += 1
        k = j
        while k < count and classes[k] == 'sp':
            k += 1
        if k >= count:
            break
        if classes[k] == 'sep':
            breaks.append(k + 1)  # SB9-11
            i = k + 1
            continue
        if classes[k] in ('scontinue', 'aterm', 'sterm'):
            i = k  # SB8a
            continue
        if kind == 'aterm':
            if k == i + 1 and classes[k] == 'numeric':
                i = k  # SB6
                continue
            if k == i + 1 and i > 0 and classes[i - 1] in ('upper', 'lower') and classes[k] == 'upper':
                i = k  # SB7
                continue
            m = k
            while m < count and classes[m] not in ('oletter', 'upper', 'lower', 'sep', 'aterm', 'sterm'):
                m += 1
            if m < count and classes[m] == 'lower':
                i = k  # SB8
                continue
        breaks.append(k)  # SB11
        i = k

    spans: list[tuple[int, int]] = []
    cursor = 0
    for unit in breaks:
        end = starts[unit]
        if end > cursor:
            spans.append((cursor, end))
            cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans


def _split_on_paragraph_breaks(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in spans:
        cursor = start
        for match in _PARAGRAPH_BREAK_SPLIT.finditer(text, start, end):
            if match.end() > cursor:
                out.append((cursor, match.end()))
            cursor = match.end()
        if cursor < end:
            out.append((cursor, end))
    return out


def segment_text(text: str) -> tuple[list[Segment], list[Paragraph]]:
    if not text.strip():
        return [], []

    spans = _split_on_paragraph_breaks(text, _sentence_spans(text))

    segments: list[Segment] = []
    paragraphs: list[Paragraph] = []
    paragraph_index = -1
    starts_new_paragraph = True
    pending_start: int | None = None

    for span_start, span_end in spans:
        raw = text[span_start:span_end]

        if count_words(raw) == 0:
            if not segments:
                if pending_start is None:
                    pending_start = span_start
                continue
            previous = segments[-1]
            previous.end = span_end
            previous.text = text[previous.start:previous.end]
            paragraphs[previous.paragraph].end = span_end
            if PARAGRAPH_BREAK.search(previous.text):
                starts_new_paragraph = True
            continue

        start = pending_start if pending_start is not None else span_start
        pending_start = None

        if starts_new_paragraph:
            paragraph_index += 1
            paragraphs.append(Paragraph(index=paragraph_index, start=start, end=span_end))
            starts_new_paragraph = False

        segment = Segment(
            index=len(segments),
            start=start,
            end=span_end,
            text=text[start:span_end],
            words=count_words(raw),
            paragraph=paragraph_index,
        )
        segments.append(segment)
        paragraph = paragraphs[paragraph_index]
        paragraph.segments.append(segment.index)
        paragraph.end = span_end

        if PARAGRAPH_BREAK.search(segment.text):
            starts_new_paragraph = True

    return segments, paragraphs
