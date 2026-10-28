"""Result shapes, mirroring lib/types.ts. Field names are snake_case; `to_dict` emits camelCase to match the browser's JSON."""

from dataclasses import asdict, dataclass, field

from .constants import ScanSettings


@dataclass
class Segment:
    index: int
    start: int
    end: int
    text: str
    words: int
    paragraph: int


@dataclass
class Paragraph:
    index: int
    start: int
    end: int
    segments: list[int] = field(default_factory=list)


@dataclass
class Window:
    index: int
    segments: list[int]
    text: str
    tokens: int
    words: int
    start: int
    end: int


@dataclass
class SentenceGroup(Window):
    passage: int = 0
    sentences: int = 0


@dataclass
class PassageScore:
    index: int
    start: int
    end: int
    score: float
    band: str
    words: int
    tokens: int
    low_confidence: bool


@dataclass
class SentenceGroupScore:
    index: int
    passage: int
    start: int
    end: int
    score: float
    raw_score: float
    band: str
    words: int
    tokens: int
    sentences: int


@dataclass
class Composition:
    human: int
    mixed: int
    ai: int
    human_words: int
    mixed_words: int
    ai_words: int
    total_words: int


@dataclass
class Confidence:
    level: str
    reasons: list[str]


@dataclass
class TextStats:
    characters: int
    words: int
    sentences: int
    paragraphs: int
    tokens: int
    reading_minutes: float


@dataclass
class DetectionResult:
    depth: str
    document_score: float
    peak_score: float
    verdict: str
    confidence: Confidence
    composition: Composition
    passages: list[PassageScore]
    groups: list[SentenceGroupScore]
    stats: TextStats
    window_count: int
    cached_window_count: int
    settings: ScanSettings
    dtype: str

    def to_dict(self) -> dict:
        return _camel(asdict(self))


def _camel_key(key: str) -> str:
    head, *rest = key.split('_')
    return head + ''.join(part.title() for part in rest)


def _camel(value):
    if isinstance(value, dict):
        return {_camel_key(k): _camel(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_camel(v) for v in value]
    return value
