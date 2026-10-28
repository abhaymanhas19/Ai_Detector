"""Constants shared with the browser detector (ailyze-nextjs, components/ai-detection/lib/types.ts)."""

from dataclasses import dataclass

MODEL_ID = 'onnx-community/tmr-ai-text-detector-ONNX'

AI_LABEL = 'ai'

MAX_WINDOW_TOKENS = 500

MIN_GROUP_TOKENS = 64

DEFAULT_LOW_CONFIDENCE_WORDS = 60

MIN_WORDS_FOR_VERDICT = 60

MIN_PASSAGES_FOR_SPREAD = 4

# The browser's two tiers: 'fast' is q8 on WASM, 'best' is fp16 on WebGPU.
# 'fp32' has no browser equivalent; it is the unquantised export.
ONNX_FILES = {
    'q8': 'onnx/model_quantized.onnx',
    'fp16': 'onnx/model_fp16.onnx',
    'fp32': 'onnx/model.onnx',
}

DEFAULT_DTYPE = 'q8'


@dataclass(frozen=True)
class ScanSettings:
    human_threshold: float = 0.25
    ai_threshold: float = 0.75
    low_confidence_words: int = DEFAULT_LOW_CONFIDENCE_WORDS


VERDICT_LABELS = {
    'very-likely-human': 'Very likely human-written',
    'probably-human': 'Probably human-written',
    'inconclusive': 'Inconclusive',
    'probably-ai': 'Probably AI-generated',
    'very-likely-ai': 'Very likely AI-generated',
    'insufficient-text': 'Not enough text for a reliable result',
}

HARD_WORD_LIMIT = 20000
