"""ONNX classifier: the same model files the browser downloads, run with onnxruntime.

Replaces the `@huggingface/transformers` text-classification pipeline in
detector.worker.ts: tokenize with padding and truncation to 512, run the
model, softmax the logits, read the `ai` probability.
"""

import json
from functools import cached_property

import numpy as np

from .constants import AI_LABEL, DEFAULT_DTYPE, MODEL_ID, ONNX_FILES

MAX_LENGTH = 512


class OnnxClassifier:
    def __init__(self, dtype: str = DEFAULT_DTYPE, model_id: str = MODEL_ID, cache_dir: str | None = None):
        if dtype not in ONNX_FILES:
            raise ValueError(f'dtype must be one of {sorted(ONNX_FILES)}, got {dtype!r}')
        self.dtype = dtype
        self.model_id = model_id
        self.cache_dir = cache_dir

    def _download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.model_id, filename, cache_dir=self.cache_dir)

    @cached_property
    def _session(self):
        import onnxruntime as ort

        return ort.InferenceSession(self._download(ONNX_FILES[self.dtype]), providers=['CPUExecutionProvider'])

    @cached_property
    def _ai_index(self) -> int:
        with open(self._download('config.json'), encoding='utf-8') as handle:
            id2label = json.load(handle)['id2label']
        for index, label in id2label.items():
            if label.lower() == AI_LABEL:
                return int(index)
        raise ValueError(f'{self.model_id} has no {AI_LABEL!r} label: {id2label}')

    def _tokenizer(self, batch: bool):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(self._download('tokenizer.json'))
        if batch:
            tokenizer.enable_truncation(max_length=MAX_LENGTH)
            tokenizer.enable_padding(pad_id=tokenizer.token_to_id('<pad>'), pad_token='<pad>')
        else:
            tokenizer.no_truncation()
            tokenizer.no_padding()
        return tokenizer

    @cached_property
    def _batch_tokenizer(self):
        return self._tokenizer(batch=True)

    @cached_property
    def _count_tokenizer(self):
        return self._tokenizer(batch=False)

    def load(self) -> None:
        """Downloads (or reads from the Hugging Face cache) and initialises everything up front."""
        _ = self._session, self._ai_index, self._batch_tokenizer, self._count_tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._count_tokenizer.encode(text, add_special_tokens=False).ids)

    def score(self, texts: list[str]) -> list[float]:
        """Probability of the `ai` label for each text."""
        if not texts:
            return []
        encodings = self._batch_tokenizer.encode_batch(texts)
        feeds = {
            'input_ids': np.array([e.ids for e in encodings], dtype=np.int64),
            'attention_mask': np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        wanted = {i.name for i in self._session.get_inputs()}
        logits = self._session.run(None, {k: v for k, v in feeds.items() if k in wanted})[0].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, self._ai_index].tolist()
