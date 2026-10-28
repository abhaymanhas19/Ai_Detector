# AI detection (Python port)

A port of the browser AI-text detector in `ailyze-nextjs`
(`src/components/ai-detection/`, which runs in a Web Worker and makes no API calls). It uses
the same model files (`onnx-community/tmr-ai-text-detector-ONNX`, a RoBERTa classifier) and
the same pipeline: sentence segmentation → balanced windows of at most 500 tokens → one
`ai` probability per window → word-weighted document score, verdict, confidence and
human/mixed/AI breakdown. A `full` scan also scores sentence groups and shifts them onto
their passage's score.

Nothing in the Django app imports it yet: no URL, view or Celery task. The dependencies
are in this folder's `requirements.txt`, not the main one.

```python
from apps.helpersapps.ai_detection import AiDetector

detector = AiDetector()            # one per process; the model downloads on first scan (~126 MB)
result = detector.scan(text)       # depth='full' by default; 'fast' skips passages and groups
result.document_score, result.verdict, result.confidence.level
result.to_dict()                   # camelCase, same shape as the browser's DetectionResult
```

```bash
python -m apps.helpersapps.ai_detection essay.txt [--depth fast|full] [--dtype q8|fp16|fp32] [--json]
```

## Parity with the browser

Checked against the TypeScript pipeline run under Node with `@huggingface/transformers`
4.2.0 on five documents (1 to 5 windows, up to 1,720 words). Verdict, confidence,
breakdown, stats, passage and group boundaries and word counts were identical, and scores
were within 2e-7. Sentence splitting and word counting matched `Intl.Segmenter` on 6,000
fuzzed strings except one: an underscore followed by a combining accent.

What can still differ:

- **q8 scores depend on batching.** The q8 export is dynamically quantised, so a window's
  score moves with whatever it is batched with, by up to 0.18 on the sample texts. q8 is
  scored in pairs, in the browser's order, to reproduce its numbers. The browser itself
  pairs only windows that aren't already cached, so after an edit a window it has not
  rescored can still change score.
- **Tiers disagree.** The `fp16` export (the browser's WebGPU tier) does not score the same
  as q8. On the 1,720-word mixed sample q8 gave 63.8% (probably AI) and fp16 gave 52.9%
  (inconclusive).
- **Chinese, Japanese and Thai word counts are approximate.** ICU splits these with a
  dictionary, which is not reproduced here. Word counts drive the 60-word verdict floor
  and how much each passage counts toward the document score.
- **Offsets are Python string indices** (code points). The browser's are UTF-16 code units,
  so they differ after an emoji or any other character outside the BMP.

The browser's calibration (`lib/calibration.ts`) applies here too: on its own test set,
typical AI prose scored 94–99% and human writing about 9%, but AI text written in a
personal voice scored about 3%. The score shows how formulaic a text reads. It does not
prove who wrote it.
