"""python -m apps.helpersapps.ai_detection FILE [--depth fast|full] [--dtype q8|fp16|fp32] [--json]

Reads stdin when FILE is '-'.
"""

import argparse
import json
import sys

from .constants import ONNX_FILES, VERDICT_LABELS
from .detector import AiDetector


def main() -> int:
    parser = argparse.ArgumentParser(description='Score how AI-like a text reads.')
    parser.add_argument('file', help="text file to scan, or '-' for stdin")
    parser.add_argument('--depth', choices=['fast', 'full'], default='full')
    parser.add_argument('--dtype', choices=sorted(ONNX_FILES), default='q8')
    parser.add_argument('--json', action='store_true', help='print the full result as JSON')
    args = parser.parse_args()

    if args.file == '-':
        text = sys.stdin.read()
    else:
        with open(args.file, encoding='utf-8') as handle:
            text = handle.read()

    result = AiDetector(dtype=args.dtype).scan(text, depth=args.depth)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f'Score:      {result.document_score:.1%}  ({VERDICT_LABELS[result.verdict]})')
    print(f'Peak:       {result.peak_score:.1%}')
    print(f'Confidence: {result.confidence.level}')
    for reason in result.confidence.reasons:
        print(f'  - {reason}')
    c = result.composition
    print(f'Breakdown:  human {c.human}%  mixed {c.mixed}%  ai {c.ai}%')
    print(f'Words: {result.stats.words}  sentences: {result.stats.sentences}  windows: {result.window_count}')
    for passage in result.passages:
        snippet = text[passage.start:passage.end].strip().replace('\n', ' ')
        print(f'  [{passage.score:.0%} {passage.band:>5}] {snippet[:90]}{"…" if len(snippet) > 90 else ""}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
