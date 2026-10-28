"""Python port of the browser AI-text detector (ailyze-nextjs, components/ai-detection).

    from apps.helpersapps.ai_detection import AiDetector
    result = AiDetector().scan(text)          # downloads the model on first use
    result.document_score, result.verdict, result.to_dict()
"""

from .constants import MODEL_ID, VERDICT_LABELS, ScanSettings
from .detector import AiDetector, EmptyText, TextTooLong
from .results import DetectionResult

__all__ = ['MODEL_ID', 'VERDICT_LABELS', 'AiDetector', 'DetectionResult', 'EmptyText', 'ScanSettings', 'TextTooLong']
