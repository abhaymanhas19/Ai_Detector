"""Logit-shift alignment of group scores onto their passage score, ported from lib/align.ts."""

import math

EPSILON = 1e-6

MAX_SHIFT = 60

ITERATIONS = 60


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _logit(value: float) -> float:
    safe = value if math.isfinite(value) else 0.5
    p = min(1 - EPSILON, max(EPSILON, safe))
    return math.log(p / (1 - p))


def _sigmoid(value: float) -> float:
    # Written so large negative inputs do not overflow math.exp.
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    z = math.exp(value)
    return z / (1 + z)


def align_to_weighted_mean(scores: list[float], weights: list[int], target: float) -> list[float]:
    """Shifts scores onto a target weighted mean without reordering them.

    The model ranks short spans well but reads too high on them in absolute
    terms, so one shift in logit space, found by bisection, re-levels a set of
    group scores onto the calibrated score of the passage they came from.
    """
    if not scores:
        return []

    goal = _clamp01(target)
    if len(scores) == 1:
        return [goal]
    if goal <= EPSILON or goal >= 1 - EPSILON:
        return [goal for _ in scores]

    logits = [_logit(s) for s in scores]
    safe_weights = [max(1, weights[i] if i < len(weights) else 1) for i in range(len(scores))]
    total_weight = sum(safe_weights)

    def mean_at(delta: float) -> float:
        return sum(_sigmoid(x + delta) * w for x, w in zip(logits, safe_weights)) / total_weight

    low, high = -MAX_SHIFT, MAX_SHIFT
    for _ in range(ITERATIONS):
        mid = (low + high) / 2
        if mean_at(mid) < goal:
            low = mid
        else:
            high = mid

    delta = (low + high) / 2
    return [_clamp01(_sigmoid(x + delta)) for x in logits]
