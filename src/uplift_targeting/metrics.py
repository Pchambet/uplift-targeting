"""Uplift evaluation curves (Qini, uplift) and their areas, with bootstrap CIs.

All functions accept optional per-row weights. That single design choice gives
the bootstrap for free: a Poisson(1) weight vector is a bootstrap replicate,
and the expensive part (sorting by score) is done once, not once per replicate.

Ties in the score are handled by evaluating the curve only at the end of each
tie block, so the result does not depend on the arbitrary order of tied rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curve:
    """A gain curve evaluated at every distinct score threshold.

    ``fraction`` is the share of the population targeted (0 to 1) and ``gain``
    the cumulative incremental outcome, both starting at the origin.
    """

    fraction: np.ndarray
    gain: np.ndarray


def _cumulative(
    score: np.ndarray, y: np.ndarray, t: np.ndarray, w: np.ndarray, order: np.ndarray | None
) -> tuple[np.ndarray, ...]:
    if order is None:
        order = np.argsort(-score, kind="stable")
    s, y, t, w = score[order], y[order], t[order], w[order]
    # Keep only the last index of each block of tied scores.
    block_end = np.r_[s[1:] != s[:-1], True]
    wt, wc = w * t, w * (1 - t)
    n_t = np.cumsum(wt)[block_end]
    n_c = np.cumsum(wc)[block_end]
    y_t = np.cumsum(wt * y)[block_end]
    y_c = np.cumsum(wc * y)[block_end]
    return n_t, n_c, y_t, y_c


def _with_origin(n: np.ndarray, gain: np.ndarray) -> Curve:
    total = n[-1]
    return Curve(np.r_[0.0, n / total], np.r_[0.0, gain])


def qini_curve(
    score: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray | None = None,
    order: np.ndarray | None = None,
) -> Curve:
    """Radcliffe's Qini curve: ``Y_T(k) - Y_C(k) * N_T(k) / N_C(k)``.

    It counts incremental outcomes among the treated units in the top-k,
    rescaling the control outcomes to the treated count.
    """
    w = np.ones_like(y, dtype=float) if weights is None else weights
    n_t, n_c, y_t, y_c = _cumulative(score, y, t, w, order)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(n_c > 0, y_t - y_c * n_t / n_c, y_t)
    return _with_origin(n_t + n_c, gain)


def uplift_curve(
    score: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray | None = None,
    order: np.ndarray | None = None,
) -> Curve:
    """Uplift (gain) curve: ``(Y_T/N_T - Y_C/N_C) * (N_T + N_C)`` in the top-k.

    Its height at k estimates the incremental outcome if the top-k were treated
    instead of left alone. Undefined points (an empty arm) are set to zero.
    """
    w = np.ones_like(y, dtype=float) if weights is None else weights
    n_t, n_c, y_t, y_c = _cumulative(score, y, t, w, order)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where((n_t > 0) & (n_c > 0), y_t / n_t - y_c / n_c, 0.0)
    return _with_origin(n_t + n_c, rate * (n_t + n_c))


def area_over_random(curve: Curve) -> float:
    """Area between a gain curve and the straight line to its end point.

    The straight line is the expected curve of random targeting, so a positive
    area means the ranking beats random. Units: outcome units times population
    share; divide by the population size to read it per customer.
    """
    random_line = curve.fraction * curve.gain[-1]
    return float(np.trapezoid(curve.gain - random_line, curve.fraction))


def qini_coefficient(
    score: np.ndarray, y: np.ndarray, t: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Qini area over random, per customer in the evaluated population."""
    curve = qini_curve(score, y, t, weights)
    n = len(y) if weights is None else float(np.sum(weights))
    return area_over_random(curve) / n


def auuc(
    score: np.ndarray, y: np.ndarray, t: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Area under the uplift curve over random, per customer."""
    curve = uplift_curve(score, y, t, weights)
    n = len(y) if weights is None else float(np.sum(weights))
    return area_over_random(curve) / n


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


def bootstrap_areas(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    n_boot: int,
    seed: int,
    kind: str = "qini",
    level: float = 0.95,
) -> tuple[dict[str, Interval], np.ndarray]:
    """Percentile-bootstrap CIs for the area of several rankings at once.

    Every model is evaluated on the same Poisson(1) weight vectors, so the
    returned replicate matrix (n_boot x n_models) also supports paired
    comparisons between models. The scores are held fixed: the interval covers
    evaluation-sample noise, not the variance of re-training the models.
    """
    curve_fn = qini_curve if kind == "qini" else uplift_curve
    names = list(scores)
    orders = {m: np.argsort(-scores[m], kind="stable") for m in names}
    point = {}
    for m in names:
        point[m] = area_over_random(curve_fn(scores[m], y, t, order=orders[m])) / len(y)
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        w = rng.poisson(1.0, size=len(y)).astype(float)
        total = w.sum()
        for j, m in enumerate(names):
            curve = curve_fn(scores[m], y, t, weights=w, order=orders[m])
            reps[b, j] = area_over_random(curve) / total
    alpha = (1 - level) / 2
    out = {
        m: Interval(
            point[m],
            float(np.quantile(reps[:, j], alpha)),
            float(np.quantile(reps[:, j], 1 - alpha)),
        )
        for j, m in enumerate(names)
    }
    return out, reps
