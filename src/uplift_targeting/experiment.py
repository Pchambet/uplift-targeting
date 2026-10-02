"""Experiment readout: randomisation checks, average effects, segment effects.

The order mirrors how a careful analyst reads an A/B/n test: first make sure
the randomisation worked (sample ratio, covariate balance), then estimate the
average effects with honest intervals, and only then look at pre-declared
segments with a multiple-testing correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

Z95 = float(stats.norm.ppf(0.975))


@dataclass(frozen=True)
class Effect:
    """A treatment-effect estimate with a normal-approximation 95% CI."""

    estimate: float
    se: float

    @property
    def low(self) -> float:
        return self.estimate - Z95 * self.se

    @property
    def high(self) -> float:
        return self.estimate + Z95 * self.se

    @property
    def p_value(self) -> float:
        if self.se == 0:
            return 0.0 if self.estimate != 0 else 1.0
        return float(2 * stats.norm.sf(abs(self.estimate / self.se)))

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "low": self.low, "high": self.high, "p_value": self.p_value}


def srm_test(arm: np.ndarray, expected_share: dict[int, float]) -> dict[str, object]:
    """Chi-square test of the observed arm counts against the design shares.

    A small p-value (sample ratio mismatch) means the assignment or the logging
    is broken, and no effect estimate should be trusted until it is explained.
    """
    labels = sorted(expected_share)
    observed = np.array([(arm == a).sum() for a in labels], dtype=float)
    expected = np.array([expected_share[a] for a in labels]) * observed.sum()
    chi2, p = stats.chisquare(observed, expected)
    return {
        "counts": {int(a): int(c) for a, c in zip(labels, observed, strict=True)},
        "chi2": float(chi2),
        "p_value": float(p),
    }


def standardized_mean_differences(X: pd.DataFrame, t: np.ndarray) -> pd.Series:
    """SMD of each covariate between treated (t=1) and control (t=0).

    |SMD| < 0.1 is the usual balance threshold; under randomisation with tens of
    thousands of units, values are expected around 0.01.
    """
    x1, x0 = X[t == 1], X[t == 0]
    pooled_sd = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
    return (x1.mean() - x0.mean()) / pooled_sd.replace(0, np.nan)


def difference_in_means(y: np.ndarray, t: np.ndarray) -> Effect:
    """Neyman estimator with the conservative unpooled variance."""
    y1, y0 = y[t == 1], y[t == 0]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return Effect(float(y1.mean() - y0.mean()), float(se))


def cuped(y: np.ndarray, t: np.ndarray, x: np.ndarray) -> Effect:
    """CUPED (Deng et al., 2013): subtract the part of Y explained by a pre-period metric.

    ``theta = cov(Y, X) / var(X)`` is estimated on the pooled sample. Because X
    is measured before randomisation, the adjusted difference in means stays
    unbiased, and its variance shrinks by a factor of about ``1 - corr(Y, X)^2``.
    """
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return difference_in_means(y - theta * (x - x.mean()), t)


def regression_adjusted(y: np.ndarray, t: np.ndarray, X: np.ndarray) -> Effect:
    """Lin (2013) estimator: OLS of Y on T, centred X and T x centred X, HC2 errors.

    The full interaction makes the adjustment unable to hurt asymptotic
    precision, even when the linear model is wrong, which is what makes it safe
    to use by default in an experiment readout.
    """
    Xc = X - X.mean(axis=0)
    design = np.column_stack([np.ones(len(y)), t, Xc, t[:, None] * Xc])
    q, r = np.linalg.qr(design)
    beta = np.linalg.solve(r, q.T @ y)
    resid = y - design @ beta
    leverage = np.sum(q**2, axis=1)
    r_inv = np.linalg.inv(r)
    bread = r_inv @ r_inv.T  # (X'X)^-1
    meat = (design * (resid**2 / (1 - leverage))[:, None]).T @ design
    cov = bread @ meat @ bread
    return Effect(float(beta[1]), float(np.sqrt(cov[1, 1])))


def segment_effects(
    y: np.ndarray, t: np.ndarray, segment: pd.Series
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-level difference in means plus a Wald test that all levels share one effect.

    The heterogeneity test is the right object for multiple-testing control:
    one p-value per pre-declared dimension, instead of hunting through levels.
    """
    rows = []
    for level in sorted(segment.unique()):
        mask = (segment == level).to_numpy()
        eff = difference_in_means(y[mask], t[mask])
        rows.append({"level": level, "n": int(mask.sum()), **eff.as_dict()})
    table = pd.DataFrame(rows)
    w = 1 / table["se"] ** 2
    pooled = np.sum(w * table["estimate"]) / np.sum(w)
    chi2 = float(np.sum(w * (table["estimate"] - pooled) ** 2))
    df = len(table) - 1
    return table, {"chi2": chi2, "df": df, "p_value": float(stats.chi2.sf(chi2, df))}


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate control)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out
