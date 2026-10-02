"""Synthetic randomised experiments with a known individual treatment effect.

Used by the tests (does a learner recover a known CATE ranking? are the policy
value estimators unbiased?) and by the simulation check in the pipeline. The
design mimics the hard part of real uplift data: the baseline outcome varies
much more across customers than the treatment effect does, and the customers
most likely to respond are not the ones most moved by the treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticExperiment:
    X: np.ndarray
    t: np.ndarray
    y: np.ndarray
    tau: np.ndarray  # true individual effect E[Y(1) - Y(0) | X]
    mu0: np.ndarray  # true baseline E[Y(0) | X]
    propensity: float


def make_experiment(
    n: int, seed: int, propensity: float = 0.5, noise: float = 1.0
) -> SyntheticExperiment:
    """Continuous-outcome experiment with heterogeneous, partly negative effects.

    ``mu0`` depends mostly on x0 (who would buy anyway) while ``tau`` depends on
    x1 and x2 (who is persuadable), so a response model ranks customers by the
    wrong variable.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    mu0 = 2.0 + 1.5 * X[:, 0] + 0.5 * X[:, 3]
    tau = 0.3 + 0.8 * np.tanh(X[:, 1]) + 0.4 * (X[:, 2] > 0)
    t = rng.binomial(1, propensity, size=n)
    y = mu0 + t * tau + noise * rng.normal(size=n)
    return SyntheticExperiment(X, t, y, tau, mu0, propensity)
