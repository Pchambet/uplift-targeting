"""Targeting policies and their off-policy value on randomised data.

A policy maps each customer to an action (0 = no e-mail, 1 = Mens, 2 = Womens).
Because the experiment randomised actions with known probabilities, the value
of *any* policy can be estimated from the logged data without deploying it:

* IPW (Horvitz-Thompson): average ``1{A = pi(X)} Y / p(A)``. Unbiased, noisy.
* Doubly robust (AIPW): the outcome model's prediction for the policy's action
  plus an IPW correction of its residual. Still unbiased with known
  propensities, and lower-variance when the outcome model is decent.

Both estimators are means of per-customer scores, so a standard error follows
from the score's standard deviation, and the *difference* between two policies
evaluated on the same customers gets a paired standard error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PolicyValue:
    estimate: float
    se: float

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        z = stats.norm.ppf(0.5 + level / 2)
        return self.estimate - z * self.se, self.estimate + z * self.se


def _mean_se(scores: np.ndarray) -> PolicyValue:
    return PolicyValue(float(np.mean(scores)), float(np.std(scores, ddof=1) / np.sqrt(len(scores))))


def ipw_scores(
    action: np.ndarray, logged: np.ndarray, y: np.ndarray, propensity: np.ndarray
) -> np.ndarray:
    """Per-customer IPW scores; ``propensity[i]`` is P(logged action | X_i)."""
    return (action == logged) * y / propensity


def dr_scores(
    action: np.ndarray,
    logged: np.ndarray,
    y: np.ndarray,
    propensity: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Per-customer doubly robust scores.

    ``mu`` has shape (n, n_actions) with cross-fitted predictions of E[Y | X, a]
    for every action; it must not have been fitted on customer i itself.
    """
    rows = np.arange(len(y))
    mu_pi = mu[rows, action]
    return mu_pi + (action == logged) * (y - mu_pi) / propensity


def value(scores: np.ndarray) -> PolicyValue:
    return _mean_se(scores)


def value_difference(scores_a: np.ndarray, scores_b: np.ndarray) -> PolicyValue:
    """Paired estimate of V(a) - V(b) on the same customers."""
    return _mean_se(scores_a - scores_b)


def top_share_policy(
    priority: np.ndarray, share: float, action_if_targeted: np.ndarray | int, seed: int = 0
) -> np.ndarray:
    """Target the ``share`` of customers with the highest priority.

    Untargeted customers get action 0. ``action_if_targeted`` can be a constant
    (always send the same e-mail) or a per-customer array (send the e-mail with
    the larger predicted effect). Ties at the cut-off are broken by a seeded
    random order, never by file order, which may be sorted by treatment arm.
    """
    n = len(priority)
    k = round(share * n)
    chosen = np.zeros(n, dtype=bool)
    if k > 0:
        tiebreak = np.random.default_rng(seed).permutation(n)
        chosen[np.lexsort((tiebreak, -priority))[:k]] = True
    targeted = np.broadcast_to(np.asarray(action_if_targeted), (n,))
    return np.where(chosen, targeted, 0).astype(np.int64)
