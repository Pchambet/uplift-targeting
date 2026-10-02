import numpy as np
import pytest

from uplift_targeting.policy import (
    dr_scores,
    ipw_scores,
    threshold_policy,
    top_share_policy,
    value,
    value_difference,
)
from uplift_targeting.synthetic import make_experiment


def test_ipw_and_dr_scores_by_hand():
    action = np.array([1, 0, 1])
    logged = np.array([1, 1, 0])
    y = np.array([2.0, 3.0, 4.0])
    p = np.full(3, 0.5)
    mu = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(ipw_scores(action, logged, y, p), [4, 0, 0])
    # unit 0: 1 + (2 - 1) / 0.5 = 3; units 1 and 2: no match, model term only.
    np.testing.assert_allclose(dr_scores(action, logged, y, p, mu), [3, 1, 3])


def test_top_share_policy_targets_the_highest_priorities():
    priority = np.array([0.1, 0.9, 0.5, 0.7])
    np.testing.assert_array_equal(top_share_policy(priority, 0.5, 1), [0, 1, 0, 1])
    np.testing.assert_array_equal(
        top_share_policy(priority, 0.5, np.array([2, 1, 2, 2])), [0, 1, 0, 2]
    )
    np.testing.assert_array_equal(top_share_policy(priority, 0.0, 1), [0, 0, 0, 0])


def test_threshold_policy_picks_best_email_only_when_profitable():
    effects = np.array([[1.0, 0.2], [0.1, 0.5], [0.05, 0.02]])
    # margin 0.4, cost 0.15: gains 0.25, 0.05, -0.13
    np.testing.assert_array_equal(threshold_policy(effects, 0.4, 0.15), [1, 2, 0])


@pytest.mark.parametrize("estimator", ["ipw", "dr_misspecified"])
def test_policy_value_estimators_are_unbiased(estimator):
    """Average estimate over many experiments equals the true policy value.

    The DR estimator is given a deliberately wrong outcome model: with known
    propensities it must stay unbiased anyway.
    """
    estimates, truths = [], []
    for rep in range(300):
        exp = make_experiment(2_000, seed=rep)
        action = (exp.X[:, 1] > 0).astype(int)  # treat customers with x1 > 0
        truths.append(np.mean(exp.mu0 + action * exp.tau))
        p = np.where(exp.t == 1, exp.propensity, 1 - exp.propensity)
        if estimator == "ipw":
            scores = ipw_scores(action, exp.t, exp.y, p)
        else:
            mu = np.column_stack([np.full(2_000, 5.0), np.full(2_000, -1.0)])
            scores = dr_scores(action, exp.t, exp.y, p, mu)
        estimates.append(value(scores).estimate)
    estimates = np.array(estimates)
    bias = estimates.mean() - np.mean(truths)
    assert abs(bias) < 3 * estimates.std(ddof=1) / np.sqrt(len(estimates))


def test_dr_with_good_model_has_smaller_variance_and_honest_ci():
    exp = make_experiment(20_000, seed=11)
    action = np.ones(20_000, dtype=int)
    p = np.full(20_000, 0.5)
    mu = np.column_stack([exp.mu0, exp.mu0 + exp.tau])
    ipw, dr = (
        value(ipw_scores(action, exp.t, exp.y, p)),
        value(dr_scores(action, exp.t, exp.y, p, mu)),
    )
    assert dr.se < 0.7 * ipw.se
    low, high = dr.ci()
    assert low < np.mean(exp.mu0 + exp.tau) < high
    nothing = np.zeros(20_000, dtype=int)
    gain = value_difference(
        dr_scores(action, exp.t, exp.y, p, mu), dr_scores(nothing, exp.t, exp.y, p, mu)
    )
    assert gain.ci()[0] < np.mean(exp.tau) < gain.ci()[1]
