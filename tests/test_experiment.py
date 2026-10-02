import numpy as np
import pandas as pd
import pytest

from uplift_targeting.experiment import (
    benjamini_hochberg,
    cuped,
    difference_in_means,
    holm,
    regression_adjusted,
    segment_effects,
    srm_test,
    standardized_mean_differences,
)


def test_difference_in_means_by_hand():
    y = np.array([3.0, 5.0, 1.0, 2.0, 3.0])
    t = np.array([1, 1, 0, 0, 0])
    eff = difference_in_means(y, t)
    # means 4 and 2; variances 2 and 1 -> se = sqrt(2/2 + 1/3)
    assert eff.estimate == pytest.approx(2.0)
    assert eff.se == pytest.approx(np.sqrt(1 + 1 / 3))


def test_lin_without_covariates_reproduces_neyman_standard_error():
    # OLS of Y on (1, T) with HC2 errors is exactly the unpooled Neyman variance.
    rng = np.random.default_rng(3)
    t = rng.integers(0, 2, 500)
    y = rng.normal(size=500) + 0.3 * t
    lin = regression_adjusted(y, t, np.empty((500, 0)))
    dim = difference_in_means(y, t)
    assert lin.estimate == pytest.approx(dim.estimate)
    assert lin.se == pytest.approx(dim.se)


def test_cuped_and_lin_shrink_the_interval_on_correlated_data():
    rng = np.random.default_rng(4)
    n, rho = 20_000, 0.7
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n) + 0.05 * t
    dim, cup = difference_in_means(y, t), cuped(y, t, x)
    lin = regression_adjusted(y, t, x[:, None])
    # Variance should fall by a factor of about 1 - rho^2 = 0.51.
    assert (cup.se / dim.se) ** 2 == pytest.approx(1 - rho**2, abs=0.03)
    assert lin.se == pytest.approx(cup.se, rel=0.02)
    assert abs(cup.estimate - 0.05) < 3 * cup.se


def test_srm_detects_a_broken_split():
    balanced = np.repeat([0, 1, 2], 1000)
    assert srm_test(balanced, {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})["p_value"] == pytest.approx(1.0)
    broken = np.r_[np.zeros(1100), np.ones(1000), np.full(900, 2)].astype(int)
    assert srm_test(broken, {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})["p_value"] < 1e-4


def test_standardized_mean_difference_by_hand():
    X = pd.DataFrame({"a": [1.0, 3.0, 0.0, 2.0]})
    t = np.array([1, 1, 0, 0])
    # means 2 and 1, both variances 2 -> SMD = 1 / sqrt(2)
    assert standardized_mean_differences(X, t)["a"] == pytest.approx(1 / np.sqrt(2))


def test_multiple_testing_adjustments_by_hand():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_segment_heterogeneity_test():
    rng = np.random.default_rng(5)
    n = 40_000
    seg = pd.Series(rng.choice(["a", "b"], n))
    t = rng.integers(0, 2, n)
    y_same = rng.normal(size=n) + 0.1 * t
    y_diff = rng.normal(size=n) + np.where(seg == "a", 0.3, -0.1) * t
    _, same = segment_effects(y_same, t, seg)
    table, diff = segment_effects(y_diff, t, seg)
    assert same["p_value"] > 0.001
    assert diff["p_value"] < 1e-6
    assert list(table["level"]) == ["a", "b"]
