import numpy as np
import pytest

from uplift_targeting.metrics import (
    area_over_random,
    auuc,
    bootstrap_areas,
    grouped_area_interval,
    qini_coefficient,
    qini_curve,
    targeted_uplift,
    uplift_curve,
)

# Four customers, ranked by score. Worked by hand in the comments of each test.
SCORE = np.array([0.9, 0.8, 0.7, 0.6])
T = np.array([1, 0, 1, 0])
Y = np.array([1.0, 0.0, 0.0, 1.0])


def test_qini_curve_matches_hand_computation():
    # k=1: one treated buyer, no control yet -> 1
    # k=2: 1 - 0 * 1/1 = 1;  k=3: 1 - 0 * 2/1 = 1;  k=4: 1 - 1 * 2/2 = 0
    curve = qini_curve(SCORE, Y, T)
    np.testing.assert_allclose(curve.fraction, [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(curve.gain, [0, 1, 1, 1, 0])
    # Random line ends at 0, so the area is the trapezoid sum 0.125 + 0.25 + 0.25 + 0.125.
    assert area_over_random(curve) == pytest.approx(0.75)
    assert qini_coefficient(SCORE, Y, T) == pytest.approx(0.75 / 4)


def test_uplift_curve_matches_hand_computation():
    # k=1: control empty -> 0;  k=2: (1/1 - 0/1) * 2 = 2;  k=3: (1/2 - 0) * 3 = 1.5;
    # k=4: (1/2 - 1/2) * 4 = 0.  Area = 0 + 0.25 + 0.4375 + 0.1875 = 0.875.
    curve = uplift_curve(SCORE, Y, T)
    np.testing.assert_allclose(curve.gain, [0, 0, 2, 1.5, 0])
    assert auuc(SCORE, Y, T) == pytest.approx(0.875 / 4)


def test_ties_are_order_invariant():
    score = np.array([1.0, 1.0, 0.0, 0.0])
    t = np.array([1, 0, 1, 0])
    y = np.array([1.0, 0.0, 0.0, 0.0])
    a = qini_curve(score, y, t)
    perm = np.array([1, 0, 3, 2])
    b = qini_curve(score[perm], y[perm], t[perm])
    np.testing.assert_allclose(a.fraction, [0, 0.5, 1])
    np.testing.assert_allclose(a.gain, b.gain)


def test_integer_weights_equal_duplicated_rows():
    rng = np.random.default_rng(0)
    n = 200
    score, t, y = rng.normal(size=n), rng.integers(0, 2, n), rng.integers(0, 2, n).astype(float)
    w = rng.integers(0, 3, n)
    rep = np.repeat(np.arange(n), w)
    assert qini_coefficient(score, y, t, w.astype(float)) == pytest.approx(
        qini_coefficient(score[rep], y[rep], t[rep])
    )


def test_oracle_ranking_beats_random_and_bootstrap_brackets_estimate():
    rng = np.random.default_rng(1)
    n = 20_000
    tau = rng.uniform(0, 0.2, n)
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.1 + t * tau).astype(float)
    scores = {"oracle": tau, "noise": rng.normal(size=n)}
    intervals, reps = bootstrap_areas(scores, y, t, n_boot=50, seed=0)
    assert intervals["oracle"].low > 0
    assert intervals["noise"].low < 0 < intervals["noise"].high
    assert reps.shape == (50, 2)
    for iv in intervals.values():
        assert iv.low <= iv.estimate <= iv.high


def test_targeted_uplift_matches_uplift_curve_and_neyman_se():
    rng = np.random.default_rng(2)
    n = 1_000
    score, t = rng.normal(size=n), rng.integers(0, 2, n)
    y = rng.binomial(1, 0.2 + 0.1 * t).astype(float)
    gain, se = targeted_uplift(score, y, t, np.array([0.5, 1.0]))
    curve = uplift_curve(score, y, t)
    assert gain[1] == pytest.approx(curve.gain[-1] / n)
    y1, y0 = y[t == 1], y[t == 0]
    expected_se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    assert se[1] == pytest.approx(expected_se)
    top = np.argsort(-score)[:500]
    yt, tt = y[top], t[top]
    assert gain[0] == pytest.approx(0.5 * (yt[tt == 1].mean() - yt[tt == 0].mean()))


def test_random_groups_interval_agrees_with_bootstrap():
    rng = np.random.default_rng(5)
    n = 60_000
    tau = rng.uniform(0, 0.2, n)
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.1 + t * tau).astype(float)
    score = tau + rng.normal(scale=0.05, size=n)
    boot, _ = bootstrap_areas({"m": score}, y, t, n_boot=200, seed=1)
    grouped = grouped_area_interval(score, y, t, n_groups=20, seed=1)
    assert grouped.estimate == pytest.approx(boot["m"].estimate)
    width_ratio = (grouped.high - grouped.low) / (boot["m"].high - boot["m"].low)
    assert 0.6 < width_ratio < 1.6


def test_targeted_uplift_ignores_file_order_inside_ties():
    # One big tie block whose rows are sorted by arm: cutting inside it by
    # position would compare treated-only rows; block-end evaluation does not.
    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.zeros(8)
    gain, _ = targeted_uplift(score, y, t, np.array([0.25, 0.5, 1.0]))
    np.testing.assert_allclose(gain, [0.0, 0.0, 0.0])


def test_unknown_curve_kind_is_an_error():
    from uplift_targeting.metrics import bootstrap_areas

    y, t = np.array([1.0, 0.0, 1.0, 0.0]), np.array([1, 0, 1, 0])
    with pytest.raises(ValueError, match="Unknown curve kind"):
        bootstrap_areas({"m": np.arange(4.0)}, y, t, n_boot=2, seed=0, kind="auuc")
