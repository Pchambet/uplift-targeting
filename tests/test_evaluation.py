"""Part C machinery on a synthetic three-arm experiment with a known best policy."""

import numpy as np
import pandas as pd
import pytest

from uplift_targeting import config
from uplift_targeting.evaluation import (
    PolicyData,
    cross_select,
    cross_selected_budget,
    split_halves,
)


def synthetic_scores(n: int = 12_000, seed: int = 0) -> pd.DataFrame:
    """Only customers with x > 0 respond to the Mens e-mail; the Womens e-mail does nothing."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    arm = rng.integers(0, 3, n)
    tau = np.where(x > 0, 2.0, 0.0)
    y = 1.0 + (arm == 1) * tau + rng.normal(size=n)
    frame = {"arm": arm, "visit": y, "conversion": y, "spend": y}
    for outcome in config.OUTCOMES:
        for base in ("lgbm", "linear"):
            frame[f"tau__{outcome}__1__Oracle__{base}"] = tau
            frame[f"tau__{outcome}__2__Oracle__{base}"] = np.zeros(n)
            frame[f"tau__{outcome}__1__Noise__{base}"] = rng.normal(size=n)
            frame[f"tau__{outcome}__2__Noise__{base}"] = rng.normal(size=n)
        for a in (0, 1, 2):
            frame[f"mu__{outcome}__{a}"] = np.ones(n)
    return pd.DataFrame(frame)


def test_halves_are_balanced_within_arms():
    arm = np.repeat([0, 1, 2], [101, 100, 99])
    half = split_halves(arm)
    for a in (0, 1, 2):
        counts = np.bincount(half[arm == a], minlength=2)
        assert abs(counts[0] - counts[1]) <= 1


def test_cross_selection_finds_the_informative_ranking():
    scores = synthetic_scores()
    cs = cross_select(scores, PolicyData.from_scores(scores, "spend"))
    assert all(name.startswith("Oracle") for name in cs.chosen.values())
    action = cs.policy(0.5)
    assert np.mean(action > 0) == pytest.approx(0.5, abs=0.01)
    # The oracle e-mails exactly the responsive half, always with the Mens e-mail.
    x_positive = scores["tau__spend__1__Oracle__lgbm"].to_numpy() > 0
    assert np.mean(action[x_positive] == 1) > 0.95


def test_selection_never_looks_at_the_half_it_serves():
    scores = synthetic_scores(seed=1)
    data = PolicyData.from_scores(scores, "spend")
    cs = cross_select(scores, data)
    scrambled = scores.copy()
    serve = cs.half == 0
    scrambled.loc[serve, "spend"] = np.random.default_rng(2).permutation(
        scrambled.loc[serve, "spend"].to_numpy()
    )
    cs2 = cross_select(scrambled, PolicyData.from_scores(scrambled, "spend"))
    assert cs2.chosen[0] == cs.chosen[0]


def test_budget_choice_responds_to_cost():
    scores = synthetic_scores(seed=3)
    data = PolicyData.from_scores(scores, "spend")
    cs = cross_select(scores, data)
    rankings = {h: cs.ranking(h) for h in (0, 1)}
    cheap = cross_selected_budget(data, cs.half, rankings, ratio=0.1)
    pricey = cross_selected_budget(data, cs.half, rankings, ratio=1.5)
    never = cross_selected_budget(data, cs.half, rankings, ratio=5.0)
    # Effect is 2 for half the base: worth e-mailing them when cost/margin < 2, nobody above.
    assert np.mean(cheap > 0) >= 0.5
    assert np.mean(pricey > 0) == pytest.approx(0.5, abs=0.06)
    assert np.mean(never > 0) == 0.0
