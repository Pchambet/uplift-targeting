"""End-to-end checks of modelling and evaluation on the 1,500-row fixture.

The fixture is a real-data sample with buyers over-represented (100 of 1,500),
so that every fold and arm contains both outcome classes.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uplift_targeting import config, evaluation, modeling
from uplift_targeting.data import load_hillstrom

FIXTURE = Path(__file__).parent / "fixtures" / "hillstrom_sample.csv"


@pytest.fixture(scope="module")
def fast_models():
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(config.LIGHTGBM_PARAMS, "n_estimators", 15)
        mp.setitem(config.LIGHTGBM_PARAMS, "min_child_samples", 20)
        mp.setattr(config, "N_WORKERS", 1)
        yield


@pytest.fixture(scope="module")
def experiment():
    return load_hillstrom(FIXTURE)


def test_held_out_outcomes_cannot_change_held_out_scores(fast_models, experiment):
    """No leakage: scrambling the outcomes of the scored rows leaves their scores unchanged."""
    folds = modeling.outer_folds(experiment, n_folds=3)
    test = folds == 0
    before = modeling.score_fold(experiment, ~test, test)
    rng = np.random.default_rng(0)
    scrambled = experiment.outcomes.copy()
    for col in scrambled:
        values = scrambled[col].to_numpy().copy()
        values[test] = rng.permutation(values[test]) + 1.0
        scrambled[col] = values
    after = modeling.score_fold(replace(experiment, outcomes=scrambled), ~test, test)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)


def test_pipeline_writes_complete_out_of_sample_results(
    fast_models, experiment, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    scores = modeling.out_of_fold_scores(experiment, modeling.outer_folds(experiment, n_folds=2))
    assert not scores.isna().any().any()
    assert len(scores) == experiment.n
    summary = evaluation.run(scores, n_boot=10)
    for name in (
        "qini.csv",
        "uplift_curves.csv",
        "deciles.csv",
        "policy_curves.csv",
        "policy_by_cost.csv",
        "selection_candidates.csv",
        "policy_summary.json",
    ):
        assert (tmp_path / name).exists(), name
    deployed = summary["spend"]["deployable"]
    assert {"Uplift model (cross-selected)", "Response model", "Blanket Mens e-mail"} == set(
        deployed
    )
    for policy in deployed.values():
        assert 0.0 <= policy["share_emailed"] <= 1.0
        assert policy["low"] <= policy["value_per_1000"] <= policy["high"]
