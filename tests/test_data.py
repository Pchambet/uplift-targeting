from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_targeting import config
from uplift_targeting.data import FEATURE_COLUMNS, build_experiment, load_hillstrom, sha256

FIXTURE = Path(__file__).parent / "fixtures" / "hillstrom_sample.csv"


@pytest.fixture(scope="module")
def experiment():
    return load_hillstrom(FIXTURE)


def test_features_are_pre_treatment_only(experiment):
    assert tuple(experiment.X.columns) == FEATURE_COLUMNS
    for outcome in ("visit", "conversion", "spend", "segment"):
        assert outcome not in experiment.X.columns
    assert experiment.X.notna().all().all()


def test_arm_coding_matches_raw_labels(experiment):
    raw = pd.read_csv(FIXTURE)
    expected = raw["segment"].map(config.ARMS).to_numpy()
    np.testing.assert_array_equal(experiment.arm, expected)
    assert set(np.unique(experiment.arm)) == {0, 1, 2}


def test_derived_segments(experiment):
    raw = experiment.raw
    both = (raw["mens"] == 1) & (raw["womens"] == 1)
    assert (raw.loc[both, "purchase_history"] == "Both lines").all()
    assert raw["history_tier"].between(1, 7).all()
    assert np.allclose(experiment.X["log_history"], np.log1p(raw["history"]))


def test_unknown_arm_is_rejected():
    raw = pd.read_csv(FIXTURE).head(3).assign(segment="Mystery E-Mail")
    with pytest.raises(ValueError, match="Unknown arms"):
        build_experiment(raw)


def test_sha256_is_stable(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"abc")
    assert sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
