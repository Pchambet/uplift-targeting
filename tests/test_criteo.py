"""The DuckDB path of Part D on a tiny gzipped CSV in the Criteo format."""

import gzip

import numpy as np
import pandas as pd
import pytest

from uplift_targeting import criteo


@pytest.fixture(scope="module")
def csv_and_parquet(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 300
    features = rng.normal(size=(n, 12)).round(6)
    features[150:200] = features[100:150]  # 50 users logged twice with the same features
    raw = pd.DataFrame(features, columns=criteo.FEATURES)
    raw["treatment"] = rng.binomial(1, 0.85, n)
    raw["conversion"] = rng.binomial(1, 0.05, n)
    raw["visit"] = np.maximum(raw["conversion"], rng.binomial(1, 0.2, n))
    raw["exposure"] = raw["treatment"] * rng.binomial(1, 0.3, n)
    folder = tmp_path_factory.mktemp("criteo")
    source = folder / "criteo.csv.gz"
    with gzip.open(source, "wt") as handle:
        raw.to_csv(handle, index=False)
    out = criteo.build_parquet(source, folder / "criteo.parquet")
    return raw, out


def test_parquet_keeps_every_row_with_compact_types(csv_and_parquet):
    raw, path = csv_and_parquet
    with criteo.connect() as con:
        types = {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()}
        n = con.execute(f"SELECT count(*) FROM '{path}'").fetchone()[0]
    assert n == len(raw)
    assert types["treatment"] == "TINYINT" and types["f0"] == "FLOAT"
    assert not path.with_suffix(".parquet.part").exists()


def test_arm_rates_and_effects_match_hand_computation(csv_and_parquet):
    raw, path = csv_and_parquet
    with criteo.connect() as con:
        summary = criteo.arm_summary(con, path)
    effects = criteo.average_effects(summary)
    for outcome in criteo.OUTCOMES:
        p1 = raw.loc[raw["treatment"] == 1, outcome].mean()
        p0 = raw.loc[raw["treatment"] == 0, outcome].mean()
        assert effects[outcome]["estimate"] == pytest.approx(p1 - p0)
        assert effects[outcome]["control_rate"] == pytest.approx(p0)
    assert summary.loc[1, "n"] == (raw["treatment"] == 1).sum()


def test_split_depends_on_features_only(csv_and_parquet):
    raw, path = csv_and_parquet
    with criteo.connect() as con:
        keyed = con.execute(f"SELECT f0, f1, row_key FROM '{path}'").df()
    first = keyed.groupby(["f0", "f1"])["row_key"].nunique()
    assert (first == 1).all()  # duplicated users share a key, so they share a side
    train, test = criteo.load_split(path)
    assert set(map(tuple, train[criteo.FEATURES].to_numpy())).isdisjoint(
        map(tuple, test[criteo.FEATURES].to_numpy())
    )
    assert 0.3 < len(test) / len(raw) < 0.7
    assert len(train) < len(raw) - len(test)  # a sample of the training half
