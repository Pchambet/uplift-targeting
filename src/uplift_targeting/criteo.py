"""Part D: the same uplift tooling on 14 million rows (Criteo Uplift v2.1).

Hillstrom is small enough that sampling noise dominates. The Criteo benchmark
(Diemert et al., 2018) logs ~14M users from incrementality tests with an 85/15
treatment/control split, so it shows how the rankings behave when noise is no
longer the binding constraint.

Resource choices, made explicit because this runs on a laptop next to other
jobs: DuckDB reads the gzip CSV once into Parquet (3 GB memory cap, 3 threads);
the split is a hash of the features, so duplicate users never straddle train
and test; models are fitted on a 2M-row reservoir sample of the training half
and evaluated on the *entire* held-out half (~7M rows).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from uplift_targeting import config
from uplift_targeting.data import download
from uplift_targeting.experiment import Effect, decile_rates
from uplift_targeting.learners import DRLearner, TLearner, make_model, predict_mean
from uplift_targeting.metrics import grouped_area_interval, targeted_uplift

CRITEO_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
FEATURES = [f"f{i}" for i in range(12)]
TRAIN_ROWS = 2_000_000
N_GROUPS = 20  # random-groups CI: one pass over 7M rows instead of a bootstrap
LGBM_OVERRIDES = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": 3,
    "deterministic": True,  # same thread count + this flag = reproducible fits
    "force_row_wise": True,
}


def raw_path() -> Path:
    return config.DATA_RAW / "criteo-uplift-v2.1.csv.gz"


def parquet_path() -> Path:
    return config.DATA_INTERIM / "criteo.parquet"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'; SET threads = 3; SET enable_progress_bar = false;")
    return con


def fetch() -> Path:
    return download(config.CRITEO_URL, raw_path(), CRITEO_SHA256)


def build_parquet() -> Path:
    """One pass over the CSV: typed columns plus a deterministic 50/50 split."""
    out = parquet_path()
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    features = ", ".join(f"CAST({f} AS FLOAT) AS {f}" for f in FEATURES)
    with connect() as con:
        con.execute(
            f"""
            COPY (
              SELECT {features},
                     CAST(treatment AS TINYINT) AS treatment,
                     CAST(visit AS TINYINT) AS visit,
                     CAST(conversion AS TINYINT) AS conversion,
                     CAST(exposure AS TINYINT) AS exposure,
                     (hash({", ".join(FEATURES)}) % 2 = 0) AS is_train
              FROM read_csv('{raw_path()}', header = true)
            ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
            """
        )
    return out


def arm_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT treatment, count(*) AS n, avg(visit) AS visit, avg(conversion) AS conversion,
               avg(exposure) AS exposure
        FROM '{parquet_path()}' GROUP BY treatment ORDER BY treatment
        """
    ).df()


def average_effects(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Intent-to-treat effects from arm-level rates (binary outcomes, Neyman SE)."""
    c, t = summary.iloc[0], summary.iloc[1]
    out = {}
    for outcome in ("visit", "conversion"):
        p1, p0 = t[outcome], c[outcome]
        se = np.sqrt(p1 * (1 - p1) / t["n"] + p0 * (1 - p0) / c["n"])
        eff = Effect(float(p1 - p0), float(se))
        out[outcome] = {**eff.as_dict(), "control_rate": float(p0)}
    return out


def _lgbm(task: str):
    model = make_model("lgbm", task)
    model.set_params(**LGBM_OVERRIDES)
    return model


class _CriteoT(TLearner):
    def outcome_model(self):
        return _lgbm("classification")


class _CriteoDR(DRLearner):
    def outcome_model(self):
        return _lgbm("classification")

    def effect_model(self):
        return _lgbm("regression")


def scores_path() -> Path:
    return config.DATA_INTERIM / "criteo_test_scores.parquet"


def score() -> pd.DataFrame:
    """Fit on the training sample, score the whole held-out half, cache the scores."""
    build_parquet()
    cols = ", ".join([*FEATURES, "treatment", "visit", "conversion"])
    with connect() as con:
        train = con.execute(
            f"SELECT {cols} FROM '{parquet_path()}' WHERE is_train "
            f"USING SAMPLE reservoir({TRAIN_ROWS} ROWS) REPEATABLE ({config.SEED % 10_000})"
        ).df()
        test = con.execute(f"SELECT {cols} FROM '{parquet_path()}' WHERE NOT is_train").df()
    Xtr, ttr = train[FEATURES].to_numpy(np.float32), train["treatment"].to_numpy()
    Xte = test[FEATURES].to_numpy(np.float32)
    e = float(ttr.mean())
    out = test[["treatment", "visit", "conversion"]].copy()
    for outcome in ("visit", "conversion"):
        ytr = train[outcome].to_numpy().astype(float)
        response = _lgbm("classification").fit(Xtr[ttr == 1], ytr[ttr == 1])
        out[f"{outcome}__Response model"] = predict_mean(response, Xte)
        out[f"{outcome}__T-learner"] = _CriteoT(propensity=e).fit(Xtr, ttr, ytr).predict(Xte)
        dr = _CriteoDR(propensity=e, n_folds=3).fit(Xtr, ttr, ytr)
        out[f"{outcome}__DR-learner"] = dr.predict(Xte)
    with connect() as con:
        con.register("scores", out)
        con.execute(f"COPY scores TO '{scores_path()}' (FORMAT parquet)")
    return out


def load_scores() -> pd.DataFrame:
    with connect() as con:
        return con.execute(f"SELECT * FROM '{scores_path()}'").df()


def evaluate(scores: pd.DataFrame) -> dict:
    """Qini with random-groups CIs and uplift curves with pointwise CIs on the held-out half."""
    with connect() as con:
        summary = arm_summary(con)
    t = scores["treatment"].to_numpy()
    grid = np.linspace(0, 1, 101)
    rows, curves, deciles = [], [], []
    for outcome in ("visit", "conversion"):
        y = scores[outcome].to_numpy().astype(float)
        rankings = {
            c.split("__")[1]: scores[c].to_numpy() for c in scores if c.startswith(f"{outcome}__")
        }
        for name, ranking in rankings.items():
            iv = grouped_area_interval(ranking, y, t, N_GROUPS, seed=config.SEED)
            # Share of users sharing the most common score: a ranking that is
            # constant for most users cannot target them.
            largest_tie = np.unique(ranking, return_counts=True)[1].max() / len(ranking)
            rows.append(
                {"outcome": outcome, "model": name, **iv.as_dict(), "largest_tie": largest_tie}
            )
            table = decile_rates(ranking, y, t, seed=config.SEED)
            deciles += [{"outcome": outcome, "model": name} | r for r in table.to_dict("records")]
            gain, se = targeted_uplift(ranking, y, t, grid)
            curves += [
                {
                    "outcome": outcome,
                    "model": name,
                    "fraction": round(float(f), 2),
                    "gain_per_1000": 1000 * g,
                    "low": 1000 * (g - 1.96 * s),
                    "high": 1000 * (g + 1.96 * s),
                }
                for f, g, s in zip(grid, gain, se, strict=True)
            ]
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.RESULTS / "criteo_qini.csv", index=False)
    pd.DataFrame(curves).to_csv(config.RESULTS / "criteo_uplift_curves.csv", index=False)
    pd.DataFrame(deciles).to_csv(config.RESULTS / "criteo_deciles.csv", index=False)
    result = {
        "n_rows": int(summary["n"].sum()),
        "treatment_share": float(summary["n"].iloc[1] / summary["n"].sum()),
        "n_train_sample": TRAIN_ROWS,
        "n_test": len(scores),
        "average_effects": average_effects(summary),
    }
    (config.RESULTS / "criteo_summary.json").write_text(json.dumps(result, indent=2))
    return result


def run() -> dict:
    return evaluate(score())
