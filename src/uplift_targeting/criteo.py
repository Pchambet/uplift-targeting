"""Part D: the same uplift tooling on 14 million rows (Criteo Uplift v2.1).

Hillstrom is small enough that sampling noise dominates. The Criteo benchmark
(Diemert et al., 2018) logs ~14M users from incrementality tests with an 85/15
treatment/control split, so it shows how the rankings behave when noise is no
longer the binding constraint.

Resource choices, made explicit because this runs on a laptop next to other
jobs: DuckDB reads the gzip CSV once into Parquet (3 GB memory cap, 3 threads);
the split is an MD5 hash of the raw feature text, so duplicate users never
straddle train and test, and the split does not depend on the DuckDB version;
models are fitted on a third of the training half, chosen by the same
features-only hash (never by the outcome), and evaluated on the *entire*
held-out half (~7M rows). Fitted scores are cached and reused while the model
settings are unchanged.
"""

from __future__ import annotations

import hashlib
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
OUTCOMES = ("visit", "conversion")
TRAIN_SAMPLE_MODULUS = 3  # one training-half row in three: about 2.3M rows
DR_FOLDS = 3
N_GROUPS = 20  # random-groups CI: one pass over 7M rows instead of a bootstrap
LGBM_OVERRIDES: dict[str, object] = {
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


def scores_path() -> Path:
    return config.DATA_INTERIM / "criteo_test_scores.parquet"


def meta_path() -> Path:
    return config.DATA_INTERIM / "criteo_meta.json"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'; SET threads = 3; SET enable_progress_bar = false;")
    return con


def fetch() -> Path:
    return download(config.CRITEO_URL, raw_path(), CRITEO_SHA256)


def build_parquet(source: Path | None = None, out: Path | None = None) -> Path:
    """One pass over the CSV: typed columns plus a features-only row key.

    ``row_key`` is the low 64 bits of the MD5 of the twelve feature values as
    written in the file. Its parity splits the data 50/50; the rest of its bits
    pick the training sample. MD5 of the raw text is stable across DuckDB
    versions, and identical users always share a key. The file is written to a
    ``.part`` path and renamed only when complete, so an interrupted run never
    leaves a truncated Parquet that later runs would trust.
    """
    source = source or raw_path()
    out = out or parquet_path()
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    features = ", ".join(f"CAST({f} AS FLOAT) AS {f}" for f in FEATURES)
    flags = ", ".join(
        f"CAST({c} AS TINYINT) AS {c}" for c in ("treatment", "visit", "conversion", "exposure")
    )
    with connect() as con:
        con.execute(
            f"""
            COPY (
              SELECT {features}, {flags},
                     md5_number_lower(concat_ws(',', {", ".join(FEATURES)})) AS row_key
              FROM read_csv('{source}', header = true, all_varchar = true)
            ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)
            """
        )
    tmp.replace(out)
    return out


def arm_summary(con: duckdb.DuckDBPyConnection, path: Path | None = None) -> pd.DataFrame:
    """Rows and outcome rates per arm, indexed by the treatment flag (0 = control)."""
    return (
        con.execute(
            f"""
            SELECT treatment, count(*) AS n, avg(visit) AS visit,
                   avg(conversion) AS conversion, avg(exposure) AS exposure
            FROM '{path or parquet_path()}' GROUP BY treatment ORDER BY treatment
            """
        )
        .df()
        .set_index("treatment")
    )


def average_effects(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Intent-to-treat effects from arm-level rates (binary outcomes, Neyman SE)."""
    c, t = summary.loc[0], summary.loc[1]
    out = {}
    for outcome in OUTCOMES:
        p1, p0 = t[outcome], c[outcome]
        se = np.sqrt(p1 * (1 - p1) / t["n"] + p0 * (1 - p0) / c["n"])
        eff = Effect(float(p1 - p0), float(se))
        out[outcome] = {**eff.as_dict(), "control_rate": float(p0)}
    return out


def settings_key() -> str:
    """Fingerprint of everything that changes the fitted scores."""
    settings = {
        "lightgbm": {**config.LIGHTGBM_PARAMS, **LGBM_OVERRIDES},
        "train_sample_modulus": TRAIN_SAMPLE_MODULUS,
        "dr_folds": DR_FOLDS,
        "seed": config.SEED,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def _read_meta() -> dict:
    return json.loads(meta_path().read_text()) if meta_path().exists() else {}


def load_split(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training sample and full held-out half, in a canonical row order."""
    cols = ", ".join([*FEATURES, "treatment", "visit", "conversion"])
    src = path or parquet_path()
    with connect() as con:
        # Canonical sort: the same rows in the same order whatever the number
        # of threads that wrote or read the Parquet file.
        train = con.execute(
            f"SELECT {cols} FROM '{src}' WHERE row_key % 2 = 0 "
            f"AND (row_key // 2) % {TRAIN_SAMPLE_MODULUS} = 0 ORDER BY {cols}"
        ).df()
        test = con.execute(f"SELECT {cols} FROM '{src}' WHERE row_key % 2 = 1 ORDER BY {cols}").df()
    return train, test


def fit_scores(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Response, T-learner and DR-learner scores of the test rows, fitted on ``train``."""
    Xtr, ttr = train[FEATURES].to_numpy(np.float32), train["treatment"].to_numpy()
    Xte = test[FEATURES].to_numpy(np.float32)
    e = float(ttr.mean())
    out = test[["treatment", "visit", "conversion"]].copy()
    for outcome in OUTCOMES:
        ytr = train[outcome].to_numpy().astype(float)
        response = make_model("lgbm", "classification", params=LGBM_OVERRIDES)
        response.fit(Xtr[ttr == 1], ytr[ttr == 1])
        out[f"{outcome}__Response model"] = predict_mean(response, Xte)
        t_learner = TLearner(propensity=e, params=LGBM_OVERRIDES).fit(Xtr, ttr, ytr)
        out[f"{outcome}__T-learner"] = t_learner.predict(Xte)
        dr = DRLearner(propensity=e, params=LGBM_OVERRIDES, n_folds=DR_FOLDS)
        out[f"{outcome}__DR-learner"] = dr.fit(Xtr, ttr, ytr).predict(Xte)
    return out


def score(refit: bool = False) -> pd.DataFrame:
    """Held-out scores: read from the cache when the model settings are unchanged."""
    build_parquet()
    meta = _read_meta()
    if not refit and scores_path().exists() and meta.get("settings") == settings_key():
        with connect() as con:
            return con.execute(f"SELECT * FROM '{scores_path()}'").df()
    train, test = load_split()
    out = fit_scores(train, test)
    with connect() as con:
        con.register("scores", out)
        con.execute(f"COPY scores TO '{scores_path()}' (FORMAT parquet)")
    meta_path().write_text(json.dumps({"n_train_sample": len(train), "settings": settings_key()}))
    return out


def evaluate(scores: pd.DataFrame) -> dict:
    """Qini with random-groups CIs and uplift curves with pointwise CIs on the held-out half."""
    with connect() as con:
        summary = arm_summary(con)
    t = scores["treatment"].to_numpy()
    grid = np.linspace(0, 1, 101)
    rows, curves, deciles = [], [], []
    for outcome in OUTCOMES:
        y = scores[outcome].to_numpy().astype(float)
        rankings = {
            c.split("__")[1]: scores[c].to_numpy() for c in scores if c.startswith(f"{outcome}__")
        }
        for name, ranking in rankings.items():
            iv = grouped_area_interval(ranking, y, t, N_GROUPS, seed=config.SEED)
            # Share of users sharing the most common score: a ranking that is
            # constant for most users cannot target them.
            largest_tie = np.unique(ranking, return_counts=True)[1].max() / len(ranking)
            # Users scored below zero: who does the model believe the treatment hurts?
            neg = ranking < 0
            rows.append(
                {
                    "outcome": outcome,
                    "model": name,
                    **iv.as_dict(),
                    "largest_tie": largest_tie,
                    "negative_share": float(neg.mean()),
                    "negative_control_rate": float(y[neg & (t == 0)].mean()) if neg.any() else None,
                    "negative_uplift": float(y[neg & (t == 1)].mean() - y[neg & (t == 0)].mean())
                    if neg.any()
                    else None,
                }
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
        "treatment_share": float(summary.loc[1, "n"] / summary["n"].sum()),
        "n_train_sample": _read_meta()["n_train_sample"],
        "n_test": len(scores),
        "average_effects": average_effects(summary),
    }
    (config.RESULTS / "criteo_summary.json").write_text(json.dumps(result, indent=2))
    return result


def run(refit: bool = False) -> dict:
    return evaluate(score(refit))
