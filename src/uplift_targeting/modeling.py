"""Part B: out-of-fold scores for every model, the input of every evaluation.

The experiment is split into ``N_OUTER_FOLDS`` folds. For each fold, every
model is trained on the other folds only and scores the held-out customers.
After the loop every customer carries a score from models that never saw
them, so Qini curves and policy values can use all 64,000 customers while
staying out-of-sample.

What is scored, per customer:

* ``tau__{outcome}__{arm}__{learner}__{base}``: predicted effect of e-mail
  ``arm`` (1 = Mens, 2 = Womens) versus no e-mail.
* ``response``: the classic response model, P(conversion | X, e-mailed),
  trained on customers who received the Mens e-mail.
* ``mu__{outcome}__{arm}``: E[outcome | X, arm] for arms 0, 1, 2, used as the
  outcome model of the doubly robust policy-value estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from uplift_targeting import config
from uplift_targeting.data import Experiment
from uplift_targeting.learners import LEARNERS, make_learner, make_model, predict_mean

BASES = ("lgbm", "linear")


def outer_folds(exp: Experiment, n_folds: int = config.N_OUTER_FOLDS) -> np.ndarray:
    """Fold id per customer, stratified on arm x conversion.

    Conversions are rare (under 1%), so stratifying keeps the number of buyers
    per fold and per arm stable.
    """
    strata = exp.arm * 2 + exp.outcomes["conversion"].to_numpy().astype(int)
    folds = np.empty(exp.n, dtype=np.int64)
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=config.SEED)
    for k, (_, test_idx) in enumerate(splitter.split(np.zeros(exp.n), strata)):
        folds[test_idx] = k
    return folds


def _task(outcome: str) -> str:
    return "regression" if outcome == "spend" else "classification"


def score_fold(exp: Experiment, train: np.ndarray, test: np.ndarray) -> dict[str, np.ndarray]:
    """All scores for the ``test`` rows, from models fitted on the ``train`` rows."""
    X = exp.X.to_numpy()
    out: dict[str, np.ndarray] = {}
    for outcome in config.OUTCOMES:
        y = exp.outcomes[outcome].to_numpy()
        for arm in (1, 2):
            fit = train & np.isin(exp.arm, [0, arm])
            t = (exp.arm[fit] == arm).astype(int)
            share = float(t.mean())  # within-pair assignment probability, ~0.5 by design
            for name in LEARNERS:
                for base in BASES:
                    learner = make_learner(name, base, _task(outcome), share)
                    learner.fit(X[fit], t, y[fit])
                    out[f"tau__{outcome}__{arm}__{name}__{base}"] = learner.predict(X[test])
        for arm in (0, 1, 2):
            fit = train & (exp.arm == arm)
            model = make_model("lgbm", _task(outcome)).fit(X[fit], y[fit])
            out[f"mu__{outcome}__{arm}"] = predict_mean(model, X[test])
    fit = train & (exp.arm == 1)
    response = make_model("lgbm", "classification").fit(
        X[fit], exp.outcomes["conversion"].to_numpy()[fit]
    )
    out["response"] = predict_mean(response, X[test])
    return out


def out_of_fold_scores(exp: Experiment, folds: np.ndarray) -> pd.DataFrame:
    fold_ids = np.unique(folds)
    results = Parallel(n_jobs=config.N_WORKERS)(
        delayed(score_fold)(exp, folds != k, folds == k) for k in fold_ids
    )
    columns: dict[str, np.ndarray] = {}
    for k, fold_scores in zip(fold_ids, results, strict=True):
        for key, values in fold_scores.items():
            columns.setdefault(key, np.full(exp.n, np.nan))[folds == k] = values
    scores = pd.DataFrame(columns)
    if scores.isna().any().any():
        raise RuntimeError("Some customers were not scored out of fold.")
    meta = pd.DataFrame({"fold": folds, "arm": exp.arm}).join(exp.outcomes)
    return pd.concat([meta, scores], axis=1)


def run(exp: Experiment) -> pd.DataFrame:
    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    scores = out_of_fold_scores(exp, outer_folds(exp))
    scores.to_pickle(config.DATA_INTERIM / "oof_scores.pkl")
    return scores


def load_scores() -> pd.DataFrame:
    return pd.read_pickle(config.DATA_INTERIM / "oof_scores.pkl")
