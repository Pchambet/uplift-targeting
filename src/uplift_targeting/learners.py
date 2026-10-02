"""Meta-learners for the conditional average treatment effect (CATE).

Each learner turns ordinary supervised models into an estimate of
``tau(x) = E[Y(1) - Y(0) | X = x]`` for a binary treatment. They differ in how
they share information between arms and how they handle the fact that the
effect is usually much smaller than the outcome itself:

* S-learner: one model with the treatment as a feature; the effect is the
  difference of two predictions. Regularisation often shrinks it to zero.
* T-learner: one model per arm. Unbiased in large samples, but the difference
  of two noisy fits is noisy.
* X-learner (Kunzel et al., 2019): imputes individual effects with the
  opposite arm's model, then smooths them.
* DR-learner (Kennedy, 2023): regresses a cross-fitted doubly robust
  pseudo-outcome on X. Its error depends on the *product* of the nuisance
  errors, which is what makes it the default choice here.
* Transformed outcome (Athey & Imbens, 2016): regresses
  ``Y (T - e) / (e (1 - e))``, whose conditional mean is ``tau(x)``. Simple and
  unbiased, but high-variance; kept as a reference point.

The propensity score is known by design in a randomised experiment, so it is
passed in rather than estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from uplift_targeting import config

Task = Literal["classification", "regression"]
BaseKind = Literal["lgbm", "linear"]


def make_model(kind: BaseKind, task: Task, seed: int = config.SEED) -> BaseEstimator:
    """Base learner factory: gradient boosting or a regularised linear model."""
    if kind == "lgbm":
        params = {**config.LIGHTGBM_PARAMS, "random_state": seed}
        if task == "classification":
            return lgb.LGBMClassifier(**params)
        return lgb.LGBMRegressor(**params)
    if task == "classification":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def predict_mean(model: BaseEstimator, X) -> np.ndarray:
    """E[Y | X] from a fitted model: the class-1 probability for classifiers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _as_array(X) -> np.ndarray:
    return np.asarray(X, dtype=float)


@dataclass
class MetaLearner:
    """Common interface: ``fit(X, t, y)`` then ``predict(X)`` returns tau-hat."""

    base: BaseKind = "lgbm"
    outcome_task: Task = "classification"
    propensity: float = 0.5
    seed: int = config.SEED
    _fitted: dict = field(default_factory=dict, init=False, repr=False)

    def outcome_model(self) -> BaseEstimator:
        return make_model(self.base, self.outcome_task, self.seed)

    def effect_model(self) -> BaseEstimator:
        # Effects and pseudo-outcomes are continuous, whatever the outcome type.
        return make_model(self.base, "regression", self.seed)

    def fit(self, X, t: np.ndarray, y: np.ndarray) -> MetaLearner:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError


class SLearner(MetaLearner):
    def fit(self, X, t, y):
        Xa = _as_array(X)
        self._fitted["m"] = self.outcome_model().fit(np.column_stack([Xa, t]), y)
        return self

    def predict(self, X):
        Xa = _as_array(X)
        model = self._fitted["m"]
        treated = predict_mean(model, np.column_stack([Xa, np.ones(len(Xa))]))
        control = predict_mean(model, np.column_stack([Xa, np.zeros(len(Xa))]))
        return treated - control


class TLearner(MetaLearner):
    def fit(self, X, t, y):
        Xa = _as_array(X)
        self._fitted["m1"] = self.outcome_model().fit(Xa[t == 1], y[t == 1])
        self._fitted["m0"] = self.outcome_model().fit(Xa[t == 0], y[t == 0])
        return self

    def mu(self, X, arm: int) -> np.ndarray:
        return predict_mean(self._fitted[f"m{arm}"], _as_array(X))

    def predict(self, X):
        return self.mu(X, 1) - self.mu(X, 0)


class XLearner(MetaLearner):
    def fit(self, X, t, y):
        Xa = _as_array(X)
        stage1 = TLearner(self.base, self.outcome_task, self.propensity, self.seed).fit(Xa, t, y)
        treated, control = t == 1, t == 0
        # Each arm's imputed effects use only the *other* arm's model, so they
        # are out-of-sample predictions by construction.
        d1 = y[treated] - stage1.mu(Xa[treated], 0)
        d0 = stage1.mu(Xa[control], 1) - y[control]
        self._fitted["tau1"] = self.effect_model().fit(Xa[treated], d1)
        self._fitted["tau0"] = self.effect_model().fit(Xa[control], d0)
        return self

    def predict(self, X):
        Xa = _as_array(X)
        g = self.propensity  # weight on the control-side estimate, as in Kunzel et al.
        return g * self._fitted["tau0"].predict(Xa) + (1 - g) * self._fitted["tau1"].predict(Xa)


def dr_pseudo_outcome(
    y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, e: float | np.ndarray
) -> np.ndarray:
    """AIPW pseudo-outcome whose conditional mean is tau(x) if mu or e is right."""
    return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)


@dataclass
class DRLearner(MetaLearner):
    n_folds: int = config.N_INNER_FOLDS

    def fit(self, X, t, y):
        Xa = _as_array(X)
        mu0, mu1 = np.empty(len(y)), np.empty(len(y))
        folds = KFold(self.n_folds, shuffle=True, random_state=self.seed)
        for fit_idx, out_idx in folds.split(Xa):
            tl = TLearner(self.base, self.outcome_task, self.propensity, self.seed)
            tl.fit(Xa[fit_idx], t[fit_idx], y[fit_idx])
            mu0[out_idx], mu1[out_idx] = tl.mu(Xa[out_idx], 0), tl.mu(Xa[out_idx], 1)
        pseudo = dr_pseudo_outcome(y, t, mu0, mu1, self.propensity)
        self._fitted["tau"] = self.effect_model().fit(Xa, pseudo)
        return self

    def predict(self, X):
        return self._fitted["tau"].predict(_as_array(X))


class TransformedOutcome(MetaLearner):
    def fit(self, X, t, y):
        e = self.propensity
        z = y * (t - e) / (e * (1 - e))
        self._fitted["tau"] = self.effect_model().fit(_as_array(X), z)
        return self

    def predict(self, X):
        return self._fitted["tau"].predict(_as_array(X))


LEARNERS: dict[str, type[MetaLearner]] = {
    "S-learner": SLearner,
    "T-learner": TLearner,
    "X-learner": XLearner,
    "DR-learner": DRLearner,
    "Transformed outcome": TransformedOutcome,
}


def make_learner(
    name: str, base: BaseKind, outcome_task: Task, propensity: float, seed: int = config.SEED
) -> MetaLearner:
    return LEARNERS[name](base=base, outcome_task=outcome_task, propensity=propensity, seed=seed)
