import numpy as np
import pytest
from scipy.stats import spearmanr

from uplift_targeting.learners import LEARNERS, dr_pseudo_outcome, make_learner
from uplift_targeting.synthetic import make_experiment


@pytest.fixture(scope="module")
def train_test():
    return make_experiment(8_000, seed=1), make_experiment(4_000, seed=2)


@pytest.mark.parametrize("name", ["DR-learner", "X-learner", "T-learner"])
def test_learner_recovers_true_cate_ranking(train_test, name):
    train, test = train_test
    learner = make_learner(name, "lgbm", "regression", train.propensity)
    tau_hat = learner.fit(train.X, train.t, train.y).predict(test.X)
    rho = spearmanr(tau_hat, test.tau).statistic
    assert rho > 0.75, f"{name}: Spearman {rho:.2f}"
    assert abs(tau_hat.mean() - test.tau.mean()) < 0.1


def test_all_learners_run_with_linear_base_and_binary_outcome():
    rng = np.random.default_rng(0)
    n = 1_500
    X = rng.normal(size=(n, 3))
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.2 + 0.1 * t * (X[:, 0] > 0)).astype(float)
    for name in LEARNERS:
        tau_hat = make_learner(name, "linear", "classification", 0.5).fit(X, t, y).predict(X)
        assert tau_hat.shape == (n,)
        assert np.isfinite(tau_hat).all()


def test_dr_pseudo_outcome_is_unbiased_with_wrong_outcome_model():
    exp = make_experiment(200_000, seed=3)
    wrong = np.zeros(len(exp.y))
    pseudo = dr_pseudo_outcome(exp.y, exp.t, wrong, wrong, exp.propensity)
    assert pseudo.mean() == pytest.approx(exp.tau.mean(), abs=0.05)


@pytest.mark.parametrize("name", ["X-learner", "DR-learner"])
def test_model_settings_reach_first_stage_models(name):
    """Overrides must apply to the nuisance models too, not only to the final stage."""
    exp = make_experiment(600, seed=4)
    params = {"n_estimators": 7, "num_leaves": 5, "min_child_samples": 10}
    learner = make_learner(name, "lgbm", "regression", exp.propensity, params=params)
    learner.fit(exp.X, exp.t, exp.y)
    stage1 = [m for k, m in learner._fitted.items() if k.startswith(("stage1", "nuisance"))]
    fitted = [m._fitted[a] for m in stage1 for a in ("m0", "m1")]
    fitted += [m for k, m in learner._fitted.items() if k.startswith("tau")]
    assert len(fitted) >= 3
    for model in fitted:
        assert {k: model.get_params()[k] for k in params} == params
