"""Parts B and C: rank quality (Qini) and the value of targeting policies.

Inputs are the out-of-fold scores from :mod:`uplift_targeting.modeling`, so
every number here is out-of-sample. Two kinds of uncertainty are reported:

* Qini / AUUC areas: percentile bootstrap over customers (scores held fixed).
* Policy values: the standard error of the mean of per-customer doubly robust
  scores; differences between policies use paired scores on the same customers.

Choosing a policy is itself a fitting step. Picking the best-looking model, or
the best-looking budget, on the data used to report its value would overstate
that value. Part C therefore uses *cross-selection*: customers are split into
two halves; everything that is chosen (the ranking model, the share of
customers to e-mail) is chosen on one half and applied to the other, and the
two out-of-half evaluations are pooled. What is evaluated is the whole
procedure "select on past data, then deploy", which is what a team would run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from uplift_targeting import config
from uplift_targeting.experiment import Z95, decile_rates
from uplift_targeting.metrics import bootstrap_areas, uplift_curve
from uplift_targeting.modeling import BASES
from uplift_targeting.policy import (
    PolicyValue,
    dr_scores,
    ipw_scores,
    top_share_policy,
    value,
    value_difference,
)

RESPONSE = "Response model"
UPLIFT = "Uplift model (cross-selected)"
TEXTBOOK = "DR-learner (LightGBM) on the target"
RANDOM = "Random (Mens e-mail)"
BLANKET = "Blanket Mens e-mail"
CURVE_GRID = np.linspace(0, 1, 101)
SHARES = np.array(config.BUDGET_GRID)
PER = 1000  # values are reported per 1,000 customers in the base
POLICY_TARGETS = ("visit", "spend")  # dense traffic signal, sparse money signal


def tau_column(outcome: str, arm: int, learner: str, base: str) -> str:
    return f"tau__{outcome}__{arm}__{learner}__{base}"


def response_column(outcome: str, arm: int) -> str:
    return f"mu__{outcome}__{arm}"


def learners_in(scores: pd.DataFrame) -> list[str]:
    return sorted({c.split("__")[3] for c in scores.columns if c.startswith("tau__")})


# ---------------------------------------------------------------- Part B: Qini


def qini_tables(
    scores: pd.DataFrame, n_boot: int = config.N_BOOTSTRAP
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Qini and AUUC over random, with bootstrap CIs, for every model x outcome x e-mail.

    The response model of the same e-mail and outcome is scored alongside the
    uplift learners, so the table shows directly whether predicting *who buys*
    is a good proxy for predicting *who is moved*. Also returns the uplift
    curves of the LightGBM models for plotting, and a paired bootstrap
    comparison of the DR-learner with the response model.
    """
    rows, curves, paired = [], [], []
    for outcome in config.OUTCOMES:
        for arm in (1, 2):
            pair = scores[scores["arm"].isin([0, arm])]
            y = pair[outcome].to_numpy()
            t = (pair["arm"] == arm).to_numpy().astype(int)
            rankings = {
                (name, base): pair[tau_column(outcome, arm, name, base)].to_numpy()
                for name in learners_in(scores)
                for base in BASES
            }
            rankings[(RESPONSE, "lgbm")] = pair[response_column(outcome, arm)].to_numpy()
            named = {f"{n}|{b}": s for (n, b), s in rankings.items()}
            seed = config.SEED + 10 * arm + config.OUTCOMES.index(outcome)
            qini, reps = bootstrap_areas(named, y, t, n_boot, seed=seed, kind="qini")
            # Paired comparison on the same bootstrap draws: DR-learner minus response model.
            names = list(named)
            diff = (
                reps[:, names.index("DR-learner|lgbm")] - reps[:, names.index(f"{RESPONSE}|lgbm")]
            )
            paired.append(
                {
                    "outcome": outcome,
                    "arm": config.ARM_LABELS[arm],
                    "dr_minus_response": qini["DR-learner|lgbm"].estimate
                    - qini[f"{RESPONSE}|lgbm"].estimate,
                    "low": float(np.quantile(diff, 0.025)),
                    "high": float(np.quantile(diff, 0.975)),
                }
            )
            auuc, _ = bootstrap_areas(named, y, t, n_boot, seed=seed, kind="uplift")
            for (name, base), score in rankings.items():
                q, u = qini[f"{name}|{base}"], auuc[f"{name}|{base}"]
                rows.append(
                    {
                        "outcome": outcome,
                        "arm": config.ARM_LABELS[arm],
                        "model": name,
                        "base": base,
                        "qini": q.estimate,
                        "qini_low": q.low,
                        "qini_high": q.high,
                        "auuc": u.estimate,
                        "auuc_low": u.low,
                        "auuc_high": u.high,
                    }
                )
                if base != "lgbm":
                    continue
                curve = uplift_curve(score, y, t)
                gain = np.interp(CURVE_GRID, curve.fraction, curve.gain) / len(y) * PER
                curves += [
                    {
                        "outcome": outcome,
                        "arm": config.ARM_LABELS[arm],
                        "model": name,
                        "fraction": round(float(f), 2),
                        "gain_per_1000": float(g),
                    }
                    for f, g in zip(CURVE_GRID, gain, strict=True)
                ]
    return pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(paired)


def decile_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Design-based look inside each ranking: who would have acted anyway?

    For each e-mail and outcome, customers in the e-mail-vs-control comparison
    are cut into deciles of the uplift score (T-learner, LightGBM) and of the
    response score. Within a decile the control arm's rate is what happens
    without the e-mail, and the gap to the e-mailed arm is the incremental
    effect: no model enters these numbers except through the ranking.
    """
    rows = []
    for arm in (1, 2):
        pair = scores[scores["arm"].isin([0, arm])]
        t = (pair["arm"] == arm).to_numpy().astype(int)
        for outcome in ("visit", "conversion"):
            y = pair[outcome].to_numpy()
            rankings = {
                "Uplift model": pair[tau_column(outcome, arm, "T-learner", "lgbm")].to_numpy(),
                RESPONSE: pair[response_column(outcome, arm)].to_numpy(),
            }
            for name, score in rankings.items():
                table = decile_rates(score, y, t, seed=config.SEED)
                key = {"arm": config.ARM_LABELS[arm], "outcome": outcome, "ranking": name}
                rows += [key | row for row in table.to_dict("records")]
    return pd.DataFrame(rows)


# ---------------------------------------------------------- Part C: policies


@dataclass(frozen=True)
class PolicyData:
    """Everything the off-policy estimators need, for one outcome."""

    logged: np.ndarray
    y: np.ndarray
    propensity: np.ndarray
    mu: np.ndarray  # (n, 3) cross-fitted E[y | X, action]

    @classmethod
    def from_scores(cls, scores: pd.DataFrame, outcome: str) -> PolicyData:
        logged = scores["arm"].to_numpy()
        mu = scores[[response_column(outcome, a) for a in (0, 1, 2)]].to_numpy()
        propensity = np.full(len(logged), config.DESIGN_PROPENSITY)
        return cls(logged, scores[outcome].to_numpy(), propensity, mu)

    def subset(self, mask: np.ndarray) -> PolicyData:
        return PolicyData(self.logged[mask], self.y[mask], self.propensity[mask], self.mu[mask])

    def scores(self, action: np.ndarray, estimator: str = "dr") -> np.ndarray:
        """Per-customer scores of ``action`` minus those of e-mailing nobody."""
        nobody = np.zeros_like(action)
        if estimator == "ipw":
            own = ipw_scores(action, self.logged, self.y, self.propensity)
            return own - ipw_scores(nobody, self.logged, self.y, self.propensity)
        own = dr_scores(action, self.logged, self.y, self.propensity, self.mu)
        return own - dr_scores(nobody, self.logged, self.y, self.propensity, self.mu)

    def incremental(self, action: np.ndarray, estimator: str = "dr") -> PolicyValue:
        """Incremental outcome per customer of ``action`` versus e-mailing nobody."""
        return value(self.scores(action, estimator))


@dataclass(frozen=True)
class Ranking:
    """A targeting strategy: who comes first, and which e-mail they get."""

    priority: np.ndarray
    email: np.ndarray  # per-customer e-mail (1 = Mens, 2 = Womens) if targeted

    def policy(self, share: float, mask: np.ndarray | None = None) -> np.ndarray:
        """Actions when the top ``share`` of customers (within ``mask``) is e-mailed."""
        if mask is None:
            return top_share_policy(self.priority, share, self.email)
        action = np.zeros(len(self.priority), dtype=np.int64)
        action[mask] = top_share_policy(self.priority[mask], share, self.email[mask])
        return action


def uplift_candidates(scores: pd.DataFrame) -> dict[str, Ranking]:
    """Every uplift model as a policy: rank by the larger predicted effect, send that e-mail.

    Candidates differ in the outcome the model was trained on (visit,
    conversion, spend), the meta-learner and the base learner. A model trained
    on a dense outcome (visits) may rank customers better for a sparse one
    (spend) than a model trained on the sparse outcome itself; the selection
    step decides on evidence.
    """
    out = {}
    for outcome in config.OUTCOMES:
        for learner in learners_in(scores):
            for base in BASES:
                mens = scores[tau_column(outcome, 1, learner, base)].to_numpy()
                womens = scores[tau_column(outcome, 2, learner, base)].to_numpy()
                out[f"{learner} | {base} | {outcome}"] = Ranking(
                    np.maximum(mens, womens), np.where(womens > mens, 2, 1)
                )
    return out


def response_ranking(scores: pd.DataFrame, target: str) -> Ranking:
    """The classic approach: e-mail the customers most likely to act after the Mens e-mail.

    For spend the response model predicts purchase (propensity to buy), the
    usual marketing score; for visits it predicts a visit.
    """
    outcome = "conversion" if target == "spend" else target
    priority = scores[response_column(outcome, 1)].to_numpy()
    return Ranking(priority, np.ones(len(priority), dtype=np.int64))


def split_halves(arm: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    """Random half (0/1) per customer, balanced within each arm."""
    rng = np.random.default_rng(seed)
    half = np.empty(len(arm), dtype=np.int64)
    for a in np.unique(arm):
        idx = rng.permutation(np.flatnonzero(arm == a))
        half[idx] = np.arange(len(idx)) % 2
    return half


def value_curve(data: PolicyData, ranking: Ranking, mask: np.ndarray) -> np.ndarray:
    """Incremental outcome per customer at every budget share, using ``mask`` rows only."""
    sub = data.subset(mask)
    return np.array([sub.incremental(ranking.policy(s, mask)[mask]).estimate for s in SHARES])


def select_ranking(data: PolicyData, candidates: dict[str, Ranking], mask: np.ndarray) -> str:
    """The candidate with the largest average incremental outcome over budgets, on ``mask``.

    For spend, profit at share s is ``margin * spend(s) - cost * s``. Averaged
    over the grid the cost term is identical for every candidate, so the choice
    does not depend on the assumed economics.
    """
    area = {name: value_curve(data, r, mask)[1:].mean() for name, r in candidates.items()}
    return max(area, key=area.get)


@dataclass(frozen=True)
class CrossSelection:
    """Rankings chosen on one half and applied to the other."""

    half: np.ndarray
    chosen: dict[int, str]  # half h is served by the candidate selected on half 1 - h
    candidates: dict[str, Ranking]

    def ranking(self, h: int) -> Ranking:
        return self.candidates[self.chosen[h]]

    def policy(self, share: float) -> np.ndarray:
        action = np.zeros(len(self.half), dtype=np.int64)
        for h in (0, 1):
            m = self.half == h
            action[m] = self.ranking(h).policy(share, m)[m]
        return action


def cross_select(scores: pd.DataFrame, data: PolicyData) -> CrossSelection:
    half = split_halves(scores["arm"].to_numpy())
    candidates = uplift_candidates(scores)
    chosen = {h: select_ranking(data, candidates, half == 1 - h) for h in (0, 1)}
    return CrossSelection(half, chosen, candidates)


def _row(target: str, policy: str, share: float, estimator: str, v: PolicyValue) -> dict:
    low, high = v.ci()
    return {
        "outcome": target,
        "policy": policy,
        "share": float(share),
        "estimator": estimator,
        "value_per_1000": PER * v.estimate,
        "low": PER * low,
        "high": PER * high,
    }


def policy_curves(
    scores: pd.DataFrame, data: PolicyData, cs: CrossSelection, target: str
) -> pd.DataFrame:
    """Incremental outcome per 1,000 customers vs share e-mailed, for each strategy.

    Margin- and cost-free on purpose: for spend, profit per 1,000 customers is
    ``margin * value - 1000 * cost * share``, applied by the figures and the page.
    """
    fixed = {
        RESPONSE: response_ranking(scores, target),
        TEXTBOOK: cs.candidates[f"DR-learner | lgbm | {target}"],
    }
    everyone = np.ones(len(scores), dtype=np.int64)
    rows = []
    for estimator in ("dr", "ipw"):
        blanket = data.incremental(everyone, estimator)
        for share in SHARES:
            v = data.incremental(cs.policy(share), estimator)
            rows.append(_row(target, UPLIFT, share, estimator, v))
            for name, ranking in fixed.items():
                v = data.incremental(ranking.policy(share), estimator)
                rows.append(_row(target, name, share, estimator, v))
            # Random targeting is, in expectation, a share of the blanket campaign.
            v = PolicyValue(share * blanket.estimate, share * blanket.se)
            rows.append(_row(target, RANDOM, share, estimator, v))
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BudgetChoice:
    """Cross-selected budget: the share to e-mail is chosen on one half, applied to the other.

    The options are every share of the grid for the half's ranking plus the
    blanket Mens campaign, so the procedure is free to conclude "do not
    target". Curves on the selection halves are computed once; choosing for a
    given cost / margin ratio is then just an argmax.
    """

    half: np.ndarray
    rankings: dict[int, Ranking]  # ranking applied to half h
    curves: dict[int, np.ndarray]  # incremental value per share, measured on half 1 - h
    blanket: dict[int, float]  # blanket Mens value, measured on half 1 - h

    @classmethod
    def build(cls, data: PolicyData, half: np.ndarray, rankings: dict[int, Ranking]):
        curves, blanket = {}, {}
        for h in (0, 1):
            select = half == 1 - h
            curves[h] = value_curve(data, rankings[h], select)
            everyone = np.ones(int(select.sum()), dtype=np.int64)
            blanket[h] = data.subset(select).incremental(everyone).estimate
        return cls(half, rankings, curves, blanket)

    def action(self, ratio: float) -> np.ndarray:
        """Policy for cost / margin = ``ratio``: maximise ``value - ratio * share``."""
        action = np.zeros(len(self.half), dtype=np.int64)
        for h in (0, 1):
            apply = self.half == h
            profit = self.curves[h] - ratio * SHARES
            if self.blanket[h] - ratio > profit.max():
                action[apply] = 1
            else:
                share = float(SHARES[int(np.argmax(profit))])
                action[apply] = self.rankings[h].policy(share, apply)[apply]
        return action


def budget_choices(
    scores: pd.DataFrame, data: PolicyData, cs: CrossSelection
) -> dict[str, BudgetChoice]:
    response = response_ranking(scores, "spend")
    return {
        UPLIFT: BudgetChoice.build(data, cs.half, {h: cs.ranking(h) for h in (0, 1)}),
        RESPONSE: BudgetChoice.build(data, cs.half, {0: response, 1: response}),
    }


def deployable(choices: dict[str, BudgetChoice], ratio: float) -> dict[str, np.ndarray]:
    """Actions of the three procedures a team could ship, at a given cost / margin."""
    n = len(next(iter(choices.values())).half)
    return {name: c.action(ratio) for name, c in choices.items()} | {
        BLANKET: np.ones(n, dtype=np.int64)
    }


def priced(v: PolicyValue, share: float, margin: float, cost: float) -> dict[str, float]:
    """Value per 1,000 customers; the e-mail cost is known, so it adds no variance."""
    est = PER * (margin * v.estimate - cost * share)
    half_width = PER * margin * Z95 * v.se
    return {"value_per_1000": est, "low": est - half_width, "high": est + half_width}


def compare(
    data: PolicyData, a: np.ndarray, b: np.ndarray, margin: float = 1.0, cost: float = 0.0
) -> dict[str, float]:
    """Paired difference a - b per 1,000 customers."""
    diff = value_difference(data.scores(a), data.scores(b))
    return priced(diff, float(np.mean(a > 0) - np.mean(b > 0)), margin, cost)


def economics_grid(data: PolicyData, choices: dict[str, BudgetChoice]) -> pd.DataFrame:
    """The deployable procedures re-run over a grid of cost / margin ratios.

    Their choices depend on the ratio only, so the report page can re-price
    any (margin, cost) pair from the nearest row.
    """
    rows = []
    for ratio in np.round(np.arange(0.0, 2.0001, 0.025), 3):  # holds 0.15 / 0.40 exactly
        for name, action in deployable(choices, ratio).items():
            v = data.incremental(action)
            rows.append(
                {
                    "cost_over_margin": ratio,
                    "policy": name,
                    "share_emailed": float(np.mean(action > 0)),
                    "spend_per_1000": PER * v.estimate,
                    "se_per_1000": PER * v.se,
                }
            )
    return pd.DataFrame(rows)


def naive_in_sample(
    scores: pd.DataFrame, data: PolicyData, cs: CrossSelection, margin: float, cost: float
) -> dict:
    """What a team that selects on the evaluation data itself would report.

    Every (candidate, share) pair and the blanket campaign are scored on all
    customers and the best is kept: the textbook winner's curse.
    """
    everything = np.ones(len(scores), dtype=bool)
    best = {"choice": BLANKET, "share": 1.0}
    everyone = np.ones(len(scores), dtype=np.int64)
    best_value = margin * data.incremental(everyone).estimate - cost
    for name, ranking in {**cs.candidates, RESPONSE: response_ranking(scores, "spend")}.items():
        for s, v in zip(SHARES, value_curve(data, ranking, everything), strict=True):
            profit = margin * v - cost * s
            if profit > best_value:
                best, best_value = {"choice": name, "share": float(s)}, profit
    return best | {"value_per_1000": PER * best_value}


def candidate_table(data: PolicyData, cs: CrossSelection, target: str) -> pd.DataFrame:
    """Average incremental outcome over budgets of every candidate, in each half."""
    rows = []
    for name, ranking in cs.candidates.items():
        learner, base, trained_on = name.split(" | ")
        row = {"target": target, "learner": learner, "base": base, "trained_on": trained_on}
        for h in (0, 1):
            area = value_curve(data, ranking, cs.half == h)[1:].mean()
            row[f"half{h}_per_1000"] = PER * area
        rows.append(row)
    return pd.DataFrame(rows).sort_values("half0_per_1000", ascending=False)


def at_budget(
    scores: pd.DataFrame,
    data: PolicyData,
    cs: CrossSelection,
    target: str,
    share: float,
    margin: float,
    cost: float,
) -> dict:
    """Uplift vs response vs random at one contact budget, with paired differences."""
    up, resp = cs.policy(share), response_ranking(scores, target).policy(share)
    everyone = np.ones(len(scores), dtype=np.int64)
    blanket = data.incremental(everyone)
    random = PolicyValue(share * blanket.estimate, share * blanket.se)
    return {
        "share": share,
        "uplift": priced(data.incremental(up), share, margin, cost),
        "response": priced(data.incremental(resp), share, margin, cost),
        "random": priced(random, share, margin, cost),
        "uplift_minus_response": compare(data, up, resp, margin, cost),
        # Random targeting's per-customer scores are share x the blanket scores.
        "uplift_minus_random": priced(
            value_difference(data.scores(up), share * data.scores(everyone)), 0.0, margin, 0.0
        ),
    }


def run(scores: pd.DataFrame, n_boot: int = config.N_BOOTSTRAP) -> dict:
    margin, cost = config.ECONOMICS.margin, config.ECONOMICS.cost_per_email
    out = config.RESULTS
    out.mkdir(parents=True, exist_ok=True)

    qini, curves, paired = qini_tables(scores, n_boot)
    qini.to_csv(out / "qini.csv", index=False)
    paired.to_csv(out / "qini_dr_vs_response.csv", index=False)
    curves.to_csv(out / "uplift_curves.csv", index=False)
    decile_table(scores).to_csv(out / "deciles.csv", index=False)

    summary: dict = {"economics": {"margin": margin, "cost_per_email": cost}}
    curves_out, candidates_out = [], []
    everyone = np.ones(len(scores), dtype=np.int64)
    for target in POLICY_TARGETS:
        data = PolicyData.from_scores(scores, target)
        cs = cross_select(scores, data)
        curves_out.append(policy_curves(scores, data, cs, target))
        candidates_out.append(candidate_table(data, cs, target))
        m, c = (margin, cost) if target == "spend" else (1.0, 0.0)
        best_email = cs.policy(1.0)
        summary[target] = {
            "selected_rankings": {f"half{h}": cs.chosen[h] for h in (0, 1)},
            "at_budget": [at_budget(scores, data, cs, target, s, m, c) for s in (0.2, 0.4)],
            "best_email_everyone_minus_blanket_mens": compare(data, best_email, everyone, m, c),
            "share_womens_when_everyone_emailed": float(np.mean(best_email == 2)),
            "blanket_mens": priced(data.incremental(everyone), 1.0, m, c),
        }
        if target != "spend":
            continue
        choices = budget_choices(scores, data, cs)
        economics_grid(data, choices).to_csv(out / "policy_by_cost.csv", index=False)
        acts = deployable(choices, cost / margin)
        summary[target]["deployable"] = {
            name: {"share_emailed": float(np.mean(a > 0))}
            | priced(data.incremental(a), float(np.mean(a > 0)), margin, cost)
            for name, a in acts.items()
        }
        summary[target]["uplift_minus_blanket"] = compare(
            data, acts[UPLIFT], acts[BLANKET], margin, cost
        )
        summary[target]["uplift_minus_response"] = compare(
            data, acts[UPLIFT], acts[RESPONSE], margin, cost
        )
        summary[target]["naive_in_sample"] = naive_in_sample(scores, data, cs, margin, cost)
    pd.concat(curves_out).to_csv(out / "policy_curves.csv", index=False)
    pd.concat(candidates_out).to_csv(out / "selection_candidates.csv", index=False)
    (out / "policy_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
