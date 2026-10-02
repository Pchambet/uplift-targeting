"""Every number and data-dependent phrase quoted in prose, formatted once.

The README and the report page are both rendered from templates whose
``{{KEY}}`` placeholders are filled from here, and here only reads
``results/``. Signs are always formatted from the value, and the few
sentences whose wording depends on the data (which segments survive, what an
in-sample contest would pick) are generated rather than written by hand.
``check_claims`` stops the build when a claim written into the templates no
longer holds for the current results.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from uplift_targeting import config
from uplift_targeting.evaluation import BLANKET, RESPONSE, UPLIFT
from uplift_targeting.experiment import Z95

ARM_PROSE = {"Mens e-mail": "the men's e-mail", "Womens e-mail": "the women's e-mail"}
LEVEL_PROSE = {
    ("purchase_history", "Mens only"): "menswear-only buyers",
    ("purchase_history", "Womens only"): "womenswear-only buyers",
    ("purchase_history", "Both lines"): "buyers of both lines",
}
DIMENSION_PROSE = {
    "purchase_history": "product-line groups",
    "newbie": "customer groups",
    "channel": "channel groups",
    "zip_code": "area types",
    "history_tier": "spend tiers",
}
PRICED_RANGE_CENTS = 60  # the cost range drawn in the cost-sensitivity figure


class StaleClaimError(RuntimeError):
    """A sentence written into a template is contradicted by the current results."""


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.RESULTS / name)


def read_json(name: str) -> dict[str, Any]:
    return json.loads((config.RESULTS / name).read_text())


def share(x: float) -> str:
    """Whole percent, rounding halves up like the report page's JavaScript."""
    return f"{math.floor(100 * x + 0.5)}%"


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def signed_money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"+${x:,.0f}"


def signed_cents(x: float) -> str:
    return f"-${abs(x):.2f}" if x < 0 else f"+${x:.2f}"


def pp(x: float, digits: int = 1) -> str:
    """A rate difference in signed percentage points."""
    return f"{100 * x:+.{digits}f} pp"


def money_interval(low: float, high: float) -> str:
    return f"[{signed_money(low)}, {signed_money(high)}]"


def count(k: int, n: int) -> str:
    return f"{k} of {n}"


def verdict(low: float, high: float) -> str:
    """How a paired difference A - B reads: A beats, loses to, or ties with B."""
    if low > 0:
        return "beats"
    if high < 0:
        return "loses to"
    return "does not detectably beat"


def _ate(ate: pd.DataFrame, arm: str, outcome: str, estimator: str) -> pd.Series:
    rows = ate[(ate["arm"] == arm) & (ate["outcome"] == outcome) & (ate["estimator"] == estimator)]
    return rows.iloc[0]


def survivor_sentence(survivor: pd.Series, levels: pd.DataFrame) -> str:
    """One Holm survivor in words: the group that stands out against the others.

    The standout level is the one furthest from the average of the other
    levels, so the sentence names the contrast that drives the test.
    """
    d = levels[
        (levels["arm"] == survivor["arm"])
        & (levels["outcome"] == survivor["outcome"])
        & (levels["dimension"] == survivor["dimension"])
    ].set_index("level")["estimate"]
    gap = {lv: abs(est - d.drop(lv).mean()) for lv, est in d.items()}
    standout = max(gap, key=gap.get)
    others = d.drop(standout)
    group = LEVEL_PROSE.get((survivor["dimension"], standout), f"the '{standout}' group")
    spread = pp(others.iloc[0]) if len(others) == 1 else f"{pp(others.min())} to {pp(others.max())}"
    return (
        f"{ARM_PROSE[survivor['arm']]} lifts the {survivor['outcome']} rate by "
        f"{pp(d[standout])} among {group}, against {spread} for the other "
        f"{DIMENSION_PROSE[survivor['dimension']]}"
    )


def readout_numbers() -> dict[str, str]:
    ate = read_table("ate.csv")
    readout = read_json("readout.json")
    dim = "Difference in means"
    mc = _ate(ate, "Mens e-mail", "conversion", dim)
    ms = _ate(ate, "Mens e-mail", "spend", dim)
    ws = _ate(ate, "Womens e-mail", "spend", dim)
    mv = _ate(ate, "Mens e-mail", "visit", dim)
    cuped = _ate(ate, "Mens e-mail", "spend", "CUPED (history)")
    lin = _ate(ate, "Mens e-mail", "visit", "Regression adjustment (Lin)")
    tests = read_table("heterogeneity_tests.csv")
    levels = read_table("segments.csv")
    survivors = tests[tests["p_holm"] < 0.05].sort_values("p_holm", ascending=False)
    outcomes = sorted(set(survivors["outcome"]))
    return {
        "N": f"{readout['n']:,}",
        "N_BUYERS": f"{readout['n_buyers']:,}",
        "SRM_P": f"{readout['srm']['p_value']:.2f}",
        "MAX_SMD": f"{readout['max_abs_smd']:.3f}",
        "MENS_CONV_PP": pp(mc["estimate"], 2),
        "MENS_CONV_CI": f"[{100 * mc['low']:.2f}, {100 * mc['high']:.2f}]",
        "MENS_CONV_LIFT": f"{100 * mc['relative_lift']:+.0f}%",
        "MENS_VISIT_PP": pp(mv["estimate"]),
        "MENS_SPEND": signed_cents(ms["estimate"]),
        "MENS_SPEND_CI": f"[{signed_cents(ms['low'])}, {signed_cents(ms['high'])}]",
        "WOMENS_SPEND": signed_cents(ws["estimate"]),
        "CUPED_SPEND_SHRINK": f"{100 * (1 - cuped['ci_width_vs_dim']):.2f}%",
        "LIN_VISIT_SHRINK": f"{100 * (1 - lin['ci_width_vs_dim']):.1f}%",
        "N_TESTS": f"{len(tests)}",
        "HOLM_SURVIVORS": count(len(survivors), len(tests)),
        "HOLM_OUTCOMES": " and ".join(f"{o}s" for o in outcomes) or "none",
        "HOLM_SURVIVOR_TEXT": "; ".join(
            survivor_sentence(r, levels) for _, r in survivors.iterrows()
        ),
    }


def model_numbers() -> dict[str, str]:
    q = read_table("qini.csv")

    def row(arm: str, outcome: str, model: str = "DR-learner", base: str = "lgbm") -> pd.Series:
        r = q[(q["arm"] == arm) & (q["outcome"] == outcome) & (q["model"] == model)]
        return r[r["base"] == base].iloc[0]

    def qini(arm: str, outcome: str, model: str = "DR-learner") -> str:
        r = row(arm, outcome, model)
        return f"{1000 * r['qini']:.1f} [{1000 * r['qini_low']:.1f}, {1000 * r['qini_high']:.1f}]"

    def above_response(arm: str, outcome: str) -> str:
        d = q[(q["arm"] == arm) & (q["outcome"] == outcome)]
        uplift = d[d["model"] != RESPONSE]
        return count(int((uplift["qini"] > row(arm, outcome, RESPONSE)["qini"]).sum()), len(uplift))

    paired = read_table("qini_dr_vs_response.csv").set_index(["arm", "outcome"])

    def diff(arm: str) -> str:
        r = paired.loc[(arm, "visit")]
        return (
            f"{1000 * r['dr_minus_response']:+.1f} "
            f"[{1000 * r['low']:+.1f}, {1000 * r['high']:+.1f}]"
        )

    def paired_verdict(arm: str) -> str:
        r = paired.loc[(arm, "visit")]
        return verdict(r["low"], r["high"])

    spend = q[q["outcome"] == "spend"]
    mens_spend = spend[spend["arm"] == "Mens e-mail"]
    n_neg = int((mens_spend["qini"] < 0).sum())
    deciles = read_table("deciles.csv").query("arm == 'Womens e-mail' and outcome == 'visit'")
    top = deciles[deciles["decile"] <= 2].groupby("ranking")[["control_rate", "treated_rate"]]
    top = top.mean()
    anyway = top["control_rate"] / top["treated_rate"]
    lift = deciles[deciles["decile"] <= 2].groupby("ranking")["uplift"].mean()
    return {
        "QINI_WOMENS_VISIT": qini("Womens e-mail", "visit"),
        "QINI_MENS_VISIT": qini("Mens e-mail", "visit"),
        "QINI_WOMENS_VISIT_RESPONSE": qini("Womens e-mail", "visit", RESPONSE),
        "QINI_MENS_VISIT_RESPONSE": qini("Mens e-mail", "visit", RESPONSE),
        "QINI_DIFF_WOMENS_VISIT": diff("Womens e-mail"),
        "QINI_DIFF_MENS_VISIT": diff("Mens e-mail"),
        "DR_VS_RESPONSE_WOMENS": paired_verdict("Womens e-mail"),
        "DR_VS_RESPONSE_MENS": paired_verdict("Mens e-mail"),
        "WOMENS_VISIT_ABOVE_RESPONSE": above_response("Womens e-mail", "visit"),
        "MENS_VISIT_ABOVE_RESPONSE": above_response("Mens e-mail", "visit"),
        "SPEND_QINI_SIGNIFICANT": count(int((spend["qini_low"] > 0).sum()), len(spend)),
        "SPEND_RANKINGS_PER_EMAIL": f"{len(mens_spend)}",
        "MENS_SPEND_QINI_NEGATIVE": (
            f"all {n_neg}" if n_neg == len(mens_spend) else count(n_neg, len(mens_spend))
        ),
        "ANYWAY_RESPONSE": f"{anyway[RESPONSE]:.0%}",
        "ANYWAY_UPLIFT": f"{anyway['Uplift model']:.0%}",
        "TOP20_LIFT_UPLIFT": f"{100 * lift['Uplift model']:.1f} pp",
        "TOP20_LIFT_RESPONSE": f"{100 * lift[RESPONSE]:.1f} pp",
    }


def cost_table() -> pd.DataFrame:
    """Profit and paired comparisons of each deployable procedure over the cost grid."""
    grid = read_table("policy_by_cost.csv")
    margin = config.ECONOMICS.margin
    grid["cost"] = grid["cost_over_margin"] * margin
    grid["profit"] = margin * grid["spend_per_1000"] - 1000 * grid["cost"] * grid["share_emailed"]
    grid["profit_low"] = grid["profit"] - Z95 * margin * grid["se_per_1000"]
    # Profit minus blanket: the cost of the e-mails not sent is known, so it adds no variance.
    grid["vs_blanket"] = margin * grid["minus_blanket_per_1000"] - 1000 * grid["cost"] * (
        grid["share_emailed"] - 1.0
    )
    grid["vs_blanket_low"] = grid["vs_blanket"] - Z95 * margin * grid["minus_blanket_se_per_1000"]
    return grid[grid["cost"] <= PRICED_RANGE_CENTS / 100 + 1e-9]


def policy_numbers() -> dict[str, str]:
    s = read_json("policy_summary.json")["spend"]
    dep = s["deployable"]
    diff = s["uplift_minus_blanket"]
    cand = read_table("selection_candidates.csv")
    rho = {
        t: spearmanr(g["half0_per_1000"], g["half1_per_1000"]).statistic
        for t, g in cand.groupby("target")
    }
    margin = config.ECONOMICS.margin
    blanket = dep[BLANKET]
    grid = cost_table()
    blanket_spend = grid[grid["policy"] == BLANKET]["spend_per_1000"].iloc[0] / 1000
    targeted = grid[grid["policy"].isin([UPLIFT, RESPONSE])]
    detectable = targeted[(targeted["profit_low"] > 0) & (targeted["vs_blanket_low"] > 0)]
    naive = s["naive_in_sample"]
    personalise = s["best_email_everyone_minus_blanket_mens"]
    if naive["share"] == 1.0 and naive["choice"] != BLANKET:
        naive_idea = "personalising which e-mail each customer gets"
        naive_oos = personalise
    else:
        naive_idea = f"e-mailing the top {share(naive['share'])} by {naive['choice']}"
        naive_oos = diff
    splits = s["split_robustness"]
    return {
        "BLANKET_VALUE": money(blanket["value_per_1000"]),
        "BLANKET_CI": f"[{money(blanket['low'])}, {money(blanket['high'])}]",
        "UPLIFT_VALUE": money(dep[UPLIFT]["value_per_1000"]),
        "UPLIFT_CI": f"[{money(dep[UPLIFT]['low'])}, {money(dep[UPLIFT]['high'])}]",
        "UPLIFT_SHARE": share(dep[UPLIFT]["share_emailed"]),
        "RESPONSE_VALUE": money(dep[RESPONSE]["value_per_1000"]),
        "RESPONSE_SHARE": share(dep[RESPONSE]["share_emailed"]),
        "RESPONSE_CI": f"[{money(dep[RESPONSE]['low'])}, {money(dep[RESPONSE]['high'])}]",
        "UPLIFT_MINUS_BLANKET": signed_money(diff["value_per_1000"]),
        "UPLIFT_MINUS_BLANKET_CI": money_interval(diff["low"], diff["high"]),
        "NAIVE_VALUE": money(naive["value_per_1000"]),
        "NAIVE_GAIN": signed_money(naive["value_per_1000"] - blanket["value_per_1000"]),
        "NAIVE_SHARE": share(naive["share"]),
        "NAIVE_N": f"{naive['n_rankings']}",
        "NAIVE_IDEA": naive_idea,
        "NAIVE_OOS": signed_money(naive_oos["value_per_1000"]),
        "NAIVE_OOS_CI": money_interval(naive_oos["low"], naive_oos["high"]),
        "N_CANDIDATES": f"{len(cand) // cand['target'].nunique()}",
        "RHO_SPEND": f"{rho['spend']:.2f}",
        "RHO_VISIT": f"{rho['visit']:.2f}",
        "BLANKET_BREAK_EVEN": f"{100 * margin * blanket_spend:.0f} cents",
        "DETECTABLE_COSTS": (
            "at no cost level"
            if detectable.empty
            else f"only at {count(detectable['cost'].nunique(), grid['cost'].nunique())} "
            "cost levels"
        ),
        "PRICED_RANGE": f"0 and {PRICED_RANGE_CENTS} cents",
        "N_SPLITS": f"{splits['n_splits']}",
        "SPLITS_BELOW": count(splits["n_below_blanket"], splits["n_splits"]),
        "SPLIT_MEAN_DIFF": signed_money(splits["uplift_minus_blanket_mean"]),
        "SPLIT_RANGE": (
            f"{signed_money(splits['uplift_minus_blanket_min'])} to "
            f"{signed_money(splits['uplift_minus_blanket_max'])}"
        ),
        "SPLIT_MEAN_VALUE": money(splits["uplift_mean_per_1000"]),
        "SPLIT_SHARE_MEDIAN": share(splits["uplift_share_median"]),
        "SPLIT_SHARE_RANGE": (
            f"{share(splits['uplift_share_min'])} to {share(splits['uplift_share_max'])}"
        ),
        "SPLIT_DISTINCT": f"{splits['n_distinct_selections']}",
    }


def criteo_numbers() -> dict[str, str]:
    summary = read_json("criteo_summary.json")
    q = read_table("criteo_qini.csv").set_index(["outcome", "model"])
    all_curves = read_table("criteo_uplift_curves.csv")
    deciles = read_table("criteo_deciles.csv").query("outcome == 'visit' and model == @RESPONSE")

    def curve(outcome: str, model: str) -> pd.Series:
        c = all_curves[(all_curves["outcome"] == outcome) & (all_curves["model"] == model)]
        return c.set_index("fraction")["gain_per_1000"]

    def top_share(outcome: str, model: str, top: float) -> str:
        c = curve(outcome, model)
        return f"{c[top] / c[1.0]:.0%}"

    def qini(outcome: str, model: str) -> str:
        return f"{1000 * q.loc[(outcome, model), 'estimate']:.2f}"

    dr = curve("conversion", "DR-learner")
    top = deciles[deciles["decile"] == 1].iloc[0]
    effects = summary["average_effects"]
    conv = q.xs("conversion")["estimate"]
    neg = q.loc[("conversion", "DR-learner")]
    return {
        "CRITEO_ROWS": f"{summary['n_rows']:,}",
        "CRITEO_TEST": f"{summary['n_test']:,}",
        "CRITEO_TRAIN": f"{summary['n_train_sample'] / 1e6:.1f}M",
        "CRITEO_CONTROL_SHARE": f"{1 - summary['treatment_share']:.0%}",
        "CRITEO_TOP20_RESPONSE_SHARE": top_share("conversion", RESPONSE, 0.2),
        "CRITEO_TOP20_DR_SHARE": top_share("conversion", "DR-learner", 0.2),
        "CRITEO_TOP20_T_SHARE": top_share("conversion", "T-learner", 0.2),
        "CRITEO_QINI_CONV_RESPONSE": qini("conversion", RESPONSE),
        "CRITEO_QINI_CONV_T": qini("conversion", "T-learner"),
        "CRITEO_QINI_CONV_DR": qini("conversion", "DR-learner"),
        "CRITEO_QINI_VISIT_RESPONSE": qini("visit", RESPONSE),
        "CRITEO_QINI_VISIT_T": qini("visit", "T-learner"),
        "CRITEO_QINI_VISIT_DR": qini("visit", "DR-learner"),
        "CRITEO_BEST_CONV": str(conv.idxmax()),
        "CRITEO_DR_TIE": f"{q.loc[('conversion', 'DR-learner'), 'largest_tie']:.0%}",
        "CRITEO_DR_TAIL": f"{(dr[1.0] - dr[0.99]) / dr[1.0]:.0%}",
        "CRITEO_DR_NEG_SHARE": f"{neg['negative_share']:.1%}",
        "CRITEO_DR_NEG_BASE": f"{100 * neg['negative_control_rate']:.1f}%",
        "CRITEO_DR_NEG_UPLIFT": pp(neg["negative_uplift"]),
        "CRITEO_TOP_DECILE_BASE": f"{100 * top['control_rate']:.0f}%",
        "CRITEO_TOP_DECILE_LIFT": pp(top["uplift"]),
        "CRITEO_TOP10_VISIT_SHARE": top_share("visit", RESPONSE, 0.1),
        "CRITEO_ATE_CONV": pp(effects["conversion"]["estimate"], 3),
        "CRITEO_ATE_VISIT": pp(effects["visit"]["estimate"], 2),
        "CRITEO_CONV_CONTROL_RATE": f"{100 * effects['conversion']['control_rate']:.2f}%",
    }


def economics_numbers() -> dict[str, str]:
    econ = config.ECONOMICS
    return {
        "MARGIN_FMT": f"{econ.margin:.0%}",
        "COST_FMT": f"${econ.cost_per_email:.2f}",
    }


def check_claims() -> None:
    """Fail loudly when the results contradict a claim the templates make in words."""
    s = read_json("policy_summary.json")["spend"]
    dep = s["deployable"]
    problems = []
    if dep[BLANKET]["low"] <= 0:
        problems.append("blanket sending is no longer detectably profitable")
    if s["uplift_minus_blanket"]["low"] > 0:
        problems.append("the cross-selected uplift procedure now beats blanket sending")
    if s["naive_in_sample"]["value_per_1000"] <= dep[BLANKET]["value_per_1000"]:
        problems.append("the in-sample contest no longer claims a gain over blanket sending")
    paired = read_table("qini_dr_vs_response.csv").set_index(["arm", "outcome"])
    if paired.loc[("Mens e-mail", "visit"), "high"] >= 0:
        problems.append("the DR-learner no longer loses to the response model on men's visits")
    q = read_table("criteo_qini.csv")
    best = q.loc[q.groupby("outcome")["estimate"].idxmax(), "model"]
    if set(best) != {RESPONSE}:
        problems.append("the response model is no longer the best Criteo ranker")
    if problems:
        raise StaleClaimError("Rewrite the templates: " + "; ".join(problems))


def load_numbers() -> dict[str, str]:
    check_claims()
    return {
        **readout_numbers(),
        **model_numbers(),
        **policy_numbers(),
        **criteo_numbers(),
        **economics_numbers(),
    }
