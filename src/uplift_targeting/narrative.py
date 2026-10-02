"""Headline numbers, formatted once, shared by the report page and the README check.

Every figure quoted in prose comes from here, and here only reads ``results/``.
``tests/test_readme_numbers.py`` asserts that each ``HEADLINE_KEYS`` string
appears verbatim in the README, so the README cannot drift from the outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from scipy.stats import spearmanr

from uplift_targeting import config
from uplift_targeting.evaluation import BLANKET, RESPONSE, UPLIFT

# Numbers quoted in the README (and checked against it by the tests).
HEADLINE_KEYS = (
    "MENS_CONV_PP",
    "MENS_CONV_CI",
    "MENS_SPEND",
    "WOMENS_SPEND",
    "CUPED_SPEND_SHRINK",
    "LIN_VISIT_SHRINK",
    "HOLM_SURVIVORS",
    "QINI_WOMENS_VISIT",
    "SPEND_QINI_SIGNIFICANT",
    "QINI_DIFF_WOMENS_VISIT",
    "QINI_DIFF_MENS_VISIT",
    "ANYWAY_RESPONSE",
    "ANYWAY_UPLIFT",
    "BLANKET_VALUE",
    "BLANKET_CI",
    "UPLIFT_VALUE",
    "UPLIFT_CI",
    "UPLIFT_SHARE",
    "RESPONSE_VALUE",
    "RESPONSE_CI",
    "UPLIFT_MINUS_BLANKET",
    "UPLIFT_MINUS_BLANKET_CI",
    "NAIVE_VALUE",
    "NAIVE_GAIN",
    "RHO_SPEND",
    "RHO_VISIT",
    "BREAK_EVEN_COST",
    "CRITEO_ROWS",
    "CRITEO_TOP20_RESPONSE_SHARE",
    "CRITEO_TOP20_DR_SHARE",
    "CRITEO_QINI_CONV_RESPONSE",
    "CRITEO_QINI_CONV_T",
    "CRITEO_QINI_CONV_DR",
)


@dataclass(frozen=True)
class Numbers:
    text: dict[str, str]


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.RESULTS / name)


def read_json(name: str) -> dict:
    return json.loads((config.RESULTS / name).read_text())


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def signed_money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"+${x:,.0f}"


def _ate(ate: pd.DataFrame, arm: str, outcome: str, estimator: str = "Difference in means"):
    return ate[(ate["arm"] == arm) & (ate["outcome"] == outcome) & (ate["estimator"] == estimator)]


def readout_numbers() -> dict[str, str]:
    ate = read_table("ate.csv")
    readout = read_json("readout.json")
    mc = _ate(ate, "Mens e-mail", "conversion").iloc[0]
    ms = _ate(ate, "Mens e-mail", "spend").iloc[0]
    ws = _ate(ate, "Womens e-mail", "spend").iloc[0]
    wc = _ate(ate, "Womens e-mail", "conversion").iloc[0]
    mv = _ate(ate, "Mens e-mail", "visit").iloc[0]
    cuped = _ate(ate, "Mens e-mail", "spend", "CUPED (history)").iloc[0]
    lin = _ate(ate, "Mens e-mail", "visit", "Regression adjustment (Lin)").iloc[0]
    tests = read_table("heterogeneity_tests.csv")
    survivors = tests[tests["p_holm"] < 0.05]
    return {
        "N": f"{readout['n']:,}",
        "SRM_P": f"{readout['srm']['p_value']:.2f}",
        "MAX_SMD": f"{readout['max_abs_smd']:.3f}",
        "MENS_CONV_PP": f"+{100 * mc['estimate']:.2f} pp",
        "MENS_CONV_CI": f"[{100 * mc['low']:.2f}, {100 * mc['high']:.2f}]",
        "MENS_CONV_LIFT": f"+{100 * mc['relative_lift']:.0f}%",
        "MENS_VISIT_PP": f"+{100 * mv['estimate']:.1f} pp",
        "MENS_SPEND": f"+${ms['estimate']:.2f}",
        "MENS_SPEND_CI": f"[${ms['low']:.2f}, ${ms['high']:.2f}]",
        "WOMENS_CONV_PP": f"+{100 * wc['estimate']:.2f} pp",
        "WOMENS_SPEND": f"+${ws['estimate']:.2f}",
        "CUPED_SPEND_SHRINK": f"{100 * (1 - cuped['ci_width_vs_dim']):.2f}%",
        "LIN_VISIT_SHRINK": f"{100 * (1 - lin['ci_width_vs_dim']):.1f}%",
        "HOLM_SURVIVORS": f"{len(survivors)} of {len(tests)}",
        "HOLM_SURVIVOR_DIMS": ", ".join(
            sorted({f"{r.arm} x {r.dimension} on {r.outcome}" for r in survivors.itertuples()})
        ),
    }


def model_numbers() -> dict[str, str]:
    q = read_table("qini.csv")

    def qini(arm: str, outcome: str, model: str = "DR-learner", base: str = "lgbm") -> str:
        r = q[(q["arm"] == arm) & (q["outcome"] == outcome) & (q["model"] == model)]
        r = r[r["base"] == base].iloc[0]
        return f"{1000 * r['qini']:.1f} [{1000 * r['qini_low']:.1f}, {1000 * r['qini_high']:.1f}]"

    paired = read_table("qini_dr_vs_response.csv").set_index(["arm", "outcome"])

    def diff(arm: str) -> str:
        r = paired.loc[(arm, "visit")]
        return (
            f"{1000 * r['dr_minus_response']:+.1f} "
            f"[{1000 * r['low']:+.1f}, {1000 * r['high']:+.1f}]"
        )

    spend = q[q["outcome"] == "spend"]
    deciles = read_table("deciles.csv").query("arm == 'Womens e-mail' and outcome == 'visit'")
    top = deciles[deciles["decile"] <= 2].groupby("ranking")[["control_rate", "treated_rate"]]
    top = top.mean()
    anyway = top["control_rate"] / top["treated_rate"]
    lift = deciles[deciles["decile"] <= 2].groupby("ranking")["uplift"].mean()
    return {
        "QINI_WOMENS_VISIT": qini("Womens e-mail", "visit"),
        "QINI_MENS_VISIT": qini("Mens e-mail", "visit"),
        "QINI_WOMENS_VISIT_RESPONSE": qini("Womens e-mail", "visit", RESPONSE),
        "SPEND_QINI_SIGNIFICANT": f"{int((spend['qini_low'] > 0).sum())} of {len(spend)}",
        "QINI_DIFF_WOMENS_VISIT": diff("Womens e-mail"),
        "QINI_DIFF_MENS_VISIT": diff("Mens e-mail"),
        "ANYWAY_RESPONSE": f"{anyway[RESPONSE]:.0%}",
        "ANYWAY_UPLIFT": f"{anyway['Uplift model']:.0%}",
        "TOP20_LIFT_UPLIFT": f"{100 * lift['Uplift model']:.1f} pp",
        "TOP20_LIFT_RESPONSE": f"{100 * lift[RESPONSE]:.1f} pp",
    }


def policy_numbers() -> dict[str, str]:
    s = read_json("policy_summary.json")["spend"]
    dep = s["deployable"]
    diff = s["uplift_minus_blanket"]
    cand = read_table("selection_candidates.csv")
    rho = {
        t: spearmanr(g["half0_per_1000"], g["half1_per_1000"]).statistic
        for t, g in cand.groupby("target")
    }
    grid = read_table("policy_by_cost.csv")
    margin = config.ECONOMICS.margin
    grid["profit"] = (
        margin * grid["spend_per_1000"]
        - 1000 * margin * grid["cost_over_margin"] * grid["share_emailed"]
    )
    wide = grid.pivot(index="cost_over_margin", columns="policy", values="profit")
    beats = wide.index[wide[UPLIFT] > wide[BLANKET]]
    break_even = 100 * margin * beats.min()
    naive = s["naive_in_sample"]
    blanket = dep[BLANKET]
    return {
        "BLANKET_VALUE": money(blanket["value_per_1000"]),
        "BLANKET_CI": f"[{money(blanket['low'])}, {money(blanket['high'])}]",
        "UPLIFT_VALUE": money(dep[UPLIFT]["value_per_1000"]),
        "UPLIFT_CI": f"[{money(dep[UPLIFT]['low'])}, {money(dep[UPLIFT]['high'])}]",
        "UPLIFT_SHARE": f"{dep[UPLIFT]['share_emailed']:.0%}",
        "RESPONSE_VALUE": money(dep[RESPONSE]["value_per_1000"]),
        "RESPONSE_SHARE": f"{dep[RESPONSE]['share_emailed']:.0%}",
        "RESPONSE_CI": f"[{money(dep[RESPONSE]['low'])}, {money(dep[RESPONSE]['high'])}]",
        "UPLIFT_MINUS_BLANKET": signed_money(diff["value_per_1000"]),
        "UPLIFT_MINUS_BLANKET_CI": f"[{signed_money(diff['low'])}, {signed_money(diff['high'])}]",
        "NAIVE_VALUE": money(naive["value_per_1000"]),
        "NAIVE_GAIN": signed_money(naive["value_per_1000"] - blanket["value_per_1000"]),
        "NAIVE_CHOICE": naive["choice"],
        "NAIVE_SHARE": f"{naive['share']:.0%}",
        "RHO_SPEND": f"{rho['spend']:.2f}",
        "RHO_VISIT": f"{rho['visit']:.2f}",
        "BREAK_EVEN_COST": f"{break_even:.0f} cents",
        "SELECTED": " and ".join(sorted(set(s["selected_rankings"].values()))),
    }


def criteo_numbers() -> dict[str, str]:
    summary = read_json("criteo_summary.json")
    q = read_table("criteo_qini.csv").set_index(["outcome", "model"])
    curves = read_table("criteo_uplift_curves.csv").query("outcome == 'conversion'")
    deciles = read_table("criteo_deciles.csv").query("outcome == 'visit' and model == @RESPONSE")

    def top20(model: str) -> str:
        c = curves[curves["model"] == model].set_index("fraction")["gain_per_1000"]
        return f"{c[0.2] / c[1.0]:.0%}"

    def qini(outcome: str, model: str) -> str:
        return f"{1000 * q.loc[(outcome, model), 'estimate']:.2f}"

    top = deciles[deciles["decile"] == 1].iloc[0]
    visit_curve = read_table("criteo_uplift_curves.csv").query(
        "outcome == 'visit' and model == @RESPONSE"
    )
    visit_curve = visit_curve.set_index("fraction")["gain_per_1000"]
    return {
        "CRITEO_ROWS": f"{summary['n_rows']:,}",
        "CRITEO_TEST": f"{summary['n_test']:,}",
        "CRITEO_TRAIN": f"{summary['n_train_sample'] / 1e6:.1f}M",
        "CRITEO_TOP20_RESPONSE_SHARE": top20(RESPONSE),
        "CRITEO_TOP20_DR_SHARE": top20("DR-learner"),
        "CRITEO_QINI_CONV_RESPONSE": qini("conversion", RESPONSE),
        "CRITEO_QINI_CONV_T": qini("conversion", "T-learner"),
        "CRITEO_QINI_CONV_DR": qini("conversion", "DR-learner"),
        "CRITEO_QINI_VISIT_RESPONSE": qini("visit", RESPONSE),
        "CRITEO_QINI_VISIT_DR": qini("visit", "DR-learner"),
        "CRITEO_DR_TIE": f"{q.loc[('conversion', 'DR-learner'), 'largest_tie']:.0%}",
        "CRITEO_TOP_DECILE_BASE": f"{100 * top['control_rate']:.0f}%",
        "CRITEO_TOP_DECILE_LIFT": f"+{100 * top['uplift']:.1f} pp",
        "CRITEO_TOP10_VISIT_SHARE": f"{visit_curve[0.1] / visit_curve[1.0]:.0%}",
        "CRITEO_ATE_CONV": f"+{100 * summary['average_effects']['conversion']['estimate']:.3f} pp",
        "CRITEO_ATE_VISIT": f"+{100 * summary['average_effects']['visit']['estimate']:.2f} pp",
    }


def load_numbers() -> Numbers:
    text = {**readout_numbers(), **model_numbers(), **policy_numbers(), **criteo_numbers()}
    return Numbers(text)
