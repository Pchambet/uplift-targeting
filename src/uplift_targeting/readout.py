"""Part A: the experiment readout, written to ``results/``.

Produces the numbers a decision memo needs: is the randomisation sound, how
large are the average effects (with and without variance reduction), and do
they differ across the segments declared in ``config.SEGMENTS``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from uplift_targeting import config
from uplift_targeting.data import Experiment
from uplift_targeting.experiment import (
    benjamini_hochberg,
    cuped,
    difference_in_means,
    holm,
    regression_adjusted,
    segment_effects,
    srm_test,
    standardized_mean_differences,
)

SEGMENT_LABELS = {
    "zip_code": {"Surburban": "Suburban"},  # the raw file misspells it
    "newbie": {0: "Returning", 1: "New"},
    "history_tier": {
        1: "$0-100",
        2: "$100-200",
        3: "$200-350",
        4: "$350-500",
        5: "$500-750",
        6: "$750-1,000",
        7: "$1,000+",
    },
}


def _pair(exp: Experiment, arm: int) -> np.ndarray:
    """Rows of one e-mail arm and the control arm."""
    return np.isin(exp.arm, [0, arm])


def ate_table(exp: Experiment) -> pd.DataFrame:
    """Average effect of each e-mail on each outcome, three estimators side by side."""
    rows = []
    covariates = exp.X.to_numpy()
    for arm in (1, 2):
        mask = _pair(exp, arm)
        t = (exp.arm[mask] == arm).astype(float)
        for outcome in config.OUTCOMES:
            y = exp.outcomes[outcome].to_numpy()[mask]
            control_mean = float(y[t == 0].mean())
            estimators = {
                "Difference in means": difference_in_means(y, t),
                "CUPED (history)": cuped(y, t, exp.X["history"].to_numpy()[mask]),
                "Regression adjustment (Lin)": regression_adjusted(y, t, covariates[mask]),
            }
            base_width = estimators["Difference in means"].se
            for name, eff in estimators.items():
                rows.append(
                    {
                        "arm": config.ARM_LABELS[arm],
                        "outcome": outcome,
                        "estimator": name,
                        "control_mean": control_mean,
                        **eff.as_dict(),
                        "relative_lift": eff.estimate / control_mean,
                        "ci_width_vs_dim": eff.se / base_width,
                    }
                )
    return pd.DataFrame(rows)


def segment_table(exp: Experiment) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-segment effects plus one heterogeneity test per (arm, outcome, dimension)."""
    level_rows, test_rows = [], []
    for arm in (1, 2):
        mask = _pair(exp, arm)
        t = (exp.arm[mask] == arm).astype(float)
        for outcome in config.OUTCOMES:
            y = exp.outcomes[outcome].to_numpy()[mask]
            for dim in config.SEGMENTS:
                seg = exp.raw[dim][mask].replace(SEGMENT_LABELS.get(dim, {})).astype(str)
                table, test = segment_effects(y, t, seg.reset_index(drop=True))
                key = {"arm": config.ARM_LABELS[arm], "outcome": outcome, "dimension": dim}
                level_rows += [{**key, **row} for row in table.to_dict("records")]
                test_rows.append({**key, **test})
    levels, tests = pd.DataFrame(level_rows), pd.DataFrame(test_rows)
    # One family = every (arm, outcome, dimension) heterogeneity test we ran.
    tests["p_holm"] = holm(tests["p_value"].to_numpy())
    tests["p_bh"] = benjamini_hochberg(tests["p_value"].to_numpy())
    levels["q_bh"] = benjamini_hochberg(levels["p_value"].to_numpy())
    return levels, tests


def balance_table(exp: Experiment) -> pd.DataFrame:
    cols = {}
    for arm in (1, 2):
        mask = _pair(exp, arm)
        t = (exp.arm[mask] == arm).astype(int)
        cols[config.ARM_LABELS[arm]] = standardized_mean_differences(exp.X[mask], t)
    return pd.DataFrame(cols).rename_axis("covariate").reset_index()


def run(exp: Experiment) -> dict:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    srm = srm_test(exp.arm, dict.fromkeys(config.ARM_LABELS, config.DESIGN_PROPENSITY))
    balance = balance_table(exp)
    ate = ate_table(exp)
    levels, tests = segment_table(exp)
    balance.to_csv(config.RESULTS / "balance.csv", index=False)
    ate.to_csv(config.RESULTS / "ate.csv", index=False)
    levels.to_csv(config.RESULTS / "segments.csv", index=False)
    tests.to_csv(config.RESULTS / "heterogeneity_tests.csv", index=False)
    summary = {
        "n": exp.n,
        "srm": srm,
        "max_abs_smd": float(balance.drop(columns="covariate").abs().max().max()),
        "n_heterogeneity_tests": len(tests),
        "n_significant_holm": int((tests["p_holm"] < 0.05).sum()),
    }
    (config.RESULTS / "readout.json").write_text(json.dumps(summary, indent=2))
    return summary
