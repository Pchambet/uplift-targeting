"""Project-wide constants: paths, data sources, economics and analysis choices.

Everything a reader might want to challenge (margin, cost per e-mail, segments,
seeds) lives here so that a single file documents the assumptions behind every
number in the README.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The repository root in a source checkout; set UPLIFT_TARGETING_ROOT to run an
# installed copy against another directory (data/, results/, docs/, site/).
ROOT = Path(os.environ.get("UPLIFT_TARGETING_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
SITE = ROOT / "site"

SEED = 20080320  # the date of the MineThatData challenge release

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)

# Treatment arms of the Hillstrom experiment. Integer codes are used everywhere
# downstream; 0 is always the control ("no e-mail") arm.
ARMS: dict[str, int] = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
ARM_LABELS: dict[int, str] = {0: "No e-mail", 1: "Mens e-mail", 2: "Womens e-mail"}
DESIGN_PROPENSITY = 1.0 / 3.0  # the challenge states a 1/3-1/3-1/3 randomisation

OUTCOMES = ("visit", "conversion", "spend")

# Segments declared before any model was fitted. Heterogeneity is only tested
# along these dimensions so that the multiple-testing correction has a fixed family.
SEGMENTS: dict[str, str] = {
    "purchase_history": "Product line bought in the past year",
    "newbie": "New customer (first purchase in the past 12 months)",
    "channel": "Purchase channel in the past year",
    "zip_code": "Area type",
    "history_tier": "Past-year spend tier",
}


@dataclass(frozen=True)
class Economics:
    """Unit economics used to turn incremental spend into incremental profit.

    The dataset ships no margins or costs. The defaults below are stated
    assumptions (apparel-retail-like gross margin; a fully loaded contact cost
    that includes fatigue and unsubscribe risk, not just the send fee). The
    report shows how the conclusions move when the cost changes.
    """

    margin: float = 0.40
    cost_per_email: float = 0.15


ECONOMICS = Economics()
COST_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

N_OUTER_FOLDS = 5  # every customer is scored by models that never saw them
N_INNER_FOLDS = 5  # cross-fitting inside the DR-learner
N_BOOTSTRAP = 500
BUDGET_GRID = tuple(round(0.05 * i, 2) for i in range(21))

LIGHTGBM_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "n_jobs": 1,  # small data: one thread per model, parallelism across folds instead
    "verbose": -1,
}
N_WORKERS = 3  # outer folds scored in parallel; keeps the CPU footprint at three cores
