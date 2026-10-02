"""Build ``site/index.html`` and render ``README.md`` from their templates.

Every number on the page and in the README is read from ``results/`` through
:mod:`uplift_targeting.narrative`, so neither can drift from the outputs. Chart data is embedded as JSON and drawn
with Plotly (jsDelivr). The profit chart is re-priced in the browser when the
reader moves the margin or cost sliders; this is exact because margin and cost
enter the policy value linearly, and the budget choices of the deployable
procedures were precomputed for every cost / margin ratio.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from uplift_targeting import config
from uplift_targeting.narrative import load_numbers

TEMPLATE = Path(__file__).with_name("report_template.html")
README_TEMPLATE = Path(__file__).with_name("readme_template.md")
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(df.to_json(orient="records", double_precision=6))


def chart_data() -> dict[str, object]:
    r = config.RESULTS
    policy = pd.read_csv(r / "policy_curves.csv").query("estimator == 'dr'")
    return {
        "economics": {"margin": config.ECONOMICS.margin, "cost": config.ECONOMICS.cost_per_email},
        "spendCurves": _records(policy.query("outcome == 'spend'").drop(columns="estimator")),
        "byCost": _records(pd.read_csv(r / "policy_by_cost.csv")),
        "upliftCurves": _records(pd.read_csv(r / "uplift_curves.csv")),
        "deciles": _records(pd.read_csv(r / "deciles.csv")),
        "segments": _records(pd.read_csv(r / "segments.csv")),
        "candidates": _records(pd.read_csv(r / "selection_candidates.csv")),
        "criteo": _records(pd.read_csv(r / "criteo_uplift_curves.csv")),
    }


def _effect(value: float, outcome: str) -> str:
    return f"${value:.3f}" if outcome == "spend" else f"{100 * value:.2f} pp"


def _bound(value: float, outcome: str) -> str:
    """A CI bound in the effect's units, without repeating the unit suffix."""
    return f"${value:.3f}" if outcome == "spend" else f"{100 * value:.2f}"


def ate_rows() -> str:
    """Average-effect table body: one group per e-mail and outcome, one row per estimator."""
    ate = pd.read_csv(config.RESULTS / "ate.csv")
    rows = []
    for (arm, outcome), d in ate.groupby(["arm", "outcome"], sort=False):
        rows.append(
            f"<tr class=group><th colspan=4 scope=rowgroup>{html.escape(arm)} · {outcome}</th></tr>"
        )
        for _, r in d.iterrows():
            rows.append(
                "<tr>"
                f"<td>{html.escape(r['estimator'])}</td>"
                f"<td class=num>{_effect(r['estimate'], outcome)}</td>"
                f"<td class=num>[{_bound(r['low'], outcome)}, {_bound(r['high'], outcome)}]</td>"
                f"<td class='num wide-only'>{r['ci_width_vs_dim']:.4f}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def qini_rows() -> str:
    q = pd.read_csv(config.RESULTS / "qini.csv")
    rows = []
    for (outcome, arm), d in q.groupby(["outcome", "arm"], sort=False):
        for _, r in d.sort_values("qini", ascending=False).iterrows():
            rows.append(
                "<tr>"
                f"<td>{arm}</td><td>{outcome}</td><td>{html.escape(r['model'])}</td>"
                f"<td>{'LightGBM' if r['base'] == 'lgbm' else 'regularised linear'}</td>"
                f"<td class=num>{1000 * r['qini']:.2f}</td>"
                f"<td class=num>{1000 * r['qini_low']:.2f} to {1000 * r['qini_high']:.2f}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def fill(template: str, replacements: dict[str, str]) -> str:
    """Replace every ``{{KEY}}``; a placeholder left unfilled is an error."""
    for key, value in replacements.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    leftover = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", template)))
    if leftover:
        raise KeyError(f"Unfilled placeholders: {leftover}")
    return template


def render_readme(numbers: dict[str, str] | None = None) -> str:
    return fill(README_TEMPLATE.read_text(), numbers or load_numbers())


def render_page(numbers: dict[str, str] | None = None) -> str:
    numbers = numbers or load_numbers()
    return fill(
        TEMPLATE.read_text(),
        {
            "PLOTLY": PLOTLY,
            "DATA": json.dumps(chart_data(), separators=(",", ":")),
            "ATE_ROWS": ate_rows(),
            "QINI_ROWS": qini_rows(),
            "MARGIN": f"{config.ECONOMICS.margin}",
            "COST": f"{config.ECONOMICS.cost_per_email}",
            **{k: html.escape(v) for k, v in numbers.items()},
        },
    )


def run() -> list[Path]:
    numbers = load_numbers()
    config.SITE.mkdir(parents=True, exist_ok=True)
    page = config.SITE / "index.html"
    page.write_text(render_page(numbers))
    readme = config.ROOT / "README.md"
    readme.write_text(render_readme(numbers))
    return [page, readme]
