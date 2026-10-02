"""Build ``site/index.html``: one self-contained page, narrative plus interactive charts.

Every number on the page is read from ``results/`` through
:mod:`uplift_targeting.narrative`. Chart data is embedded as JSON and drawn
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
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", double_precision=6))


def chart_data() -> dict:
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


def ate_rows() -> str:
    ate = pd.read_csv(config.RESULTS / "ate.csv")
    rows = []
    for _, r in ate.iterrows():
        o = r["outcome"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(r['arm'])}</td><td>{r['outcome']}</td>"
            f"<td>{html.escape(r['estimator'])}</td>"
            f"<td class=num>{_effect(r['estimate'], o)}</td>"
            f"<td class=num>{_effect(r['low'], o)} to {_effect(r['high'], o)}</td>"
            f"<td class=num>{r['ci_width_vs_dim']:.4f}</td>"
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


def run() -> Path:
    numbers = load_numbers().text
    page = TEMPLATE.read_text()
    replacements = {
        "{{PLOTLY}}": PLOTLY,
        "{{DATA}}": json.dumps(chart_data(), separators=(",", ":")),
        "{{ATE_ROWS}}": ate_rows(),
        "{{QINI_ROWS}}": qini_rows(),
        "{{MARGIN}}": f"{config.ECONOMICS.margin}",
        "{{COST}}": f"{config.ECONOMICS.cost_per_email}",
        "{{COST_FMT}}": f"${config.ECONOMICS.cost_per_email:.2f}",
        "{{MARGIN_FMT}}": f"{config.ECONOMICS.margin:.0%}",
        **{f"{{{{{k}}}}}": html.escape(v) for k, v in numbers.items()},
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    leftover = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", page)))
    if leftover:
        raise KeyError(f"Unfilled placeholders in the report template: {leftover}")
    config.SITE.mkdir(parents=True, exist_ok=True)
    out = config.SITE / "index.html"
    out.write_text(page)
    return out
