"""Static README figures, drawn only from the tables in ``results/``.

Titles state the finding and are filled with measured numbers, so a re-run
that moves the numbers moves the titles with them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from uplift_targeting import config
from uplift_targeting.evaluation import BLANKET, RANDOM, RESPONSE, UPLIFT
from uplift_targeting.narrative import read_json, read_table, signed_money

INK, TEAL, AMBER, SLATE, GRID = "#0f172a", "#0d9488", "#d97706", "#64748b", "#e2e8f0"
LIGHT = "#cbd5e1"
SHORT_DIM = {
    "purchase_history": "Bought",
    "newbie": "Customer",
    "channel": "Channel",
    "zip_code": "Area",
    "history_tier": "Spend",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SLATE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "text.parse_math": False,  # dollar amounts are text, not TeX
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def _headline(fig: plt.Figure, title: str, subtitle: str, y: float = 0.98) -> None:
    fig.text(0.01, y, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, y - 0.075, subtitle, ha="left", va="top", fontsize=9.5, color=SLATE)


def _band(ax, x, y, low, high, color, label=None, ls="-", lw=2.2):
    ax.fill_between(x, low, high, color=color, alpha=0.13, lw=0)
    ax.plot(x, y, color=color, lw=lw, ls=ls, label=label)


def _end_label(ax, x, y, text, color, dy=0):
    ax.annotate(
        text,
        (x, y),
        xytext=(6, dy),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="bold",
        va="center",
    )


def profit_curves(margin: float, cost: float) -> pd.DataFrame:
    """Profit per 1,000 customers from the margin-free spend curves."""
    df = read_table("policy_curves.csv").query("outcome == 'spend' and estimator == 'dr'").copy()
    for col in ("value_per_1000", "low", "high"):
        df[col] = margin * df[col] - 1000 * cost * df["share"]
    return df


def hero() -> Path:
    s = read_json("policy_summary.json")
    econ = s["economics"]
    df = profit_curves(econ["margin"], econ["cost_per_email"])
    criteo = read_table("criteo_uplift_curves.csv").query("outcome == 'conversion'")
    cs = read_json("criteo_summary.json")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"wspace": 0.3})

    styles = {
        UPLIFT: (TEAL, "-", "Uplift model"),
        RESPONSE: (AMBER, "-", "Response model"),
        RANDOM: (SLATE, "--", "Random"),
    }
    for policy, (color, ls, label) in styles.items():
        d = df[df["policy"] == policy].sort_values("share")
        x = 100 * d["share"]
        _band(left, x, d["value_per_1000"], d["low"], d["high"], color, label=label, ls=ls)
    left.legend(loc="upper left", fontsize=9)
    left.axhline(0, color=INK, lw=0.8)
    blanket = s["spend"]["deployable"][BLANKET]
    left.annotate(
        f"E-mailing everyone: ${blanket['value_per_1000']:,.0f}\n"
        f"(95% CI ${blanket['low']:,.0f} to ${blanket['high']:,.0f})",
        (100, blanket["value_per_1000"]),
        xytext=(-150, 62),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": SLATE, "lw": 0.8},
    )
    left.set_xlim(0, 100)
    left.set_xlabel("Customers e-mailed (% of base, ranked by each policy)")
    left.set_ylabel("Incremental profit vs no e-mail ($ per 1,000 customers)")
    left.set_title("Hillstrom: 64,000 customers, 578 buyers", pad=8)

    cstyles = {
        "DR-learner": (TEAL, "DR-learner (uplift)"),
        RESPONSE: (AMBER, "Response model"),
    }
    for model, (color, label) in cstyles.items():
        d = criteo[criteo["model"] == model]
        x = 100 * d["fraction"]
        _band(right, x, d["gain_per_1000"], d["low"], d["high"], color, label=label)
    total = criteo[criteo["fraction"] == 1.0]["gain_per_1000"].iloc[0]
    right.plot([0, 100], [0, total], color=SLATE, ls="--", lw=1.6, label="Random")
    right.legend(loc="lower right", fontsize=9)
    at20 = criteo[criteo["fraction"] == 0.2].set_index("model")["gain_per_1000"]
    right.annotate(
        f"Top 20% by response model:\n{at20[RESPONSE] / total:.0%} of all incremental conversions",
        (20, at20[RESPONSE]),
        xytext=(14, 22),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": SLATE, "lw": 0.8},
    )
    right.set_xlim(0, 100)
    right.set_ylim(bottom=0)
    right.set_xlabel("Users targeted (% of held-out half, ranked by each model)")
    right.set_ylabel("Incremental conversions per 1,000 users")
    right.set_title(
        f"Criteo: {cs['n_rows'] / 1e6:.0f}M users, {cs['n_test'] / 1e6:.1f}M held out", pad=8
    )

    gain = s["spend"]["uplift_minus_blanket"]
    _headline(
        fig,
        "Whether to target, and with which model, is a question the experiment answers",
        f"Left: no targeting rule beats e-mailing everyone; the cross-selected uplift procedure is "
        f"{signed_money(gain['value_per_1000'])} per 1,000 vs blanket (95% CI "
        f"{signed_money(gain['low'])} to {signed_money(gain['high'])}). Right: effects are "
        f"concentrated where the\nbaseline rate is high, "
        f"so the plain response model is the best uplift ranker. Out-of-sample estimates, 95% CI "
        f"bands; Hillstrom profit assumes a {econ['margin']:.0%} margin and "
        f"${econ['cost_per_email']:.2f} per e-mail.",
        y=1.08,
    )
    return _save(fig, "hero.png")


def uplift_curves() -> Path:
    """Part B: visit uplift curves for both e-mails."""
    df = read_table("uplift_curves.csv").query("outcome == 'visit'")
    q = read_table("qini.csv").query("outcome == 'visit' and base == 'lgbm'")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, gridspec_kw={"wspace": 0.08})
    others = ["Transformed outcome", "S-learner", "T-learner", "X-learner"]
    for ax, arm in zip(axes, ("Womens e-mail", "Mens e-mail"), strict=True):
        d = df[df["arm"] == arm]
        for model in others:
            m = d[d["model"] == model]
            ax.plot(100 * m["fraction"], m["gain_per_1000"], color=LIGHT, lw=1.1)
        for model, color in (("DR-learner", TEAL), (RESPONSE, AMBER)):
            m = d[d["model"] == model]
            ax.plot(100 * m["fraction"], m["gain_per_1000"], color=color, lw=2.3)
            qi = q[(q["arm"] == arm) & (q["model"] == model)].iloc[0]
            ax.text(
                0.97,
                0.16 if model == "DR-learner" else 0.07,
                f"{model}: Qini ×1,000 {1000 * qi['qini']:.1f} "
                f"[{1000 * qi['qini_low']:.1f}, {1000 * qi['qini_high']:.1f}]",
                transform=ax.transAxes,
                ha="right",
                color=color,
                fontsize=9,
                fontweight="bold",
            )
        end = d[d["fraction"] == 1.0]["gain_per_1000"].iloc[0]
        ax.plot([0, 100], [0, end], color=SLATE, ls="--", lw=1.3)
        ax.set_title(arm, pad=6)
        ax.set_xlabel("Customers e-mailed (% of base, ranked by model)")
        ax.set_xlim(0, 100)
    axes[0].set_ylabel("Incremental visits per 1,000 customers")
    _headline(
        fig,
        "The Womens e-mail moves a findable subset of customers; for the Mens e-mail, ranking by response does better",
        "Visit uplift curves, out-of-fold LightGBM scores. Grey: other meta-learners. Dashed: "
        "random targeting. Bootstrap 95% CI on the Qini coefficient.",
        y=1.06,
    )
    return _save(fig, "uplift_curves.png")


def deciles() -> Path:
    df = read_table("deciles.csv").query("arm == 'Womens e-mail' and outcome == 'visit'")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, ranking, color in zip(axes, (RESPONSE, "Uplift model"), (AMBER, TEAL), strict=True):
        d = df[df["ranking"] == ranking]
        x = d["decile"].to_numpy()
        ax.bar(
            x - 0.2,
            100 * d["control_rate"],
            width=0.38,
            color=LIGHT,
            label="Visit without the e-mail (control)",
        )
        ax.bar(
            x + 0.2,
            100 * d["uplift"],
            width=0.38,
            color=color,
            label="Extra visits caused by the e-mail",
        )
        ax.errorbar(
            x + 0.2,
            100 * d["uplift"],
            yerr=[100 * (d["uplift"] - d["uplift_low"]), 100 * (d["uplift_high"] - d["uplift"])],
            fmt="none",
            ecolor=INK,
            lw=0.8,
            capsize=2,
        )
        ax.set_xticks(x)
        ax.set_xlabel(f"{ranking} decile (1 = top-ranked)")
        ax.set_title(ranking, pad=6)
        ax.legend(loc="upper right", fontsize=8.5)
    axes[0].set_ylabel("Visit rate (%)")
    top = df[df["decile"] <= 2].groupby("ranking")[["control_rate", "treated_rate", "uplift"]]
    top = top.mean()
    anyway = top["control_rate"] / top["treated_rate"]
    _headline(
        fig,
        f"In the response model's top 20%, {anyway[RESPONSE]:.0%} of e-mailed visitors would have "
        f"come anyway ({anyway['Uplift model']:.0%} for the uplift model)",
        f"Womens e-mail vs no e-mail. Top-20% incremental visit rate: uplift model "
        f"{100 * top.loc['Uplift model', 'uplift']:.1f} pp, response model "
        f"{100 * top.loc[RESPONSE, 'uplift']:.1f} pp. Bars: design-based rates per decile, "
        f"95% CI.",
        y=1.07,
    )
    return _save(fig, "deciles.png")


def selection() -> Path:
    """Do candidate rankings replicate from one half of the customers to the other?"""
    df = read_table("selection_candidates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.28})
    labels = {
        "visit": ("Target: visits", "incremental visits"),
        "spend": ("Target: spend", "incremental $ spend"),
    }
    rhos = {}
    for ax, target in zip(axes, ("visit", "spend"), strict=True):
        d = df[df["target"] == target]
        rho = spearmanr(d["half0_per_1000"], d["half1_per_1000"]).statistic
        rhos[target] = rho
        colors = d["trained_on"].map({"visit": TEAL, "conversion": AMBER, "spend": SLATE})
        ax.scatter(
            d["half0_per_1000"],
            d["half1_per_1000"],
            c=colors,
            s=34,
            edgecolor="white",
            lw=0.8,
            zorder=3,
        )
        lo = min(d["half0_per_1000"].min(), d["half1_per_1000"].min())
        hi = max(d["half0_per_1000"].max(), d["half1_per_1000"].max())
        ax.plot([lo, hi], [lo, hi], color=SLATE, lw=0.8, ls=":")
        title, unit = labels[target]
        ax.set_title(f"{title}: rank correlation {rho:.2f}", pad=6)
        ax.set_xlabel(f"Half A: mean {unit} per 1,000 over budgets")
        ax.set_ylabel(f"Half B: mean {unit} per 1,000")
    for name, color in (
        ("trained on visits", TEAL),
        ("on conversions", AMBER),
        ("on spend", SLATE),
    ):
        axes[1].scatter([], [], color=color, s=34, label=f"Candidate {name}")
    axes[1].legend(loc="lower right", fontsize=8.5)
    _headline(
        fig,
        "Which uplift model looks best for revenue depends on which customers you look at",
        f"30 candidate rankings (5 learners x 2 base models x 3 training outcomes), each scored "
        f"on two random halves of the customers.\nRank agreement between halves: "
        f"{rhos['spend']:.2f} for spend vs {rhos['visit']:.2f} for visits. Picking the best on "
        f"the data used to report it would overstate its value.",
        y=1.07,
    )
    return _save(fig, "selection.png")


def cost_sensitivity() -> Path:
    df = read_table("policy_by_cost.csv")
    margin = config.ECONOMICS.margin
    df = df[df["cost_over_margin"] * margin <= 0.6 + 1e-9].copy()
    df["cost"] = df["cost_over_margin"] * margin
    df["profit"] = margin * df["spend_per_1000"] - 1000 * df["cost"] * df["share_emailed"]
    df["half"] = margin * 1.96 * df["se_per_1000"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    styles = {
        BLANKET: (SLATE, "--", "E-mail everyone (Mens)"),
        RESPONSE: (AMBER, "-", RESPONSE),
        UPLIFT: (TEAL, "-", "Uplift model"),
    }
    for policy, (color, ls, label) in styles.items():
        d = df[df["policy"] == policy]
        x = 100 * d["cost"]
        _band(ax, x, d["profit"], d["profit"] - d["half"], d["profit"] + d["half"], color, ls=ls)
        _end_label(
            ax,
            x.iloc[-1],
            d["profit"].iloc[-1],
            label,
            color,
            dy={BLANKET: 0, RESPONSE: 8, UPLIFT: -8}[policy],
        )
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(100 * config.ECONOMICS.cost_per_email, color=SLATE, lw=0.8, ls=":")
    ax.set_xlabel("Fully loaded cost per e-mail (cents)")
    ax.set_ylabel("Incremental profit ($ per 1,000 customers)")
    ax.set_xlim(0, 60)
    _headline(
        fig,
        "Targeting only earns its keep once an e-mail costs about as much as it returns",
        f"Each procedure picks how many to e-mail on one half and is scored on the other. "
        f"Margin {margin:.0%}; dotted line: the default cost. 95% CI bands.",
        y=1.08,
    )
    fig.subplots_adjust(right=0.8)
    return _save(fig, "cost_sensitivity.png")


def criteo() -> Path:
    df = read_table("criteo_uplift_curves.csv")
    q = read_table("criteo_qini.csv")
    dec = read_table("criteo_deciles.csv").query("outcome == 'visit' and model == @RESPONSE")
    cs = read_json("criteo_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.25})
    styles = {RESPONSE: AMBER, "T-learner": SLATE, "DR-learner": TEAL}
    for ax, outcome in zip(axes, ("visit", "conversion"), strict=True):
        d = df[df["outcome"] == outcome]
        for k, (model, color) in enumerate(styles.items()):
            m = d[d["model"] == model]
            _band(ax, 100 * m["fraction"], m["gain_per_1000"], m["low"], m["high"], color, lw=2)
            qi = q[(q["outcome"] == outcome) & (q["model"] == model)].iloc[0]
            ax.text(
                0.97,
                0.26 - 0.09 * k,
                f"{model}: Qini ×1,000 {1000 * qi['estimate']:.2f} "
                f"[{1000 * qi['low']:.2f}, {1000 * qi['high']:.2f}]",
                transform=ax.transAxes,
                ha="right",
                color=color,
                fontsize=8.8,
                fontweight="bold",
            )
        end = d[d["fraction"] == 1.0]["gain_per_1000"].iloc[0]
        ax.plot([0, 100], [0, end], color=SLATE, ls=":", lw=1.3)
        ax.set_title(outcome.capitalize(), pad=6)
        ax.set_xlabel("Users targeted (% of held-out half, ranked by model)")
        ax.set_ylabel(f"Incremental {outcome}s per 1,000 users")
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
    top = dec[dec["decile"] == 1].iloc[0]
    _headline(
        fig,
        "At 14M users the effect tracks the baseline rate, and no meta-learner beats the response model",
        f"Criteo Uplift v2.1, fitted on {cs['n_train_sample'] / 1e6:.1f}M rows of one half, "
        f"evaluated on the other {cs['n_test'] / 1e6:.1f}M. The response "
        f"model's top decile visits {100 * top['control_rate']:.0f}% of the time untreated and "
        f"gains {100 * top['uplift']:.1f} pp when treated.\nBands: pointwise 95% CI; Qini: 20 "
        f"random-group 95% CI. Dotted: random targeting.",
        y=1.08,
    )
    return _save(fig, "criteo.png")


def segments() -> Path:
    lv = read_table("segments.csv")
    tests = read_table("heterogeneity_tests.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True, gridspec_kw={"wspace": 0.06})
    dims = list(config.SEGMENTS)
    for ax, outcome, unit in zip(
        axes, ("visit", "conversion"), ("visit", "conversion"), strict=True
    ):
        labels, ypos, y = [], [], 0.0
        for dim in dims:
            levels = lv[(lv["dimension"] == dim) & (lv["outcome"] == outcome)]
            for level in levels["level"].unique():
                for k, (arm, color) in enumerate((("Mens e-mail", TEAL), ("Womens e-mail", AMBER))):
                    r = levels[(levels["level"] == level) & (levels["arm"] == arm)].iloc[0]
                    yy = -(y + 0.22 * (k - 0.5) * 2)
                    ax.errorbar(
                        100 * r["estimate"],
                        yy,
                        xerr=[
                            [100 * (r["estimate"] - r["low"])],
                            [100 * (r["high"] - r["estimate"])],
                        ],
                        fmt="o",
                        color=color,
                        ms=4,
                        lw=1.2,
                        label=arm
                        if (dim == dims[0] and level == levels["level"].unique()[0])
                        else None,
                    )
                labels.append(f"{SHORT_DIM[dim]}: {level}")
                ypos.append(-y)
                y += 1
            t = tests[(tests["dimension"] == dim) & (tests["outcome"] == outcome)]
            p = t["p_holm"].min()
            ax.text(
                0.99,
                -(y - len(levels["level"].unique()) / 2 - 0.5),
                "Holm p < 0.001"
                if p < 0.001
                else ("Holm p = 1.00" if p > 0.995 else f"Holm p ≥ {p:.2f}"),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
                color=SLATE,
            )
            y += 0.7
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(f"Effect on {unit} rate", pad=6)
        ax.set_xlabel("Percentage points (e-mail minus control), 95% CI")
    axes[0].set_yticks(ypos, labels)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper right",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.99, 0.965),
    )
    n_conv = tests[tests["outcome"] != "visit"]
    n_sig = int((n_conv["p_holm"] < 0.05).sum())
    n_visit = int((tests[tests["outcome"] == "visit"]["p_holm"] < 0.05).sum())
    _headline(
        fig,
        "Pre-declared segments: visits differ by product line; purchases show no "
        "detectable heterogeneity",
        f"{n_visit} of 10 visit tests and {n_sig} of {len(n_conv)} conversion/spend tests "
        f"survive a Holm correction over all {len(tests)} heterogeneity tests.",
        y=1.05,
    )
    return _save(fig, "segments.png")


def run() -> list[Path]:
    _style()
    return [
        hero(),
        uplift_curves(),
        deciles(),
        selection(),
        cost_sensitivity(),
        segments(),
        criteo(),
    ]
