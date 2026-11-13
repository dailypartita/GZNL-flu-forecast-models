#!/usr/bin/env python3
"""Academic-style reliability figures from the rolling-origin backtest."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "time-series.csv"
FORECAST = ROOT / "output" / "GZNL-FluLGBMCQR" / "2026-11-09-GZNL-FluLGBMCQR.csv"

QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.95, 0.975, 0.99,
])
QCOLS = [f"q_{q}" for q in QUANTILES]

# Okabe–Ito, colorblind-safe
COL = {
    "lightgbm_cqr": "#0072B2",
    "lightgbm": "#56B4E9",
    "ensemble_cqr": "#E69F00",
    "ensemble": "#F0E442",
    "clim_persist": "#009E73",
    "clim_persist_cqr": "#CC79A7",
    "seasonal_naive": "#999999",
}
LABEL = {
    "lightgbm_cqr": "LightGBM+CQR (submitted)",
    "lightgbm": "LightGBM",
    "ensemble_cqr": "Ensemble+CQR",
    "ensemble": "Ensemble",
    "clim_persist": "Climatology+persistence",
    "clim_persist_cqr": "Climatology+persistence+CQR",
    "seasonal_naive": "Seasonal naive",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.unicode_minus": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_fair() -> pd.DataFrame:
    pred = pd.read_csv(RESULTS / "backtest_predictions.csv", parse_dates=["reference_date", "target_end_date"])
    keys = (
        pred.loc[pred["model"] == "lightgbm_cqr", ["horizon", "reference_date"]]
        .drop_duplicates()
    )
    fair = pred.merge(keys, on=["horizon", "reference_date"], how="inner")
    return fair


def pit_values(df: pd.DataFrame) -> np.ndarray:
    qs = df[QCOLS].to_numpy(dtype=float)
    y = df["actual"].to_numpy(dtype=float)
    pits = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        q = qs[i]
        yi = y[i]
        if yi <= q[0]:
            pits[i] = 0.0 if yi < q[0] else float(QUANTILES[0])
        elif yi >= q[-1]:
            pits[i] = 1.0 if yi > q[-1] else float(QUANTILES[-1])
        else:
            pits[i] = float(np.interp(yi, q, QUANTILES))
    return pits


def panel_label(ax, text: str) -> None:
    ax.text(
        0.0, 1.04, text, transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="bottom", ha="left",
    )


def save(fig, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path)
        paths.append(path)
    return paths


def fig_accuracy(fair: pd.DataFrame) -> None:
    models = ["lightgbm_cqr", "lightgbm", "ensemble_cqr", "clim_persist", "seasonal_naive"]
    g = (
        fair[fair["model"].isin(models)]
        .groupby(["model", "horizon"], as_index=False)
        .agg(WIS=("wis", "mean"), MAE=("mae", "mean"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True)
    for ax, metric, ylab in zip(axes, ["WIS", "MAE"], ["Weighted interval score (pp)", "Mean absolute error (pp)"]):
        for m in models:
            sub = g[g["model"] == m].sort_values("horizon")
            is_sub = m == "lightgbm_cqr"
            ax.plot(
                sub["horizon"], sub[metric],
                color=COL[m],
                lw=1.9 if is_sub else 1.2,
                ls="-" if is_sub else ("--" if m == "lightgbm" else "-"),
                marker="o" if is_sub else ("D" if m == "lightgbm" else "o"),
                ms=4.0 if is_sub else 3.4,
                mfc=COL[m] if is_sub else "white",
                mew=1.0,
                label=LABEL[m],
                zorder=4 if is_sub else 3,
            )
        ax.set_xlabel("Horizon (weeks)")
        ax.set_ylabel(ylab)
        ax.set_xticks(range(-1, 7))
        ax.yaxis.set_major_locator(MultipleLocator(1 if metric == "MAE" else 1))
        ax.set_xlim(-1.3, 6.3)
        ax.set_ylim(0, None)
        ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig1_accuracy_by_horizon")
    plt.close(fig)


def fig_calibration(fair: pd.DataFrame) -> None:
    models = ["lightgbm", "lightgbm_cqr"]
    cov = (
        fair[fair["model"].isin(models)]
        .groupby(["model", "horizon"], as_index=False)
        .agg(c50=("cover50", "mean"), c80=("cover80", "mean"), c95=("cover95", "mean"))
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85))

    # (a, b) coverage
    specs = [
        ("c50", 0.50, "50% interval coverage"),
        ("c80", 0.80, "80% interval coverage"),
        ("c95", 0.95, "95% interval coverage"),
    ]
    # Put all three coverages in one panel instead? User wants academic proof.
    # Better: one coverage panel with 3 nominals, one PIT, one width.
    axes[0].remove()
    axes[1].remove()
    axes[2].remove()
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35))
    ax = axes[0]
    styles = {
        ("lightgbm", "c50"): dict(color=COL["lightgbm"], ls="--", marker="s", ms=3.5, lw=1.1),
        ("lightgbm", "c80"): dict(color=COL["lightgbm"], ls="--", marker="o", ms=3.5, lw=1.1),
        ("lightgbm", "c95"): dict(color=COL["lightgbm"], ls="--", marker="^", ms=3.5, lw=1.1),
        ("lightgbm_cqr", "c50"): dict(color=COL["lightgbm_cqr"], ls="-", marker="s", ms=3.8, lw=1.6),
        ("lightgbm_cqr", "c80"): dict(color=COL["lightgbm_cqr"], ls="-", marker="o", ms=3.8, lw=1.6),
        ("lightgbm_cqr", "c95"): dict(color=COL["lightgbm_cqr"], ls="-", marker="^", ms=3.8, lw=1.6),
    }
    # Cleaner coverage panel: only CQR solid vs raw dashed, three levels as different markers, plus nominal hlines
    for m in models:
        sub = cov[cov["model"] == m].sort_values("horizon")
        for col, nom, _ in specs:
            key = (m, col)
            ax.plot(sub["horizon"], sub[col], **styles[key], zorder=3)
    for nom, ls in ((0.5, ":"), (0.8, ":"), (0.95, ":")):
        ax.axhline(nom, color="#555555", ls=ls, lw=0.7, zorder=1)
    ax.set_xlabel("Horizon (weeks)")
    ax.set_ylabel("Empirical coverage")
    ax.set_xticks(range(-1, 7))
    ax.set_ylim(0.38, 1.05)
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    legend_elems = [
        Line2D([0], [0], color=COL["lightgbm"], ls="--", lw=1.2, label="LightGBM"),
        Line2D([0], [0], color=COL["lightgbm_cqr"], ls="-", lw=1.6, label="LightGBM+CQR"),
        Line2D([0], [0], color="#333333", marker="s", ls="none", ms=4, label="50%"),
        Line2D([0], [0], color="#333333", marker="o", ls="none", ms=4, label="80%"),
        Line2D([0], [0], color="#333333", marker="^", ls="none", ms=4, label="95%"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", bbox_to_anchor=(0.0, 1.12), ncol=2, fontsize=6.4, handlelength=1.6, borderpad=0.15, labelspacing=0.25, columnspacing=0.9, frameon=False)
    panel_label(ax, "a")

    # (b) PIT
    ax = axes[1]
    pits_raw = pit_values(fair[fair["model"] == "lightgbm"])
    pits_cqr = pit_values(fair[fair["model"] == "lightgbm_cqr"])
    bins = np.linspace(0, 1, 11)
    ax.hist(pits_raw, bins=bins, density=True, histtype="step", color=COL["lightgbm"], lw=1.2, label="LightGBM")
    ax.hist(pits_cqr, bins=bins, density=True, color=COL["lightgbm_cqr"], alpha=0.55, edgecolor=COL["lightgbm_cqr"], lw=0.6, label="LightGBM+CQR")
    ax.axhline(1.0, color="#555555", ls=":", lw=0.8, label="Uniform")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability integral transform")
    ax.set_ylabel("Density")
    ax.set_ylim(0, 2.2)
    ax.legend(loc="upper right", fontsize=6.8)
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    panel_label(ax, "b")

    # (c) sharpness: mean 95% width
    ax = axes[2]
    tmp = fair[fair["model"].isin(models)].copy()
    tmp["w95"] = tmp["q_0.975"] - tmp["q_0.025"]
    tmp["w50"] = tmp["q_0.75"] - tmp["q_0.25"]
    w = tmp.groupby(["model", "horizon"], as_index=False).agg(w95=("w95", "mean"), w50=("w50", "mean"))
    for m, lw in (("lightgbm", 1.2), ("lightgbm_cqr", 1.7)):
        sub = w[w["model"] == m].sort_values("horizon")
        ax.plot(sub["horizon"], sub["w95"], color=COL[m], lw=lw, marker="o", ms=3.5, label=LABEL[m] + ", 95%")
        ax.plot(sub["horizon"], sub["w50"], color=COL[m], lw=lw, ls="--", marker="s", ms=3.2, label=LABEL[m] + ", 50%")
    ax.set_xlabel("Horizon (weeks)")
    ax.set_ylabel("Mean interval width (pp)")
    ax.set_xticks(range(-1, 7))
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.8)
    panel_label(ax, "c")

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig2_calibration")
    plt.close(fig)


def fig_backtest_series(fair: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True)
    for ax, h, lab in zip(axes, [1, 4], ["a", "b"]):
        sub = fair[(fair["model"] == "lightgbm_cqr") & (fair["horizon"] == h)].sort_values("target_end_date")
        t = sub["target_end_date"]
        ax.fill_between(t, sub["q_0.025"], sub["q_0.975"], color=COL["lightgbm_cqr"], alpha=0.18, lw=0, label="95% interval", zorder=1)
        ax.fill_between(t, sub["q_0.25"], sub["q_0.75"], color=COL["lightgbm_cqr"], alpha=0.32, lw=0, label="50% interval", zorder=2)
        ax.plot(t, sub["q_0.5"], color=COL["lightgbm_cqr"], lw=1.15, label="Median forecast", zorder=3)
        ax.plot(t, sub["actual"], color="black", lw=1.0, marker="o", ms=2.0, mew=0, label="Observed", zorder=4)
        ax.set_ylabel("ILI specimen positivity (pp)")
        ax.set_ylim(0, None)
        ax.xaxis.set_major_locator(YearLocator(base=1))
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.set_xlim(pd.Timestamp("2023-12-15"), pd.Timestamp("2027-01-20"))
        ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
        ax.set_title(f"Horizon = +{h} week{'s' if h != 1 else ''}", loc="left", fontsize=9, pad=14)
        panel_label(ax, lab)
        n = len(sub)
        cov95 = sub["cover95"].mean()
        cov50 = sub["cover50"].mean()
        mae = sub["mae"].mean()
        ax.text(
            0.99, 0.97, f"n = {n}\nMAE = {mae:.2f} pp\n50%/95% coverage = {cov50:.2f}/{cov95:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
        )
    axes[1].set_xlabel("Target week ending")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig3_backtest_timeseries")
    plt.close(fig)


def fig_submission_fan() -> None:
    ts = pd.read_csv(DATA, parse_dates=["target_end_date"])
    flu = ts[(ts["target"] == "wk inc flu prop ili") & (ts["location"] == "CN")].copy()
    flu = flu.sort_values("target_end_date")
    hist_start = pd.Timestamp("2025-01-06")
    hist = flu[flu["target_end_date"] >= hist_start]

    fc = pd.read_csv(FORECAST, parse_dates=["target_end_date", "reference_date"])
    fc["output_type_id"] = fc["output_type_id"].astype(float)
    wide = fc.pivot(index="target_end_date", columns="output_type_id", values="value").sort_index()

    last = flu["target_end_date"].max()
    last_y = float(flu.loc[flu["target_end_date"] == last, "observation"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.plot(hist["target_end_date"], hist["observation"], color="black", lw=1.15, label="Observed", zorder=4)
    t = wide.index
    ax.fill_between(t, wide[0.025], wide[0.975], color=COL["lightgbm_cqr"], alpha=0.18, lw=0, label="95% interval", zorder=1)
    ax.fill_between(t, wide[0.25], wide[0.75], color=COL["lightgbm_cqr"], alpha=0.32, lw=0, label="50% interval", zorder=2)
    ax.plot(t, wide[0.5], color=COL["lightgbm_cqr"], lw=1.5, marker="o", ms=3.5, label="Median forecast", zorder=3)
    # connect last obs to first forecast median
    ax.plot([last, t[0]], [last_y, wide[0.5].iloc[0]], color=COL["lightgbm_cqr"], lw=1.0, ls=":", zorder=3)
    ax.set_ylabel("ILI specimen positivity (pp)")
    ax.set_xlabel("Week ending")
    ax.set_ylim(0, max(float(hist["observation"].max()), float(wide[0.975].max())) * 1.12)
    ax.axvline(pd.Timestamp("2026-11-09"), color="#555555", ls="--", lw=0.7, zorder=1)
    ymin, ymax = ax.get_ylim()
    ax.text(
        pd.Timestamp("2026-11-16"), ymax * 0.97,
        "Reference date\n2026-11-09",
        color="#555555", fontsize=7, va="top",
    )
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Submitted GZNL-FluLGBMCQR forecast, reference date 2026-11-09", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_submission_forecast")
    plt.close(fig)


def fig_reliability_diagram(fair: pd.DataFrame) -> None:
    """Observed coverage versus nominal interval level (all horizons pooled)."""
    levels = [
        (0.50, "q_0.25", "q_0.75"),
        (0.70, "q_0.15", "q_0.85"),
        (0.80, "q_0.1", "q_0.9"),
        (0.90, "q_0.05", "q_0.95"),
        (0.95, "q_0.025", "q_0.975"),
        (0.98, "q_0.01", "q_0.99"),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for model, marker, ms, lw, z in (
        ("lightgbm", "s", 5, 1.1, 2),
        ("lightgbm_cqr", "o", 6, 1.6, 3),
    ):
        sub = fair[fair["model"] == model]
        xs, ys = [], []
        for nom, lo, hi in levels:
            cov = ((sub["actual"] >= sub[lo]) & (sub["actual"] <= sub[hi])).mean()
            xs.append(nom)
            ys.append(cov)
        ax.plot(xs, ys, color=COL[model], marker=marker, ms=ms, lw=lw, label=LABEL[model], zorder=z)
    ax.plot([0.45, 1.0], [0.45, 1.0], color="#555555", ls=":", lw=0.8, zorder=1, label="Perfect calibration")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.grid(color="#dddddd", lw=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Pooled over horizons −1 to 6 (n = 1008)", loc="left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_reliability_diagram")
    plt.close(fig)


def main() -> None:
    apply_style()
    fair = load_fair()
    assert fair[fair["model"] == "lightgbm_cqr"].shape[0] == 1008
    fig_accuracy(fair)
    fig_calibration(fair)
    fig_backtest_series(fair)
    fig_submission_fan()
    fig_reliability_diagram(fair)
    print("Wrote figures to", OUT)
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:40s} {p.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()
