"""Paper-ready plotting functions for TurnZero evaluation.

Produces reliability diagrams, top-k comparison bar charts, and stratified
comparison tables (JSON + optional LaTeX).

All figures saved at 300 DPI in both PDF and PNG.
Reference: docs/PROJECT_BIBLE.md Section 4.3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)

_DPI = 300


def _save_fig(fig: plt.Figure, out_path: str | Path) -> None:
    """Save figure as both PNG and PDF next to *out_path*."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=_DPI)
    fig.savefig(out.with_suffix(".pdf"), dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def reliability_diagram(
    probs: np.ndarray,
    action90_true: np.ndarray,
    out_path: str | Path,
    title: str = "",
    n_bins: int = 15,
) -> None:
    """Two-panel reliability diagram.

    Top panel: accuracy vs confidence per bin (with identity diagonal).
    Bottom panel: histogram of confidence values.

    Args:
        probs: (N, 90) predicted probabilities.
        action90_true: (N,) ground-truth action90 ids.
        out_path: base path for saved figure (PNG + PDF).
        title: optional figure title.
        n_bins: number of equal-width confidence bins.
    """
    probs = np.asarray(probs, dtype=np.float64)
    action90_true = np.asarray(action90_true, dtype=np.int64)

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == action90_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    acc_per_bin = np.full(n_bins, np.nan)
    conf_per_bin = np.full(n_bins, np.nan)
    count_per_bin = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        n_in = mask.sum()
        count_per_bin[i] = n_in
        if n_in > 0:
            acc_per_bin[i] = correct[mask].mean()
            conf_per_bin[i] = confidence[mask].mean()

    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(5, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # -- Top: reliability --
    ax_rel.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Perfect")
    valid = ~np.isnan(acc_per_bin)
    ax_rel.bar(
        bin_centers[valid],
        acc_per_bin[valid],
        width=1.0 / n_bins * 0.85,
        color="#4c72b0",
        edgecolor="white",
        linewidth=0.5,
        label="Model",
    )
    ax_rel.set_ylabel("Accuracy")
    ax_rel.set_ylim(0, 1)
    ax_rel.legend(loc="upper left", frameon=False)
    if title:
        ax_rel.set_title(title)

    # -- Bottom: confidence histogram --
    ax_hist.bar(
        bin_centers,
        count_per_bin,
        width=1.0 / n_bins * 0.85,
        color="#4c72b0",
        edgecolor="white",
        linewidth=0.5,
    )
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(0, 1)

    fig.tight_layout()
    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Top-k comparison bar chart
# ---------------------------------------------------------------------------


def topk_comparison_bar(
    results_dict: dict[str, dict[str, float]],
    out_path: str | Path,
) -> None:
    """Grouped bar chart comparing top-1 / top-3 / top-5 across models.

    Args:
        results_dict: ``{"ModelName": metrics_dict, ...}`` where each
            *metrics_dict* contains keys like ``"overall/top1_action90"``, etc.
        out_path: base path for saved figure (PNG + PDF).
    """
    model_names = list(results_dict.keys())
    n_models = len(model_names)

    metric_keys = ["overall/top1_action90", "overall/top3_action90", "overall/top5_action90"]
    labels = ["Top-1", "Top-3", "Top-5"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    x = np.arange(n_models)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(max(4, 1.8 * n_models), 4.5))

    for j, (key, label, color) in enumerate(zip(metric_keys, labels, colors)):
        values = []
        for model in model_names:
            v = results_dict[model].get(key, 0.0)
            values.append(v * 100)  # percentage
        ax.bar(x + j * bar_width, values, bar_width, label=label, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Action-90 Top-k Accuracy (Tier 1)")
    ax.legend(frameon=False)
    ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.15))

    fig.tight_layout()
    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Stratified comparison table
# ---------------------------------------------------------------------------

# Metrics to include in the table, in display order
_TABLE_METRICS = [
    "top1_action90",
    "top3_action90",
    "top5_action90",
    "top1_lead2",
    "top3_lead2",
    "nll_action90",
    "nll_lead2",
    "brier_action90",
    "brier_lead2",
    "ece_action90",
    "ece_lead2",
]

_STRATA = ["overall", "mirror", "non_mirror"]

_METRIC_FMT: dict[str, str] = {
    "top1_action90": ".1%",
    "top3_action90": ".1%",
    "top5_action90": ".1%",
    "top1_lead2": ".1%",
    "top3_lead2": ".1%",
    "nll_action90": ".4f",
    "nll_lead2": ".4f",
    "brier_action90": ".4f",
    "brier_lead2": ".4f",
    "ece_action90": ".4f",
    "ece_lead2": ".4f",
}


def stratified_table(
    results_dict: dict[str, dict[str, Any]],
    out_path_json: str | Path,
    out_path_latex: str | Path | None = None,
) -> dict[str, Any]:
    """Write a stratified comparison table as JSON (and optionally LaTeX).

    Args:
        results_dict: ``{"ModelName": metrics_dict, ...}``.
        out_path_json: path for JSON output.
        out_path_latex: optional path for LaTeX output.

    Returns:
        The table data structure written to JSON.
    """
    model_names = list(results_dict.keys())

    # Build nested dict: {model: {stratum: {metric: value}}}
    table: dict[str, dict[str, dict[str, float | None]]] = {}
    for model in model_names:
        table[model] = {}
        metrics = results_dict[model]
        for stratum in _STRATA:
            row: dict[str, float | None] = {}
            for m in _TABLE_METRICS:
                key = f"{stratum}/{m}"
                row[m] = metrics.get(key)
            table[model][stratum] = row

    out_json = Path(out_path_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2)

    if out_path_latex is not None:
        _write_latex_table(table, model_names, out_path_latex)

    return table


def _write_latex_table(
    table: dict[str, dict[str, dict[str, float | None]]],
    model_names: list[str],
    out_path: str | Path,
) -> None:
    """Write a LaTeX booktabs table comparing models across strata."""
    n_models = len(model_names)
    col_spec = "ll" + "r" * n_models

    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = "Stratum & Metric & " + " & ".join(model_names) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for stratum in _STRATA:
        stratum_label = stratum.replace("_", " ").title()
        first = True
        for m in _TABLE_METRICS:
            prefix = stratum_label if first else ""
            first = False
            fmt = _METRIC_FMT.get(m, ".4f")
            cells = []
            for model in model_names:
                v = table[model][stratum].get(m)
                if v is None:
                    cells.append("--")
                else:
                    cells.append(f"{v:{fmt}}")
            row = f"{prefix} & {m} & " + " & ".join(cells) + r" \\"
            lines.append(row)
        lines.append(r"\midrule")

    # Replace last midrule with bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
