#!/usr/bin/env python3
"""Generate new figures for the rewrite (Step 7).

Produces:
  1. Adaptation effectiveness bar chart (win rates: changed vs stayed, by prior outcome)
  2. Opponent-context stratified comparison (base vs sequential vs opp_ctx)
  3. Post-win/post-loss subset model comparison

Usage:
    cd /home/walter/CS229/turnzero
    PYTHONPATH=. .venv/bin/python scripts/run_rewrite_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.eval.plots import COLORS, _save_fig, setup_plotting

setup_plotting()

OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"
EVAL_DIR = ROOT / "outputs" / "eval"


def _load(name: str) -> dict:
    with open(EVAL_DIR / name) as f:
        return json.load(f)


# ── Figure 1: Adaptation Effectiveness ────────────────────────────────

def fig_adaptation_effectiveness() -> None:
    data = _load("adaptation_effectiveness.json")

    conditions = ["After Loss", "After Win"]
    changed_rates = [
        data["after_loss"]["changed_win_rate"]["rate"],
        data["after_win"]["changed_win_rate"]["rate"],
    ]
    stayed_rates = [
        data["after_loss"]["stayed_win_rate"]["rate"],
        data["after_win"]["stayed_win_rate"]["rate"],
    ]
    # Wilson CI half-widths
    changed_err = [
        [
            data[k]["changed_win_rate"]["rate"] - data[k]["changed_win_rate"]["ci_lo"],
            data[k]["changed_win_rate"]["ci_hi"] - data[k]["changed_win_rate"]["rate"],
        ]
        for k in ("after_loss", "after_win")
    ]
    stayed_err = [
        [
            data[k]["stayed_win_rate"]["rate"] - data[k]["stayed_win_rate"]["ci_lo"],
            data[k]["stayed_win_rate"]["ci_hi"] - data[k]["stayed_win_rate"]["rate"],
        ]
        for k in ("after_loss", "after_win")
    ]

    x = np.arange(len(conditions))
    w = 0.3

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        x - w / 2, [r * 100 for r in changed_rates], w,
        yerr=np.array(changed_err).T * 100,
        label="Changed leads", color=COLORS[0], edgecolor="white", linewidth=0.5,
        capsize=4,
    )
    ax.bar(
        x + w / 2, [r * 100 for r in stayed_rates], w,
        yerr=np.array(stayed_err).T * 100,
        label="Kept leads", color=COLORS[1], edgecolor="white", linewidth=0.5,
        capsize=4,
    )

    ax.axhline(50, color="gray", ls="--", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Game 2 Win Rate (%)")
    ax.set_title("Does Changing Leads Help?")
    ax.legend(frameon=False)
    ax.set_ylim(30, 65)

    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "adaptation_effectiveness")
    print("  Saved adaptation_effectiveness")


# ── Figure 2: Opponent-Context Stratified Comparison ──────────────────

def fig_opponent_context_comparison() -> None:
    data = _load("opponent_context/opponent_context_results.json")

    strata_keys = [
        ("games_2_3", "Games 2-3"),
        ("after_win_all", "After Win"),
        ("after_loss_all", "After Loss"),
    ]
    models = ["base", "sequential", "opponent_context"]
    model_labels = ["Base", "Sequential", "Opp. Context"]
    colors = [COLORS[0], COLORS[2], COLORS[3]]

    x = np.arange(len(strata_keys))
    n_models = len(models)
    w = 0.22

    fig, ax = plt.subplots(figsize=(6, 4))
    for j, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        vals = []
        for skey, _ in strata_keys:
            v = data["strata"][skey].get(model, {}).get("top1_action90", 0)
            vals.append(v * 100)
        ax.bar(
            x + (j - n_models / 2 + 0.5) * w, vals, w,
            label=label, color=color, edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in strata_keys])
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Sequential vs Opponent-Context Model")
    ax.legend(frameon=False)
    ax.set_ylim(0, 16)

    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "opponent_context_comparison")
    print("  Saved opponent_context_comparison")


# ── Figure 3: Post-Win / Post-Loss Subset Models ─────────────────────

def fig_subset_models() -> None:
    pw = _load("opponent_context/post_win_results.json")
    pl = _load("opponent_context/post_loss_results.json")

    strata_keys = [
        ("after_win_all", "After Win"),
        ("after_loss_all", "After Loss"),
    ]

    # For each stratum, compare: base, sequential, post_win-only, post_loss-only
    model_sources = [
        ("Base", "base", pw),  # base is same in both files
        ("Sequential", "sequential", pw),
        ("Post-Win Only", "post_win", pw),
        ("Post-Loss Only", "post_loss", pl),
    ]
    colors_fig = [COLORS[0], COLORS[2], COLORS[1], COLORS[3]]

    x = np.arange(len(strata_keys))
    n = len(model_sources)
    w = 0.18

    fig, ax = plt.subplots(figsize=(6, 4))
    for j, (label, key, src) in enumerate(model_sources):
        vals = []
        for skey, _ in strata_keys:
            v = src["strata"][skey].get(key, {}).get("top1_action90", 0)
            vals.append(v * 100)
        ax.bar(
            x + (j - n / 2 + 0.5) * w, vals, w,
            label=label, color=colors_fig[j], edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in strata_keys])
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Subset-Trained Models vs Full Sequential")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 16)

    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "subset_model_comparison")
    print("  Saved subset_model_comparison")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating rewrite figures...")
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    fig_adaptation_effectiveness()
    fig_opponent_context_comparison()
    fig_subset_models()

    print("\nDone. All figures saved to outputs/plots/paper/")


if __name__ == "__main__":
    main()
