#!/usr/bin/env python3
"""Per-team analysis: team linearity, predictability, and archetypes.

Groups test predictions by species-6 composition (sorted 6-species tuple,
ignoring moves/items/abilities) and computes per-team metrics to reveal
which teams are predictable (low entropy, linear game plan) and which are
flexible/hard (high entropy, many viable plans).

Note: core_cluster_a (4/6 species overlap + union-find) is too coarse —
the VGC meta is a small-world network and 91% of data falls into one
giant connected component. Species-6 composition gives 10K+ unique teams,
~200 with n >= 20 Tier 1 examples. This is the right granularity.

Outputs:
    outputs/eval/cluster_analysis.json
    outputs/plots/paper/cluster_entropy_vs_accuracy.{png,pdf}
    outputs/plots/paper/cluster_entropy_histogram.{png,pdf}

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/run_cluster_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
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
_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.data.io_utils import read_jsonl

DATA_A = ROOT / "data" / "assembled" / "regime_a"
ENS_NPZ = ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz"
OUT_EVAL = ROOT / "outputs" / "eval"
OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"

MIN_TIER1 = 20  # minimum Tier 1 examples per team to include


def _save_fig(fig: plt.Figure, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=_DPI)
    fig.savefig(out.with_suffix(".pdf"), dpi=_DPI)
    plt.close(fig)
    print(f"  Saved: {out.with_suffix('.png')}")


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  Per-Team Analysis (Week 5, Task 0)")
    print("=" * 70)

    # --- Load ensemble predictions ---
    print("\nLoading ensemble predictions...")
    ens = np.load(ENS_NPZ)
    probs = ens["probs"]                      # (N, 90)
    entropy = ens["entropy"]                   # (N,)
    mi = ens["mi"]                             # (N,)
    confidence = ens["confidence"]             # (N,)
    action90_true = ens["action90_true"]       # (N,)
    bring4_observed = ens["bring4_observed"]   # (N,)

    N = len(probs)
    predicted = probs.argmax(axis=1)
    correct_top1 = (predicted == action90_true)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    correct_top3 = np.any(top3 == action90_true[:, None], axis=1)
    tier1 = bring4_observed.astype(bool)

    print(f"  {N:,} test examples, {tier1.sum():,} Tier 1")

    # --- Load species compositions from test JSONL ---
    print("Loading test JSONL for species compositions...")
    examples = list(read_jsonl(DATA_A / "test.jsonl"))
    assert len(examples) == N, f"Length mismatch: {len(examples)} vs {N}"

    # Group by sorted species-6 tuple (exact 6 species, order-invariant)
    species_keys: list[str] = []
    species_per_example: list[list[str]] = []
    for ex in examples:
        sp = [mon["species"] for mon in ex["team_a"]["pokemon"]]
        species_per_example.append(sp)
        species_keys.append("|".join(sorted(sp)))
    del examples

    # --- Group by species composition ---
    print("Grouping by species-6 composition...")
    team_to_idx: dict[str, list[int]] = {}
    for i, key in enumerate(species_keys):
        team_to_idx.setdefault(key, []).append(i)

    print(f"  {len(team_to_idx):,} unique species compositions in test set")

    # --- Compute per-team metrics ---
    team_metrics: dict[str, dict] = {}
    for key, indices in team_to_idx.items():
        idx = np.array(indices)
        t1_mask = tier1[idx]
        n_tier1 = int(t1_mask.sum())

        if n_tier1 < MIN_TIER1:
            continue

        t1_idx = idx[t1_mask]

        # Accuracy
        top1_acc = float(correct_top1[t1_idx].mean())
        top3_acc = float(correct_top3[t1_idx].mean())

        # Uncertainty
        mean_ent = float(entropy[t1_idx].mean())
        mean_mi_val = float(mi[t1_idx].mean())
        mean_conf = float(confidence[t1_idx].mean())

        # Species list (sorted, from the key)
        species = key.split("|")

        team_metrics[key] = {
            "n_examples": len(idx),
            "n_tier1": n_tier1,
            "top1_action90": top1_acc,
            "top3_action90": top3_acc,
            "mean_entropy": mean_ent,
            "mean_mi": mean_mi_val,
            "mean_confidence": mean_conf,
            "species": species,
        }

    n_valid = len(team_metrics)
    total_covered = sum(m["n_tier1"] for m in team_metrics.values())
    print(f"  {n_valid} teams with >= {MIN_TIER1} Tier 1 examples")
    print(f"  Covering {total_covered:,} / {tier1.sum():,} Tier 1 examples "
          f"({total_covered / tier1.sum() * 100:.1f}%)")

    # --- Save JSON ---
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    json_path = OUT_EVAL / "cluster_analysis.json"
    with open(json_path, "w") as f:
        json.dump(team_metrics, f, indent=2)
    print(f"\nSaved: {json_path}")

    # --- Extract arrays for plotting ---
    keys = list(team_metrics.keys())
    ent_arr = np.array([team_metrics[k]["mean_entropy"] for k in keys])
    top1_arr = np.array([team_metrics[k]["top1_action90"] for k in keys]) * 100
    top3_arr = np.array([team_metrics[k]["top3_action90"] for k in keys]) * 100
    n_arr = np.array([team_metrics[k]["n_tier1"] for k in keys])

    # --- Figure 1: Entropy vs Accuracy scatter ---
    print("\nGenerating scatter plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Size: proportional to sqrt(n) for visual scaling
    sizes = np.sqrt(n_arr) * 3
    sizes = np.clip(sizes, 8, 200)

    # Left: entropy vs top-1
    ax1.scatter(ent_arr, top1_arr, s=sizes, alpha=0.5, color=_COLORS[0],
                edgecolor="white", linewidth=0.3, rasterized=True)
    r_top1 = np.corrcoef(ent_arr, top1_arr)[0, 1]
    z = np.polyfit(ent_arr, top1_arr, 1)
    x_fit = np.linspace(ent_arr.min(), ent_arr.max(), 100)
    ax1.plot(x_fit, np.polyval(z, x_fit), "--", color=_COLORS[3], lw=1.5,
             label=f"r = {r_top1:.3f}")
    ax1.axhline(100 / 90, ls=":", color="gray", alpha=0.5, lw=1)
    ax1.text(ent_arr.max() - 0.02, 100 / 90 + 0.3, "Random", fontsize=8,
             color="gray", ha="right")
    ax1.set_xlabel("Team Mean Entropy (nats)")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_title("Entropy vs Top-1 Accuracy")
    ax1.legend(frameon=False, fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: entropy vs top-3
    ax2.scatter(ent_arr, top3_arr, s=sizes, alpha=0.5, color=_COLORS[2],
                edgecolor="white", linewidth=0.3, rasterized=True)
    r_top3 = np.corrcoef(ent_arr, top3_arr)[0, 1]
    z3 = np.polyfit(ent_arr, top3_arr, 1)
    ax2.plot(x_fit, np.polyval(z3, x_fit), "--", color=_COLORS[3], lw=1.5,
             label=f"r = {r_top3:.3f}")
    ax2.axhline(300 / 90, ls=":", color="gray", alpha=0.5, lw=1)
    ax2.text(ent_arr.max() - 0.02, 300 / 90 + 0.5, "Random", fontsize=8,
             color="gray", ha="right")
    ax2.set_xlabel("Team Mean Entropy (nats)")
    ax2.set_ylabel("Top-3 Accuracy (%)")
    ax2.set_title("Entropy vs Top-3 Accuracy")
    ax2.legend(frameon=False, fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"Per-Team Predictability ({n_valid} teams, n ≥ {MIN_TIER1})",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "cluster_entropy_vs_accuracy")

    # --- Figure 2: Entropy histogram with annotations ---
    print("Generating entropy histogram...")
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.hist(ent_arr, bins=40, color=_COLORS[0], edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(ent_arr.mean(), ls="--", color=_COLORS[3], lw=1.5,
               label=f"Mean = {ent_arr.mean():.3f}")
    ax.set_xlabel("Team Mean Entropy (nats)")
    ax.set_ylabel("Number of Teams")
    ax.set_title("Distribution of Team Linearity (Lower Entropy = More Predictable)")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate extremes
    sorted_by_ent = sorted(team_metrics.items(), key=lambda x: x[1]["mean_entropy"])
    most_linear = sorted_by_ent[:3]
    most_flexible = sorted_by_ent[-3:]

    linear_text = "Most linear:\n"
    for _, m in most_linear:
        sp = ", ".join(m["species"][:3])
        linear_text += f"  {sp}... (H={m['mean_entropy']:.2f})\n"
    ax.text(0.02, 0.95, linear_text.strip(), transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_COLORS[2],
                      alpha=0.15, edgecolor="none"))

    flex_text = "Most flexible:\n"
    for _, m in most_flexible:
        sp = ", ".join(m["species"][:3])
        flex_text += f"  {sp}... (H={m['mean_entropy']:.2f})\n"
    ax.text(0.98, 0.95, flex_text.strip(), transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_COLORS[3],
                      alpha=0.15, edgecolor="none"))

    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "cluster_entropy_histogram")

    # --- Print summary tables ---
    print(f"\n{'=' * 70}")
    print("  PER-TEAM ANALYSIS SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {n_valid} teams with >= {MIN_TIER1} Tier 1 examples")
    print(f"  Entropy range: [{ent_arr.min():.3f}, {ent_arr.max():.3f}]")
    print(f"  Entropy mean:  {ent_arr.mean():.3f} +/- {ent_arr.std():.3f}")
    print(f"  Correlation (entropy vs top-1): r = {r_top1:.3f}")
    print(f"  Correlation (entropy vs top-3): r = {r_top3:.3f}")

    header = f"  {'H':>6} {'Top-1':>7} {'Top-3':>7} {'N':>5}  Species"

    # Top 5 most linear teams (lowest entropy)
    print(f"\n  {'─' * 66}")
    print("  TOP 5 MOST LINEAR TEAMS (lowest entropy = most predictable)")
    print(f"  {'─' * 66}")
    print(header)
    for _, m in sorted_by_ent[:5]:
        sp = ", ".join(m["species"][:4])
        print(f"  {m['mean_entropy']:>6.3f} {m['top1_action90']*100:>6.1f}% "
              f"{m['top3_action90']*100:>6.1f}% {m['n_tier1']:>5}  {sp}...")

    # Top 5 most flexible teams (highest entropy)
    print(f"\n  {'─' * 66}")
    print("  TOP 5 MOST FLEXIBLE TEAMS (highest entropy = least predictable)")
    print(f"  {'─' * 66}")
    print(header)
    for _, m in sorted_by_ent[-5:][::-1]:
        sp = ", ".join(m["species"][:4])
        print(f"  {m['mean_entropy']:>6.3f} {m['top1_action90']*100:>6.1f}% "
              f"{m['top3_action90']*100:>6.1f}% {m['n_tier1']:>5}  {sp}...")

    # Top 5 most accurate
    sorted_by_acc = sorted(team_metrics.items(),
                           key=lambda x: x[1]["top1_action90"], reverse=True)
    print(f"\n  {'─' * 66}")
    print("  TOP 5 MOST ACCURATE TEAMS")
    print(f"  {'─' * 66}")
    print(header)
    for _, m in sorted_by_acc[:5]:
        sp = ", ".join(m["species"][:4])
        print(f"  {m['mean_entropy']:>6.3f} {m['top1_action90']*100:>6.1f}% "
              f"{m['top3_action90']*100:>6.1f}% {m['n_tier1']:>5}  {sp}...")

    # Bottom 5 least accurate
    print(f"\n  {'─' * 66}")
    print("  BOTTOM 5 LEAST ACCURATE TEAMS")
    print(f"  {'─' * 66}")
    print(header)
    for _, m in sorted_by_acc[-5:][::-1]:
        sp = ", ".join(m["species"][:4])
        print(f"  {m['mean_entropy']:>6.3f} {m['top1_action90']*100:>6.1f}% "
              f"{m['top3_action90']*100:>6.1f}% {m['n_tier1']:>5}  {sp}...")

    # Distribution stats
    print(f"\n  {'─' * 66}")
    print("  DISTRIBUTION STATS")
    print(f"  {'─' * 66}")
    zero_top1 = (top1_arr == 0).sum()
    above_10 = (top1_arr > 10).sum()
    above_20 = (top1_arr > 20).sum()
    print(f"  Teams with 0% top-1:   {zero_top1} / {n_valid}")
    print(f"  Teams with >10% top-1: {above_10} / {n_valid}")
    print(f"  Teams with >20% top-1: {above_20} / {n_valid}")
    print(f"  Median top-1: {np.median(top1_arr):.1f}%")
    print(f"  Median top-3: {np.median(top3_arr):.1f}%")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s")


if __name__ == "__main__":
    main()
