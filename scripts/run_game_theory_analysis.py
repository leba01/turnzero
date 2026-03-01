#!/usr/bin/env python3
"""Game-theoretic analysis: evaluate ensemble as distributional estimator.

Reframes evaluation from point prediction to distributional estimation.
Computes JS divergence between model predictions and empirical mixed strategies,
stratifies accuracy by matchup consensus level, and decomposes entropy to show
where prediction is structurally possible vs impossible.

Outputs:
    outputs/eval/game_theory_analysis.json
    outputs/plots/paper/game_theory_js_comparison.{png,pdf}
    outputs/plots/paper/game_theory_consensus_accuracy.{png,pdf}
    outputs/plots/paper/game_theory_entropy_scatter.{png,pdf}

Usage:
    cd /home/walter/CS229/turnzero
    PYTHONPATH=. .venv/bin/python scripts/run_game_theory_analysis.py
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.data.io_utils import read_jsonl  # noqa: E402
from turnzero.eval.metrics import _MARGIN_MATRIX  # noqa: E402
from turnzero.eval.plots import COLORS as _COLORS  # noqa: E402
from turnzero.eval.plots import _save_fig, setup_plotting  # noqa: E402

setup_plotting()

DATA_A = ROOT / "data" / "assembled" / "regime_a"
ENS_NPZ = ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz"
OUT_EVAL = ROOT / "outputs" / "eval"
OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"

MIN_MATCHUP_N = 10  # minimum training observations per matchup


def _species_key(team: dict) -> str:
    """Readable species key: sorted species joined by '|'."""
    return "|".join(sorted(mon["species"] for mon in team["pokemon"]))


def _matchup_key(ex: dict) -> str:
    """Ordered matchup key: (team_a_species, team_b_species)."""
    return _species_key(ex["team_a"]) + "||" + _species_key(ex["team_b"])


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits for a single distribution."""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  Game-Theoretic Analysis")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Task 1: Load data
    # ------------------------------------------------------------------
    print("\nLoading ensemble predictions...")
    ens = np.load(ENS_NPZ)
    probs = ens["probs"]                # (N, 90)
    action90_true = ens["action90_true"]  # (N,)
    lead2_true = ens["lead2_true"]      # (N,)
    bring4_observed = ens["bring4_observed"]  # (N,)
    N = len(probs)
    tier1 = bring4_observed.astype(bool)
    print(f"  {N:,} test examples, {tier1.sum():,} Tier 1")

    print("Loading test JSONL for matchup keys...")
    test_examples = list(read_jsonl(DATA_A / "test.jsonl"))
    assert len(test_examples) == N, f"Length mismatch: {len(test_examples)} vs {N}"

    test_matchup_keys = [_matchup_key(ex) for ex in test_examples]
    del test_examples

    # ------------------------------------------------------------------
    # Task 2: Build per-matchup empirical distributions from TRAINING data
    # ------------------------------------------------------------------
    print("\nBuilding empirical distributions from training data...")
    matchup_counts: dict[str, np.ndarray] = collections.defaultdict(
        lambda: np.zeros(90)
    )
    matchup_n: dict[str, int] = collections.defaultdict(int)
    # Also accumulate per-team-A marginals and global popularity
    team_a_counts: dict[str, np.ndarray] = collections.defaultdict(
        lambda: np.zeros(90)
    )
    team_a_n: dict[str, int] = collections.defaultdict(int)
    global_counts = np.zeros(90)
    n_train = 0

    for ex in read_jsonl(DATA_A / "train.jsonl"):
        action_id = ex["label"]["action90_id"]
        mk = _matchup_key(ex)
        matchup_counts[mk][action_id] += 1
        matchup_n[mk] += 1

        ta_key = _species_key(ex["team_a"])
        team_a_counts[ta_key][action_id] += 1
        team_a_n[ta_key] += 1

        global_counts[action_id] += 1
        n_train += 1

    print(f"  {n_train:,} training examples")
    print(f"  {len(matchup_counts):,} unique matchups in training")

    # Filter to well-sampled matchups
    well_sampled = {}
    for mk, counts in matchup_counts.items():
        if matchup_n[mk] >= MIN_MATCHUP_N:
            well_sampled[mk] = counts / counts.sum()

    print(f"  {len(well_sampled):,} matchups with n >= {MIN_MATCHUP_N}")

    # Build team-A marginal distributions
    team_a_dists = {}
    for ta_key, counts in team_a_counts.items():
        if counts.sum() > 0:
            team_a_dists[ta_key] = counts / counts.sum()

    # Global popularity prior
    popularity_prior = global_counts / global_counts.sum()

    # ------------------------------------------------------------------
    # Task 3: Map test examples to matchup distributions
    # ------------------------------------------------------------------
    print("\nMapping test examples to matchup distributions...")
    covered_indices = []
    covered_empirical = []
    covered_team_a_dists = []

    for i, mk in enumerate(test_matchup_keys):
        if mk in well_sampled:
            covered_indices.append(i)
            covered_empirical.append(well_sampled[mk])
            ta_key = mk.split("||")[0]
            covered_team_a_dists.append(
                team_a_dists.get(ta_key, popularity_prior)
            )

    covered_indices = np.array(covered_indices)
    covered_empirical = np.array(covered_empirical)  # (M, 90)
    covered_team_a_dists = np.array(covered_team_a_dists)  # (M, 90)
    n_covered = len(covered_indices)
    print(f"  {n_covered:,} / {N:,} test examples in well-sampled matchups "
          f"({n_covered / N * 100:.1f}%)")

    # ------------------------------------------------------------------
    # Task 4: JS divergence (action-90)
    # ------------------------------------------------------------------
    print("\nComputing JS divergence (action-90)...")
    uniform_90 = np.ones(90) / 90
    js_ensemble = np.zeros(n_covered)
    js_uniform = np.zeros(n_covered)
    js_team_a = np.zeros(n_covered)
    js_popularity = np.zeros(n_covered)

    for j in range(n_covered):
        emp = covered_empirical[j]
        # scipy jensenshannon returns sqrt(JSD), square it
        js_ensemble[j] = jensenshannon(probs[covered_indices[j]], emp) ** 2
        js_uniform[j] = jensenshannon(uniform_90, emp) ** 2
        js_team_a[j] = jensenshannon(covered_team_a_dists[j], emp) ** 2
        js_popularity[j] = jensenshannon(popularity_prior, emp) ** 2

    js_results_a90 = {
        "ensemble": float(js_ensemble.mean()),
        "uniform": float(js_uniform.mean()),
        "team_a_marginal": float(js_team_a.mean()),
        "popularity_prior": float(js_popularity.mean()),
    }
    print(f"  Ensemble JSD:    {js_results_a90['ensemble']:.6f}")
    print(f"  Uniform JSD:     {js_results_a90['uniform']:.6f}")
    print(f"  Team-A JSD:      {js_results_a90['team_a_marginal']:.6f}")
    print(f"  Popularity JSD:  {js_results_a90['popularity_prior']:.6f}")

    # ------------------------------------------------------------------
    # Task 5: Entropy decomposition
    # ------------------------------------------------------------------
    print("\nEntropy decomposition...")
    matchup_entropies = {}
    matchup_test_indices: dict[str, list[int]] = collections.defaultdict(list)

    for j, idx in enumerate(covered_indices):
        mk = test_matchup_keys[idx]
        if mk not in matchup_entropies:
            matchup_entropies[mk] = _entropy(well_sampled[mk])
        matchup_test_indices[mk].append(idx)

    # Stratify by entropy
    predicted = probs.argmax(axis=1)
    correct_top1 = predicted == action90_true
    top3 = np.argsort(probs, axis=1)[:, -3:]
    correct_top3 = np.any(top3 == action90_true[:, None], axis=1)

    strata_def = {
        "deterministic": lambda h: h < 2.0,
        "moderate": lambda h: 2.0 <= h <= 3.0,
        "diverse": lambda h: h > 3.0,
    }
    entropy_strata = {}
    for sname, cond in strata_def.items():
        mks = [mk for mk, h in matchup_entropies.items() if cond(h)]
        test_idx = []
        for mk in mks:
            test_idx.extend(matchup_test_indices[mk])
        test_idx = np.array(test_idx) if test_idx else np.array([], dtype=int)
        # Filter to tier1 for accuracy
        if len(test_idx) > 0:
            t1_mask = tier1[test_idx]
            t1_idx = test_idx[t1_mask]
        else:
            t1_idx = np.array([], dtype=int)
        entropy_strata[sname] = {
            "n_matchups": len(mks),
            "n_test": int(len(t1_idx)),
            "top1": float(correct_top1[t1_idx].mean()) if len(t1_idx) > 0 else 0.0,
            "top3": float(correct_top3[t1_idx].mean()) if len(t1_idx) > 0 else 0.0,
        }
        print(f"  {sname}: {len(mks)} matchups, {len(t1_idx)} Tier 1 test, "
              f"top1={entropy_strata[sname]['top1']:.3f}, "
              f"top3={entropy_strata[sname]['top3']:.3f}")

    # ------------------------------------------------------------------
    # Task 6: Consensus buckets (action-90)
    # ------------------------------------------------------------------
    print("\nConsensus buckets (action-90)...")
    consensus_a90 = {}
    bucket_def = {
        "high": lambda m: m > 0.30,
        "moderate": lambda m: 0.10 <= m <= 0.30,
        "none": lambda m: m < 0.10,
    }
    for bname, cond in bucket_def.items():
        test_idx = []
        for j, idx in enumerate(covered_indices):
            mk = test_matchup_keys[idx]
            max_p = covered_empirical[j].max()
            if cond(max_p):
                test_idx.append(idx)
        test_idx = np.array(test_idx) if test_idx else np.array([], dtype=int)
        if len(test_idx) > 0:
            t1_mask = tier1[test_idx]
            t1_idx = test_idx[t1_mask]
        else:
            t1_idx = np.array([], dtype=int)
        consensus_a90[bname] = {
            "n_test": int(len(t1_idx)),
            "top1": float(correct_top1[t1_idx].mean()) if len(t1_idx) > 0 else 0.0,
            "top3": float(correct_top3[t1_idx].mean()) if len(t1_idx) > 0 else 0.0,
        }
        print(f"  {bname}: {len(t1_idx)} Tier 1 test, "
              f"top1={consensus_a90[bname]['top1']:.3f}, "
              f"top3={consensus_a90[bname]['top3']:.3f}")

    # ------------------------------------------------------------------
    # Task 7: Lead-2 marginalized analysis
    # ------------------------------------------------------------------
    print("\nLead-2 marginalized analysis...")
    lead2_probs = probs @ _MARGIN_MATRIX  # (N, 15)
    lead2_empirical = covered_empirical @ _MARGIN_MATRIX  # (M, 15)
    lead2_team_a = covered_team_a_dists @ _MARGIN_MATRIX  # (M, 15)
    lead2_popularity = popularity_prior @ _MARGIN_MATRIX  # (15,)
    uniform_15 = np.ones(15) / 15

    lead2_predicted = lead2_probs.argmax(axis=1)
    lead2_correct_top1 = lead2_predicted == lead2_true
    lead2_top3 = np.argsort(lead2_probs, axis=1)[:, -3:]
    lead2_correct_top3 = np.any(lead2_top3 == lead2_true[:, None], axis=1)

    # JS divergence (lead-2)
    js_l2_ensemble = np.zeros(n_covered)
    js_l2_uniform = np.zeros(n_covered)
    js_l2_team_a = np.zeros(n_covered)
    js_l2_popularity = np.zeros(n_covered)

    for j in range(n_covered):
        emp_l2 = lead2_empirical[j]
        js_l2_ensemble[j] = jensenshannon(
            lead2_probs[covered_indices[j]], emp_l2
        ) ** 2
        js_l2_uniform[j] = jensenshannon(uniform_15, emp_l2) ** 2
        js_l2_team_a[j] = jensenshannon(lead2_team_a[j], emp_l2) ** 2
        js_l2_popularity[j] = jensenshannon(lead2_popularity, emp_l2) ** 2

    js_results_l2 = {
        "ensemble": float(js_l2_ensemble.mean()),
        "uniform": float(js_l2_uniform.mean()),
        "team_a_marginal": float(js_l2_team_a.mean()),
        "popularity_prior": float(js_l2_popularity.mean()),
    }
    print(f"  Ensemble JSD:    {js_results_l2['ensemble']:.6f}")
    print(f"  Uniform JSD:     {js_results_l2['uniform']:.6f}")
    print(f"  Team-A JSD:      {js_results_l2['team_a_marginal']:.6f}")
    print(f"  Popularity JSD:  {js_results_l2['popularity_prior']:.6f}")

    # Consensus buckets (lead-2)
    consensus_l2 = {}
    for bname, cond in bucket_def.items():
        test_idx = []
        for j, idx in enumerate(covered_indices):
            max_p = lead2_empirical[j].max()
            if cond(max_p):
                test_idx.append(idx)
        test_idx = np.array(test_idx) if test_idx else np.array([], dtype=int)
        consensus_l2[bname] = {
            "n_test": int(len(test_idx)),
            "top1": float(lead2_correct_top1[test_idx].mean()) if len(test_idx) > 0 else 0.0,
            "top3": float(lead2_correct_top3[test_idx].mean()) if len(test_idx) > 0 else 0.0,
        }
        print(f"  Lead-2 {bname}: {len(test_idx)} test, "
              f"top1={consensus_l2[bname]['top1']:.3f}, "
              f"top3={consensus_l2[bname]['top3']:.3f}")

    # ------------------------------------------------------------------
    # Task 8: Save JSON
    # ------------------------------------------------------------------
    print("\nSaving results...")
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    results = {
        "matchup_coverage": {
            "n_train_matchups": len(matchup_counts),
            "n_well_sampled": len(well_sampled),
            "n_test_covered": n_covered,
            "n_test_total": N,
        },
        "js_divergence_action90": js_results_a90,
        "js_divergence_lead2": js_results_l2,
        "entropy_strata": entropy_strata,
        "consensus_buckets_action90": consensus_a90,
        "consensus_buckets_lead2": consensus_l2,
    }
    json_path = OUT_EVAL / "game_theory_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # ------------------------------------------------------------------
    # Task 9: Figures
    # ------------------------------------------------------------------
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: JS divergence comparison ---
    print("\nGenerating JS divergence comparison plot...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = ["Ensemble", "Team-A\nMarginal", "Popularity\nPrior", "Uniform"]
    keys = ["ensemble", "team_a_marginal", "popularity_prior", "uniform"]
    x = np.arange(len(methods))
    width = 0.35

    vals_a90 = [js_results_a90[k] for k in keys]
    vals_l2 = [js_results_l2[k] for k in keys]

    bars1 = ax.bar(x - width / 2, vals_a90, width, label="Action-90",
                   color=_COLORS[0], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, vals_l2, width, label="Lead-2",
                   color=_COLORS[2], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Mean JS Divergence")
    ax.set_title("Distributional Quality: Model vs Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "game_theory_js_comparison")

    # --- Figure 2: Consensus accuracy ---
    print("Generating consensus accuracy plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    bucket_labels = ["High\n(>0.30)", "Moderate\n(0.10–0.30)", "None\n(<0.10)"]
    bucket_keys = ["high", "moderate", "none"]
    x = np.arange(len(bucket_labels))
    width = 0.35

    # Action-90
    t1_a90 = [consensus_a90[k]["top1"] * 100 for k in bucket_keys]
    t3_a90 = [consensus_a90[k]["top3"] * 100 for k in bucket_keys]
    ax1.bar(x - width / 2, t1_a90, width, label="Top-1", color=_COLORS[0],
            edgecolor="white", linewidth=0.5)
    ax1.bar(x + width / 2, t3_a90, width, label="Top-3", color=_COLORS[1],
            edgecolor="white", linewidth=0.5)
    for i, (v1, v3) in enumerate(zip(t1_a90, t3_a90)):
        ax1.text(i - width / 2, v1 + 0.3, f"{v1:.1f}%", ha="center",
                 va="bottom", fontsize=8)
        ax1.text(i + width / 2, v3 + 0.3, f"{v3:.1f}%", ha="center",
                 va="bottom", fontsize=8)
    # Annotate sample sizes
    for i, k in enumerate(bucket_keys):
        ax1.text(i, -0.02 * ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else -1,
                 f"n={consensus_a90[k]['n_test']:,}", ha="center",
                 va="top", fontsize=7, color="gray")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Action-90: Accuracy by Consensus")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bucket_labels)
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Lead-2
    t1_l2 = [consensus_l2[k]["top1"] * 100 for k in bucket_keys]
    t3_l2 = [consensus_l2[k]["top3"] * 100 for k in bucket_keys]
    ax2.bar(x - width / 2, t1_l2, width, label="Top-1", color=_COLORS[0],
            edgecolor="white", linewidth=0.5)
    ax2.bar(x + width / 2, t3_l2, width, label="Top-3", color=_COLORS[1],
            edgecolor="white", linewidth=0.5)
    for i, (v1, v3) in enumerate(zip(t1_l2, t3_l2)):
        ax2.text(i - width / 2, v1 + 0.3, f"{v1:.1f}%", ha="center",
                 va="bottom", fontsize=8)
        ax2.text(i + width / 2, v3 + 0.3, f"{v3:.1f}%", ha="center",
                 va="bottom", fontsize=8)
    for i, k in enumerate(bucket_keys):
        ax2.text(i, -0.02 * ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else -1,
                 f"n={consensus_l2[k]['n_test']:,}", ha="center",
                 va="top", fontsize=7, color="gray")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Lead-2: Accuracy by Consensus")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bucket_labels)
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Accuracy Conditioned on Empirical Consensus", fontsize=13,
                 y=1.02)
    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "game_theory_consensus_accuracy")

    # --- Figure 3: Entropy scatter ---
    print("Generating entropy scatter plot...")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-matchup: entropy, accuracy, n_test, consensus bucket
    scatter_h = []
    scatter_acc = []
    scatter_n = []
    scatter_color = []

    for mk, h in matchup_entropies.items():
        idx_list = matchup_test_indices[mk]
        idx_arr = np.array(idx_list)
        t1_mask = tier1[idx_arr]
        t1_idx = idx_arr[t1_mask]
        if len(t1_idx) < 3:
            continue
        acc = float(correct_top1[t1_idx].mean()) * 100
        max_p = well_sampled[mk].max()
        if max_p > 0.30:
            color = _COLORS[0]
        elif max_p >= 0.10:
            color = _COLORS[1]
        else:
            color = _COLORS[3]
        scatter_h.append(h)
        scatter_acc.append(acc)
        scatter_n.append(len(t1_idx))
        scatter_color.append(color)

    scatter_h = np.array(scatter_h)
    scatter_acc = np.array(scatter_acc)
    scatter_n = np.array(scatter_n)
    sizes = np.clip(np.sqrt(scatter_n) * 5, 10, 250)

    ax.scatter(scatter_h, scatter_acc, s=sizes, c=scatter_color, alpha=0.6,
               edgecolor="white", linewidth=0.3, rasterized=True)

    # Reference lines
    overall_acc = float(correct_top1[tier1].mean()) * 100
    ax.axhline(overall_acc, ls="--", color="gray", lw=1, alpha=0.7,
               label=f"Overall top-1 = {overall_acc:.1f}%")
    ax.axhline(100 / 90, ls=":", color="gray", lw=1, alpha=0.5)
    ax.text(0.5, 100 / 90 + 0.5, "Random (1.1%)", fontsize=8, color="gray")

    # Legend for consensus colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS[0],
               markersize=8, label="High consensus (>0.30)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS[1],
               markersize=8, label="Moderate (0.10–0.30)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS[3],
               markersize=8, label="No consensus (<0.10)"),
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=9,
              loc="upper right")

    ax.set_xlabel("Matchup Empirical Entropy (bits)")
    ax.set_ylabel("Model Top-1 Accuracy (%)")
    ax.set_title("Per-Matchup: Entropy vs Predictability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "game_theory_entropy_scatter")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  GAME-THEORETIC ANALYSIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Matchup coverage: {len(well_sampled):,} well-sampled matchups")
    print(f"  Test coverage: {n_covered:,} / {N:,} ({n_covered / N * 100:.1f}%)")
    print(f"\n  JS Divergence (action-90):")
    for method, val in js_results_a90.items():
        print(f"    {method:20s}: {val:.6f}")
    print(f"\n  JS Divergence (lead-2):")
    for method, val in js_results_l2.items():
        print(f"    {method:20s}: {val:.6f}")
    print(f"\n  Consensus → Accuracy (action-90):")
    for bname in bucket_keys:
        b = consensus_a90[bname]
        print(f"    {bname:10s}: top1={b['top1']:.3f}, "
              f"top3={b['top3']:.3f} (n={b['n_test']:,})")
    print(f"\n  Consensus → Accuracy (lead-2):")
    for bname in bucket_keys:
        b = consensus_l2[bname]
        print(f"    {bname:10s}: top1={b['top1']:.3f}, "
              f"top3={b['top3']:.3f} (n={b['n_test']:,})")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s")


if __name__ == "__main__":
    main()
