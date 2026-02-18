#!/usr/bin/env python3
"""Supplementary analyses: top-k curve, bring-4 vs lead-2 decomposition,
speed control hypothesis.

All three use existing ensemble predictions — no GPU, no new inference.

Outputs:
    outputs/plots/paper/topk_accuracy_curve.{png,pdf}
    outputs/eval/supplementary_analysis.json
    (+ printed tables for PAPER_ANALYSIS.md)

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/run_supplementary_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
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

from turnzero.action_space import ACTION_TABLE
from turnzero.data.io_utils import read_jsonl

ENS_NPZ = ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz"
DATA_A = ROOT / "data" / "assembled" / "regime_a"
CLUSTER_JSON = ROOT / "outputs" / "eval" / "cluster_analysis.json"
OUT_EVAL = ROOT / "outputs" / "eval"
OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"


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
    print("  Supplementary Analyses")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading ensemble predictions...")
    ens = np.load(ENS_NPZ)
    probs = ens["probs"]                      # (N, 90)
    action90_true = ens["action90_true"]       # (N,)
    bring4_observed = ens["bring4_observed"]   # (N,)

    N = len(probs)
    tier1 = bring4_observed.astype(bool)
    N_t1 = int(tier1.sum())
    print(f"  {N:,} test examples, {N_t1:,} Tier 1")

    # Tier 1 subset
    probs_t1 = probs[tier1]
    true_t1 = action90_true[tier1]

    results: dict = {}

    # =================================================================
    # Analysis 1: Top-k accuracy curve (k=1..90)
    # =================================================================
    print("\n--- Analysis 1: Top-k Accuracy Curve ---")

    # Sort predictions by descending probability for each example
    sorted_indices = np.argsort(probs_t1, axis=1)[:, ::-1]  # (N_t1, 90)
    true_col = true_t1[:, None]  # (N_t1, 1)

    topk_acc = []
    for k in range(1, 91):
        topk = sorted_indices[:, :k]
        hit = np.any(topk == true_col, axis=1)
        topk_acc.append(float(hit.mean()) * 100)

    topk_acc = np.array(topk_acc)

    # Find key milestones
    k_50 = int(np.searchsorted(topk_acc, 50.0)) + 1  # first k where acc >= 50%
    k_75 = int(np.searchsorted(topk_acc, 75.0)) + 1
    k_90 = int(np.searchsorted(topk_acc, 90.0)) + 1

    print(f"  Top-1:  {topk_acc[0]:.1f}%")
    print(f"  Top-3:  {topk_acc[2]:.1f}%")
    print(f"  Top-5:  {topk_acc[4]:.1f}%")
    print(f"  Top-10: {topk_acc[9]:.1f}%")
    print(f"  Top-20: {topk_acc[19]:.1f}%")
    print(f"  50% coverage at k = {k_50}")
    print(f"  75% coverage at k = {k_75}")
    print(f"  90% coverage at k = {k_90}")

    # Random baseline: top-k accuracy = k/90
    random_topk = np.arange(1, 91) / 90.0 * 100

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = np.arange(1, 91)
    ax.plot(ks, topk_acc, "-", color=_COLORS[0], lw=2, label="Ensemble")
    ax.plot(ks, random_topk, "--", color="gray", lw=1, alpha=0.7, label="Random")

    # Annotate milestones
    for k_val, label in [(1, f"k=1: {topk_acc[0]:.1f}%"),
                          (3, f"k=3: {topk_acc[2]:.1f}%"),
                          (k_50, f"k={k_50}: 50%")]:
        ax.plot(k_val, topk_acc[k_val - 1], "o", color=_COLORS[3], ms=6, zorder=5)
        # Offset annotations to avoid overlap
        y_off = 3 if k_val < 10 else -5
        ax.annotate(label, (k_val, topk_acc[k_val - 1]),
                     textcoords="offset points", xytext=(8, y_off),
                     fontsize=8, color=_COLORS[3])

    ax.set_xlabel("k (number of predictions)")
    ax.set_ylabel("Top-k Accuracy (%)")
    ax.set_title("Coverage Curve: How Many Predictions to Capture the Expert's Choice?")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1, 90)
    ax.set_ylim(0, 102)

    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "topk_accuracy_curve")

    results["topk_curve"] = {
        "topk_accuracy": topk_acc.tolist(),
        "k_for_50pct": k_50,
        "k_for_75pct": k_75,
        "k_for_90pct": k_90,
    }

    # =================================================================
    # Analysis 2: Bring-4 vs Lead-2 decomposition
    # =================================================================
    print("\n--- Analysis 2: Bring-4 vs Lead-2 Decomposition ---")

    # Build mapping from action90 → bring4_id and lead_given_bring4_id
    # bring4 = which 4 of 6 are brought (C(6,4) = 15 possibilities)
    # For each bring4, lead2 = which 2 of the 4 lead (C(4,2) = 6 possibilities)

    # Map each action90 to its bring-4 group (the set of 4 brought)
    bring4_groups: dict[frozenset, int] = {}
    action90_to_bring4 = np.zeros(90, dtype=int)
    action90_to_lead_within_bring4 = np.zeros(90, dtype=int)

    for aid, (lead, back) in enumerate(ACTION_TABLE):
        brought = frozenset(lead + back)
        if brought not in bring4_groups:
            bring4_groups[brought] = len(bring4_groups)
        b4id = bring4_groups[brought]
        action90_to_bring4[aid] = b4id

        # Within this bring4, which of the 6 lead arrangements is this?
        brought_sorted = sorted(brought)
        lead_combos = list(combinations(brought_sorted, 2))
        lead_sorted = tuple(sorted(lead))
        action90_to_lead_within_bring4[aid] = lead_combos.index(lead_sorted)

    assert len(bring4_groups) == 15, f"Expected 15 bring-4 groups, got {len(bring4_groups)}"

    # Compute bring-4 probabilities by marginalizing over lead arrangements
    # For each example, sum probs over all 6 actions that share the same bring-4
    bring4_probs = np.zeros((N_t1, 15))
    for aid in range(90):
        b4id = action90_to_bring4[aid]
        bring4_probs[:, b4id] += probs_t1[:, aid]

    # True bring-4 labels
    true_bring4 = action90_to_bring4[true_t1]

    # Bring-4 accuracy (15-way)
    bring4_pred = bring4_probs.argmax(axis=1)
    bring4_top1 = float((bring4_pred == true_bring4).mean()) * 100

    bring4_top3_idx = np.argsort(bring4_probs, axis=1)[:, -3:]
    bring4_top3 = float(np.any(bring4_top3_idx == true_bring4[:, None], axis=1).mean()) * 100

    # Lead-2 accuracy (given correct bring-4)
    # For examples where we got the bring-4 right, did we also get the lead pair right?
    correct_bring4_mask = (bring4_pred == true_bring4)
    n_correct_bring4 = int(correct_bring4_mask.sum())

    if n_correct_bring4 > 0:
        # Among examples where bring-4 is correct, check if the top action
        # within that bring-4 group matches the true lead arrangement
        true_lead_within = action90_to_lead_within_bring4[true_t1]
        pred_action90 = probs_t1.argmax(axis=1)
        pred_lead_within = action90_to_lead_within_bring4[pred_action90]

        lead_correct_given_bring4 = float(
            (pred_lead_within[correct_bring4_mask] == true_lead_within[correct_bring4_mask]).mean()
        ) * 100
    else:
        lead_correct_given_bring4 = 0.0

    # Also compute lead-2 marginal accuracy (already have from main eval, but recompute)
    # Lead-2: which 2 of 6 lead, C(6,2)=15 possibilities
    lead2_groups: dict[tuple, int] = {}
    action90_to_lead2 = np.zeros(90, dtype=int)
    for aid, (lead, back) in enumerate(ACTION_TABLE):
        lead_key = tuple(sorted(lead))
        if lead_key not in lead2_groups:
            lead2_groups[lead_key] = len(lead2_groups)
        action90_to_lead2[aid] = lead2_groups[lead_key]

    assert len(lead2_groups) == 15

    lead2_probs = np.zeros((N_t1, 15))
    for aid in range(90):
        l2id = action90_to_lead2[aid]
        lead2_probs[:, l2id] += probs_t1[:, aid]

    true_lead2 = action90_to_lead2[true_t1]
    lead2_pred = lead2_probs.argmax(axis=1)
    lead2_top1 = float((lead2_pred == true_lead2).mean()) * 100

    lead2_top3_idx = np.argsort(lead2_probs, axis=1)[:, -3:]
    lead2_top3 = float(np.any(lead2_top3_idx == true_lead2[:, None], axis=1).mean()) * 100

    print(f"\n  Action-90 (joint, 90-way):")
    print(f"    Top-1: {topk_acc[0]:.1f}%   Top-3: {topk_acc[2]:.1f}%")
    print(f"\n  Bring-4 (marginal, 15-way):")
    print(f"    Top-1: {bring4_top1:.1f}%   Top-3: {bring4_top3:.1f}%")
    print(f"\n  Lead-2 (marginal, 15-way):")
    print(f"    Top-1: {lead2_top1:.1f}%   Top-3: {lead2_top3:.1f}%")
    print(f"\n  Lead arrangement | bring-4 correct (6-way):")
    print(f"    Accuracy: {lead_correct_given_bring4:.1f}% "
          f"(n = {n_correct_bring4:,} examples where bring-4 was correct)")

    # Interpretation
    print(f"\n  Interpretation:")
    print(f"    Bring-4 top-1 ({bring4_top1:.1f}%) vs Lead-2 top-1 ({lead2_top1:.1f}%):")
    if bring4_top1 > lead2_top1:
        print(f"    → Model is better at predicting WHO to bring than WHO to lead")
    else:
        print(f"    → Model is better at predicting WHO to lead than WHO to bring")

    results["decomposition"] = {
        "action90_top1": topk_acc[0],
        "action90_top3": topk_acc[2],
        "bring4_top1": bring4_top1,
        "bring4_top3": bring4_top3,
        "lead2_top1": lead2_top1,
        "lead2_top3": lead2_top3,
        "lead_given_correct_bring4": lead_correct_given_bring4,
        "n_correct_bring4": n_correct_bring4,
    }

    # =================================================================
    # Analysis 3: Speed control hypothesis
    # =================================================================
    print("\n--- Analysis 3: Speed Control Hypothesis ---")

    # Load cluster analysis to check entropy by team composition
    with open(CLUSTER_JSON) as f:
        cluster_data = json.load(f)

    # Speed control moves from the lexicon
    speed_control_moves = {
        "Tailwind", "Trick Room", "Icy Wind", "Electroweb",
        "Scary Face", "Bulldoze", "Drum Beating",
    }

    # Trick Room specifically (the most constraining speed control)
    trick_room_indicators = {
        "Trick Room", "Indeedee-F", "Hatterene", "Dusclops",
        "Oranguru", "Porygon2", "Gothitelle", "Farigiraf",
    }

    # Load test examples to check for speed control moves
    print("  Loading test JSONL for move analysis...")
    examples = list(read_jsonl(DATA_A / "test.jsonl"))

    # Build species key → move info
    team_speed_control: dict[str, dict] = {}
    team_has_trick_room: dict[str, bool] = {}

    for ex in examples:
        sp_key = "|".join(sorted(mon["species"] for mon in ex["team_a"]["pokemon"]))

        if sp_key in team_speed_control:
            continue

        all_moves = []
        species_list = []
        for mon in ex["team_a"]["pokemon"]:
            species_list.append(mon["species"])
            for move in mon.get("moves", []):
                if move and move != "UNK":
                    all_moves.append(move)

        sc_moves = speed_control_moves.intersection(all_moves)
        has_tr = "Trick Room" in all_moves or any(
            s in trick_room_indicators for s in species_list
        )

        team_speed_control[sp_key] = {
            "speed_control_moves": sorted(sc_moves),
            "n_speed_control": len(sc_moves),
            "has_trick_room": has_tr,
        }
        team_has_trick_room[sp_key] = has_tr

    del examples

    # Now cross-reference with cluster analysis (which has entropy)
    tr_entropies = []
    non_tr_entropies = []
    tr_accs = []
    non_tr_accs = []

    for key, metrics in cluster_data.items():
        has_tr = team_has_trick_room.get(key, False)
        ent = metrics["mean_entropy"]
        acc = metrics["top1_action90"]

        if has_tr:
            tr_entropies.append(ent)
            tr_accs.append(acc)
        else:
            non_tr_entropies.append(ent)
            non_tr_accs.append(acc)

    tr_entropies = np.array(tr_entropies) if tr_entropies else np.array([])
    non_tr_entropies = np.array(non_tr_entropies) if non_tr_entropies else np.array([])
    tr_accs = np.array(tr_accs) if tr_accs else np.array([])
    non_tr_accs = np.array(non_tr_accs) if non_tr_accs else np.array([])

    print(f"\n  Trick Room teams: {len(tr_entropies)} / {len(cluster_data)}")
    if len(tr_entropies) > 0:
        print(f"    Mean entropy: {tr_entropies.mean():.3f} ± {tr_entropies.std():.3f}")
        print(f"    Mean top-1:   {tr_accs.mean()*100:.1f}%")
    print(f"\n  Non-Trick Room teams: {len(non_tr_entropies)} / {len(cluster_data)}")
    if len(non_tr_entropies) > 0:
        print(f"    Mean entropy: {non_tr_entropies.mean():.3f} ± {non_tr_entropies.std():.3f}")
        print(f"    Mean top-1:   {non_tr_accs.mean()*100:.1f}%")

    if len(tr_entropies) > 0 and len(non_tr_entropies) > 0:
        delta_ent = non_tr_entropies.mean() - tr_entropies.mean()
        print(f"\n  Entropy delta (non-TR minus TR): {delta_ent:+.3f} nats")
        if delta_ent > 0:
            print(f"  → Trick Room teams are MORE predictable (lower entropy)")
        else:
            print(f"  → Trick Room teams are LESS predictable (higher entropy)")

    # Also check: teams with ANY speed control
    sc_entropies = []
    no_sc_entropies = []
    for key, metrics in cluster_data.items():
        sc_info = team_speed_control.get(key, {})
        n_sc = sc_info.get("n_speed_control", 0)
        if n_sc > 0:
            sc_entropies.append(metrics["mean_entropy"])
        else:
            no_sc_entropies.append(metrics["mean_entropy"])

    sc_entropies = np.array(sc_entropies) if sc_entropies else np.array([])
    no_sc_entropies = np.array(no_sc_entropies) if no_sc_entropies else np.array([])

    print(f"\n  Teams with speed control moves: {len(sc_entropies)}")
    if len(sc_entropies) > 0:
        print(f"    Mean entropy: {sc_entropies.mean():.3f}")
    print(f"  Teams without speed control moves: {len(no_sc_entropies)}")
    if len(no_sc_entropies) > 0:
        print(f"    Mean entropy: {no_sc_entropies.mean():.3f}")

    results["speed_control"] = {
        "n_trick_room_teams": len(tr_entropies),
        "n_non_trick_room_teams": len(non_tr_entropies),
        "tr_mean_entropy": float(tr_entropies.mean()) if len(tr_entropies) > 0 else None,
        "non_tr_mean_entropy": float(non_tr_entropies.mean()) if len(non_tr_entropies) > 0 else None,
        "tr_mean_top1": float(tr_accs.mean()) if len(tr_accs) > 0 else None,
        "non_tr_mean_top1": float(non_tr_accs.mean()) if len(non_tr_accs) > 0 else None,
        "n_with_speed_control": len(sc_entropies),
        "n_without_speed_control": len(no_sc_entropies),
        "sc_mean_entropy": float(sc_entropies.mean()) if len(sc_entropies) > 0 else None,
        "no_sc_mean_entropy": float(no_sc_entropies.mean()) if len(no_sc_entropies) > 0 else None,
    }

    # --- Save results ---
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    out_path = OUT_EVAL / "supplementary_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s")


if __name__ == "__main__":
    main()
