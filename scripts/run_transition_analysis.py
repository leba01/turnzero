#!/usr/bin/env python3
"""Lead transition matrix analysis: where do players go after wins vs losses?

Builds 15x15 transition matrices conditioned on prior outcome and computes
conditional entropy H(g2_lead | g1_lead, outcome).

Outputs:
    outputs/eval/transition_analysis.json
    outputs/plots/paper/transition_heatmaps.{png,pdf}

Usage:
    cd /home/walter/CS229/turnzero
    PYTHONPATH=. .venv/bin/python scripts/run_transition_analysis.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from scripts.run_bo3_adaptation import (
    RAW_PATH,
    OUT_EVAL,
    OUT_PLOTS,
    GameInfo,
    extract_bo3_linkage,
    parse_game_info,
)
from turnzero.eval.plots import COLORS as _COLORS, _save_fig, setup_plotting

setup_plotting()

TOP_K = 15  # top lead pairs to include in the matrix


def _canonical_pair(leads: frozenset[str]) -> str:
    """Sort and join a lead pair for a canonical key."""
    return "|".join(sorted(leads))


def build_transition_records(
    set_to_games: dict[str, dict[int, str]],
    raw_data: dict[str, Any],
) -> list[dict]:
    """Build (g1_pair, g2_pair, prior_result) records."""
    game_cache: dict[str, GameInfo] = {}
    parse_ok = parse_fail = 0

    for sid, games in set_to_games.items():
        gnums = sorted(games.keys())
        if len(gnums) < 2:
            continue
        for gn in gnums:
            bid = games[gn]
            if bid in game_cache:
                continue
            _, log_text = raw_data[bid]
            gi = parse_game_info(bid, sid, gn, log_text)
            if gi is not None:
                game_cache[bid] = gi
                parse_ok += 1
            else:
                parse_fail += 1

    print(f"  Parsed: {parse_ok:,} ok, {parse_fail:,} failed")

    records = []
    for sid, games in set_to_games.items():
        gnums = sorted(games.keys())
        for i in range(len(gnums) - 1):
            g1_bid = games[gnums[i]]
            g2_bid = games[gnums[i + 1]]
            if g1_bid not in game_cache or g2_bid not in game_cache:
                continue
            g1 = game_cache[g1_bid]
            g2 = game_cache[g2_bid]
            if g1.player_names != g2.player_names:
                continue
            if g1.winner_side is None:
                continue

            for side in ("p1", "p2"):
                prior = "won" if g1.winner_side == side else "lost"
                records.append({
                    "g1_pair": _canonical_pair(g1.leads[side]),
                    "g2_pair": _canonical_pair(g2.leads[side]),
                    "prior_result": prior,
                })

    return records


def compute_transition_matrices(
    records: list[dict], top_pairs: list[str]
) -> dict:
    """Build transition matrices and compute entropy."""
    pair_to_idx = {p: i for i, p in enumerate(top_pairs)}
    k = len(top_pairs)

    results = {}
    for outcome in ("won", "lost"):
        subset = [r for r in records if r["prior_result"] == outcome]
        mat = np.zeros((k, k), dtype=int)
        for r in subset:
            i = pair_to_idx.get(r["g1_pair"])
            j = pair_to_idx.get(r["g2_pair"])
            if i is not None and j is not None:
                mat[i, j] += 1

        # Row-normalize for conditional probabilities
        row_sums = mat.sum(axis=1, keepdims=True)
        prob = np.where(row_sums > 0, mat / row_sums, 0.0)

        # Conditional entropy: H(g2 | g1, outcome) = sum_i P(g1=i) * H(g2 | g1=i)
        total = mat.sum()
        entropy = 0.0
        if total > 0:
            for i in range(k):
                p_i = row_sums[i, 0] / total
                if p_i == 0:
                    continue
                h_i = 0.0
                for j in range(k):
                    if prob[i, j] > 0:
                        h_i -= prob[i, j] * np.log2(prob[i, j])
                entropy += p_i * h_i

        # Diagonal concentration
        diag_sum = np.diag(mat).sum()
        diag_frac = diag_sum / total if total > 0 else 0.0

        results[outcome] = {
            "counts": mat.tolist(),
            "probs": prob.tolist(),
            "conditional_entropy": float(entropy),
            "diagonal_fraction": float(diag_frac),
            "n_transitions": int(total),
            "n_subset": len(subset),
        }

    return results


def make_heatmaps(
    matrices: dict, top_pairs: list[str]
) -> plt.Figure:
    """Two-panel heatmap: after-win (diagonal) vs after-loss (dispersed)."""
    # Shorten pair names for display
    short = [p.replace("|", "\n") for p in top_pairs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, outcome, title in [
        (ax1, "won", "(a) After Win — repeat leads"),
        (ax2, "lost", "(b) After Loss — explore leads"),
    ]:
        probs = np.array(matrices[outcome]["probs"])
        im = ax.imshow(probs, cmap="YlOrRd", aspect="auto", vmin=0,
                       vmax=max(0.5, probs.max()))
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, fontsize=5.5, rotation=90)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=5.5)
        ax.set_xlabel("Game 2 Lead Pair", fontsize=9)
        ax.set_ylabel("Game 1 Lead Pair", fontsize=9)
        ax.set_title(title, fontsize=11)

        ent = matrices[outcome]["conditional_entropy"]
        diag = matrices[outcome]["diagonal_fraction"]
        ax.text(0.02, 0.98, f"H = {ent:.2f} bits\nDiag = {diag*100:.0f}%",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.colorbar(ax2.images[0], ax=[ax1, ax2], shrink=0.6, label="P(G2 pair | G1 pair)")
    fig.tight_layout()
    return fig


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  Lead Transition Matrix Analysis")
    print("=" * 70)

    print(f"\nLoading {RAW_PATH.name}...")
    with open(RAW_PATH) as f:
        raw_data = json.load(f)
    print(f"  {len(raw_data):,} battles loaded")

    print("\nExtracting BO3 set linkage...")
    set_to_games = extract_bo3_linkage(raw_data)
    multi = {sid: g for sid, g in set_to_games.items() if len(g) >= 2}
    print(f"  {len(multi):,} sets with >= 2 games")

    print("\nBuilding transition records...")
    records = build_transition_records(multi, raw_data)
    print(f"  {len(records):,} transition records")

    # Find top-K lead pairs
    all_pairs: Counter[str] = Counter()
    for r in records:
        all_pairs[r["g1_pair"]] += 1
        all_pairs[r["g2_pair"]] += 1
    top_pairs = [p for p, _ in all_pairs.most_common(TOP_K)]
    print(f"\n  Top {TOP_K} lead pairs (by total frequency):")
    for i, p in enumerate(top_pairs):
        print(f"    {i+1:2d}. {p} ({all_pairs[p]:,})")

    print("\nComputing transition matrices...")
    matrices = compute_transition_matrices(records, top_pairs)

    # Save
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    out_path = OUT_EVAL / "transition_analysis.json"
    output = {
        "top_pairs": top_pairs,
        "matrices": matrices,
        "n_records": len(records),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Generate figure
    print("\nGenerating heatmaps...")
    fig = make_heatmaps(matrices, top_pairs)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    _save_fig(fig, OUT_PLOTS / "transition_heatmaps")

    # Summary
    print(f"\n{'=' * 70}")
    print("  TRANSITION ANALYSIS SUMMARY")
    print(f"{'=' * 70}")

    for outcome in ("won", "lost"):
        m = matrices[outcome]
        print(f"\n  After {outcome}:")
        print(f"    Conditional entropy: {m['conditional_entropy']:.3f} bits")
        print(f"    Diagonal fraction:   {m['diagonal_fraction']*100:.1f}%")
        print(f"    Transitions in top-{TOP_K}: {m['n_transitions']:,} / {m['n_subset']:,}")

    diff = matrices["lost"]["conditional_entropy"] - matrices["won"]["conditional_entropy"]
    print(f"\n  Entropy difference (lost - won): {diff:+.3f} bits")
    print(f"  → After-loss transitions are {'more' if diff > 0 else 'less'} dispersed")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
