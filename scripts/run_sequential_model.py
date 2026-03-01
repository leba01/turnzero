#!/usr/bin/env python3
"""BO3 Sequential Model — end-to-end pipeline.

Trains a 5-member SequentialOTSTransformer ensemble that conditions on
Game 1 outcome and leads, then evaluates whether within-set adaptation
is predictable.

Stages:
  1. Build BO3 context table (if not cached)
  2. Train 5 ensemble members
  3. Evaluate: stratified comparison (game_num × prior_result × mirror)
  4. Output: outputs/eval/sequential_bo3/

Usage:
    cd /home/walter/CS229/turnzero
    PYTHONPATH=. .venv/bin/python scripts/run_sequential_model.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.data.bo3_context import build_context_lookup, save_context_lookup
from turnzero.data.dataset import Vocab
from turnzero.data.sequential_dataset import SequentialVGCDataset
from turnzero.eval.metrics import _MARGIN_MATRIX, compute_metrics
from turnzero.eval.plots import COLORS, _save_fig, setup_plotting
from turnzero.models.train import load_config, train
from turnzero.uq.ensemble import (
    _collect_probs,
    _entropy,
    _load_model_from_checkpoint,
    load_ensemble_predictions,
)

setup_plotting()

# ── Paths ─────────────────────────────────────────────────────────────
RAW_PATH = ROOT / "data" / "raw" / "logs-gen9vgc2025reggbo3.json"
DATA_A = ROOT / "data" / "assembled" / "regime_a"
CONTEXT_PATH = DATA_A / "bo3_context.json"

CONFIGS_DIR = ROOT / "configs" / "sequential"
RUNS_DIR = ROOT / "outputs" / "runs"
OUT_DIR = ROOT / "outputs" / "eval" / "sequential_bo3"
OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"

BASE_PREDS = ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 4


# ── Stage 1: Build context table ─────────────────────────────────────

def stage_build_context() -> None:
    """Build BO3 context lookup table from raw logs."""
    if CONTEXT_PATH.exists():
        print(f"Context table already exists at {CONTEXT_PATH}")
        with open(CONTEXT_PATH) as f:
            ctx = json.load(f)
        print(f"  {len(ctx):,} entries")
        return

    print("Building BO3 context table...")
    t0 = time.time()

    # Load all raw BO3 logs
    raw_paths = list((ROOT / "data" / "raw").glob("logs-gen9vgc*bo3*.json"))
    all_raw: dict = {}
    for p in raw_paths:
        print(f"  Loading {p.name}...")
        with open(p) as f:
            data = json.load(f)
        all_raw.update(data)
    print(f"  Total raw battles: {len(all_raw):,}")

    context = build_context_lookup(all_raw)
    save_context_lookup(context, CONTEXT_PATH)
    print(f"  Built in {time.time() - t0:.1f}s")


# ── Stage 2: Train ensemble ──────────────────────────────────────────

def stage_train() -> list[Path]:
    """Train 5 sequential ensemble members. Returns checkpoint paths."""
    ckpt_paths = []
    for i in range(1, 6):
        cfg_path = CONFIGS_DIR / f"member_{i:03d}.yaml"
        out_dir = RUNS_DIR / f"sequential_{i:03d}"
        ckpt_path = out_dir / "best.pt"

        if ckpt_path.exists():
            print(f"\nMember {i}: checkpoint exists at {ckpt_path}")
            ckpt_paths.append(ckpt_path)
            continue

        print(f"\n{'='*60}")
        print(f"Training sequential member {i}/5")
        print(f"{'='*60}")

        config = load_config(cfg_path)
        best = train(config, out_dir)
        ckpt_paths.append(best)

    return ckpt_paths


# ── Stage 3: Evaluate ────────────────────────────────────────────────

def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def stage_evaluate(ckpt_paths: list[Path]) -> dict:
    """Run sequential ensemble on test set, compare with base ensemble."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load vocab from first member
    vocab_path = ckpt_paths[0].parent / "vocab.json"
    if not vocab_path.exists():
        vocab_path = DATA_A / "vocab.json"
    vocab = Vocab.load(vocab_path)

    # Build test loader with context
    test_ds = SequentialVGCDataset(
        DATA_A / "test.jsonl", vocab, context_path=CONTEXT_PATH,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    print(f"\nTest set: {len(test_ds):,} examples")

    # Collect context fields for stratification
    all_game_num = []
    all_prior_result = []
    for ex in test_ds.examples:
        bid = ex.get("battle_id", "")
        ctx = test_ds._context.get(bid)
        if ctx is not None:
            side = test_ds._resolve_perspective(ex, ctx)
            if side is not None:
                raw_gn = ctx["game_num"]
                all_game_num.append(max(0, raw_gn - 1))
                all_prior_result.append(ctx["sides"][side]["prior_result"])
                continue
        all_game_num.append(0)
        all_prior_result.append(0)

    game_num_arr = np.array(all_game_num)
    prior_result_arr = np.array(all_prior_result)

    # Run sequential ensemble inference
    M = len(ckpt_paths)
    member_probs_list = []
    labels_dict = None

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"Sequential member {i+1}/{M}: {ckpt_path.parent.name}")
        model = _load_model_from_checkpoint(ckpt_path, DEVICE)
        probs_m, ld = _collect_probs(model, test_loader, DEVICE)
        member_probs_list.append(probs_m)
        if labels_dict is None:
            labels_dict = ld
        del model
        torch.cuda.empty_cache()

    assert labels_dict is not None
    member_probs = np.stack(member_probs_list, axis=0)
    seq_probs = member_probs.mean(axis=0)

    # Load base ensemble predictions for comparison
    base_data = load_ensemble_predictions(BASE_PREDS)
    base_probs = base_data["probs"]

    action90_true = labels_dict["action90_true"]
    lead2_true = labels_dict["lead2_true"]
    bring4 = labels_dict["bring4_observed"].astype(bool)
    is_mirror = labels_dict["is_mirror"].astype(bool)

    # ── Compute metrics for each stratum ──
    margin_matrix = _MARGIN_MATRIX

    def _stratum_metrics(probs: np.ndarray, mask: np.ndarray, label: str) -> dict:
        if mask.sum() == 0:
            return {"n": 0, "label": label}
        p = probs[mask]
        a90 = action90_true[mask]
        l2 = lead2_true[mask]
        b4 = bring4[mask]
        mir = is_mirror[mask]

        top1 = (p.argmax(axis=1) == a90).mean()
        top3 = np.array([a90[i] in np.argsort(p[i])[-3:] for i in range(len(a90))]).mean()
        nll = -np.log(np.clip(p[np.arange(len(a90)), a90], 1e-10, 1.0)).mean()

        # Lead-2 metrics
        lead2_probs = p @ margin_matrix
        lead2_top1 = (lead2_probs.argmax(axis=1) == l2).mean()

        return {
            "n": int(mask.sum()),
            "label": label,
            "top1_action90": float(top1),
            "top3_action90": float(top3),
            "nll_action90": float(nll),
            "top1_lead2": float(lead2_top1),
        }

    results: dict = {"strata": {}}

    # Overall
    all_mask = np.ones(len(action90_true), dtype=bool)
    results["overall"] = {
        "base": _stratum_metrics(base_probs, all_mask, "all"),
        "sequential": _stratum_metrics(seq_probs, all_mask, "all"),
    }

    # By game_num
    for gn, gn_label in [(0, "game1_or_non_bo3"), (1, "game2"), (2, "game3")]:
        mask = game_num_arr == gn
        results["strata"][gn_label] = {
            "base": _stratum_metrics(base_probs, mask, gn_label),
            "sequential": _stratum_metrics(seq_probs, mask, gn_label),
        }

    # By game_num × prior_result (games 2-3 only)
    for gn, gn_label in [(1, "game2"), (2, "game3")]:
        for pr, pr_label in [(1, "after_win"), (2, "after_loss")]:
            mask = (game_num_arr == gn) & (prior_result_arr == pr)
            key = f"{gn_label}_{pr_label}"
            results["strata"][key] = {
                "base": _stratum_metrics(base_probs, mask, key),
                "sequential": _stratum_metrics(seq_probs, mask, key),
            }

    # Games 2-3 combined
    mask_g23 = game_num_arr > 0
    results["strata"]["games_2_3"] = {
        "base": _stratum_metrics(base_probs, mask_g23, "games_2_3"),
        "sequential": _stratum_metrics(seq_probs, mask_g23, "games_2_3"),
    }

    # After-loss combined (key comparison)
    mask_after_loss = prior_result_arr == 2
    results["strata"]["after_loss_all"] = {
        "base": _stratum_metrics(base_probs, mask_after_loss, "after_loss_all"),
        "sequential": _stratum_metrics(seq_probs, mask_after_loss, "after_loss_all"),
    }

    # Context coverage stats
    n_with_context = int((game_num_arr > 0).sum())
    results["context_coverage"] = {
        "total_test": len(game_num_arr),
        "with_context": n_with_context,
        "game1_or_non_bo3": int((game_num_arr == 0).sum()),
        "game2": int((game_num_arr == 1).sum()),
        "game3": int((game_num_arr == 2).sum()),
        "after_win": int((prior_result_arr == 1).sum()),
        "after_loss": int((prior_result_arr == 2).sum()),
    }

    # Save results
    results_path = OUT_DIR / "sequential_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


# ── Stage 4: Figures ─────────────────────────────────────────────────

def stage_figures(results: dict) -> None:
    """Generate comparison figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    # Figure: Base vs Sequential by game_num
    strata_keys = ["game1_or_non_bo3", "game2", "game3"]
    strata_labels = ["Game 1 / Non-BO3", "Game 2", "Game 3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(strata_keys))
    w = 0.32

    # Panel A: Top-1 accuracy
    base_top1 = [results["strata"][k]["base"].get("top1_action90", 0) * 100
                 for k in strata_keys]
    seq_top1 = [results["strata"][k]["sequential"].get("top1_action90", 0) * 100
                for k in strata_keys]

    bars1 = ax1.bar(x - w/2, base_top1, w, label="Base Ensemble", color=COLORS[0],
                    edgecolor="white")
    bars2 = ax1.bar(x + w/2, seq_top1, w, label="Sequential", color=COLORS[3],
                    edgecolor="white")

    for bar, val in zip(bars1, base_top1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, seq_top1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax1.set_ylabel("Action-90 Top-1 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strata_labels, fontsize=9)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("(a) Top-1 Accuracy by Game Number", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Top-1 by prior result (games 2-3 only)
    cross_keys = ["game2_after_win", "game2_after_loss", "game3_after_win", "game3_after_loss"]
    cross_labels = ["G2\nAfter Win", "G2\nAfter Loss", "G3\nAfter Win", "G3\nAfter Loss"]

    x2 = np.arange(len(cross_keys))
    base_cross = []
    seq_cross = []
    for k in cross_keys:
        s = results["strata"].get(k, {})
        base_cross.append(s.get("base", {}).get("top1_action90", 0) * 100)
        seq_cross.append(s.get("sequential", {}).get("top1_action90", 0) * 100)

    bars3 = ax2.bar(x2 - w/2, base_cross, w, label="Base", color=COLORS[0],
                    edgecolor="white")
    bars4 = ax2.bar(x2 + w/2, seq_cross, w, label="Sequential", color=COLORS[3],
                    edgecolor="white")

    for bar, val in zip(bars3, base_cross):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars4, seq_cross):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Action-90 Top-1 (%)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cross_labels, fontsize=9)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_title("(b) Top-1 by Prior Result", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, OUT_DIR / "sequential_comparison")
    _save_fig(fig, OUT_PLOTS / "sequential_bo3_comparison")
    plt.close(fig)


# ── Stage 5: Print summary ───────────────────────────────────────────

def stage_summary(results: dict) -> None:
    """Print a human-readable summary."""
    print(f"\n{'='*70}")
    print("  SEQUENTIAL BO3 MODEL — RESULTS SUMMARY")
    print(f"{'='*70}")

    cov = results["context_coverage"]
    print(f"\n  Context coverage:")
    print(f"    Total test examples:  {cov['total_test']:,}")
    print(f"    With BO3 context:     {cov['with_context']:,} "
          f"({cov['with_context']/cov['total_test']*100:.1f}%)")
    print(f"    Game 2:               {cov['game2']:,}")
    print(f"    Game 3:               {cov['game3']:,}")
    print(f"    After win:            {cov['after_win']:,}")
    print(f"    After loss:           {cov['after_loss']:,}")

    print(f"\n  {'Stratum':<25} {'Base Top-1':>10} {'Seq Top-1':>10} {'Delta':>8} {'N':>8}")
    print(f"  {'-'*65}")

    for key in ["game1_or_non_bo3", "game2", "game3", "games_2_3",
                "after_loss_all", "game2_after_win", "game2_after_loss",
                "game3_after_win", "game3_after_loss"]:
        s = results["strata"].get(key, {})
        b = s.get("base", {})
        sq = s.get("sequential", {})
        bt1 = b.get("top1_action90", 0)
        st1 = sq.get("top1_action90", 0)
        n = sq.get("n", 0)
        delta = st1 - bt1
        sign = "+" if delta > 0 else ""
        print(f"  {key:<25} {bt1*100:9.2f}% {st1*100:9.2f}% {sign}{delta*100:6.2f}pp {n:>8,}")

    # Overall
    b_all = results["overall"]["base"]
    s_all = results["overall"]["sequential"]
    print(f"\n  {'Overall':<25} {b_all['top1_action90']*100:9.2f}% "
          f"{s_all['top1_action90']*100:9.2f}% "
          f"{(s_all['top1_action90']-b_all['top1_action90'])*100:+6.2f}pp "
          f"{s_all['n']:>8,}")

    # Key claim
    g23 = results["strata"].get("games_2_3", {})
    g23_base = g23.get("base", {}).get("top1_action90", 0)
    g23_seq = g23.get("sequential", {}).get("top1_action90", 0)
    delta_g23 = g23_seq - g23_base

    print(f"\n  KEY FINDING:")
    if delta_g23 > 0.001:
        print(f"    Sequential model recovers {delta_g23*100:+.2f} pp on Games 2-3.")
        print(f"    Some within-set adaptation IS predictable given prior game info.")
    elif delta_g23 < -0.001:
        print(f"    Sequential model shows {delta_g23*100:+.2f} pp on Games 2-3.")
        print(f"    Prior game info does not help — adaptation is truly unpredictable.")
    else:
        print(f"    No meaningful difference ({delta_g23*100:+.2f} pp on Games 2-3).")
        print(f"    Prior game info provides negligible signal.")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("="*70)
    print("  BO3 Sequential Model Pipeline")
    print("="*70)

    # Stage 1: Context table
    print(f"\n{'─'*40}")
    print("  Stage 1: Build context table")
    print(f"{'─'*40}")
    stage_build_context()

    # Stage 2: Train
    print(f"\n{'─'*40}")
    print("  Stage 2: Train sequential ensemble")
    print(f"{'─'*40}")
    ckpt_paths = stage_train()

    # Stage 3: Evaluate
    print(f"\n{'─'*40}")
    print("  Stage 3: Evaluate")
    print(f"{'─'*40}")
    results = stage_evaluate(ckpt_paths)

    # Stage 4: Figures
    print(f"\n{'─'*40}")
    print("  Stage 4: Generate figures")
    print(f"{'─'*40}")
    stage_figures(results)

    # Stage 5: Summary
    stage_summary(results)

    dt = time.time() - t0
    print(f"\nTotal pipeline time: {dt:.1f}s")


if __name__ == "__main__":
    main()
