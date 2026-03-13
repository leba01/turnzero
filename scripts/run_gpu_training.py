#!/usr/bin/env python3
"""GPU Training Pipeline — Steps 5e + 6.

Trains three model families:
  1. Opponent-context ensemble (5 members) — Step 5e
  2. Post-win sequential ensemble (5 members) — Step 6
  3. Post-loss sequential ensemble (5 members) — Step 6

Then evaluates all three with stratified comparison against the base
and sequential ensembles.

Usage:
    cd /home/walter/CS229/turnzero
    PYTHONPATH=. .venv/bin/python scripts/run_gpu_training.py
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

from turnzero.data.bo3_context import load_context_lookup
from turnzero.data.dataset import Vocab
from turnzero.data.sequential_dataset import (
    OpponentContextVGCDataset,
    SequentialVGCDataset,
)
from turnzero.eval.metrics import _MARGIN_MATRIX
from turnzero.eval.plots import COLORS, _save_fig, setup_plotting
from turnzero.models.train import load_config, train
from turnzero.uq.ensemble import (
    _collect_probs,
    _load_model_from_checkpoint,
    load_ensemble_predictions,
)

setup_plotting()

# ── Paths ─────────────────────────────────────────────────────────────
DATA_A = ROOT / "data" / "assembled" / "regime_a"
CONTEXT_PATH = DATA_A / "bo3_context.json"
RUNS_DIR = ROOT / "outputs" / "runs"
OUT_DIR = ROOT / "outputs" / "eval" / "opponent_context"
OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"
BASE_PREDS = ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 4

# Model families to train
FAMILIES = {
    "opponent_context": {
        "configs_dir": ROOT / "configs" / "opponent_context",
        "runs_prefix": "opp_ctx",
        "n_members": 5,
    },
    "post_win": {
        "configs_dir": ROOT / "configs" / "post_win",
        "runs_prefix": "post_win",
        "n_members": 5,
    },
    "post_loss": {
        "configs_dir": ROOT / "configs" / "post_loss",
        "runs_prefix": "post_loss",
        "n_members": 5,
    },
}


# ── Stage 1: Train all families ──────────────────────────────────────

def train_family(name: str, info: dict) -> list[Path]:
    """Train all members of a model family. Returns checkpoint paths."""
    configs_dir = info["configs_dir"]
    prefix = info["runs_prefix"]
    ckpt_paths = []

    for i in range(1, info["n_members"] + 1):
        cfg_path = configs_dir / f"member_{i:03d}.yaml"
        out_dir = RUNS_DIR / f"{prefix}_{i:03d}"
        ckpt_path = out_dir / "best.pt"

        if ckpt_path.exists():
            print(f"\n  {name} member {i}: checkpoint exists at {ckpt_path}")
            ckpt_paths.append(ckpt_path)
            continue

        print(f"\n{'='*60}")
        print(f"  Training {name} member {i}/{info['n_members']}")
        print(f"{'='*60}")

        config = load_config(cfg_path)
        best = train(config, out_dir)
        ckpt_paths.append(best)

    return ckpt_paths


# ── Stage 2: Evaluate ────────────────────────────────────────────────

def evaluate_ensemble(
    family_name: str,
    ckpt_paths: list[Path],
    test_ds: SequentialVGCDataset,
    test_loader: DataLoader,
    game_num_arr: np.ndarray,
    prior_result_arr: np.ndarray,
    base_probs: np.ndarray,
    seq_probs: np.ndarray | None,
) -> dict:
    """Run ensemble inference and compute stratified metrics."""
    M = len(ckpt_paths)
    member_probs_list = []
    labels_dict = None

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"  {family_name} member {i+1}/{M}: {ckpt_path.parent.name}")
        model = _load_model_from_checkpoint(ckpt_path, DEVICE)
        probs_m, ld = _collect_probs(model, test_loader, DEVICE)
        member_probs_list.append(probs_m)
        if labels_dict is None:
            labels_dict = ld
        del model
        torch.cuda.empty_cache()

    assert labels_dict is not None
    member_probs = np.stack(member_probs_list, axis=0)
    ens_probs = member_probs.mean(axis=0)

    action90_true = labels_dict["action90_true"]
    lead2_true = labels_dict["lead2_true"]
    margin_matrix = _MARGIN_MATRIX

    def _metrics(probs: np.ndarray, mask: np.ndarray, label: str) -> dict:
        if mask.sum() == 0:
            return {"n": 0, "label": label}
        p = probs[mask]
        a90 = action90_true[mask]
        l2 = lead2_true[mask]
        top1 = float((p.argmax(axis=1) == a90).mean())
        top3_indices = np.argsort(p, axis=1)[:, -3:]
        top3 = float((top3_indices == a90[:, None]).any(axis=1).mean())
        nll = float(-np.log(np.clip(
            p[np.arange(len(a90)), a90], 1e-10, 1.0,
        )).mean())
        lead2_probs = p @ margin_matrix
        lead2_top1 = float((lead2_probs.argmax(axis=1) == l2).mean())
        return {
            "n": int(mask.sum()), "label": label,
            "top1_action90": top1, "top3_action90": top3,
            "nll_action90": nll, "top1_lead2": lead2_top1,
        }

    results: dict = {"strata": {}}
    all_mask = np.ones(len(action90_true), dtype=bool)

    models = {"base": base_probs, family_name: ens_probs}
    if seq_probs is not None:
        models["sequential"] = seq_probs

    results["overall"] = {k: _metrics(v, all_mask, "all") for k, v in models.items()}

    strata_defs = [
        ("game1_or_non_bo3", game_num_arr == 0),
        ("game2", game_num_arr == 1),
        ("game3", game_num_arr == 2),
        ("games_2_3", game_num_arr > 0),
        ("after_win_all", prior_result_arr == 1),
        ("after_loss_all", prior_result_arr == 2),
        ("game2_after_win", (game_num_arr == 1) & (prior_result_arr == 1)),
        ("game2_after_loss", (game_num_arr == 1) & (prior_result_arr == 2)),
        ("game3_after_win", (game_num_arr == 2) & (prior_result_arr == 1)),
        ("game3_after_loss", (game_num_arr == 2) & (prior_result_arr == 2)),
    ]
    for key, mask in strata_defs:
        results["strata"][key] = {k: _metrics(v, mask, key) for k, v in models.items()}

    return results


def build_context_arrays(test_ds: SequentialVGCDataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract game_num and prior_result arrays for stratification."""
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
    return np.array(all_game_num), np.array(all_prior_result)


def print_summary(name: str, results: dict) -> None:
    """Print a human-readable summary table."""
    models = list(results["overall"].keys())
    print(f"\n{'='*80}")
    print(f"  {name.upper()} — RESULTS SUMMARY")
    print(f"{'='*80}")

    header = f"  {'Stratum':<25}"
    for m in models:
        header += f" {m:>12}"
    header += f" {'N':>8}"
    print(header)
    print(f"  {'-'*75}")

    strata_order = [
        "game1_or_non_bo3", "game2", "game3", "games_2_3",
        "after_win_all", "after_loss_all",
        "game2_after_win", "game2_after_loss",
        "game3_after_win", "game3_after_loss",
    ]
    for key in strata_order:
        s = results["strata"].get(key, {})
        line = f"  {key:<25}"
        n = 0
        for m in models:
            val = s.get(m, {}).get("top1_action90", 0)
            line += f" {val*100:11.2f}%"
            n = s.get(m, {}).get("n", n)
        line += f" {n:>8,}"
        print(line)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  GPU Training Pipeline — Steps 5e + 6")
    print("=" * 70)

    # ── Train all families ──
    all_ckpts: dict[str, list[Path]] = {}
    for name, info in FAMILIES.items():
        print(f"\n{'─'*50}")
        print(f"  Training family: {name}")
        print(f"{'─'*50}")
        all_ckpts[name] = train_family(name, info)

    # ── Evaluate ──
    print(f"\n{'─'*50}")
    print("  Evaluation")
    print(f"{'─'*50}")

    # Load vocab from first opponent_context member
    vocab_path = all_ckpts["opponent_context"][0].parent / "vocab.json"
    if not vocab_path.exists():
        vocab_path = DATA_A / "vocab.json"
    vocab = Vocab.load(vocab_path)

    # Build test loaders
    # Opponent-context needs OpponentContextVGCDataset
    opp_test_ds = OpponentContextVGCDataset(
        DATA_A / "test.jsonl", vocab, context_path=CONTEXT_PATH, n_opp_slots=4,
    )
    opp_test_loader = DataLoader(
        opp_test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )

    # Sequential test loader (for post-win/post-loss eval — use full test set)
    seq_test_ds = SequentialVGCDataset(
        DATA_A / "test.jsonl", vocab, context_path=CONTEXT_PATH,
    )
    seq_test_loader = DataLoader(
        seq_test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )

    game_num_arr, prior_result_arr = build_context_arrays(seq_test_ds)
    print(f"\nTest set: {len(seq_test_ds):,} examples")
    print(f"  With context: {(game_num_arr > 0).sum():,}")

    # Load base ensemble predictions
    base_data = load_ensemble_predictions(BASE_PREDS)
    base_probs = base_data["probs"]

    # Load sequential ensemble predictions (from prior Step 5 run)
    seq_results_path = ROOT / "outputs" / "eval" / "sequential_bo3" / "sequential_results.json"
    seq_probs = None
    if seq_results_path.exists():
        # Recompute sequential probs from checkpoints
        seq_ckpts = [
            RUNS_DIR / f"sequential_{i:03d}" / "best.pt" for i in range(1, 6)
        ]
        if all(p.exists() for p in seq_ckpts):
            print("\nLoading sequential ensemble for comparison...")
            seq_member_probs = []
            for i, cp in enumerate(seq_ckpts):
                model = _load_model_from_checkpoint(cp, DEVICE)
                pm, _ = _collect_probs(model, seq_test_loader, DEVICE)
                seq_member_probs.append(pm)
                del model
                torch.cuda.empty_cache()
            seq_probs = np.stack(seq_member_probs, axis=0).mean(axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate opponent-context
    print("\nEvaluating opponent_context ensemble...")
    opp_results = evaluate_ensemble(
        "opponent_context", all_ckpts["opponent_context"],
        opp_test_ds, opp_test_loader,
        game_num_arr, prior_result_arr,
        base_probs, seq_probs,
    )
    with open(OUT_DIR / "opponent_context_results.json", "w") as f:
        json.dump(opp_results, f, indent=2)
    print_summary("opponent_context", opp_results)

    # Evaluate post-win and post-loss (on full test set for fair comparison)
    for family_name in ("post_win", "post_loss"):
        print(f"\nEvaluating {family_name} ensemble...")
        fam_results = evaluate_ensemble(
            family_name, all_ckpts[family_name],
            seq_test_ds, seq_test_loader,
            game_num_arr, prior_result_arr,
            base_probs, seq_probs,
        )
        with open(OUT_DIR / f"{family_name}_results.json", "w") as f:
            json.dump(fam_results, f, indent=2)
        print_summary(family_name, fam_results)

    dt = time.time() - t0
    print(f"\nTotal pipeline time: {dt:.1f}s")


if __name__ == "__main__":
    main()
