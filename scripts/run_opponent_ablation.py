#!/usr/bin/env python3
"""Opponent ablation experiment: does the model use opponent info?

Zeroes out team_b (opponent) or team_a (self) at inference time on the
frozen ensemble and measures accuracy delta. If accuracy barely drops
without opponent info, the model primarily learns "what this team usually
does" — not matchup-conditional strategy.

Three conditions:
  baseline       — no modification (loaded from cached predictions)
  team_b_zeroed  — opponent team set to all-UNK (index 0)
  team_a_zeroed  — own team set to all-UNK (symmetric control)

Usage:
    cd /home/walter/CS229/turnzero
    PYTHONPATH=. .venv/bin/python scripts/run_opponent_ablation.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.constants import LOG_EPS as _EPS
from turnzero.data.dataset import Vocab, VGCDataset
from turnzero.eval.metrics import compute_metrics
from turnzero.eval.plots import COLORS, _DPI, _save_fig, setup_plotting
from turnzero.uq.ensemble import _load_model_from_checkpoint

setup_plotting()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_A = ROOT / "data" / "assembled" / "regime_a"
ENSEMBLE_DIR = ROOT / "outputs" / "runs"
ENSEMBLE_MEMBERS = [
    ENSEMBLE_DIR / f"ensemble_{i:03d}" / "best.pt" for i in range(1, 6)
]
VOCAB_PATH = ENSEMBLE_DIR / "ensemble_001" / "vocab.json"
CACHED_PREDS = ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz"

OUT_EVAL = ROOT / "outputs" / "eval"
OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 4

CONDITIONS = ["baseline", "team_b_zeroed", "team_a_zeroed"]


# ---------------------------------------------------------------------------
# Ablated ensemble inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect_probs_ablated(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    zero_key: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run inference with one team tensor zeroed out.

    Parameters
    ----------
    zero_key : str
        "team_a" or "team_b" — which tensor to replace with zeros.
    """
    all_probs, all_a90, all_l2, all_b4, all_mir = [], [], [], [], []

    for batch in loader:
        team_a = batch["team_a"]
        team_b = batch["team_b"]

        if zero_key == "team_a":
            team_a = torch.zeros_like(team_a)
        else:
            team_b = torch.zeros_like(team_b)

        team_a = team_a.to(device, non_blocking=True)
        team_b = team_b.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(team_a, team_b)

        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_a90.append(batch["action90_label"].numpy())
        all_l2.append(batch["lead2_label"].numpy())
        all_b4.append(batch["bring4_observed"].numpy())
        all_mir.append(batch["is_mirror"].numpy())

    return (
        np.concatenate(all_probs, axis=0).astype(np.float64),
        {
            "action90_true": np.concatenate(all_a90),
            "lead2_true": np.concatenate(all_l2),
            "bring4_observed": np.concatenate(all_b4),
            "is_mirror": np.concatenate(all_mir),
        },
    )


def run_ablated_condition(
    zero_key: str,
    ckpt_paths: list[Path],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run ensemble inference with one team zeroed, return metrics."""
    M = len(ckpt_paths)
    member_probs_list: list[np.ndarray] = []
    labels_dict = None

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"  Member {i + 1}/{M}: {ckpt_path.parent.name}")
        model = _load_model_from_checkpoint(ckpt_path, device)
        probs_m, ld = _collect_probs_ablated(model, loader, device, zero_key)
        member_probs_list.append(probs_m)
        if labels_dict is None:
            labels_dict = ld
        del model
        torch.cuda.empty_cache()

    assert labels_dict is not None
    p_bar = np.stack(member_probs_list, axis=0).mean(axis=0)

    metrics = compute_metrics(
        probs=p_bar,
        action90_true=labels_dict["action90_true"],
        lead2_true=labels_dict["lead2_true"],
        bring4_observed=labels_dict["bring4_observed"],
        is_mirror=labels_dict["is_mirror"],
    )
    confidence = p_bar.max(axis=1)
    entropy = -np.sum(p_bar * np.log(p_bar + _EPS), axis=-1)
    metrics["mean_confidence"] = float(confidence.mean())
    metrics["mean_entropy"] = float(entropy.mean())
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_opponent_ablation(results: dict[str, dict], out_dir: Path) -> None:
    """Grouped bar chart: top-1/3/5 accuracy across the three conditions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Baseline", "Opponent zeroed", "Self zeroed"]
    metric_keys = ["overall/top1_action90", "overall/top3_action90", "overall/top5_action90"]
    metric_labels = ["Top-1", "Top-3", "Top-5"]

    x = np.arange(len(labels))
    width = 0.22

    for j, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        vals = [results[c].get(mk, 0) * 100 for c in CONDITIONS]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=ml, color=COLORS[j])
        # Annotate delta on ablated bars
        base_val = vals[0]
        for k in (1, 2):
            delta = vals[k] - base_val
            ax.annotate(
                f"{delta:+.1f}pp",
                xy=(x[k] + (j - 1) * width, vals[k]),
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Opponent Ablation: Action-90 Accuracy (Tier 1)")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_fig(fig, out_dir / "opponent_ablation")
    print(f"  Saved: {out_dir / 'opponent_ablation.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  Opponent Ablation Experiment")
    print("=" * 70)

    # --- Verify paths ---
    for p in ENSEMBLE_MEMBERS:
        if not p.exists():
            print(f"ERROR: Checkpoint not found: {p}")
            sys.exit(1)
    if not CACHED_PREDS.exists():
        print(f"ERROR: Cached predictions not found: {CACHED_PREDS}")
        sys.exit(1)

    # --- Build test DataLoader (needed for ablated conditions) ---
    print(f"\nLoading test data from {DATA_A} ...")
    vocab = Vocab.load(VOCAB_PATH)
    test_ds = VGCDataset(DATA_A / "test.jsonl", vocab)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    print(f"Test set: {len(test_ds)} examples, {len(test_loader)} batches")
    print(f"Ensemble: {len(ENSEMBLE_MEMBERS)} members | Device: {DEVICE}")

    results: dict[str, dict] = {}

    # --- Condition 1: baseline (from cache) ---
    print(f"\n{'─' * 60}")
    print("Condition: baseline (from cached predictions)")
    print(f"{'─' * 60}")
    cached = np.load(CACHED_PREDS, allow_pickle=False)
    baseline_metrics = compute_metrics(
        probs=cached["probs"],
        action90_true=cached["action90_true"],
        lead2_true=cached["lead2_true"],
        bring4_observed=cached["bring4_observed"],
        is_mirror=cached["is_mirror"],
    )
    baseline_metrics["mean_confidence"] = float(cached["confidence"].mean())
    baseline_metrics["mean_entropy"] = float(cached["entropy"].mean())
    results["baseline"] = baseline_metrics
    t1 = baseline_metrics.get("overall/top1_action90", 0)
    print(f"  top1={t1:.1%}  top3={baseline_metrics.get('overall/top3_action90', 0):.1%}")

    # --- Condition 2: team_b zeroed ---
    print(f"\n{'─' * 60}")
    print("Condition: team_b_zeroed (opponent hidden)")
    print(f"{'─' * 60}")
    results["team_b_zeroed"] = run_ablated_condition(
        "team_b", ENSEMBLE_MEMBERS, test_loader, DEVICE,
    )

    # --- Condition 3: team_a zeroed ---
    print(f"\n{'─' * 60}")
    print("Condition: team_a_zeroed (self hidden — sanity check)")
    print(f"{'─' * 60}")
    results["team_a_zeroed"] = run_ablated_condition(
        "team_a", ENSEMBLE_MEMBERS, test_loader, DEVICE,
    )

    # --- Compute deltas ---
    base_t1 = results["baseline"].get("overall/top1_action90", 0)
    base_t3 = results["baseline"].get("overall/top3_action90", 0)
    base_nll = results["baseline"].get("overall/nll_action90", 0)

    deltas = {}
    for cond in ("team_b_zeroed", "team_a_zeroed"):
        m = results[cond]
        deltas[cond] = {
            "top1_delta_pp": (m.get("overall/top1_action90", 0) - base_t1) * 100,
            "top3_delta_pp": (m.get("overall/top3_action90", 0) - base_t3) * 100,
            "nll_delta": m.get("overall/nll_action90", 0) - base_nll,
        }

    # --- Save JSON ---
    out_json: dict = {}
    for cond in CONDITIONS:
        out_json[cond] = {
            k: float(v) if hasattr(v, "__float__") else v
            for k, v in results[cond].items()
        }
    out_json["deltas"] = deltas

    json_path = OUT_EVAL / "opponent_ablation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # --- Plot ---
    print("\nGenerating figure...")
    plot_opponent_ablation(results, OUT_PLOTS)

    # --- Summary table ---
    print(f"\n{'=' * 70}")
    print("  OPPONENT ABLATION SUMMARY")
    print(f"{'=' * 70}\n")

    header = f"{'Condition':<20} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'NLL':>8} {'Conf':>8}"
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        m = results[cond]
        print(
            f"{cond:<20} "
            f"{m.get('overall/top1_action90', 0) * 100:>6.1f}% "
            f"{m.get('overall/top3_action90', 0) * 100:>6.1f}% "
            f"{m.get('overall/top5_action90', 0) * 100:>6.1f}% "
            f"{m.get('overall/nll_action90', 0):>8.3f} "
            f"{m.get('mean_confidence', 0):>8.4f}"
        )

    print("\nDeltas from baseline:")
    for cond in ("team_b_zeroed", "team_a_zeroed"):
        d = deltas[cond]
        print(
            f"  {cond:<20} "
            f"Top-1: {d['top1_delta_pp']:>+6.1f}pp  "
            f"Top-3: {d['top3_delta_pp']:>+6.1f}pp  "
            f"NLL: {d['nll_delta']:>+7.3f}"
        )

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s ({dt / 60:.1f} min)")


if __name__ == "__main__":
    main()
