#!/usr/bin/env python3
"""Run moves-hidden stress test (Week 4, Task 0).

Loads ensemble checkpoints + temperature, builds test DataLoader,
runs stress test across all masking configurations, saves results
JSON + plots, and prints a summary table.

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/run_stress_test.py
"""

import json
import sys
import time
from pathlib import Path

import torch

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.data.dataset import Vocab, VGCDataset
from turnzero.eval.robustness import (
    MASK_ORDER,
    plot_stress_test,
    run_stress_test,
)
from turnzero.uq.temperature import TemperatureScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_A = ROOT / "data" / "assembled" / "regime_a"
ENSEMBLE_DIR = ROOT / "outputs" / "runs"
ENSEMBLE_MEMBERS = [
    ENSEMBLE_DIR / f"ensemble_{i:03d}" / "best.pt" for i in range(1, 6)
]
TEMP_JSON = ROOT / "outputs" / "calibration" / "run_001" / "temperature.json"
VOCAB_PATH = ROOT / "outputs" / "runs" / "ensemble_001" / "vocab.json"

OUT_EVAL = ROOT / "outputs" / "eval"
OUT_PLOTS = ROOT / "outputs" / "plots" / "week4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 4


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  Moves-Hidden Stress Test (Week 4, Task 0)")
    print("=" * 70)

    # --- Load temperature ---
    scaler = TemperatureScaler.load(TEMP_JSON)
    print(f"\nTemperature: T={scaler.T:.4f}")

    # --- Build test DataLoader ---
    print(f"\nLoading test data from {DATA_A} ...")
    vocab = Vocab.load(VOCAB_PATH)
    test_ds = VGCDataset(DATA_A / "test.jsonl", vocab)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Test set: {len(test_ds)} examples, {len(test_loader)} batches")

    # --- Verify checkpoint paths ---
    for p in ENSEMBLE_MEMBERS:
        if not p.exists():
            print(f"ERROR: Checkpoint not found: {p}")
            sys.exit(1)
    print(f"Ensemble: {len(ENSEMBLE_MEMBERS)} members")
    print(f"Device: {DEVICE}")

    # --- Run stress test ---
    results = run_stress_test(
        ckpt_paths=ENSEMBLE_MEMBERS,
        loader=test_loader,
        device=DEVICE,
        temperature=scaler.T,
        mask_configs=MASK_ORDER,
        seed=42,
    )

    # --- Save results JSON ---
    # Convert any numpy types for JSON serialization
    results_json = {}
    for cfg_name, metrics in results.items():
        results_json[cfg_name] = {
            k: float(v) if hasattr(v, "__float__") else v
            for k, v in metrics.items()
        }

    json_path = OUT_EVAL / "stress_test.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # --- Generate plots ---
    print("\nGenerating plots...")
    plot_stress_test(results, OUT_PLOTS)

    # --- Print summary table ---
    print(f"\n{'=' * 70}")
    print("  STRESS TEST SUMMARY")
    print(f"{'=' * 70}")
    print()

    # Header
    header = f"{'Config':<20} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'NLL':>8} {'Conf':>8} {'Entropy':>8}"
    print(header)
    print("-" * len(header))

    for cfg in MASK_ORDER:
        if cfg not in results:
            continue
        m = results[cfg]
        t1 = m.get("overall/top1_action90", 0) * 100
        t3 = m.get("overall/top3_action90", 0) * 100
        t5 = m.get("overall/top5_action90", 0) * 100
        nll = m.get("overall/nll_action90", 0)
        conf = m.get("mean_confidence", 0)
        ent = m.get("mean_entropy", 0)
        print(f"{cfg:<20} {t1:>6.1f}% {t3:>6.1f}% {t5:>6.1f}% {nll:>8.3f} {conf:>8.4f} {ent:>8.3f}")

    # Degradation from baseline
    if "baseline" in results and len(results) > 1:
        print()
        print("Degradation from baseline:")
        base = results["baseline"]
        base_t1 = base.get("overall/top1_action90", 0)
        base_t3 = base.get("overall/top3_action90", 0)
        base_nll = base.get("overall/nll_action90", 0)

        for cfg in MASK_ORDER:
            if cfg == "baseline" or cfg not in results:
                continue
            m = results[cfg]
            dt1 = (m.get("overall/top1_action90", 0) - base_t1) * 100
            dt3 = (m.get("overall/top3_action90", 0) - base_t3) * 100
            dnll = m.get("overall/nll_action90", 0) - base_nll
            print(f"  {cfg:<20} Top-1: {dt1:>+6.1f}pp  Top-3: {dt3:>+6.1f}pp  NLL: {dnll:>+7.3f}")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s ({dt / 60:.1f} min)")


if __name__ == "__main__":
    main()
