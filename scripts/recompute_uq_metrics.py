#!/usr/bin/env python3
"""Recompute corrected UQ metrics from saved ensemble predictions.

Computes:
  1. Adaptive ECE (equal-mass binning, Nixon et al. 2019)
  2. E-AURC (excess AURC = AURC - optimal AURC)
  3. OOD AUROC (entropy-based ID/OOD discrimination)
  4. Brier score (already stored, just surfaced)

Items 1-2 and 4 use saved Regime A predictions (no GPU needed).
Item 3 re-runs Regime B ensemble inference (~2 min GPU).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from turnzero.eval.metrics import _ece, _ece_adaptive, compute_metrics
from turnzero.eval.risk_coverage import risk_coverage_curve


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load Regime A ensemble predictions
    # ------------------------------------------------------------------
    npz_path = ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz"
    print(f"Loading Regime A predictions from {npz_path}")
    d = np.load(npz_path)
    probs = d["probs"]  # (N, 90)
    action90_true = d["action90_true"]
    bring4_observed = d["bring4_observed"]

    # Tier 1 only
    tier1 = bring4_observed.astype(bool)
    p_t1 = probs[tier1]
    y_t1 = action90_true[tier1]
    conf_t1 = p_t1.max(axis=1)
    correct_t1 = p_t1.argmax(axis=1) == y_t1
    entropy_a = d["entropy"]  # (N,) — all examples, for AUROC

    print(f"  Regime A: {len(probs):,} total, {tier1.sum():,} Tier 1")

    # ------------------------------------------------------------------
    # 2. Adaptive ECE
    # ------------------------------------------------------------------
    ece_fixed = _ece(conf_t1, correct_t1, n_bins=15)
    ece_adaptive = _ece_adaptive(conf_t1, correct_t1, n_bins=15)
    print(f"\n--- Adaptive ECE ---")
    print(f"  ECE (equal-width, 15 bins): {ece_fixed:.6f}")
    print(f"  ECE (adaptive, 15 bins):    {ece_adaptive:.6f}")

    # ------------------------------------------------------------------
    # 3. E-AURC (top-1 and top-3)
    # ------------------------------------------------------------------
    print(f"\n--- E-AURC ---")
    results = {}
    for k_val in (1, 3):
        rc = risk_coverage_curve(p_t1, y_t1, k=k_val)
        aurc = rc["aurc"]
        aurc_opt = rc["aurc_optimal"]
        e_aurc = rc["e_aurc"]
        print(f"  Top-{k_val}: AURC={aurc:.6f}, AURC_opt={aurc_opt:.6f}, E-AURC={e_aurc:.6f}")
        results[f"top{k_val}"] = {
            "aurc": aurc,
            "aurc_optimal": aurc_opt,
            "e_aurc": e_aurc,
        }

    # ------------------------------------------------------------------
    # 4. Brier score (already computed, just surface it)
    # ------------------------------------------------------------------
    metrics_a = compute_metrics(
        probs=probs,
        action90_true=action90_true,
        lead2_true=d["lead2_true"],
        bring4_observed=bring4_observed,
        is_mirror=d["is_mirror"],
    )
    brier_ens = metrics_a["overall/brier_action90"]
    ece_adaptive_full = metrics_a["overall/ece_adaptive_action90"]
    print(f"\n--- Brier score ---")
    print(f"  Ensemble Brier (action90): {brier_ens:.6f}")
    print(f"  Adaptive ECE (from compute_metrics): {ece_adaptive_full:.6f}")

    # ------------------------------------------------------------------
    # 5. OOD AUROC via entropy
    # ------------------------------------------------------------------
    print(f"\n--- OOD AUROC ---")
    print("  Re-running Regime B ensemble inference...")

    from sklearn.metrics import roc_auc_score

    from turnzero.data.dataset import VGCDataset, Vocab
    from turnzero.uq.ensemble import ensemble_predict

    DATA_B = ROOT / "data" / "assembled" / "regime_b"
    ENSEMBLE_DIR = ROOT / "outputs" / "runs"
    ENSEMBLE_MEMBERS = [
        ENSEMBLE_DIR / f"ensemble_{i:03d}" / "best.pt" for i in range(1, 6)
    ]

    vocab = Vocab.load(ENSEMBLE_MEMBERS[0].parent / "vocab.json")
    test_b_ds = VGCDataset(DATA_B / "test.jsonl", vocab)
    test_b_loader = torch.utils.data.DataLoader(
        test_b_ds, batch_size=512, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False,
    )
    print(f"  Regime B test set: {len(test_b_ds):,} examples")

    t0 = time.time()
    ens_preds_b = ensemble_predict(
        ckpt_paths=ENSEMBLE_MEMBERS,
        loader=test_b_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    dt = time.time() - t0
    print(f"  Inference done in {dt:.1f}s")

    entropy_b = ens_preds_b["entropy"]

    # AUROC: ID=0 (Regime A), OOD=1 (Regime B)
    # Higher entropy → more likely OOD
    labels_auroc = np.concatenate([
        np.zeros(len(entropy_a)),
        np.ones(len(entropy_b)),
    ])
    scores_auroc = np.concatenate([entropy_a, entropy_b])
    auroc = roc_auc_score(labels_auroc, scores_auroc)
    print(f"  AUROC (entropy, ID vs OOD): {auroc:.4f}")

    # Also compute with MI
    mi_a = d["mi"]
    mi_b = ens_preds_b["mi"]
    scores_mi = np.concatenate([mi_a, mi_b])
    auroc_mi = roc_auc_score(labels_auroc, scores_mi)
    print(f"  AUROC (MI, ID vs OOD):      {auroc_mi:.4f}")

    # ------------------------------------------------------------------
    # 6. Summary JSON
    # ------------------------------------------------------------------
    summary = {
        "adaptive_ece": {
            "ece_fixed_15bin": round(ece_fixed, 6),
            "ece_adaptive_15bin": round(ece_adaptive, 6),
        },
        "e_aurc": {
            "top1": {k: round(v, 6) for k, v in results["top1"].items()},
            "top3": {k: round(v, 6) for k, v in results["top3"].items()},
        },
        "brier_action90": round(brier_ens, 6),
        "ood_auroc": {
            "entropy": round(auroc, 4),
            "mi": round(auroc_mi, 4),
            "n_id": int(len(entropy_a)),
            "n_ood": int(len(entropy_b)),
        },
    }

    out_path = ROOT / "outputs" / "eval" / "uq_corrected_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("CORRECTED UQ METRICS SUMMARY")
    print("=" * 60)
    print(f"  Adaptive ECE (action90):     {ece_adaptive:.4f}")
    print(f"  E-AURC top-1:               {results['top1']['e_aurc']:.4f}")
    print(f"  E-AURC top-3:               {results['top3']['e_aurc']:.4f}")
    print(f"  Brier (action90):            {brier_ens:.4f}")
    print(f"  OOD AUROC (entropy):         {auroc:.4f}")
    print(f"  OOD AUROC (MI):              {auroc_mi:.4f}")


if __name__ == "__main__":
    main()
