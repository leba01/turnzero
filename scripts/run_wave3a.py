#!/usr/bin/env python3
"""Wave 3A: Full Week 3 evaluation pipeline.

Runs all evaluations WITHOUT modifying source code:
1. Ensemble evaluation on Regime A test
2. Risk-coverage curves (both risk defs, all 4 models)
3. Bootstrap CIs (B=1000, seed=42)
4. Regime B OOD evaluation
5. Comprehensive summary

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/run_wave3a.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.data.dataset import VGCDataset, Vocab, build_dataloaders
from turnzero.data.io_utils import read_jsonl
from turnzero.eval.bootstrap import cluster_bootstrap_ci
from turnzero.eval.metrics import compute_metrics
from turnzero.eval.risk_coverage import (
    plot_risk_coverage,
    risk_coverage_curve,
)
from turnzero.models.baselines import LogisticBaseline, PopularityBaseline
from turnzero.models.train import validate
from turnzero.uq.ensemble import (
    ensemble_predict,
    load_ensemble_predictions,
    save_ensemble_predictions,
)
from turnzero.uq.temperature import TemperatureScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_A = ROOT / "data" / "assembled" / "regime_a"
DATA_B = ROOT / "data" / "assembled" / "regime_b"
CKPT_SINGLE = ROOT / "outputs" / "runs" / "run_001" / "best.pt"
VOCAB_SINGLE = ROOT / "outputs" / "runs" / "run_001" / "vocab.json"
TEMP_JSON = ROOT / "outputs" / "calibration" / "run_001" / "temperature.json"
ENSEMBLE_DIR = ROOT / "outputs" / "runs"
ENSEMBLE_MEMBERS = [
    ENSEMBLE_DIR / f"ensemble_{i:03d}" / "best.pt" for i in range(1, 6)
]

OUT_ENSEMBLE = ROOT / "outputs" / "ensemble"
OUT_EVAL = ROOT / "outputs" / "eval"
OUT_PLOTS = ROOT / "outputs" / "plots" / "week3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 4


def _save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Ensemble evaluation on Regime A test
# ═══════════════════════════════════════════════════════════════════════════

def step1_ensemble_eval():
    print("\n" + "=" * 70)
    print("STEP 1: Ensemble evaluation on Regime A test")
    print("=" * 70)

    # Load temperature
    scaler = TemperatureScaler.load(TEMP_JSON)
    T = scaler.T
    print(f"  Temperature: T = {T:.4f}")

    # Build test DataLoader using ensemble_001's vocab (all trained on same data)
    vocab = Vocab.load(ENSEMBLE_MEMBERS[0].parent / "vocab.json")
    test_ds = VGCDataset(DATA_A / "test.jsonl", vocab)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"  Test set: {len(test_ds):,} examples")

    # Run ensemble prediction with temperature scaling
    t0 = time.time()
    ens_preds = ensemble_predict(
        ckpt_paths=ENSEMBLE_MEMBERS,
        loader=test_loader,
        device=DEVICE,
        temperature=T,
    )
    dt = time.time() - t0
    print(f"  Ensemble inference done in {dt:.1f}s")

    # Save predictions
    OUT_ENSEMBLE.mkdir(parents=True, exist_ok=True)
    save_ensemble_predictions(ens_preds, OUT_ENSEMBLE / "ensemble_predictions.npz")

    # Compute metrics
    metrics = compute_metrics(
        probs=ens_preds["probs"],
        action90_true=ens_preds["action90_true"],
        lead2_true=ens_preds["lead2_true"],
        bring4_observed=ens_preds["bring4_observed"],
        is_mirror=ens_preds["is_mirror"],
    )
    _save_json(metrics, OUT_ENSEMBLE / "test_metrics.json")

    # Print key metrics
    print(f"\n  Ensemble Test Metrics (Regime A):")
    print(f"    Action90 Top-1 (Tier 1):  {metrics['overall/top1_action90']:.1%}")
    print(f"    Action90 Top-3 (Tier 1):  {metrics['overall/top3_action90']:.1%}")
    print(f"    Action90 Top-5 (Tier 1):  {metrics['overall/top5_action90']:.1%}")
    print(f"    Lead-2 Top-1 (all):       {metrics['overall/top1_lead2']:.1%}")
    print(f"    Lead-2 Top-3 (all):       {metrics['overall/top3_lead2']:.1%}")
    print(f"    NLL Action90:             {metrics['overall/nll_action90']:.4f}")
    print(f"    ECE Action90:             {metrics['overall/ece_action90']:.4f}")
    print(f"    Mean entropy:             {ens_preds['entropy'].mean():.4f}")
    print(f"    Mean MI (epistemic):      {ens_preds['mi'].mean():.4f}")
    print(f"    Mean confidence:          {ens_preds['confidence'].mean():.4f}")

    return ens_preds, metrics, test_loader, vocab


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Risk-coverage curves
# ═══════════════════════════════════════════════════════════════════════════

def step2_risk_coverage(ens_preds, test_loader, vocab):
    print("\n" + "=" * 70)
    print("STEP 2: Risk-coverage curves (all models, both risk defs)")
    print("=" * 70)

    # We need Tier 1 (bring4_observed) predictions for action90 risk-coverage
    tier1 = ens_preds["bring4_observed"].astype(bool)
    action90_true_t1 = ens_preds["action90_true"][tier1]
    ens_probs_t1 = ens_preds["probs"][tier1]

    print(f"  Tier 1 examples: {tier1.sum():,} / {len(tier1):,}")

    # --- Baseline: Popularity ---
    print("\n  Computing popularity baseline predictions...")
    train_examples = list(read_jsonl(DATA_A / "train.jsonl"))
    train_labels = np.array([ex["label"]["action90_id"] for ex in train_examples])
    pop = PopularityBaseline()
    pop.fit(train_labels)
    pop_probs_full = pop.predict(len(ens_preds["action90_true"]))
    pop_probs_t1 = pop_probs_full[tier1]

    # --- Baseline: Logistic regression ---
    print("  Computing logistic baseline predictions (fitting may take a minute)...")
    test_examples = list(read_jsonl(DATA_A / "test.jsonl"))
    t0 = time.time()
    log_bl = LogisticBaseline(C=1.0, max_iter=1000)
    log_bl.fit(train_examples, train_labels)
    log_probs_full = log_bl.predict_proba(test_examples)
    log_probs_t1 = log_probs_full[tier1]
    print(f"  Logistic baseline fit + predict done in {time.time() - t0:.1f}s")

    # Free memory
    del train_examples, train_labels, test_examples
    import gc; gc.collect()

    # --- Single transformer (with temp scaling) ---
    print("  Computing single transformer predictions...")
    scaler = TemperatureScaler.load(TEMP_JSON)
    T = scaler.T

    from turnzero.models.transformer import ModelConfig, OTSTransformer
    ckpt = torch.load(CKPT_SINGLE, map_location=DEVICE, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    single_vocab = Vocab.load(VOCAB_SINGLE)
    model = OTSTransformer(ckpt["vocab_sizes"], model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # Build test loader with single model's vocab
    single_test_ds = VGCDataset(DATA_A / "test.jsonl", single_vocab)
    single_test_loader = torch.utils.data.DataLoader(
        single_test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    _, single_probs, single_labels = validate(model, single_test_loader, criterion, DEVICE)

    # Apply temperature scaling to single model probs
    # Need logits for proper temp scaling - recompute from probs is lossy
    # Instead, collect logits directly
    from turnzero.uq.temperature import collect_logits
    single_logits, _ = collect_logits(model, single_test_loader, DEVICE)
    single_probs_cal = scaler.calibrate(single_logits)

    single_probs_t1 = single_probs_cal[tier1]

    del model, single_logits
    torch.cuda.empty_cache()

    # --- Compute risk-coverage curves ---
    all_models = {
        "Popularity": pop_probs_t1,
        "Logistic": log_probs_t1,
        "Single Transformer": single_probs_t1,
        "Ensemble (5-member)": ens_probs_t1,
    }

    rc_results = {}

    # Risk def 1: 1 - top1_accuracy
    print("\n  Computing risk-coverage curves (Top-1 risk)...")
    rc_top1 = {}
    for name, probs in all_models.items():
        rc = risk_coverage_curve(probs, action90_true_t1, k=1, n_thresholds=200)
        rc_top1[name] = rc
        print(f"    {name}: AURC = {rc['aurc']:.4f}")
    rc_results["top1"] = {
        name: {
            "aurc": rc["aurc"],
            "operating_points": rc["operating_points"],
        }
        for name, rc in rc_top1.items()
    }

    # Risk def 2: P(expert not in top-3)
    print("\n  Computing risk-coverage curves (Top-3 risk)...")
    rc_top3 = {}
    for name, probs in all_models.items():
        rc = risk_coverage_curve(probs, action90_true_t1, k=3, n_thresholds=200)
        rc_top3[name] = rc
        print(f"    {name}: AURC = {rc['aurc']:.4f}")
    rc_results["top3"] = {
        name: {
            "aurc": rc["aurc"],
            "operating_points": rc["operating_points"],
        }
        for name, rc in rc_top3.items()
    }

    # Save curves + operating points
    _save_json(rc_results, OUT_EVAL / "risk_coverage.json")

    # Plot risk-coverage curves
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    plot_risk_coverage(
        rc_top1,
        OUT_PLOTS / "risk_coverage_top1",
        title="Risk-Coverage: Action-90 Top-1 (Tier 1)",
        risk_label="Risk (1 - Top-1 Accuracy)",
    )
    print(f"  Saved: {OUT_PLOTS / 'risk_coverage_top1.png'}")

    plot_risk_coverage(
        rc_top3,
        OUT_PLOTS / "risk_coverage_top3",
        title="Risk-Coverage: Action-90 Top-3 (Tier 1)",
        risk_label="Risk (P(expert not in top-3))",
    )
    print(f"  Saved: {OUT_PLOTS / 'risk_coverage_top3.png'}")

    return rc_results, pop_probs_full, log_probs_full, single_probs_cal


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════════

def step3_bootstrap_cis(ens_preds):
    print("\n" + "=" * 70)
    print("STEP 3: Cluster-aware bootstrap CIs (B=1000, seed=42)")
    print("=" * 70)

    # Extract cluster IDs from test JSONL
    test_examples = list(read_jsonl(DATA_A / "test.jsonl"))
    cluster_ids = np.array([
        ex["split_keys"]["core_cluster_a"] for ex in test_examples
    ])
    del test_examples

    n_clusters = len(np.unique(cluster_ids))
    print(f"  Unique clusters: {n_clusters:,}")
    print(f"  Examples: {len(cluster_ids):,}")

    t0 = time.time()
    cis = cluster_bootstrap_ci(
        probs=ens_preds["probs"],
        action90_true=ens_preds["action90_true"],
        lead2_true=ens_preds["lead2_true"],
        bring4_observed=ens_preds["bring4_observed"],
        is_mirror=ens_preds["is_mirror"],
        cluster_ids=cluster_ids,
        n_bootstrap=1000,
        ci_level=0.95,
        seed=42,
    )
    dt = time.time() - t0
    print(f"\n  Bootstrap done in {dt:.1f}s")

    _save_json(cis, OUT_EVAL / "bootstrap_cis.json")

    # Print key CIs
    print("\n  Key 95% CIs (Ensemble, Regime A test):")
    for key in [
        "overall/top1_action90",
        "overall/top3_action90",
        "overall/top5_action90",
        "overall/nll_action90",
        "overall/ece_action90",
        "overall/top1_lead2",
        "overall/top3_lead2",
    ]:
        if key in cis:
            ci = cis[key]
            print(f"    {key}: {ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]")

    return cis


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Regime B OOD evaluation
# ═══════════════════════════════════════════════════════════════════════════

def step4_ood_eval(ens_preds_a):
    print("\n" + "=" * 70)
    print("STEP 4: Regime B (OOD) evaluation")
    print("=" * 70)

    # Load temperature
    scaler = TemperatureScaler.load(TEMP_JSON)
    T = scaler.T

    # Build Regime B test loader using ensemble vocab
    # (same vocab — trained on Regime A data, evaluating OOD)
    vocab = Vocab.load(ENSEMBLE_MEMBERS[0].parent / "vocab.json")
    test_b_ds = VGCDataset(DATA_B / "test.jsonl", vocab)
    test_b_loader = torch.utils.data.DataLoader(
        test_b_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"  Regime B test set: {len(test_b_ds):,} examples")

    # Run ensemble prediction
    t0 = time.time()
    ens_preds_b = ensemble_predict(
        ckpt_paths=ENSEMBLE_MEMBERS,
        loader=test_b_loader,
        device=DEVICE,
        temperature=T,
    )
    dt = time.time() - t0
    print(f"  Ensemble inference on Regime B done in {dt:.1f}s")

    # Compute metrics
    metrics_b = compute_metrics(
        probs=ens_preds_b["probs"],
        action90_true=ens_preds_b["action90_true"],
        lead2_true=ens_preds_b["lead2_true"],
        bring4_observed=ens_preds_b["bring4_observed"],
        is_mirror=ens_preds_b["is_mirror"],
    )

    # Comparison: Regime A vs Regime B
    # Compute abstention rate at a threshold derived from Regime A
    # Use the 80% coverage threshold from ensemble top-3 risk-coverage
    conf_a = ens_preds_a["confidence"]
    conf_b = ens_preds_b["confidence"]

    # Find threshold giving ~80% coverage on Regime A
    tau_80 = np.percentile(conf_a, 20)  # 20th percentile → 80% coverage

    abstain_rate_a = float((conf_a < tau_80).mean())
    abstain_rate_b = float((conf_b < tau_80).mean())

    comparison = {
        "regime_a": {
            "n_examples": int(len(conf_a)),
            "mean_entropy": float(ens_preds_a["entropy"].mean()),
            "std_entropy": float(ens_preds_a["entropy"].std()),
            "mean_mi": float(ens_preds_a["mi"].mean()),
            "std_mi": float(ens_preds_a["mi"].std()),
            "mean_confidence": float(conf_a.mean()),
            "abstain_rate_at_tau80": abstain_rate_a,
            "top1_action90": float(metrics_a["overall/top1_action90"]) if "metrics_a" in dir() else None,
            "top3_action90": float(metrics_a["overall/top3_action90"]) if "metrics_a" in dir() else None,
        },
        "regime_b": {
            "n_examples": int(len(conf_b)),
            "mean_entropy": float(ens_preds_b["entropy"].mean()),
            "std_entropy": float(ens_preds_b["entropy"].std()),
            "mean_mi": float(ens_preds_b["mi"].mean()),
            "std_mi": float(ens_preds_b["mi"].std()),
            "mean_confidence": float(conf_b.mean()),
            "abstain_rate_at_tau80": abstain_rate_b,
            "top1_action90": float(metrics_b.get("overall/top1_action90", 0)),
            "top3_action90": float(metrics_b.get("overall/top3_action90", 0)),
        },
        "tau_80_threshold": float(tau_80),
        "metrics_b": metrics_b,
    }

    _save_json(comparison, OUT_EVAL / "ood_comparison.json")

    print(f"\n  OOD Comparison (Regime A vs Regime B):")
    print(f"    {'Metric':<30} {'Regime A':>12} {'Regime B':>12} {'Delta':>10}")
    print(f"    {'─' * 64}")
    print(f"    {'Mean entropy':<30} {ens_preds_a['entropy'].mean():>12.4f} {ens_preds_b['entropy'].mean():>12.4f} {ens_preds_b['entropy'].mean() - ens_preds_a['entropy'].mean():>+10.4f}")
    print(f"    {'Mean MI (epistemic)':<30} {ens_preds_a['mi'].mean():>12.4f} {ens_preds_b['mi'].mean():>12.4f} {ens_preds_b['mi'].mean() - ens_preds_a['mi'].mean():>+10.4f}")
    print(f"    {'Mean confidence':<30} {conf_a.mean():>12.4f} {conf_b.mean():>12.4f} {conf_b.mean() - conf_a.mean():>+10.4f}")
    print(f"    {'Abstain rate (tau@80%cov)':<30} {abstain_rate_a:>12.1%} {abstain_rate_b:>12.1%} {abstain_rate_b - abstain_rate_a:>+10.1%}")
    if metrics_b.get("overall/top1_action90") is not None:
        print(f"    {'Action90 Top-1':<30} {comparison['regime_a'].get('top1_action90', 0) or 0:>12.1%} {metrics_b['overall/top1_action90']:>12.1%}")
        print(f"    {'Action90 Top-3':<30} {comparison['regime_a'].get('top3_action90', 0) or 0:>12.1%} {metrics_b['overall/top3_action90']:>12.1%}")

    return comparison, ens_preds_b


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Comprehensive summary
# ═══════════════════════════════════════════════════════════════════════════

def step5_summary(ens_metrics, single_metrics, rc_results, cis, ood_comparison):
    print("\n" + "=" * 70)
    print("STEP 5: COMPREHENSIVE WEEK 3 SUMMARY")
    print("=" * 70)

    # Load baseline metrics for comparison
    with open(ROOT / "outputs" / "baselines" / "metrics_popularity.json") as f:
        pop_metrics = json.load(f)
    with open(ROOT / "outputs" / "baselines" / "metrics_logistic.json") as f:
        log_metrics = json.load(f)

    print("\n" + "─" * 80)
    print("TABLE 1: Model Comparison — Action-90 Metrics (Tier 1, Overall)")
    print("─" * 80)
    print(f"  {'Model':<25} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'NLL':>8} {'ECE':>8}")
    print(f"  {'─' * 65}")

    rows = [
        ("Popularity", pop_metrics),
        ("Logistic", log_metrics),
        ("Single Transformer", single_metrics),
        ("Ensemble (5-member)", ens_metrics),
    ]
    for name, m in rows:
        t1 = m.get("overall/top1_action90", 0)
        t3 = m.get("overall/top3_action90", 0)
        t5 = m.get("overall/top5_action90", 0)
        nll = m.get("overall/nll_action90", 0)
        ece = m.get("overall/ece_action90", 0)
        print(f"  {name:<25} {t1:>7.1%} {t3:>7.1%} {t5:>7.1%} {nll:>8.3f} {ece:>8.4f}")

    # With CIs for ensemble
    if cis:
        print(f"\n  Ensemble 95% CIs:")
        for key in ["overall/top1_action90", "overall/top3_action90", "overall/nll_action90", "overall/ece_action90"]:
            if key in cis:
                ci = cis[key]
                print(f"    {key}: {ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]")

    print("\n" + "─" * 80)
    print("TABLE 2: Lead-2 Metrics (All Examples, Overall)")
    print("─" * 80)
    print(f"  {'Model':<25} {'Top-1':>8} {'Top-3':>8} {'NLL':>8} {'ECE':>8}")
    print(f"  {'─' * 50}")
    for name, m in rows:
        t1 = m.get("overall/top1_lead2", 0)
        t3 = m.get("overall/top3_lead2", 0)
        nll = m.get("overall/nll_lead2", 0)
        ece = m.get("overall/ece_lead2", 0)
        print(f"  {name:<25} {t1:>7.1%} {t3:>7.1%} {nll:>8.3f} {ece:>8.4f}")

    print("\n" + "─" * 80)
    print("TABLE 3: Risk-Coverage Operating Points (Action-90, Tier 1)")
    print("─" * 80)
    for risk_type, risk_label in [("top1", "Top-1"), ("top3", "Top-3")]:
        print(f"\n  {risk_label} Risk:")
        print(f"  {'Model':<25} {'AURC':>8} {'Risk@95%':>10} {'Risk@80%':>10} {'Risk@60%':>10}")
        print(f"  {'─' * 63}")
        for name, rc_data in rc_results[risk_type].items():
            aurc = rc_data["aurc"]
            ops = rc_data["operating_points"]
            r95 = ops["95"]["risk"] if not np.isnan(ops["95"]["risk"]) else float("nan")
            r80 = ops["80"]["risk"] if not np.isnan(ops["80"]["risk"]) else float("nan")
            r60 = ops["60"]["risk"] if not np.isnan(ops["60"]["risk"]) else float("nan")
            print(f"  {name:<25} {aurc:>8.4f} {r95:>9.1%} {r80:>9.1%} {r60:>9.1%}")

    print("\n" + "─" * 80)
    print("TABLE 4: OOD Comparison (Regime A vs Regime B)")
    print("─" * 80)
    ra = ood_comparison["regime_a"]
    rb = ood_comparison["regime_b"]
    print(f"  {'Metric':<30} {'Regime A':>12} {'Regime B':>12}")
    print(f"  {'─' * 54}")
    print(f"  {'N examples':<30} {ra['n_examples']:>12,} {rb['n_examples']:>12,}")
    print(f"  {'Mean entropy':<30} {ra['mean_entropy']:>12.4f} {rb['mean_entropy']:>12.4f}")
    print(f"  {'Mean MI (epistemic)':<30} {ra['mean_mi']:>12.4f} {rb['mean_mi']:>12.4f}")
    print(f"  {'Mean confidence':<30} {ra['mean_confidence']:>12.4f} {rb['mean_confidence']:>12.4f}")
    print(f"  {'Abstain rate @ tau80':<30} {ra['abstain_rate_at_tau80']:>11.1%} {rb['abstain_rate_at_tau80']:>11.1%}")

    print("\n" + "─" * 80)
    print("TABLE 5: Calibration — Temperature Scaling Effect")
    print("─" * 80)
    cal_report = json.load(open(ROOT / "outputs" / "calibration" / "run_001" / "calibration_report.json"))
    print(f"  Fitted T: {cal_report['T']:.4f}")
    print(f"  Val NLL:  {cal_report['val_nll_before']:.4f} → {cal_report['val_nll_after']:.4f}")
    print(f"  Val ECE:  {cal_report['val_ece_before']:.4f} → {cal_report['val_ece_after']:.4f}")

    print("\n" + "─" * 80)
    print("OUTPUT ARTIFACTS:")
    print("─" * 80)
    artifacts = [
        OUT_ENSEMBLE / "test_metrics.json",
        OUT_ENSEMBLE / "ensemble_predictions.npz",
        OUT_EVAL / "risk_coverage.json",
        OUT_EVAL / "bootstrap_cis.json",
        OUT_EVAL / "ood_comparison.json",
        OUT_PLOTS / "risk_coverage_top1.png",
        OUT_PLOTS / "risk_coverage_top1.pdf",
        OUT_PLOTS / "risk_coverage_top3.png",
        OUT_PLOTS / "risk_coverage_top3.pdf",
    ]
    for p in artifacts:
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.relative_to(ROOT)}")

    print("\n" + "=" * 70)
    print("WEEK 3 EVALUATION COMPLETE")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    overall_t0 = time.time()

    # Step 1: Ensemble eval
    ens_preds, ens_metrics, test_loader, vocab = step1_ensemble_eval()

    # Step 2: Risk-coverage curves
    rc_results, pop_probs, log_probs, single_probs = step2_risk_coverage(
        ens_preds, test_loader, vocab
    )

    # Step 3: Bootstrap CIs
    cis = step3_bootstrap_cis(ens_preds)

    # Load single model metrics for comparison in summary
    with open(ROOT / "outputs" / "eval" / "run_001" / "test_metrics.json") as f:
        single_metrics = json.load(f)

    # Step 4: OOD eval — need metrics_a for comparison
    # Patch the Regime A metrics into the comparison function scope
    metrics_a = ens_metrics

    # We need to make metrics_a accessible in step4. Restructure slightly:
    ood_comparison, ens_preds_b = step4_ood_eval(ens_preds)
    # Fix the Regime A metrics in comparison
    ood_comparison["regime_a"]["top1_action90"] = float(ens_metrics.get("overall/top1_action90", 0))
    ood_comparison["regime_a"]["top3_action90"] = float(ens_metrics.get("overall/top3_action90", 0))
    _save_json(ood_comparison, OUT_EVAL / "ood_comparison.json")

    # Step 5: Summary
    step5_summary(ens_metrics, single_metrics, rc_results, cis, ood_comparison)

    total_time = time.time() - overall_t0
    print(f"\nTotal wall time: {total_time:.0f}s ({total_time/60:.1f} min)")
