#!/usr/bin/env python3
"""Generate the definitive paper figure set for TurnZero.

Produces publication-quality figures from all evaluation artifacts:
  1. model_comparison — Bar chart: action-90 top-1/3/5, all 4 models
  2. reliability_diagram — Calibration diagram for ensemble (action-90)
  3. risk_coverage — Risk-coverage curves (top-1 and top-3, all models)
  4. stress_test — Degradation plot from masking experiment
  5. ood_comparison — OOD comparison (Regime A vs B)
  6. uncertainty_decomposition — Entropy vs MI scatter, colored by correctness

All saved to outputs/plots/paper/ as PNG + PDF at 300 DPI.

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/run_final_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

matplotlib.use("Agg")  # non-interactive backend

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

PAPER_DIR = ROOT / "outputs" / "plots" / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, out_path: str | Path) -> None:
    """Save figure as both PNG and PDF next to *out_path*."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=_DPI)
    fig.savefig(out.with_suffix(".pdf"), dpi=_DPI)
    plt.close(fig)
    print(f"  Saved: {out.with_suffix('.png')}")
    print(f"  Saved: {out.with_suffix('.pdf')}")


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ===================================================================
# Figure 1: Model Comparison Bar Chart
# ===================================================================

def fig1_model_comparison() -> None:
    """Bar chart: action-90 top-1/3/5 for all 4 models."""
    print("\n[1/6] Model Comparison Bar Chart")

    pop = _load_json(ROOT / "outputs" / "baselines" / "metrics_popularity.json")
    log = _load_json(ROOT / "outputs" / "baselines" / "metrics_logistic.json")
    single = _load_json(ROOT / "outputs" / "eval" / "run_001" / "test_metrics.json")
    ensemble = _load_json(ROOT / "outputs" / "ensemble" / "test_metrics.json")

    model_names = ["Popularity", "Logistic", "Single\nTransformer", "Ensemble\n(5-member)"]
    metrics_list = [pop, log, single, ensemble]

    metric_keys = [
        "overall/top1_action90",
        "overall/top3_action90",
        "overall/top5_action90",
    ]
    labels = ["Top-1", "Top-3", "Top-5"]
    colors = [_COLORS[0], _COLORS[1], _COLORS[2]]

    n_models = len(model_names)
    x = np.arange(n_models)
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for j, (key, label, color) in enumerate(zip(metric_keys, labels, colors)):
        values = [m.get(key, 0.0) * 100 for m in metrics_list]
        bars = ax.bar(
            x + j * bar_width, values, bar_width,
            label=label, color=color, edgecolor="white", linewidth=0.5,
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Action-90 Top-k Accuracy (Tier 1, Regime A Test)")
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, 28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, PAPER_DIR / "model_comparison")


# ===================================================================
# Figure 2: Reliability Diagram (Ensemble)
# ===================================================================

def fig2_reliability_diagram() -> None:
    """Calibration / reliability diagram for the ensemble on action-90."""
    print("\n[2/6] Reliability Diagram")

    # Load ensemble predictions
    ens = np.load(ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz")
    probs = ens["probs"]  # (N, 90)
    action90_true = ens["action90_true"]  # (N,)
    bring4_observed = ens["bring4_observed"]  # (N,)

    # Tier 1 only
    tier1 = bring4_observed.astype(bool)
    probs_t1 = probs[tier1]
    labels_t1 = action90_true[tier1]

    confidence = probs_t1.max(axis=1)
    predicted = probs_t1.argmax(axis=1)
    correct = (predicted == labels_t1).astype(np.float64)

    n_bins = 15
    bin_edges = np.linspace(0.0, confidence.max() * 1.05, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    acc_per_bin = np.full(n_bins, np.nan)
    conf_per_bin = np.full(n_bins, np.nan)
    count_per_bin = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        n_in = mask.sum()
        count_per_bin[i] = n_in
        if n_in > 0:
            acc_per_bin[i] = correct[mask].mean()
            conf_per_bin[i] = confidence[mask].mean()

    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(5.5, 5.5),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # Top: reliability
    # Diagonal — use range of actual confidence values
    max_conf = bin_edges[-1]
    ax_rel.plot([0, max_conf], [0, max_conf], ls="--", color="gray", lw=1,
                label="Perfect calibration")
    valid = ~np.isnan(acc_per_bin)
    ax_rel.bar(
        bin_centers[valid], acc_per_bin[valid],
        width=(bin_edges[1] - bin_edges[0]) * 0.85,
        color=_COLORS[0], edgecolor="white", linewidth=0.5, label="Ensemble",
    )
    # Overlay mean confidence per bin as points
    valid_conf = ~np.isnan(conf_per_bin)
    ax_rel.scatter(
        conf_per_bin[valid_conf], acc_per_bin[valid_conf],
        marker="o", s=20, color=_COLORS[3], zorder=5, label="Mean conf.",
    )
    ax_rel.set_ylabel("Accuracy")
    ax_rel.set_ylim(0, max(0.2, np.nanmax(acc_per_bin) * 1.3))
    ax_rel.legend(loc="upper left", frameon=False, fontsize=9)
    ax_rel.set_title("Ensemble Reliability Diagram (Action-90, Tier 1)")
    ax_rel.spines["top"].set_visible(False)
    ax_rel.spines["right"].set_visible(False)

    # Bottom: histogram
    ax_hist.bar(
        bin_centers, count_per_bin,
        width=(bin_edges[1] - bin_edges[0]) * 0.85,
        color=_COLORS[0], edgecolor="white", linewidth=0.5,
    )
    ax_hist.set_xlabel("Confidence (max p)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(bin_edges[0], bin_edges[-1])
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, PAPER_DIR / "reliability_diagram")


# ===================================================================
# Figure 3: Risk-Coverage Curves
# ===================================================================

def fig3_risk_coverage() -> None:
    """Risk-coverage curves for all 4 models (top-1 and top-3)."""
    print("\n[3/6] Risk-Coverage Curves")

    # We need to recompute the full curves, since risk_coverage.json only has
    # operating points and AURC. Load predictions and recompute.
    from turnzero.eval.risk_coverage import risk_coverage_curve

    ens = np.load(ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz")
    tier1 = ens["bring4_observed"].astype(bool)
    labels_t1 = ens["action90_true"][tier1]
    ens_probs_t1 = ens["probs"][tier1]

    # We need baselines + single model predictions for comparison
    # For a clean figure using only available data, we compute from
    # saved operating points from risk_coverage.json.
    # But to get smooth curves, we need the full coverage/risk arrays.
    # We must recompute from scratch for each model.

    # Load baseline predictions
    from turnzero.data.dataset import VGCDataset, Vocab
    from turnzero.data.io_utils import read_jsonl
    from turnzero.models.baselines import LogisticBaseline, PopularityBaseline
    import torch

    DATA_A = ROOT / "data" / "assembled" / "regime_a"

    # Popularity baseline
    train_examples = list(read_jsonl(DATA_A / "train.jsonl"))
    train_labels = np.array([ex["label"]["action90_id"] for ex in train_examples])
    pop = PopularityBaseline()
    pop.fit(train_labels)
    pop_probs_full = pop.predict(len(ens["action90_true"]))
    pop_probs_t1 = pop_probs_full[tier1]

    # Logistic baseline
    test_examples = list(read_jsonl(DATA_A / "test.jsonl"))
    log_bl = LogisticBaseline(C=1.0, max_iter=1000)
    log_bl.fit(train_examples, train_labels)
    log_probs_full = log_bl.predict_proba(test_examples)
    log_probs_t1 = log_probs_full[tier1]

    del train_examples, train_labels, test_examples
    import gc
    gc.collect()

    # Single transformer
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from turnzero.models.transformer import OTSTransformer
    from turnzero.uq.temperature import collect_logits

    CKPT_SINGLE = ROOT / "outputs" / "runs" / "run_001" / "best.pt"
    VOCAB_SINGLE = ROOT / "outputs" / "runs" / "run_001" / "vocab.json"

    model = OTSTransformer.load_from_checkpoint(CKPT_SINGLE, DEVICE)
    vocab = Vocab.load(VOCAB_SINGLE)

    test_ds = VGCDataset(DATA_A / "test.jsonl", vocab)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    logits, _ = collect_logits(model, test_loader, DEVICE)
    from turnzero.uq.temperature import _softmax
    single_probs_t1 = _softmax(logits.astype(np.float64))[tier1]

    del model, logits
    torch.cuda.empty_cache()

    # Compute risk-coverage curves
    model_probs = {
        "Popularity": pop_probs_t1,
        "Logistic": log_probs_t1,
        "Single Transformer": single_probs_t1,
        "Ensemble (5-member)": ens_probs_t1,
    }

    for k_val, risk_label_str, suffix in [
        (1, "Risk (1 - Top-1 Accuracy)", "top1"),
        (3, "Risk (P(expert not in top-3))", "top3"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for i, (name, probs) in enumerate(model_probs.items()):
            rc = risk_coverage_curve(probs, labels_t1, k=k_val, n_thresholds=300)
            cov = rc["coverage"]
            risk = rc["risk"]
            aurc = rc["aurc"]
            color = _COLORS[i % len(_COLORS)]

            valid = ~np.isnan(risk)
            if aurc == 0.0 or np.isnan(aurc):
                label = name  # degenerate curve (e.g. constant confidence)
            else:
                label = f"{name} (AURC={aurc:.3f})"
            ax.plot(cov[valid], risk[valid], color=color, lw=1.8, label=label)

        ax.set_xlabel("Coverage", fontsize=11)
        ax.set_ylabel(risk_label_str, fontsize=11)
        ax.set_xlim(1.0, 0.0)  # reversed
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.set_title(f"Risk-Coverage: Action-90 Top-{k_val} (Tier 1)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        _save_fig(fig, PAPER_DIR / f"risk_coverage_{suffix}")


# ===================================================================
# Figure 4: Stress Test Degradation Plot
# ===================================================================

def fig4_stress_test() -> None:
    """Clean version of the stress test degradation plot."""
    print("\n[4/6] Stress Test Degradation Plot")

    stress = _load_json(ROOT / "outputs" / "eval" / "stress_test.json")

    # Ordering from least to most severe masking
    order = [
        ("baseline", "Baseline"),
        ("tera", "Tera"),
        ("items", "Items"),
        ("moves_2", "Moves (2/4)"),
        ("moves_4", "Moves (4/4)"),
        ("moves_4+items", "Moves+Items"),
        ("all_but_species", "Species Only"),
    ]

    # Extract metrics
    x_labels = []
    top1_vals = []
    top3_vals = []
    top5_vals = []
    nll_vals = []
    conf_vals = []

    for key, label in order:
        if key not in stress:
            continue
        m = stress[key]
        x_labels.append(label)
        top1_vals.append(m.get("overall/top1_action90", 0) * 100)
        top3_vals.append(m.get("overall/top3_action90", 0) * 100)
        top5_vals.append(m.get("overall/top5_action90", 0) * 100)
        nll_vals.append(m.get("overall/nll_action90", 0))
        conf_vals.append(m.get("mean_confidence", 0) * 100)

    x = np.arange(len(x_labels))

    # Two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Top-k accuracy
    ax1.plot(x, top1_vals, "o-", color=_COLORS[0], lw=1.8, markersize=5, label="Top-1")
    ax1.plot(x, top3_vals, "s-", color=_COLORS[1], lw=1.8, markersize=5, label="Top-3")
    ax1.plot(x, top5_vals, "^-", color=_COLORS[2], lw=1.8, markersize=5, label="Top-5")
    # Random baseline (1/90, 3/90, 5/90)
    ax1.axhline(100 / 90, ls=":", color="gray", alpha=0.5, lw=1)
    ax1.text(len(x) - 0.5, 100 / 90 + 0.2, "Random (1/90)", fontsize=7,
             color="gray", ha="right")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Top-k Accuracy vs Masking Level")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: NLL and confidence
    color_nll = _COLORS[3]
    color_conf = _COLORS[0]

    ax2.plot(x, nll_vals, "o-", color=color_nll, lw=1.8, markersize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("NLL (Action-90)", color=color_nll)
    ax2.tick_params(axis="y", labelcolor=color_nll)
    ax2.set_title("NLL and Confidence vs Masking Level")
    ax2.spines["top"].set_visible(False)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, conf_vals, "s--", color=color_conf, lw=1.5, markersize=4)
    ax2_twin.set_ylabel("Mean Confidence (%)", color=color_conf)
    ax2_twin.tick_params(axis="y", labelcolor=color_conf)
    ax2_twin.spines["top"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, PAPER_DIR / "stress_test")


# ===================================================================
# Figure 5: OOD Comparison (Regime A vs B)
# ===================================================================

def fig5_ood_comparison() -> None:
    """OOD comparison: Regime A vs Regime B uncertainty shifts."""
    print("\n[5/6] OOD Comparison")

    ood = _load_json(ROOT / "outputs" / "eval" / "ood_comparison.json")
    ra = ood["regime_a"]
    rb = ood["regime_b"]

    # 2-panel figure: uncertainty metrics and performance metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Uncertainty metrics
    labels_unc = ["Entropy", "Mutual\nInformation", "Confidence"]
    ra_vals = [ra["mean_entropy"], ra["mean_mi"], ra["mean_confidence"]]
    rb_vals = [rb["mean_entropy"], rb["mean_mi"], rb["mean_confidence"]]

    x_unc = np.arange(len(labels_unc))
    w = 0.3
    bars_a = ax1.bar(x_unc - w / 2, ra_vals, w, label="Regime A (in-dist)",
                     color=_COLORS[0], edgecolor="white", linewidth=0.5)
    bars_b = ax1.bar(x_unc + w / 2, rb_vals, w, label="Regime B (OOD)",
                     color=_COLORS[3], edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars_a, ra_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_b, rb_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x_unc)
    ax1.set_xticklabels(labels_unc, fontsize=10)
    ax1.set_title("Uncertainty Metrics")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_ylim(0, max(max(ra_vals), max(rb_vals)) * 1.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2: Abstention rate + accuracy comparison
    labels_perf = ["Top-1\nAccuracy", "Top-3\nAccuracy", "Abstention\nRate"]
    ra_perf = [
        ra.get("top1_action90", 0) * 100,
        ra.get("top3_action90", 0) * 100,
        ra["abstain_rate_at_tau80"] * 100,
    ]
    rb_perf = [
        rb.get("top1_action90", 0) * 100,
        rb.get("top3_action90", 0) * 100,
        rb["abstain_rate_at_tau80"] * 100,
    ]

    x_perf = np.arange(len(labels_perf))
    bars_a2 = ax2.bar(x_perf - w / 2, ra_perf, w, label="Regime A (in-dist)",
                      color=_COLORS[0], edgecolor="white", linewidth=0.5)
    bars_b2 = ax2.bar(x_perf + w / 2, rb_perf, w, label="Regime B (OOD)",
                      color=_COLORS[3], edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars_a2, ra_perf):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_b2, rb_perf):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x_perf)
    ax2.set_xticklabels(labels_perf, fontsize=10)
    ax2.set_title("Performance and Abstention")
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_ylim(0, max(max(ra_perf), max(rb_perf)) * 1.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Within-Core (Regime A) vs OOD (Regime B)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, PAPER_DIR / "ood_comparison")


# ===================================================================
# Figure 6: Uncertainty Decomposition (Entropy vs MI scatter)
# ===================================================================

def fig6_uncertainty_decomposition() -> None:
    """Entropy vs MI scatter, colored by correctness."""
    print("\n[6/6] Uncertainty Decomposition")

    ens = np.load(ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz")
    probs = ens["probs"]
    action90_true = ens["action90_true"]
    bring4_observed = ens["bring4_observed"]
    entropy = ens["entropy"]
    mi = ens["mi"]

    # Tier 1 only
    tier1 = bring4_observed.astype(bool)
    probs_t1 = probs[tier1]
    labels_t1 = action90_true[tier1]
    entropy_t1 = entropy[tier1]
    mi_t1 = mi[tier1]

    # Top-1 correctness
    predicted = probs_t1.argmax(axis=1)
    correct_t1 = (predicted == labels_t1)

    # Top-3 correctness
    top3 = np.argsort(probs_t1, axis=1)[:, -3:]
    correct_top3 = np.any(top3 == labels_t1[:, None], axis=1)

    # Subsample for readable scatter (plot up to 5000 points)
    rng = np.random.default_rng(42)
    n = len(entropy_t1)
    if n > 5000:
        idx = rng.choice(n, 5000, replace=False)
    else:
        idx = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: Entropy vs MI, colored by top-1 correctness
    ax1.scatter(
        entropy_t1[idx][~correct_t1[idx]], mi_t1[idx][~correct_t1[idx]],
        s=4, alpha=0.3, color=_COLORS[3], label="Incorrect", rasterized=True,
    )
    ax1.scatter(
        entropy_t1[idx][correct_t1[idx]], mi_t1[idx][correct_t1[idx]],
        s=4, alpha=0.3, color=_COLORS[2], label="Correct (Top-1)", rasterized=True,
    )
    ax1.set_xlabel("Predictive Entropy H(p)")
    ax1.set_ylabel("Mutual Information (MI)")
    ax1.set_title("Entropy vs MI (Top-1 Correctness)")
    ax1.legend(frameon=False, fontsize=9, markerscale=3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2: Entropy vs MI, colored by top-3 correctness
    ax2.scatter(
        entropy_t1[idx][~correct_top3[idx]], mi_t1[idx][~correct_top3[idx]],
        s=4, alpha=0.3, color=_COLORS[3], label="Not in Top-3", rasterized=True,
    )
    ax2.scatter(
        entropy_t1[idx][correct_top3[idx]], mi_t1[idx][correct_top3[idx]],
        s=4, alpha=0.3, color=_COLORS[2], label="In Top-3", rasterized=True,
    )
    ax2.set_xlabel("Predictive Entropy H(p)")
    ax2.set_ylabel("Mutual Information (MI)")
    ax2.set_title("Entropy vs MI (Top-3 Correctness)")
    ax2.legend(frameon=False, fontsize=9, markerscale=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Uncertainty Decomposition: Total (Entropy) vs Epistemic (MI)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, PAPER_DIR / "uncertainty_decomposition")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("TurnZero -- Final Paper Figure Generation")
    print("=" * 70)

    t0 = time.time()

    fig1_model_comparison()
    fig2_reliability_diagram()
    fig3_risk_coverage()
    fig4_stress_test()
    fig5_ood_comparison()
    fig6_uncertainty_decomposition()

    dt = time.time() - t0

    print("\n" + "=" * 70)
    print(f"All figures saved to: {PAPER_DIR}")
    print(f"Total time: {dt:.1f}s ({dt / 60:.1f} min)")
    print("=" * 70)

    # Verify all outputs exist
    expected = [
        "model_comparison",
        "reliability_diagram",
        "risk_coverage_top1",
        "risk_coverage_top3",
        "stress_test",
        "ood_comparison",
        "uncertainty_decomposition",
    ]

    print("\nOutput verification:")
    all_ok = True
    for name in expected:
        png = PAPER_DIR / f"{name}.png"
        pdf = PAPER_DIR / f"{name}.pdf"
        png_ok = png.exists()
        pdf_ok = pdf.exists()
        status = "OK" if (png_ok and pdf_ok) else "MISSING"
        if not (png_ok and pdf_ok):
            all_ok = False
        print(f"  [{status}] {name}.png  ({png_ok})  {name}.pdf  ({pdf_ok})")

    if all_ok:
        print("\nAll figures generated successfully.")
    else:
        print("\nWARNING: Some figures are missing!")
        sys.exit(1)
