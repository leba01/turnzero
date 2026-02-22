#!/usr/bin/env python3
"""Wave 3B: Comprehensive Week 3 plots + tables.

Generates all paper-ready figures from Wave 3A artifacts:
1. Reliability diagrams (single model vs ensemble, overlay)
2. Updated comparison table with bootstrap CIs (JSON + LaTeX)
3. Uncertainty histograms (confidence, entropy, MI; mirror vs non-mirror)
4. Within-core vs OOD comparison bar chart + table

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/run_wave3b.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

OUT_PLOTS = ROOT / "outputs" / "plots" / "week3"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=_DPI)
    fig.savefig(out.with_suffix(".pdf"), dpi=_DPI)
    plt.close(fig)
    print(f"  Saved: {out.with_suffix('.png')}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Reliability diagrams — single model vs ensemble (overlay)
# ═══════════════════════════════════════════════════════════════════════════


def _reliability_data(probs, labels, n_bins=15):
    """Compute reliability diagram data."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
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

    return {
        "bin_centers": bin_centers,
        "acc_per_bin": acc_per_bin,
        "count_per_bin": count_per_bin,
        "confidence": confidence,
    }


def plot_reliability_comparison(models_data, out_path, title="", n_bins=15):
    """Overlay reliability diagram comparing multiple models.

    models_data: dict of {name: (probs, labels)}
    """
    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(5.5, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Diagonal
    ax_rel.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Perfect")

    bin_width = 1.0 / n_bins
    n_models = len(models_data)
    bar_width = bin_width * 0.85 / n_models

    for i, (name, (probs, labels)) in enumerate(models_data.items()):
        rd = _reliability_data(probs, labels, n_bins)
        color = _COLORS[i % len(_COLORS)]
        valid = ~np.isnan(rd["acc_per_bin"])

        offset = (i - n_models / 2 + 0.5) * bar_width
        ax_rel.bar(
            rd["bin_centers"][valid] + offset,
            rd["acc_per_bin"][valid],
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=name,
            alpha=0.8,
        )

        # Histogram
        ax_hist.bar(
            rd["bin_centers"] + offset,
            rd["count_per_bin"],
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.6,
        )

    ax_rel.set_ylabel("Accuracy")
    ax_rel.set_ylim(0, 1)
    ax_rel.legend(loc="upper left", frameon=False, fontsize=9)
    if title:
        ax_rel.set_title(title)

    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(0, 1)

    fig.tight_layout()
    _save_fig(fig, out_path)


def step1_reliability_diagrams():
    print("\n" + "=" * 70)
    print("STEP 1: Reliability diagrams")
    print("=" * 70)

    from turnzero.uq.ensemble import load_ensemble_predictions

    # Load ensemble predictions
    ens = load_ensemble_predictions(ROOT / "outputs" / "ensemble" / "ensemble_predictions.npz")
    tier1 = ens["bring4_observed"].astype(bool)
    ens_probs_t1 = ens["probs"][tier1]
    action90_true_t1 = ens["action90_true"][tier1]

    # Load single model predictions — recompute from checkpoint
    import torch
    from turnzero.data.dataset import VGCDataset, Vocab
    from turnzero.models.transformer import OTSTransformer
    from turnzero.uq.temperature import collect_logits

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_SINGLE = ROOT / "outputs" / "runs" / "run_001" / "best.pt"
    VOCAB_SINGLE = ROOT / "outputs" / "runs" / "run_001" / "vocab.json"
    DATA_A = ROOT / "data" / "assembled" / "regime_a"

    model = OTSTransformer.load_from_checkpoint(CKPT_SINGLE, DEVICE)
    vocab = Vocab.load(VOCAB_SINGLE)

    test_ds = VGCDataset(DATA_A / "test.jsonl", vocab)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    logits, _ = collect_logits(model, test_loader, DEVICE)

    from turnzero.uq.temperature import _softmax
    single_probs_t1 = _softmax(logits)[tier1]

    del model, logits
    torch.cuda.empty_cache()

    # Plot: Single model vs Ensemble
    print("  Plotting: single model vs ensemble reliability...")
    plot_reliability_comparison(
        {
            "Single Transformer": (single_probs_t1, action90_true_t1),
            "Ensemble (5-member)": (ens_probs_t1, action90_true_t1),
        },
        OUT_PLOTS / "reliability_single_vs_ensemble",
        title="Reliability: Single Model vs Ensemble (Action-90, Tier 1)",
    )

    return ens, tier1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Comparison table with bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════════

_TABLE_METRICS = [
    "top1_action90",
    "top3_action90",
    "top5_action90",
    "top1_lead2",
    "top3_lead2",
    "nll_action90",
    "nll_lead2",
    "ece_action90",
    "ece_lead2",
]

_METRIC_FMT = {
    "top1_action90": (".1%", True),   # (format, is_percentage)
    "top3_action90": (".1%", True),
    "top5_action90": (".1%", True),
    "top1_lead2": (".1%", True),
    "top3_lead2": (".1%", True),
    "nll_action90": (".3f", False),
    "nll_lead2": (".3f", False),
    "ece_action90": (".4f", False),
    "ece_lead2": (".4f", False),
}

_METRIC_DISPLAY = {
    "top1_action90": "Action90 Top-1",
    "top3_action90": "Action90 Top-3",
    "top5_action90": "Action90 Top-5",
    "top1_lead2": "Lead-2 Top-1",
    "top3_lead2": "Lead-2 Top-3",
    "nll_action90": "NLL (Action90)",
    "nll_lead2": "NLL (Lead-2)",
    "ece_action90": "ECE (Action90)",
    "ece_lead2": "ECE (Lead-2)",
}


def step2_comparison_table():
    print("\n" + "=" * 70)
    print("STEP 2: Comparison table with bootstrap CIs")
    print("=" * 70)

    # Load all model metrics
    with open(ROOT / "outputs" / "baselines" / "metrics_popularity.json") as f:
        pop = json.load(f)
    with open(ROOT / "outputs" / "baselines" / "metrics_logistic.json") as f:
        log = json.load(f)
    with open(ROOT / "outputs" / "eval" / "run_001" / "test_metrics.json") as f:
        single = json.load(f)
    with open(ROOT / "outputs" / "ensemble" / "test_metrics.json") as f:
        ensemble = json.load(f)
    with open(ROOT / "outputs" / "eval" / "bootstrap_cis.json") as f:
        cis = json.load(f)

    models = {
        "Popularity": pop,
        "Logistic": log,
        "Single Transformer": single,
        "Ensemble (5-member)": ensemble,
    }

    # Build table with CIs for ensemble
    table_data = {}
    for stratum in ["overall", "mirror", "non_mirror"]:
        table_data[stratum] = {}
        for m in _TABLE_METRICS:
            key = f"{stratum}/{m}"
            row = {}
            for model_name, metrics in models.items():
                v = metrics.get(key)
                if v is None:
                    row[model_name] = None
                    continue

                fmt_str, is_pct = _METRIC_FMT[m]
                if model_name == "Ensemble (5-member)" and key in cis:
                    ci = cis[key]
                    if is_pct:
                        row[model_name] = f"{v:{fmt_str}} [{ci['lo']:{fmt_str}}, {ci['hi']:{fmt_str}}]"
                    else:
                        row[model_name] = f"{v:{fmt_str}} [{ci['lo']:{fmt_str}}, {ci['hi']:{fmt_str}}]"
                else:
                    row[model_name] = f"{v:{fmt_str}}"
            table_data[stratum][m] = row

    # Save JSON table
    json_path = OUT_PLOTS / "comparison_table_with_cis.json"
    with open(json_path, "w") as f:
        json.dump(table_data, f, indent=2)
    print(f"  Saved: {json_path}")

    # Write LaTeX table (overall stratum only for paper)
    model_names = list(models.keys())
    lines = []
    lines.append(r"\begin{tabular}{l" + "r" * len(model_names) + "}")
    lines.append(r"\toprule")
    header = "Metric & " + " & ".join(model_names) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for m in _TABLE_METRICS:
        display = _METRIC_DISPLAY.get(m, m)
        cells = []
        for model_name in model_names:
            v = table_data["overall"][m].get(model_name)
            cells.append(v if v else "--")
        row = f"{display} & " + " & ".join(cells) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    latex_path = OUT_PLOTS / "comparison_table_with_cis.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {latex_path}")

    # Print table to stdout
    print(f"\n  {'Metric':<20}", end="")
    for name in model_names:
        print(f"  {name:>28}", end="")
    print()
    print(f"  {'─' * (20 + 30 * len(model_names))}")
    for m in _TABLE_METRICS:
        display = _METRIC_DISPLAY.get(m, m)
        print(f"  {display:<20}", end="")
        for name in model_names:
            v = table_data["overall"][m].get(name, "--")
            print(f"  {v:>28}", end="")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Uncertainty histograms
# ═══════════════════════════════════════════════════════════════════════════


def step3_uncertainty_histograms(ens, tier1):
    print("\n" + "=" * 70)
    print("STEP 3: Uncertainty histograms")
    print("=" * 70)

    confidence = ens["confidence"]
    entropy = ens["entropy"]
    mi = ens["mi"]
    is_mirror = ens["is_mirror"].astype(bool)

    # 3a: Three-panel uncertainty histogram (confidence, entropy, MI)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Confidence
    ax = axes[0]
    ax.hist(confidence, bins=50, color=_COLORS[0], edgecolor="white",
            linewidth=0.5, alpha=0.8, density=True)
    ax.set_xlabel("Confidence (max p)")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution")
    ax.axvline(np.median(confidence), color="red", ls="--", lw=1,
               label=f"Median={np.median(confidence):.3f}")
    ax.legend(frameon=False, fontsize=8)

    # Entropy
    ax = axes[1]
    ax.hist(entropy, bins=50, color=_COLORS[1], edgecolor="white",
            linewidth=0.5, alpha=0.8, density=True)
    ax.set_xlabel("Predictive Entropy H(p)")
    ax.set_title("Entropy Distribution")
    ax.axvline(np.median(entropy), color="red", ls="--", lw=1,
               label=f"Median={np.median(entropy):.3f}")
    ax.legend(frameon=False, fontsize=8)

    # Mutual Information
    ax = axes[2]
    ax.hist(mi, bins=50, color=_COLORS[2], edgecolor="white",
            linewidth=0.5, alpha=0.8, density=True)
    ax.set_xlabel("Mutual Information (MI)")
    ax.set_title("Epistemic Uncertainty")
    ax.axvline(np.median(mi), color="red", ls="--", lw=1,
               label=f"Median={np.median(mi):.4f}")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Ensemble Uncertainty Decomposition (Regime A Test)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "uncertainty_histograms")

    # 3b: Mirror vs non-mirror overlay
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, data, xlabel, title in [
        (axes[0], confidence, "Confidence", "Confidence: Mirror vs Non-Mirror"),
        (axes[1], entropy, "Entropy", "Entropy: Mirror vs Non-Mirror"),
        (axes[2], mi, "Mutual Information", "MI: Mirror vs Non-Mirror"),
    ]:
        bins = np.linspace(data.min(), data.max(), 50)
        ax.hist(data[is_mirror], bins=bins, color=_COLORS[0], edgecolor="white",
                linewidth=0.5, alpha=0.6, density=True, label="Mirror")
        ax.hist(data[~is_mirror], bins=bins, color=_COLORS[3], edgecolor="white",
                linewidth=0.5, alpha=0.6, density=True, label="Non-Mirror")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "uncertainty_mirror_vs_nonmirror")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Within-core vs OOD comparison
# ═══════════════════════════════════════════════════════════════════════════


def step4_ood_comparison():
    print("\n" + "=" * 70)
    print("STEP 4: Within-core vs OOD comparison")
    print("=" * 70)

    with open(ROOT / "outputs" / "eval" / "ood_comparison.json") as f:
        ood = json.load(f)

    ra = ood["regime_a"]
    rb = ood["regime_b"]

    # Bar chart comparing key metrics
    metrics = [
        ("Mean Entropy", ra["mean_entropy"], rb["mean_entropy"]),
        ("Mean MI", ra["mean_mi"], rb["mean_mi"]),
        ("Mean Confidence", ra["mean_confidence"], rb["mean_confidence"]),
        ("Abstain Rate\n(@80% cov)", ra["abstain_rate_at_tau80"], rb["abstain_rate_at_tau80"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, (label, va, vb) in zip(axes, metrics):
        bars = ax.bar(
            ["Within-Core\n(Regime A)", "OOD\n(Regime B)"],
            [va, vb],
            color=[_COLORS[0], _COLORS[3]],
            edgecolor="white",
            linewidth=0.5,
            width=0.5,
        )
        ax.set_title(label, fontsize=10)
        # Add value labels
        for bar, val in zip(bars, [va, vb]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}" if val < 1 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylim(0, max(va, vb) * 1.2)

    fig.suptitle("Within-Core vs OOD: Ensemble Uncertainty Shifts", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "ood_comparison_bars")

    # Also save a comparison table as JSON
    comparison_table = {
        "metrics": {
            "mean_entropy": {"regime_a": ra["mean_entropy"], "regime_b": rb["mean_entropy"],
                             "delta": rb["mean_entropy"] - ra["mean_entropy"]},
            "mean_mi": {"regime_a": ra["mean_mi"], "regime_b": rb["mean_mi"],
                        "delta": rb["mean_mi"] - ra["mean_mi"]},
            "mean_confidence": {"regime_a": ra["mean_confidence"], "regime_b": rb["mean_confidence"],
                                "delta": rb["mean_confidence"] - ra["mean_confidence"]},
            "abstain_rate_tau80": {"regime_a": ra["abstain_rate_at_tau80"], "regime_b": rb["abstain_rate_at_tau80"],
                                   "delta": rb["abstain_rate_at_tau80"] - ra["abstain_rate_at_tau80"]},
            "top1_action90": {"regime_a": ra.get("top1_action90"), "regime_b": rb.get("top1_action90")},
            "top3_action90": {"regime_a": ra.get("top3_action90"), "regime_b": rb.get("top3_action90")},
        },
    }

    table_path = OUT_PLOTS / "ood_comparison_table.json"
    with open(table_path, "w") as f:
        json.dump(comparison_table, f, indent=2)
    print(f"  Saved: {table_path}")

    # Print
    print(f"\n  {'Metric':<25} {'Regime A':>12} {'Regime B':>12} {'Delta':>10}")
    print(f"  {'─' * 59}")
    for label, va, vb in metrics:
        label_clean = label.replace("\n", " ")
        print(f"  {label_clean:<25} {va:>12.4f} {vb:>12.4f} {vb - va:>+10.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Updated top-k comparison with ensemble
# ═══════════════════════════════════════════════════════════════════════════


def step5_topk_bar():
    print("\n" + "=" * 70)
    print("STEP 5: Updated top-k comparison bar chart")
    print("=" * 70)

    with open(ROOT / "outputs" / "baselines" / "metrics_popularity.json") as f:
        pop = json.load(f)
    with open(ROOT / "outputs" / "baselines" / "metrics_logistic.json") as f:
        log = json.load(f)
    with open(ROOT / "outputs" / "eval" / "run_001" / "test_metrics.json") as f:
        single = json.load(f)
    with open(ROOT / "outputs" / "ensemble" / "test_metrics.json") as f:
        ensemble = json.load(f)

    models = {
        "Popularity": pop,
        "Logistic": log,
        "Single\nTransformer": single,
        "Ensemble\n(5-member)": ensemble,
    }

    from turnzero.eval.plots import topk_comparison_bar

    # Need to adapt keys for the bar chart function
    models_clean = {}
    for name, m in models.items():
        models_clean[name] = m

    topk_comparison_bar(models_clean, OUT_PLOTS / "topk_comparison_all_models")
    print(f"  Saved: {OUT_PLOTS / 'topk_comparison_all_models.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Confidence vs accuracy scatter (binned)
# ═══════════════════════════════════════════════════════════════════════════


def step6_confidence_accuracy(ens, tier1):
    print("\n" + "=" * 70)
    print("STEP 6: Confidence vs accuracy (binned scatter)")
    print("=" * 70)

    confidence_t1 = ens["confidence"][tier1]
    probs_t1 = ens["probs"][tier1]
    labels_t1 = ens["action90_true"][tier1]
    mi_t1 = ens["mi"][tier1]

    # Top-1 correct
    correct_top1 = (probs_t1.argmax(axis=1) == labels_t1)
    # Top-3 correct
    top3 = np.argsort(probs_t1, axis=1)[:, -3:]
    correct_top3 = np.any(top3 == labels_t1[:, None], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Confidence vs top-1 accuracy (binned)
    ax = axes[0]
    n_bins = 20
    bin_edges = np.linspace(confidence_t1.min(), confidence_t1.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    acc1_bins = []
    for i in range(n_bins):
        mask = (confidence_t1 >= bin_edges[i]) & (confidence_t1 < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidence_t1 >= bin_edges[i]) & (confidence_t1 <= bin_edges[i + 1])
        if mask.sum() > 10:
            acc1_bins.append(correct_top1[mask].mean())
        else:
            acc1_bins.append(np.nan)
    acc1_bins = np.array(acc1_bins)
    valid = ~np.isnan(acc1_bins)
    ax.plot(bin_centers[valid], acc1_bins[valid], "o-", color=_COLORS[0],
            markersize=4, lw=1.5, label="Top-1 Acc")
    ax.set_xlabel("Ensemble Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Confidence vs Accuracy (Binned)")
    ax.legend(frameon=False, fontsize=9)

    # Right: MI vs top-3 accuracy
    ax = axes[1]
    bin_edges_mi = np.linspace(mi_t1.min(), mi_t1.max(), n_bins + 1)
    bin_centers_mi = 0.5 * (bin_edges_mi[:-1] + bin_edges_mi[1:])
    acc3_bins = []
    for i in range(n_bins):
        mask = (mi_t1 >= bin_edges_mi[i]) & (mi_t1 < bin_edges_mi[i + 1])
        if i == n_bins - 1:
            mask = (mi_t1 >= bin_edges_mi[i]) & (mi_t1 <= bin_edges_mi[i + 1])
        if mask.sum() > 10:
            acc3_bins.append(correct_top3[mask].mean())
        else:
            acc3_bins.append(np.nan)
    acc3_bins = np.array(acc3_bins)
    valid = ~np.isnan(acc3_bins)
    ax.plot(bin_centers_mi[valid], acc3_bins[valid], "s-", color=_COLORS[3],
            markersize=4, lw=1.5, label="Top-3 Acc")
    ax.set_xlabel("Mutual Information (Epistemic)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Epistemic Uncertainty vs Accuracy (Binned)")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    _save_fig(fig, OUT_PLOTS / "confidence_accuracy_binned")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    t0 = time.time()

    # Step 1: Reliability diagrams (needs GPU for single model inference)
    ens, tier1 = step1_reliability_diagrams()

    # Step 2: Comparison table with CIs
    step2_comparison_table()

    # Step 3: Uncertainty histograms
    step3_uncertainty_histograms(ens, tier1)

    # Step 4: OOD comparison
    step4_ood_comparison()

    # Step 5: Updated top-k bar chart
    step5_topk_bar()

    # Step 6: Confidence-accuracy scatter
    step6_confidence_accuracy(ens, tier1)

    dt = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Wave 3B complete. Total time: {dt:.1f}s ({dt/60:.1f} min)")
    print(f"{'=' * 70}")

    # List all generated plots
    print("\nGenerated plots:")
    for p in sorted(OUT_PLOTS.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}")
