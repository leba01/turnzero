"""Risk-coverage curves for selective prediction.

Sweeps a confidence threshold and reports coverage vs risk,
including AURC and operating-point summaries.

Reference: docs/PROJECT_BIBLE.md Section 4.4
"""

from __future__ import annotations

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


def _save_fig(fig: plt.Figure, out_path: str | Path) -> None:
    """Save figure as both PNG and PDF next to *out_path*."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=_DPI)
    fig.savefig(out.with_suffix(".pdf"), dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Risk-coverage computation
# ---------------------------------------------------------------------------


def risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    k: int = 1,
    n_thresholds: int = 200,
) -> dict:
    """Sweep confidence threshold and compute coverage vs risk.

    Confidence = max_y p(y|x).
    Coverage = fraction where confidence >= threshold.
    Risk = 1 - top_k_accuracy on the non-abstained subset.

    Args:
        probs: (N, C) predicted probabilities.
        labels: (N,) ground-truth class indices.
        k: use top-k accuracy for risk (1 for top-1, 3 for top-3).
        n_thresholds: number of threshold values to sweep.

    Returns:
        Dict with keys:
            coverage: (T,) array of coverage fractions.
            risk: (T,) array of risk values.
            thresholds: (T,) array of confidence thresholds.
            aurc: float — area under risk-coverage curve (trapezoidal).
            operating_points: dict with keys "95", "80", "60" giving
                the threshold, risk, and exact coverage at those levels.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    N = len(probs)
    assert probs.shape[0] == N and probs.ndim == 2

    confidence = probs.max(axis=1)  # (N,)

    # Pre-compute per-example correctness: is true label in top-k?
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]  # (N, k)
    correct = np.any(top_k_preds == labels[:, None], axis=1)  # (N,) bool

    # Sweep thresholds from 0 to max confidence
    thresholds = np.linspace(0.0, float(confidence.max()) + 1e-9, n_thresholds)

    coverage_arr = np.empty(n_thresholds)
    risk_arr = np.empty(n_thresholds)

    for i, tau in enumerate(thresholds):
        mask = confidence >= tau
        n_covered = mask.sum()
        if n_covered == 0:
            coverage_arr[i] = 0.0
            risk_arr[i] = np.nan
        else:
            coverage_arr[i] = n_covered / N
            risk_arr[i] = 1.0 - float(correct[mask].mean())

    # AURC: trapezoidal integration over non-NaN points
    valid = ~np.isnan(risk_arr)
    if valid.sum() >= 2:
        # Sort by coverage descending for proper integration
        cov_valid = coverage_arr[valid]
        risk_valid = risk_arr[valid]
        order = np.argsort(cov_valid)
        aurc = float(np.trapz(risk_valid[order], cov_valid[order]))
    else:
        aurc = float("nan")

    # Operating points at target coverage levels
    operating_points: dict[str, dict[str, float]] = {}
    for target_pct in ("95", "80", "60"):
        target_cov = int(target_pct) / 100.0
        # Find threshold that gives coverage closest to target (from above)
        diffs = coverage_arr - target_cov
        # Among thresholds where coverage >= target, pick the one closest
        above = diffs >= -1e-9
        if above.any():
            # Closest coverage that is >= target
            idx = np.argmin(np.abs(diffs[above]))
            actual_idx = np.where(above)[0][idx]
        else:
            # No threshold achieves this coverage; use lowest threshold
            actual_idx = 0

        op: dict[str, float] = {
            "threshold": float(thresholds[actual_idx]),
            "coverage": float(coverage_arr[actual_idx]),
            "risk": float(risk_arr[actual_idx])
            if not np.isnan(risk_arr[actual_idx])
            else float("nan"),
        }
        operating_points[target_pct] = op

    return {
        "coverage": coverage_arr,
        "risk": risk_arr,
        "thresholds": thresholds,
        "aurc": aurc,
        "operating_points": operating_points,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color cycle for multi-model comparison
_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]


def plot_risk_coverage(
    curves: dict[str, dict],
    out_path: str | Path,
    title: str = "",
    risk_label: str = "Risk (1 - Top-1 Acc)",
) -> None:
    """Multi-model risk-coverage plot.

    One line per model. X-axis = coverage (1.0 to 0.0, reversed).
    Y-axis = risk. Includes AURC in legend labels.

    Args:
        curves: ``{"Model A": rc_dict, "Model B": rc_dict, ...}`` where
            each *rc_dict* is the output of :func:`risk_coverage_curve`.
        out_path: base path for saved figure (PNG + PDF).
        title: optional figure title.
        risk_label: Y-axis label.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, (name, rc) in enumerate(curves.items()):
        cov = rc["coverage"]
        risk = rc["risk"]
        aurc = rc["aurc"]
        color = _COLORS[i % len(_COLORS)]

        # Filter out NaN risk values for plotting
        valid = ~np.isnan(risk)
        label = f"{name} (AURC={aurc:.4f})"
        ax.plot(cov[valid], risk[valid], color=color, lw=1.8, label=label)

    ax.set_xlabel("Coverage")
    ax.set_ylabel(risk_label)
    ax.set_xlim(1.0, 0.0)  # reversed: high coverage on left
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    _save_fig(fig, out_path)
