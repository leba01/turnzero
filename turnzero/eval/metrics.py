"""Metric computation for the TurnZero evaluation harness.

Computes proper scoring rules (NLL, Brier), classification accuracy (top-k),
and calibration (ECE) across overall / mirror / non-mirror strata.

Reference: docs/PROJECT_BIBLE.md Sections 4.3-4.4
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from turnzero.action_space import ACTION_TABLE

# ---------------------------------------------------------------------------
# Lead-pair marginalization: 90-way action probs → 15-way lead-2 probs
# ---------------------------------------------------------------------------

LEAD_PAIRS: list[tuple[int, int]] = list(combinations(range(6), 2))
LEAD_PAIR_TO_IDX: dict[tuple[int, int], int] = {
    pair: idx for idx, pair in enumerate(LEAD_PAIRS)
}

# ACTION90_TO_LEAD2[i] = lead-pair index for action i
ACTION90_TO_LEAD2: np.ndarray = np.array(
    [LEAD_PAIR_TO_IDX[ACTION_TABLE[i][0]] for i in range(90)], dtype=np.int64
)

# (90, 15) matrix: lead2_probs = probs @ _MARGIN_MATRIX
_MARGIN_MATRIX: np.ndarray = np.zeros((90, 15), dtype=np.float64)
for _i in range(90):
    _MARGIN_MATRIX[_i, ACTION90_TO_LEAD2[_i]] = 1.0


def _marginalize_to_lead2(probs: np.ndarray) -> np.ndarray:
    """Marginalize (N, 90) action probs to (N, 15) lead-pair probs."""
    return probs @ _MARGIN_MATRIX


# ---------------------------------------------------------------------------
# Helper metrics
# ---------------------------------------------------------------------------

_EPS = 1e-12  # clamp for log


def _topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of examples where the true label is in the top-k predictions."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]  # (N, k) highest-prob ids
    return float(np.mean(np.any(top_k_preds == labels[:, None], axis=1)))


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood: -log p(y_true)."""
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p_true, _EPS, None))))


def _brier(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean ||p - onehot(y)||^2."""
    onehot = np.zeros((len(labels), n_classes), dtype=np.float64)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _ece(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error with equal-width confidence bins.

    Args:
        confidence: (N,) max predicted probability per example.
        correct: (N,) bool — whether top-1 prediction matches the true label.
        n_bins: number of equal-width bins in [0, 1].

    Returns:
        Weighted average of |accuracy - confidence| across bins.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confidence)
    if n_total == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].astype(np.float64).mean())
        ece += (n_in_bin / n_total) * abs(avg_acc - avg_conf)
    return ece


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    probs: np.ndarray,
    action90_true: np.ndarray,
    lead2_true: np.ndarray,
    bring4_observed: np.ndarray,
    is_mirror: np.ndarray,
) -> dict[str, float | int]:
    """Compute all evaluation metrics with mirror/non-mirror stratification.

    Args:
        probs: (N, 90) predicted 90-way action probabilities.
        action90_true: (N,) ground-truth action90 ids.
        lead2_true: (N,) ground-truth lead-pair index (0..14).
        bring4_observed: (N,) bool — True for Tier 1 (fully observed bring-4).
        is_mirror: (N,) bool — True when core_cluster_a == core_cluster_b.

    Returns:
        Dict keyed like ``"overall/top1_action90"``, ``"mirror/nll_lead2"``, etc.
        Also includes ``"<stratum>/n_examples"`` and ``"<stratum>/n_tier1"`` counts.
    """
    probs = np.asarray(probs, dtype=np.float64)
    action90_true = np.asarray(action90_true, dtype=np.int64)
    lead2_true = np.asarray(lead2_true, dtype=np.int64)
    bring4_observed = np.asarray(bring4_observed, dtype=bool)
    is_mirror = np.asarray(is_mirror, dtype=bool)

    N = len(probs)
    assert probs.shape == (N, 90), f"Expected (N, 90) probs, got {probs.shape}"

    lead2_probs = _marginalize_to_lead2(probs)

    strata: dict[str, np.ndarray] = {
        "overall": np.ones(N, dtype=bool),
        "mirror": is_mirror,
        "non_mirror": ~is_mirror,
    }

    results: dict[str, float | int] = {}

    for name, mask in strata.items():
        n_stratum = int(mask.sum())
        tier1_mask = mask & bring4_observed
        n_tier1 = int(tier1_mask.sum())

        results[f"{name}/n_examples"] = n_stratum
        results[f"{name}/n_tier1"] = n_tier1

        # --- Action-90 metrics (Tier 1 only) ---
        if n_tier1 > 0:
            p_a = probs[tier1_mask]
            y_a = action90_true[tier1_mask]

            results[f"{name}/top1_action90"] = _topk_accuracy(p_a, y_a, 1)
            results[f"{name}/top3_action90"] = _topk_accuracy(p_a, y_a, 3)
            results[f"{name}/top5_action90"] = _topk_accuracy(p_a, y_a, 5)
            results[f"{name}/nll_action90"] = _nll(p_a, y_a)
            results[f"{name}/brier_action90"] = _brier(p_a, y_a, 90)

            conf_a = p_a.max(axis=1)
            correct_a = p_a.argmax(axis=1) == y_a
            results[f"{name}/ece_action90"] = _ece(conf_a, correct_a)

        # --- Lead-2 metrics (all examples in stratum) ---
        if n_stratum > 0:
            p_l = lead2_probs[mask]
            y_l = lead2_true[mask]

            results[f"{name}/top1_lead2"] = _topk_accuracy(p_l, y_l, 1)
            results[f"{name}/top3_lead2"] = _topk_accuracy(p_l, y_l, 3)
            results[f"{name}/nll_lead2"] = _nll(p_l, y_l)
            results[f"{name}/brier_lead2"] = _brier(p_l, y_l, 15)

            conf_l = p_l.max(axis=1)
            correct_l = p_l.argmax(axis=1) == y_l
            results[f"{name}/ece_lead2"] = _ece(conf_l, correct_l)

    return results
