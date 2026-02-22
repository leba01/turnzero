"""Post-hoc temperature scaling (Guo et al., 2017).

Fits a single scalar T > 0 on validation logits to minimize NLL.
At inference: calibrated_probs = softmax(logits / T)

Reference: docs/PROJECT_BIBLE.md Section 4.2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from turnzero.eval.metrics import _ece


# ---------------------------------------------------------------------------
# Softmax + NLL helpers (pure numpy, operates on pre-collected logits)
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis=-1."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _mean_nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    """Mean NLL at temperature T: mean(-log softmax(logits/T)[y_true])."""
    probs = _softmax(logits / T)
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p_true, _EPS, None))))


# ---------------------------------------------------------------------------
# TemperatureScaler
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Post-hoc temperature scaling (Guo et al., 2017).

    Fits a single scalar T > 0 on validation logits to minimize NLL.
    At inference: calibrated_probs = softmax(logits / T)

    Note: Temperature scaling is not applied in the final pipeline
    (T~1.0 on the ensemble confirms it is already well-calibrated).
    This class is retained for documentation and experimentation.
    """

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> dict:
        """Fit T on validation set using scipy.optimize.minimize_scalar (Brent).

        Args:
            logits: (N, 90) raw logits from model.
            labels: (N,) ground truth action90 ids.

        Returns:
            dict with T, val_nll_before, val_nll_after, val_ece_before,
            val_ece_after.
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        # --- Metrics before calibration (T=1.0) ---
        nll_before = _mean_nll(logits, labels, T=1.0)
        probs_before = _softmax(logits)
        conf_before = probs_before.max(axis=1)
        correct_before = probs_before.argmax(axis=1) == labels
        ece_before = _ece(conf_before, correct_before)

        # --- Optimize T via bounded scalar minimization ---
        result = minimize_scalar(
            lambda T: _mean_nll(logits, labels, T),
            bounds=(0.01, 50.0),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 200},
        )
        self.T = float(result.x)

        # --- Metrics after calibration ---
        nll_after = _mean_nll(logits, labels, self.T)
        probs_after = _softmax(logits / self.T)
        conf_after = probs_after.max(axis=1)
        correct_after = probs_after.argmax(axis=1) == labels
        ece_after = _ece(conf_after, correct_after)

        return {
            "T": self.T,
            "val_nll_before": nll_before,
            "val_nll_after": nll_after,
            "val_ece_before": ece_before,
            "val_ece_after": ece_after,
        }

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling: softmax(logits / T).

        Args:
            logits: (N, 90) raw logits.

        Returns:
            (N, 90) calibrated probabilities.
        """
        return _softmax(np.asarray(logits, dtype=np.float64) / self.T)

    def save(self, path: str | Path) -> None:
        """Save temperature.json artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"T": self.T}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        """Load from temperature.json."""
        with open(path) as f:
            data = json.load(f)
        scaler = cls()
        scaler.T = float(data["T"])
        return scaler


# ---------------------------------------------------------------------------
# Logit collection helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run inference and return raw logits (N, 90) and labels dict.

    Like validate() in train.py but returns logits instead of probs.

    Args:
        model: trained OTSTransformer (already on device, eval mode).
        loader: DataLoader yielding batches with team_a, team_b, labels.
        device: torch device.

    Returns:
        logits: (N, 90) float32 numpy array of raw logits.
        labels_dict: dict with keys action90_true, lead2_true,
            bring4_observed, is_mirror — each (N,) numpy array.
    """
    model.eval()

    all_logits: list[np.ndarray] = []
    all_action90: list[np.ndarray] = []
    all_lead2: list[np.ndarray] = []
    all_bring4: list[np.ndarray] = []
    all_mirror: list[np.ndarray] = []

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(team_a, team_b)

        # Collect in FP32
        all_logits.append(logits.float().cpu().numpy())
        all_action90.append(batch["action90_label"].numpy())
        all_lead2.append(batch["lead2_label"].numpy())
        all_bring4.append(batch["bring4_observed"].numpy())
        all_mirror.append(batch["is_mirror"].numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_dict = {
        "action90_true": np.concatenate(all_action90, axis=0),
        "lead2_true": np.concatenate(all_lead2, axis=0),
        "bring4_observed": np.concatenate(all_bring4, axis=0),
        "is_mirror": np.concatenate(all_mirror, axis=0),
    }
    return logits_np, labels_dict
