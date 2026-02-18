"""Deep ensemble prediction and uncertainty decomposition.

Loads M independently-trained checkpoints, runs inference, averages
softmax probabilities, and decomposes uncertainty into aleatoric (mean
member entropy) and epistemic (mutual information) components.

Reference:
  Lakshminarayanan et al., 2017. "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles." NeurIPS.
  docs/PROJECT_BIBLE.md Section 4.1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from turnzero.models.transformer import ModelConfig, OTSTransformer

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model_from_ckpt(
    ckpt_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load an OTSTransformer from a checkpoint (same pattern as evaluate_checkpoint)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = OTSTransformer(ckpt["vocab_sizes"], model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy H(p) = -sum(p * log(p + eps)) along last axis."""
    return -np.sum(probs * np.log(probs + _EPS), axis=-1)


@torch.no_grad()
def _collect_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run inference with one model, return calibrated probs and labels.

    Returns
    -------
    probs : (N, 90) float64 array
    labels_dict : dict with action90_true, lead2_true, bring4_observed, is_mirror
    """
    all_probs: list[np.ndarray] = []
    all_action90: list[np.ndarray] = []
    all_lead2: list[np.ndarray] = []
    all_bring4: list[np.ndarray] = []
    all_mirror: list[np.ndarray] = []

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(team_a, team_b)

        # Apply temperature and softmax in FP32
        scaled = logits.float() / temperature
        probs = torch.softmax(scaled, dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_action90.append(batch["action90_label"].numpy())
        all_lead2.append(batch["lead2_label"].numpy())
        all_bring4.append(batch["bring4_observed"].numpy())
        all_mirror.append(batch["is_mirror"].numpy())

    probs_np = np.concatenate(all_probs, axis=0).astype(np.float64)
    labels_dict = {
        "action90_true": np.concatenate(all_action90, axis=0),
        "lead2_true": np.concatenate(all_lead2, axis=0),
        "bring4_observed": np.concatenate(all_bring4, axis=0),
        "is_mirror": np.concatenate(all_mirror, axis=0),
    }
    return probs_np, labels_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensemble_predict(
    ckpt_paths: list[str | Path],
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Load M checkpoints, run inference, return averaged predictions.

    For each checkpoint:
    - Load model from ckpt
    - Run forward pass on all batches
    - Collect softmax(logits / T)

    Then:
    - Average probs across M members → p_bar
    - Compute uncertainty decomposition

    Models are loaded and discarded one at a time to avoid OOM.

    Parameters
    ----------
    ckpt_paths : list of paths
        Paths to best.pt checkpoints for each ensemble member.
    loader : DataLoader
        Test/val DataLoader (shuffle=False, drop_last=False).
    device : torch.device
        GPU or CPU device.
    temperature : float
        Temperature for softmax calibration (default 1.0).

    Returns
    -------
    dict with keys:
        "probs": (N, 90) — averaged calibrated probs p_bar
        "member_probs": (M, N, 90) — per-member probs
        "entropy": (N,) — H(p_bar), predictive entropy
        "member_entropy": (N,) — mean H(p_m), aleatoric proxy
        "mi": (N,) — H(p_bar) - mean H(p_m), epistemic (mutual info)
        "confidence": (N,) — max p_bar per example
        "action90_true": (N,) int
        "lead2_true": (N,) int
        "bring4_observed": (N,) bool
        "is_mirror": (N,) bool
    """
    M = len(ckpt_paths)
    member_probs_list: list[np.ndarray] = []
    labels_dict: dict[str, np.ndarray] | None = None

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"Ensemble member {i + 1}/{M}: {Path(ckpt_path).parent.name}")

        model = _load_model_from_ckpt(ckpt_path, device)
        probs_m, ld = _collect_probs(model, loader, device, temperature)
        member_probs_list.append(probs_m)

        # Labels are identical across members; keep from first
        if labels_dict is None:
            labels_dict = ld

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    assert labels_dict is not None

    # (M, N, 90)
    member_probs = np.stack(member_probs_list, axis=0)

    # Averaged probs: p_bar = (1/M) * sum_m p_m
    p_bar = member_probs.mean(axis=0)  # (N, 90)

    # Predictive entropy: H(p_bar)
    entropy = _entropy(p_bar)  # (N,)

    # Mean member entropy: (1/M) * sum_m H(p_m) — aleatoric proxy
    member_entropies = np.array([_entropy(member_probs[m]) for m in range(M)])
    member_entropy = member_entropies.mean(axis=0)  # (N,)

    # Mutual information: H(p_bar) - mean H(p_m) — epistemic proxy
    mi = entropy - member_entropy  # (N,)

    # Confidence: max p_bar per example
    confidence = p_bar.max(axis=1)  # (N,)

    print(f"\nEnsemble summary (M={M}, N={p_bar.shape[0]}):")
    print(f"  Mean predictive entropy: {entropy.mean():.4f}")
    print(f"  Mean aleatoric (member H): {member_entropy.mean():.4f}")
    print(f"  Mean epistemic (MI):       {mi.mean():.4f}")
    print(f"  Mean confidence:           {confidence.mean():.4f}")

    return {
        "probs": p_bar,
        "member_probs": member_probs,
        "entropy": entropy,
        "member_entropy": member_entropy,
        "mi": mi,
        "confidence": confidence,
        "action90_true": labels_dict["action90_true"],
        "lead2_true": labels_dict["lead2_true"],
        "bring4_observed": labels_dict["bring4_observed"],
        "is_mirror": labels_dict["is_mirror"],
    }


def save_ensemble_predictions(preds: dict, out_path: str | Path) -> None:
    """Save ensemble predictions to .npz for later analysis."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **preds)
    print(f"Ensemble predictions saved to {out_path}")


def load_ensemble_predictions(path: str | Path) -> dict[str, np.ndarray]:
    """Load ensemble predictions from .npz."""
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}
