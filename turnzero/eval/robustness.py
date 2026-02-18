"""Moves-hidden stress test: test-time ablation of OTS fields.

Replaces OTS fields with UNK tokens (index 0) at test time and measures
performance degradation. Verifies graceful degradation when partial OTS
is available (e.g., datasets without |showteam|).

Field layout per mon: [species=0, item=1, ability=2, tera=3,
                        move0=4, move1=5, move2=6, move3=7]

Reference: docs/PROJECT_BIBLE.md Section 6.3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from turnzero.eval.metrics import compute_metrics
from turnzero.models.transformer import ModelConfig, OTSTransformer

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
_EPS = 1e-12

# ---------------------------------------------------------------------------
# Masking configurations
# ---------------------------------------------------------------------------

# Maps config name -> list of column indices to zero out (or special handling)
# Field layout: species=0, item=1, ability=2, tera=3, move0=4..move3=7
MASK_CONFIGS: dict[str, list[int]] = {
    "baseline": [],                         # no masking (control)
    "moves_2": [4, 5, 6, 7],               # special: randomly pick 2 of 4 move cols
    "moves_4": [4, 5, 6, 7],               # hide all 4 moves
    "items": [1],                           # hide items
    "tera": [3],                            # hide tera types
    "moves_4+items": [1, 4, 5, 6, 7],      # hide moves + items
    "all_but_species": [1, 2, 3, 4, 5, 6, 7],  # hide everything except species
}

# Ordered by severity for plotting
MASK_ORDER: list[str] = [
    "baseline",
    "items",
    "tera",
    "moves_2",
    "moves_4",
    "moves_4+items",
    "all_but_species",
]


def mask_batch(
    batch: dict[str, Any],
    mask_config: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Apply test-time masking to a batch. Replaces fields with UNK (0).

    Operates on the (B, 6, 8) team tensors. Clones before modifying.

    Parameters
    ----------
    batch : dict
        Batch with "team_a" and "team_b" keys, each (B, 6, 8) LongTensor.
    mask_config : str
        Name of the masking config (key in MASK_CONFIGS).
    rng : np.random.Generator
        Random number generator (used for "moves_2" random selection).

    Returns
    -------
    dict with the same keys, team tensors cloned and masked.
    """
    if mask_config == "baseline":
        return batch

    cols = MASK_CONFIGS[mask_config]
    masked = dict(batch)  # shallow copy

    for key in ("team_a", "team_b"):
        t = batch[key].clone()  # (B, 6, 8)

        if mask_config == "moves_2":
            # Randomly pick 2 of columns [4,5,6,7] per mon per example
            B, M, _ = t.shape
            move_cols = np.array([4, 5, 6, 7])
            for b in range(B):
                for m in range(M):
                    chosen = rng.choice(move_cols, size=2, replace=False)
                    t[b, m, chosen[0]] = 0
                    t[b, m, chosen[1]] = 0
        else:
            # Zero out all specified columns
            t[:, :, cols] = 0

        masked[key] = t

    return masked


# ---------------------------------------------------------------------------
# Internal helpers (mirrors ensemble.py patterns)
# ---------------------------------------------------------------------------

def _load_model_from_ckpt(
    ckpt_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load an OTSTransformer from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = OTSTransformer(ckpt["vocab_sizes"], model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _collect_probs_masked(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    mask_config: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run inference with masking applied per batch.

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
        # Apply masking before moving to device
        batch = mask_batch(batch, mask_config, rng)

        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(team_a, team_b)

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

def run_stress_test(
    ckpt_paths: list[str | Path],
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    mask_configs: list[str] | None = None,
    seed: int = 42,
) -> dict[str, dict]:
    """Run moves-hidden stress test across all masking configurations.

    For each mask config, runs ensemble inference (load each member, collect
    probs with masking, average across members), then computes metrics.

    Parameters
    ----------
    ckpt_paths : list of paths
        Paths to best.pt checkpoints for each ensemble member.
    loader : DataLoader
        Test DataLoader (shuffle=False, drop_last=False).
    device : torch.device
        GPU or CPU.
    temperature : float
        Temperature for softmax calibration.
    mask_configs : list of str or None
        Masking configs to test. If None, uses MASK_ORDER.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict mapping mask config name to compute_metrics result dict.
    """
    if mask_configs is None:
        mask_configs = MASK_ORDER

    M = len(ckpt_paths)
    results: dict[str, dict] = {}

    for cfg_name in mask_configs:
        print(f"\n{'─' * 60}")
        print(f"Masking config: {cfg_name}")
        print(f"{'─' * 60}")

        # Use a fresh RNG per config for reproducibility
        rng = np.random.default_rng(seed)

        member_probs_list: list[np.ndarray] = []
        labels_dict: dict[str, np.ndarray] | None = None

        for i, ckpt_path in enumerate(ckpt_paths):
            print(f"  Member {i + 1}/{M}: {Path(ckpt_path).parent.name}")

            model = _load_model_from_ckpt(ckpt_path, device)

            # Reset RNG per member so each member sees the same masks
            member_rng = np.random.default_rng(seed)
            probs_m, ld = _collect_probs_masked(
                model, loader, device, temperature, cfg_name, member_rng,
            )
            member_probs_list.append(probs_m)

            if labels_dict is None:
                labels_dict = ld

            del model
            torch.cuda.empty_cache()

        assert labels_dict is not None

        # Average across ensemble members
        p_bar = np.stack(member_probs_list, axis=0).mean(axis=0)  # (N, 90)

        # Compute metrics
        metrics = compute_metrics(
            probs=p_bar,
            action90_true=labels_dict["action90_true"],
            lead2_true=labels_dict["lead2_true"],
            bring4_observed=labels_dict["bring4_observed"],
            is_mirror=labels_dict["is_mirror"],
        )

        # Add summary uncertainty stats
        confidence = p_bar.max(axis=1)
        entropy = -np.sum(p_bar * np.log(p_bar + _EPS), axis=-1)
        metrics["mean_confidence"] = float(confidence.mean())
        metrics["mean_entropy"] = float(entropy.mean())

        results[cfg_name] = metrics

        # Print key metrics
        t1 = f"top1={metrics.get('overall/top1_action90', 0):.1%}"
        t3 = f"top3={metrics.get('overall/top3_action90', 0):.1%}"
        t5 = f"top5={metrics.get('overall/top5_action90', 0):.1%}"
        nll = f"nll={metrics.get('overall/nll_action90', 0):.3f}"
        conf = f"conf={metrics['mean_confidence']:.4f}"
        print(f"  Result: {t1}  {t3}  {t5}  {nll}  {conf}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, out_path: str | Path) -> None:
    """Save figure as both PNG and PDF next to *out_path*."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=_DPI)
    fig.savefig(out.with_suffix(".pdf"), dpi=_DPI)
    plt.close(fig)


_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def plot_stress_test(
    results: dict[str, dict],
    out_dir: str | Path,
) -> None:
    """Generate stress test degradation plots.

    Plot 1: top-1, top-3, top-5 accuracy (action90, Tier 1) vs masking level.
    Plot 2: NLL + mean confidence vs masking level.

    Parameters
    ----------
    results : dict
        Output of run_stress_test: mask config name -> metrics dict.
    out_dir : str or Path
        Directory for output figures.
    """
    out_dir = Path(out_dir)

    # Order configs by severity
    configs = [c for c in MASK_ORDER if c in results]
    x_labels = configs
    x = np.arange(len(configs))

    # Extract metrics
    top1 = [results[c].get("overall/top1_action90", 0) * 100 for c in configs]
    top3 = [results[c].get("overall/top3_action90", 0) * 100 for c in configs]
    top5 = [results[c].get("overall/top5_action90", 0) * 100 for c in configs]
    nll = [results[c].get("overall/nll_action90", 0) for c in configs]
    conf = [results[c].get("mean_confidence", 0) for c in configs]

    # --- Plot 1: Accuracy degradation ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, top1, "o-", color=_COLORS[0], lw=2, markersize=6, label="Top-1")
    ax.plot(x, top3, "s-", color=_COLORS[1], lw=2, markersize=6, label="Top-3")
    ax.plot(x, top5, "^-", color=_COLORS[2], lw=2, markersize=6, label="Top-5")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Stress Test: Action-90 Accuracy vs Masking Level (Tier 1)")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save_fig(fig, out_dir / "stress_test_degradation")
    print(f"  Saved: {out_dir / 'stress_test_degradation.png'}")

    # --- Plot 2: NLL + Confidence ---
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_nll = _COLORS[3]
    color_conf = _COLORS[0]

    ax1.plot(x, nll, "o-", color=color_nll, lw=2, markersize=6, label="NLL")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right")
    ax1.set_ylabel("NLL (Action-90)", color=color_nll)
    ax1.tick_params(axis="y", labelcolor=color_nll)

    ax2 = ax1.twinx()
    ax2.plot(x, conf, "s-", color=color_conf, lw=2, markersize=6, label="Mean Confidence")
    ax2.set_ylabel("Mean Confidence", color=color_conf)
    ax2.tick_params(axis="y", labelcolor=color_conf)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center left")

    ax1.set_title("Stress Test: NLL + Confidence vs Masking Level")

    fig.tight_layout()
    _save_fig(fig, out_dir / "stress_test_confidence")
    print(f"  Saved: {out_dir / 'stress_test_confidence.png'}")
