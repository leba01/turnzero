"""Explanation modules: marginals, formatted output, and feature sensitivity.

Provides three lightweight explanation components for the coach demo:
  1. compute_marginals — per-mon lead/bring probs + 15-way lead pair probs
  2. format_marginals — pretty-print marginals with species names
  3. feature_sensitivity — KL-divergence under field masking (counterfactual)

Reference: docs/PROJECT_BIBLE.md Sections 5.4, 5.5
           docs/WEEK4_PLAN.md Task 2
"""

from __future__ import annotations

import numpy as np
import torch

from turnzero.action_space import ACTION_TABLE
from turnzero.eval.metrics import LEAD_PAIRS as _LEAD_PAIRS
from turnzero.eval.metrics import _MARGIN_MATRIX as _LEAD_MARGIN_MATRIX

# ---------------------------------------------------------------------------
# Pre-computed masks: (90, 6) binary matrices
# ---------------------------------------------------------------------------

# LEAD_MASK[a, i] = 1 if mon i is in the lead pair of action a
LEAD_MASK: np.ndarray = np.zeros((90, 6), dtype=np.float64)
# BRING_MASK[a, i] = 1 if mon i is in the bring-4 of action a
BRING_MASK: np.ndarray = np.zeros((90, 6), dtype=np.float64)

for _a, (_lead, _back) in enumerate(ACTION_TABLE):
    for _i in _lead:
        LEAD_MASK[_a, _i] = 1.0
        BRING_MASK[_a, _i] = 1.0
    for _i in _back:
        BRING_MASK[_a, _i] = 1.0

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Marginal extraction (Bible 5.4)
# ---------------------------------------------------------------------------

def compute_marginals(probs_90: np.ndarray) -> dict[str, np.ndarray]:
    """From 90-way action probs, extract per-mon lead/bring marginals.

    Args:
        probs_90: (90,) single example or (N, 90) batch of action probs.

    Returns:
        Dict with keys:
          - ``"lead_probs"``: (6,) or (N, 6) — P(mon i is led)
          - ``"bring_probs"``: (6,) or (N, 6) — P(mon i is brought)
          - ``"lead_pair_probs"``: (15,) or (N, 15) — P(lead pair j)
    """
    probs = np.asarray(probs_90, dtype=np.float64)
    squeeze = False
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]  # (1, 90)
        squeeze = True

    assert probs.shape[-1] == 90, f"Expected last dim=90, got {probs.shape}"

    lead_probs = probs @ LEAD_MASK          # (N, 6)
    bring_probs = probs @ BRING_MASK        # (N, 6)
    lead_pair_probs = probs @ _LEAD_MARGIN_MATRIX  # (N, 15)

    if squeeze:
        lead_probs = lead_probs[0]
        bring_probs = bring_probs[0]
        lead_pair_probs = lead_pair_probs[0]

    return {
        "lead_probs": lead_probs,
        "bring_probs": bring_probs,
        "lead_pair_probs": lead_pair_probs,
    }


# ---------------------------------------------------------------------------
# Pretty-print marginals (Bible 5.4)
# ---------------------------------------------------------------------------

def format_marginals(
    marginals: dict[str, np.ndarray],
    species_names: list[str],
) -> str:
    """Pretty-print per-mon lead/bring marginals with species names.

    Args:
        marginals: output from :func:`compute_marginals` (single example).
        species_names: list of 6 species names matching mon indices.

    Returns:
        Multi-line formatted string, sorted by probability descending.
    """
    lead_probs = np.asarray(marginals["lead_probs"])
    bring_probs = np.asarray(marginals["bring_probs"])
    lead_pair_probs = np.asarray(marginals["lead_pair_probs"])

    lines: list[str] = []

    # Per-mon lead probs (sorted descending)
    lines.append("  Lead Probabilities:")
    order = np.argsort(lead_probs)[::-1]
    for i in order:
        lines.append(f"    P({species_names[i]} leads) = {lead_probs[i]:.0%}")

    # Per-mon bring probs (sorted descending)
    lines.append("  Bring Probabilities:")
    order = np.argsort(bring_probs)[::-1]
    for i in order:
        lines.append(f"    P({species_names[i]} brought) = {bring_probs[i]:.0%}")

    # Top lead pairs
    lines.append("  Top Lead Pairs:")
    pair_order = np.argsort(lead_pair_probs)[::-1]
    for rank, j in enumerate(pair_order[:5]):
        pair = _LEAD_PAIRS[j]
        names = f"{species_names[pair[0]]} + {species_names[pair[1]]}"
        lines.append(f"    {names}: {lead_pair_probs[j]:.1%}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature masking sensitivity (Bible 5.5, Counterfactual 2)
# ---------------------------------------------------------------------------

# Field groups in the (B, 6, 8) team tensor:
#   [species=0, item=1, ability=2, tera=3, move0=4, move1=5, move2=6, move3=7]
_FIELD_GROUPS: dict[str, list[int]] = {
    "species": [0],
    "items": [1],
    "ability": [2],
    "tera": [3],
    "moves": [4, 5, 6, 7],
}


@torch.no_grad()
def feature_sensitivity(
    models: list,
    team_a_tensor: torch.Tensor,
    team_b_tensor: torch.Tensor,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    """Mask one field group at a time on team_b, measure KL from baseline.

    For each field group (species, items, ability, tera, moves), replaces
    those columns in team_b with 0 (UNK index) and runs the ensemble
    forward pass.  Reports KL(p_base || p_masked) for each group.

    Higher KL = the prediction depends more on that field type.

    Args:
        models: list of OTSTransformer ensemble members (already on device,
                eval mode).
        team_a_tensor: (1, 6, 8) LongTensor on device.
        team_b_tensor: (1, 6, 8) LongTensor on device.
        temperature: calibrated temperature scalar.
        device: torch device.

    Returns:
        Dict mapping field group name to KL divergence (float).
        Example: ``{"species": 0.05, "items": 0.03, ...}``
    """

    def _ensemble_probs(ta: torch.Tensor, tb: torch.Tensor) -> np.ndarray:
        """Run ensemble forward and return mean probs (90,) as numpy."""
        member_probs = []
        for model in models:
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(ta, tb)
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            member_probs.append(probs)
        stacked = torch.stack(member_probs, dim=0)  # (M, 1, 90)
        p_bar = stacked.mean(dim=0).cpu().numpy()[0]  # (90,)
        return p_bar

    # Baseline
    p_base = _ensemble_probs(team_a_tensor, team_b_tensor)

    # Per-group masking
    results: dict[str, float] = {}
    for group_name, cols in _FIELD_GROUPS.items():
        tb_masked = team_b_tensor.clone()
        for col in cols:
            tb_masked[:, :, col] = 0  # UNK index
        p_masked = _ensemble_probs(team_a_tensor, tb_masked)

        # KL(p_base || p_masked) = sum(p_base * log(p_base / p_masked))
        kl = float(np.sum(
            p_base * np.log((p_base + _EPS) / (p_masked + _EPS))
        ))
        results[group_name] = kl

    return results
