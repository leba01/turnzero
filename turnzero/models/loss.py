"""Loss computation for training, supporting multiple loss modes."""

from __future__ import annotations

import torch
import torch.nn as nn

from turnzero.constants import LOG_EPS


def compute_loss(
    logits: torch.Tensor,
    action90_labels: torch.Tensor,
    criterion: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    loss_mode: str,
    margin_matrix: torch.Tensor | None,
) -> torch.Tensor:
    """Compute loss for a single batch, supporting all three loss modes.

    Parameters
    ----------
    loss_mode : str
        ``"action90_all"`` or ``"tier1_only"`` -- standard CE on action90.
        ``"multitask"`` -- Tier 1: action90 CE, Tier 2: lead-2 CE via marginalization.
    """
    if loss_mode in ("action90_all", "tier1_only"):
        return criterion(logits, action90_labels)

    # multitask: split Tier 1 (action90 CE) and Tier 2 (lead-2 CE)
    bring4 = batch["bring4_observed"].to(device).bool()
    lead2_labels = batch["lead2_label"].to(device, non_blocking=True)
    n = logits.size(0)
    loss = torch.tensor(0.0, device=device)

    if bring4.any():
        loss_t1 = criterion(logits[bring4], action90_labels[bring4])
        loss = loss + loss_t1 * (bring4.sum() / n)

    tier2 = ~bring4
    if tier2.any():
        # Marginalize in probability space (FP32 for numerical stability)
        probs_90 = torch.softmax(logits[tier2].float(), dim=-1)
        lead2_probs = probs_90 @ margin_matrix  # (n_t2, 15)
        log_lead2 = torch.log(lead2_probs.clamp(min=LOG_EPS))
        loss_t2 = nn.functional.nll_loss(log_lead2, lead2_labels[tier2])
        loss = loss + loss_t2 * (tier2.sum() / n)

    return loss
