"""Training loop for OTSTransformer.

Supports:
  - YAML config loading
  - Deterministic seeding (torch, numpy, random, CUDA)
  - Mixed precision (BF16 autocast on CUDA)
  - torch.compile() for speed
  - AdamW + CosineAnnealingLR
  - Cross-entropy with label smoothing
  - Early stopping on val NLL
  - Best-checkpoint saving
  - Training curves (JSON) + run metadata

Reference: docs/WEEK2_PLAN.md Task 5-6, docs/PROJECT_BIBLE.md Section 3
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from turnzero.data.dataset import Vocab, build_dataloaders
from turnzero.eval.metrics import _MARGIN_MATRIX, compute_metrics
from turnzero.models.transformer import ModelConfig, OTSTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a plain dict."""
    import yaml

    with open(yaml_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed torch, numpy, random, and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Git hash
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    """Return short git hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss_mode: str = "action90_all",
    margin_matrix: torch.Tensor | None = None,
) -> float:
    """Train for one epoch. Returns average loss.

    Parameters
    ----------
    loss_mode : str
        ``"action90_all"`` — CE on action90 for all examples (default).
        ``"tier1_only"`` — CE on action90 (dataset already filtered).
        ``"multitask"`` — Tier 1: action90 CE, Tier 2: lead-2 CE via marginalization.
    margin_matrix : (90, 15) Tensor or None
        Required when loss_mode="multitask". Maps action-90 probs → lead-2 probs.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)
        action90_labels = batch["action90_label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(team_a, team_b)

            if loss_mode in ("action90_all", "tier1_only"):
                loss = criterion(logits, action90_labels)
            else:
                # multitask: split Tier 1 and Tier 2
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
                    log_lead2 = torch.log(lead2_probs.clamp(min=1e-12))
                    loss_t2 = nn.functional.nll_loss(log_lead2, lead2_labels[tier2])
                    loss = loss + loss_t2 * (tier2.sum() / n)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_mode: str = "action90_all",
    margin_matrix: torch.Tensor | None = None,
) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    """Validate on a split.

    Parameters
    ----------
    loss_mode, margin_matrix : same semantics as train_one_epoch.

    Returns
    -------
    avg_loss : float
        Mean loss (matching training loss computation for consistent early stopping).
    probs : (N, 90) ndarray
        Softmax probabilities.
    labels_dict : dict
        Keys: action90_true, lead2_true, bring4_observed, is_mirror.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_probs: list[np.ndarray] = []
    all_action90: list[np.ndarray] = []
    all_lead2: list[np.ndarray] = []
    all_bring4: list[np.ndarray] = []
    all_mirror: list[np.ndarray] = []

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)
        action90_labels = batch["action90_label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(team_a, team_b)

            if loss_mode in ("action90_all", "tier1_only"):
                loss = criterion(logits, action90_labels)
            else:
                # multitask: mirror training loss for consistent early stopping
                bring4 = batch["bring4_observed"].to(device).bool()
                lead2_labels = batch["lead2_label"].to(device, non_blocking=True)
                n = logits.size(0)
                loss = torch.tensor(0.0, device=device)

                if bring4.any():
                    loss_t1 = criterion(logits[bring4], action90_labels[bring4])
                    loss = loss + loss_t1 * (bring4.sum() / n)

                tier2 = ~bring4
                if tier2.any():
                    probs_90 = torch.softmax(logits[tier2].float(), dim=-1)
                    lead2_probs = probs_90 @ margin_matrix
                    log_lead2 = torch.log(lead2_probs.clamp(min=1e-12))
                    loss_t2 = nn.functional.nll_loss(log_lead2, lead2_labels[tier2])
                    loss = loss + loss_t2 * (tier2.sum() / n)

        total_loss += loss.item()
        n_batches += 1

        # Softmax in FP32 for numerical stability
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_action90.append(batch["action90_label"].numpy())
        all_lead2.append(batch["lead2_label"].numpy())
        all_bring4.append(batch["bring4_observed"].numpy())
        all_mirror.append(batch["is_mirror"].numpy())

    avg_loss = total_loss / max(n_batches, 1)
    probs = np.concatenate(all_probs, axis=0)
    labels_dict = {
        "action90_true": np.concatenate(all_action90, axis=0),
        "lead2_true": np.concatenate(all_lead2, axis=0),
        "bring4_observed": np.concatenate(all_bring4, axis=0),
        "is_mirror": np.concatenate(all_mirror, axis=0),
    }
    return avg_loss, probs, labels_dict


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict[str, Any], out_dir: str | Path) -> Path:
    """Full training loop.

    Parameters
    ----------
    config : dict
        Loaded YAML config with 'model', 'training', 'data' sections.
    out_dir : path
        Output directory for checkpoints, curves, and metadata.

    Returns
    -------
    Path to best checkpoint.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_model = config["model"]
    cfg_train = config["training"]
    cfg_data = config["data"]

    loss_mode = cfg_train.get("loss_mode", "action90_all")
    assert loss_mode in ("action90_all", "multitask", "tier1_only"), (
        f"Unknown loss_mode: {loss_mode}"
    )

    seed = cfg_train["seed"]
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loss mode: {loss_mode}")

    # --- Data ---
    split_dir = cfg_data["split_dir"]
    print(f"Loading data from {split_dir} ...")
    train_loader, val_loader, test_loader, vocab = build_dataloaders(
        split_dir=split_dir,
        batch_size=cfg_train["batch_size"],
        num_workers=cfg_train["num_workers"],
        tier1_only=(loss_mode == "tier1_only"),
    )
    print(f"Vocab: {vocab}")
    print(f"Train: {len(train_loader.dataset):,} examples, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset):,} examples, {len(val_loader)} batches")
    print(f"Test:  {len(test_loader.dataset):,} examples, {len(test_loader)} batches")

    # Save vocab alongside checkpoint
    vocab.save(out_dir / "vocab.json")

    # --- Model ---
    model_cfg = ModelConfig(
        d_model=cfg_model["d_model"],
        n_layers=cfg_model["n_layers"],
        n_heads=cfg_model["n_heads"],
        d_ff=cfg_model["d_ff"],
        dropout=cfg_model["dropout"],
        pool=cfg_model["pool"],
    )
    model = OTSTransformer(vocab.vocab_sizes, model_cfg)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # torch.compile for speed (graceful fallback if Triton/compiler unavailable)
    try:
        compiled_model = torch.compile(model)
        # Force a test compilation with a dummy forward pass
        _dummy_a = torch.zeros(1, 6, 8, dtype=torch.long, device=device)
        _dummy_b = torch.zeros(1, 6, 8, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            compiled_model(_dummy_a, _dummy_b)
        del _dummy_a, _dummy_b
        print("torch.compile() applied")
    except Exception as e:
        compiled_model = model
        print(f"torch.compile() skipped ({type(e).__name__}: {e})")

    # --- Optimizer + scheduler ---
    optimizer = AdamW(
        compiled_model.parameters(),
        lr=cfg_train["lr"],
        weight_decay=cfg_train["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg_train["max_epochs"],
    )

    # --- Loss ---
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg_train["label_smoothing"])

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler("cuda")

    # --- Margin matrix for multitask loss ---
    margin_matrix = None
    if loss_mode == "multitask":
        margin_matrix = torch.from_numpy(_MARGIN_MATRIX).float().to(device)

    # --- Training loop ---
    max_epochs = cfg_train["max_epochs"]
    patience = cfg_train["patience"]
    best_val_nll = float("inf")
    epochs_no_improve = 0
    best_epoch = -1

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_nll": [],
        "lr": [],
    }

    best_ckpt_path = out_dir / "best.pt"

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val NLL':>10}  {'LR':>10}  {'Best':>5}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            compiled_model, train_loader, optimizer, criterion, scaler, device,
            loss_mode=loss_mode, margin_matrix=margin_matrix,
        )

        # Validate
        val_nll, _, _ = validate(
            compiled_model, val_loader, criterion, device,
            loss_mode=loss_mode, margin_matrix=margin_matrix,
        )

        # LR step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Record
        history["train_loss"].append(train_loss)
        history["val_nll"].append(val_nll)
        history["lr"].append(current_lr)

        # Check improvement
        is_best = val_nll < best_val_nll
        if is_best:
            best_val_nll = val_nll
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best checkpoint (unwrap compiled model)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_sizes": vocab.vocab_sizes,
                    "model_config": {
                        "d_model": model_cfg.d_model,
                        "n_layers": model_cfg.n_layers,
                        "n_heads": model_cfg.n_heads,
                        "d_ff": model_cfg.d_ff,
                        "dropout": model_cfg.dropout,
                        "pool": model_cfg.pool,
                    },
                    "config": config,
                    "loss_mode": loss_mode,
                    "epoch": epoch,
                    "val_nll": val_nll,
                },
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1

        dt = time.time() - t0
        mark = "*" if is_best else ""
        print(f"{epoch:5d}  {train_loss:10.4f}  {val_nll:10.4f}  {current_lr:10.2e}  {mark:>5}  {dt:5.1f}s")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    print(f"\nBest val NLL: {best_val_nll:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to {best_ckpt_path}")

    # --- Save training curves ---
    curves_path = out_dir / "training_curves.json"
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training curves saved to {curves_path}")

    # --- Save run metadata ---
    metadata = {
        "config": config,
        "loss_mode": loss_mode,
        "seed": seed,
        "git_hash": _git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "total_epochs": len(history["train_loss"]),
        "vocab_sizes": vocab.vocab_sizes,
        "train_examples": len(train_loader.dataset),
        "val_examples": len(val_loader.dataset),
        "test_examples": len(test_loader.dataset),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    meta_path = out_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Run metadata saved to {meta_path}")

    return best_ckpt_path


# ---------------------------------------------------------------------------
# Evaluation from checkpoint
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    model_ckpt: str | Path,
    test_split: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Load a checkpoint, run inference on a test split, compute + save metrics.

    Parameters
    ----------
    model_ckpt : path
        Path to best.pt checkpoint.
    test_split : path
        Path to test.jsonl.
    out_dir : path
        Where to save metrics JSON and plots.

    Returns
    -------
    metrics dict
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    vocab_sizes = ckpt["vocab_sizes"]
    model_config = ckpt["model_config"]
    config = ckpt["config"]

    model_cfg = ModelConfig(**model_config)
    model = OTSTransformer(vocab_sizes, model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load vocab from checkpoint dir or from test split dir
    ckpt_dir = Path(model_ckpt).parent
    vocab_path = ckpt_dir / "vocab.json"
    if not vocab_path.exists():
        # Try split dir
        split_dir = Path(config["data"]["split_dir"])
        vocab_path = split_dir / "vocab.json"
    vocab = Vocab.load(vocab_path)

    # Build test loader
    from turnzero.data.dataset import VGCDataset
    test_ds = VGCDataset(test_split, vocab)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    print(f"Test: {len(test_ds):,} examples, {len(test_loader)} batches")

    # Run validation
    criterion = nn.CrossEntropyLoss()
    val_nll, probs, labels_dict = validate(model, test_loader, criterion, device)

    # Compute metrics
    metrics = compute_metrics(
        probs=probs,
        action90_true=labels_dict["action90_true"],
        lead2_true=labels_dict["lead2_true"],
        bring4_observed=labels_dict["bring4_observed"],
        is_mirror=labels_dict["is_mirror"],
    )
    metrics["test_nll"] = val_nll

    # Save metrics
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test metrics saved to {metrics_path}")

    # Print summary
    print(f"\n{'─' * 50}")
    print(f"Test NLL (CE):              {val_nll:.4f}")
    print(f"Action90 Top-1 (Tier 1):    {metrics.get('overall/top1_action90', 0):.1%}")
    print(f"Action90 Top-3 (Tier 1):    {metrics.get('overall/top3_action90', 0):.1%}")
    print(f"Action90 Top-5 (Tier 1):    {metrics.get('overall/top5_action90', 0):.1%}")
    print(f"Lead-2 Top-1 (all):         {metrics.get('overall/top1_lead2', 0):.1%}")
    print(f"Lead-2 Top-3 (all):         {metrics.get('overall/top3_lead2', 0):.1%}")
    print(f"NLL Action90 (Tier 1):      {metrics.get('overall/nll_action90', 0):.4f}")
    print(f"NLL Lead-2 (all):           {metrics.get('overall/nll_lead2', 0):.4f}")
    print(f"ECE Action90 (Tier 1):      {metrics.get('overall/ece_action90', 0):.4f}")
    print(f"Tier 1 examples:            {metrics.get('overall/n_tier1', 0):,}")
    print(f"Total test examples:        {metrics.get('overall/n_examples', 0):,}")
    print(f"{'─' * 50}")

    # Generate reliability diagram
    from turnzero.eval.plots import reliability_diagram
    tier1_mask = labels_dict["bring4_observed"].astype(bool)
    if tier1_mask.sum() > 0:
        reliability_diagram(
            probs[tier1_mask],
            labels_dict["action90_true"][tier1_mask],
            out_dir / "reliability_action90",
            title="Action-90 Reliability (Tier 1)",
        )
        print(f"Reliability diagram saved to {out_dir}/reliability_action90.png")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_train(config_path: str, out_dir: str) -> None:
    """CLI entry point for training."""
    config = load_config(config_path)
    train(config, out_dir)


def run_eval(model_ckpt: str, test_split: str, out_dir: str) -> None:
    """CLI entry point for evaluation."""
    evaluate_checkpoint(model_ckpt, test_split, out_dir)
