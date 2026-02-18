#!/usr/bin/env python3
"""Generate all Week 2 comparison plots and tables.

Loads saved metrics (popularity, logistic, transformer) and generates:
  1. topk_comparison_bar — grouped bars for all 3 models
  2. stratified_table — overall/mirror/non_mirror breakdown as JSON + LaTeX
  3. reliability_diagram — for the transformer model (requires inference)
  4. training_curves — from the training log JSON

Saves everything to outputs/plots/week2/.
Prints a summary table to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from turnzero.eval.plots import (
    reliability_diagram,
    stratified_table,
    topk_comparison_bar,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS = PROJECT_ROOT / "outputs"
BASELINES = OUTPUTS / "baselines"
EVAL_DIR = OUTPUTS / "eval" / "run_001"
RUN_DIR = OUTPUTS / "runs" / "run_001"
WEEK2_DIR = OUTPUTS / "plots" / "week2"
WEEK2_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIR = PROJECT_ROOT / "data" / "assembled" / "regime_a"


# ---------------------------------------------------------------------------
# 1. Load all saved metrics
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


print("=" * 70)
print("TurnZero Week 2 — Plot Generation")
print("=" * 70)
print()

metrics_pop = load_json(BASELINES / "metrics_popularity.json")
metrics_log = load_json(BASELINES / "metrics_logistic.json")
metrics_tfm = load_json(EVAL_DIR / "test_metrics.json")
training_curves = load_json(RUN_DIR / "training_curves.json")
run_meta = load_json(RUN_DIR / "run_metadata.json")

print(f"Loaded popularity metrics   : {BASELINES / 'metrics_popularity.json'}")
print(f"Loaded logistic metrics     : {BASELINES / 'metrics_logistic.json'}")
print(f"Loaded transformer metrics  : {EVAL_DIR / 'test_metrics.json'}")
print(f"Loaded training curves      : {RUN_DIR / 'training_curves.json'}")
print()

results_dict = {
    "Popularity": metrics_pop,
    "Logistic": metrics_log,
    "Transformer": metrics_tfm,
}


# ---------------------------------------------------------------------------
# 2a. Top-k comparison bar chart
# ---------------------------------------------------------------------------
print("[1/4] Generating top-k comparison bar chart ...")
topk_comparison_bar(results_dict, WEEK2_DIR / "topk_comparison_bar")
print(f"  -> {WEEK2_DIR / 'topk_comparison_bar.png'}")
print(f"  -> {WEEK2_DIR / 'topk_comparison_bar.pdf'}")
print()


# ---------------------------------------------------------------------------
# 2b. Stratified comparison table (JSON + LaTeX)
# ---------------------------------------------------------------------------
print("[2/4] Generating stratified comparison table ...")
table = stratified_table(
    results_dict,
    out_path_json=WEEK2_DIR / "stratified_table.json",
    out_path_latex=WEEK2_DIR / "stratified_table.tex",
)
print(f"  -> {WEEK2_DIR / 'stratified_table.json'}")
print(f"  -> {WEEK2_DIR / 'stratified_table.tex'}")
print()


# ---------------------------------------------------------------------------
# 2c. Reliability diagram — requires running inference on test set
# ---------------------------------------------------------------------------
print("[3/4] Generating transformer reliability diagram ...")
print("  Loading model checkpoint and running inference on test set ...")

import torch
from turnzero.models.transformer import OTSTransformer, ModelConfig
from turnzero.data.dataset import VGCDataset, Vocab
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = RUN_DIR / "best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

vocab_sizes = ckpt["vocab_sizes"]
model_config = ckpt["model_config"]

model = OTSTransformer(vocab_sizes, ModelConfig(**model_config))
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

# Load vocab
vocab_path = RUN_DIR / "vocab.json"
if not vocab_path.exists():
    vocab_path = SPLIT_DIR / "vocab.json"
vocab = Vocab.load(vocab_path)

# Build test dataset
test_path = SPLIT_DIR / "test.jsonl"
test_ds = VGCDataset(test_path, vocab)
test_loader = DataLoader(
    test_ds,
    batch_size=1024,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# Run inference
all_probs = []
all_action90 = []
all_lead2 = []
all_bring4 = []
all_mirror = []

with torch.no_grad():
    for batch in test_loader:
        team_a = batch["team_a"].to(device)
        team_b = batch["team_b"].to(device)
        logits = model(team_a, team_b)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_action90.append(batch["action90_label"].numpy())
        all_lead2.append(batch["lead2_label"].numpy())
        all_bring4.append(batch["bring4_observed"].numpy())
        all_mirror.append(batch["is_mirror"].numpy())

probs_arr = np.concatenate(all_probs, axis=0)
action90_arr = np.concatenate(all_action90, axis=0)
lead2_arr = np.concatenate(all_lead2, axis=0)
bring4_arr = np.concatenate(all_bring4, axis=0)
mirror_arr = np.concatenate(all_mirror, axis=0)

# Reliability diagram on Tier 1 (bring4_observed) subset
tier1_mask = bring4_arr.astype(bool)
reliability_diagram(
    probs_arr[tier1_mask],
    action90_arr[tier1_mask],
    out_path=WEEK2_DIR / "reliability_transformer_action90",
    title="Transformer — Action-90 Reliability (Tier 1)",
    n_bins=15,
)
print(f"  -> {WEEK2_DIR / 'reliability_transformer_action90.png'}")
print(f"  -> {WEEK2_DIR / 'reliability_transformer_action90.pdf'}")

# Also save a lead-2 reliability diagram
from turnzero.eval.metrics import _marginalize_to_lead2

lead2_probs = _marginalize_to_lead2(probs_arr)
lead2_preds = lead2_probs.argmax(axis=1)
lead2_conf = lead2_probs.max(axis=1)

# Build a quick lead-2 reliability using the same infrastructure
# (We need to reshape for the reliability_diagram fn which expects (N, C))
reliability_diagram(
    lead2_probs,
    lead2_arr,
    out_path=WEEK2_DIR / "reliability_transformer_lead2",
    title="Transformer — Lead-2 Reliability",
    n_bins=15,
)
print(f"  -> {WEEK2_DIR / 'reliability_transformer_lead2.png'}")
print(f"  -> {WEEK2_DIR / 'reliability_transformer_lead2.pdf'}")
print()


# ---------------------------------------------------------------------------
# 2d. Training curves plot
# ---------------------------------------------------------------------------
print("[4/4] Generating training curves plot ...")

train_loss = training_curves["train_loss"]
val_nll = training_curves["val_nll"]
lr_schedule = training_curves["lr"]
epochs = list(range(1, len(train_loss) + 1))
best_epoch = run_meta["best_epoch"] + 1  # 0-indexed in metadata

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: loss curves
ax1.plot(epochs, train_loss, label="Train CE Loss", color="#4c72b0", linewidth=1.5)
ax1.plot(epochs, val_nll, label="Val NLL", color="#dd8452", linewidth=1.5)
ax1.axvline(best_epoch, ls="--", color="gray", alpha=0.7, label=f"Best epoch ({best_epoch})")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend(frameon=False, fontsize=9)
ax1.set_xlim(1, len(epochs))

# Right: learning rate
ax2.plot(epochs, [lr * 1e4 for lr in lr_schedule], color="#55a868", linewidth=1.5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Learning Rate (×10⁻⁴)")
ax2.set_title("Learning Rate Schedule")
ax2.set_xlim(1, len(epochs))

fig.tight_layout()

out_train = WEEK2_DIR / "training_curves"
fig.savefig(out_train.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out_train.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"  -> {out_train.with_suffix('.png')}")
print(f"  -> {out_train.with_suffix('.pdf')}")
print()


# ---------------------------------------------------------------------------
# 3. Print summary table to stdout
# ---------------------------------------------------------------------------
print("=" * 70)
print("WEEK 2 MODEL COMPARISON — SUMMARY TABLE")
print("=" * 70)
print()

# Header
header_fmt = "{:<14s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
row_fmt = "{:<14s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"

print("--- Action-90 Metrics (Tier 1 subset, bring4_observed=True) ---")
print()
print(header_fmt.format(
    "Model", "Top-1", "Top-3", "Top-5", "NLL", "Brier", "ECE", "N"
))
print("-" * 78)

for name, m in results_dict.items():
    for stratum in ["overall", "mirror", "non_mirror"]:
        label = f"{name}" if stratum == "overall" else f"  {stratum}"
        t1 = m.get(f"{stratum}/top1_action90")
        t3 = m.get(f"{stratum}/top3_action90")
        t5 = m.get(f"{stratum}/top5_action90")
        nll = m.get(f"{stratum}/nll_action90")
        brier = m.get(f"{stratum}/brier_action90")
        ece = m.get(f"{stratum}/ece_action90")
        n = m.get(f"{stratum}/n_tier1", 0)

        def fmt_pct(v):
            return f"{v*100:.1f}%" if v is not None else "--"

        def fmt_f(v):
            return f"{v:.4f}" if v is not None else "--"

        print(row_fmt.format(
            label, fmt_pct(t1), fmt_pct(t3), fmt_pct(t5),
            fmt_f(nll), fmt_f(brier), fmt_f(ece), str(n)
        ))
    print()

print()
print("--- Lead-2 Metrics (all examples) ---")
print()
print(header_fmt.format(
    "Model", "Top-1", "Top-3", "", "NLL", "Brier", "ECE", "N"
))
print("-" * 78)

for name, m in results_dict.items():
    for stratum in ["overall", "mirror", "non_mirror"]:
        label = f"{name}" if stratum == "overall" else f"  {stratum}"
        t1 = m.get(f"{stratum}/top1_lead2")
        t3 = m.get(f"{stratum}/top3_lead2")
        nll = m.get(f"{stratum}/nll_lead2")
        brier = m.get(f"{stratum}/brier_lead2")
        ece = m.get(f"{stratum}/ece_lead2")
        n = m.get(f"{stratum}/n_examples", 0)

        def fmt_pct(v):
            return f"{v*100:.1f}%" if v is not None else "--"

        def fmt_f(v):
            return f"{v:.4f}" if v is not None else "--"

        print(row_fmt.format(
            label, fmt_pct(t1), fmt_pct(t3), "",
            fmt_f(nll), fmt_f(brier), fmt_f(ece), str(n)
        ))
    print()

print()
print("--- Transformer Training Summary ---")
print(f"  Architecture : d_model={run_meta['config']['model']['d_model']}, "
      f"layers={run_meta['config']['model']['n_layers']}, "
      f"heads={run_meta['config']['model']['n_heads']}, "
      f"d_ff={run_meta['config']['model']['d_ff']}")
print(f"  Parameters   : {run_meta['n_params']:,}")
print(f"  Best epoch   : {run_meta['best_epoch'] + 1} / {run_meta['total_epochs']} "
      f"(val NLL = {run_meta['best_val_nll']:.4f})")
print(f"  Device       : {run_meta['cuda_device']}")
print(f"  Train/Val/Test: {run_meta['train_examples']:,} / "
      f"{run_meta['val_examples']:,} / {run_meta['test_examples']:,}")
print()

# Improvement summary
print("--- Improvement vs Baselines (Overall, Action-90) ---")
pop_t1 = metrics_pop.get("overall/top1_action90", 0)
log_t1 = metrics_log.get("overall/top1_action90", 0)
tfm_t1 = metrics_tfm.get("overall/top1_action90", 0)
print(f"  Top-1: Popularity {pop_t1*100:.1f}% -> Logistic {log_t1*100:.1f}% "
      f"-> Transformer {tfm_t1*100:.1f}%  "
      f"(+{(tfm_t1-pop_t1)*100:.1f}pp vs pop, +{(tfm_t1-log_t1)*100:.1f}pp vs log)")

pop_t3 = metrics_pop.get("overall/top3_action90", 0)
log_t3 = metrics_log.get("overall/top3_action90", 0)
tfm_t3 = metrics_tfm.get("overall/top3_action90", 0)
print(f"  Top-3: Popularity {pop_t3*100:.1f}% -> Logistic {log_t3*100:.1f}% "
      f"-> Transformer {tfm_t3*100:.1f}%  "
      f"(+{(tfm_t3-pop_t3)*100:.1f}pp vs pop, +{(tfm_t3-log_t3)*100:.1f}pp vs log)")

pop_nll = metrics_pop.get("overall/nll_action90", 0)
log_nll = metrics_log.get("overall/nll_action90", 0)
tfm_nll = metrics_tfm.get("overall/nll_action90", 0)
print(f"  NLL  : Popularity {pop_nll:.4f} -> Logistic {log_nll:.4f} "
      f"-> Transformer {tfm_nll:.4f}  "
      f"({tfm_nll-pop_nll:+.4f} vs pop, {tfm_nll-log_nll:+.4f} vs log)")

print()
print("=" * 70)
print(f"All plots saved to: {WEEK2_DIR}")
print("=" * 70)
