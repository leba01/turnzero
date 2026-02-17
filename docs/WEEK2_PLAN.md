# Week 2 Plan: Model + Training Loop + Eval Harness

## Setup on PC

```bash
# 1. Clone the repo
git clone https://github.com/leba01/turnzero.git
cd turnzero

# 2. Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install in dev mode with all deps
pip install -e ".[dev]"

# 4. Install ML deps (new for week 2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install pandas numpy matplotlib scikit-learn

# 5. Run tests to verify code works
pytest
```

## Task 0: Download Full Dataset + Re-run Pipeline

Week 1 only used `logs-gen9vgc2024regg.json` (ladder, 165 MB, ~21K battles).
The bible calls for **BO3 tournament data as primary** + ladder for volume.

### Step 0a: Download from HuggingFace

The dataset is at `cameronangliss/vgc-battle-logs`. Download these files into `data/raw/`:

```bash
mkdir -p data/raw
pip install huggingface_hub

# Primary: BO3 tournament files (high-quality expert games)
huggingface-cli download cameronangliss/vgc-battle-logs logs-gen9vgc2024reggbo3.json --repo-type dataset --local-dir data/raw
huggingface-cli download cameronangliss/vgc-battle-logs logs-gen9vgc2025reggbo3.json --repo-type dataset --local-dir data/raw

# Supplement: Reg G ladder for volume
huggingface-cli download cameronangliss/vgc-battle-logs logs-gen9vgc2024regg.json --repo-type dataset --local-dir data/raw
```

Expected sizes:
- `logs-gen9vgc2024reggbo3.json` — ~394 KB (small but high quality)
- `logs-gen9vgc2025reggbo3.json` — ~1.37 GB (bulk of tournament data)
- `logs-gen9vgc2024regg.json` — ~165 MB (ladder supplement)

### Step 0b: Parse each file separately

The parser takes one file at a time. Parse each, then concatenate before canonicalization.

```bash
# Parse each raw file into its own temp directory
turnzero parse --raw_path data/raw/logs-gen9vgc2024reggbo3.json --out_dir data/parsed/gen9vgc2024reggbo3
turnzero parse --raw_path data/raw/logs-gen9vgc2025reggbo3.json --out_dir data/parsed/gen9vgc2025reggbo3
turnzero parse --raw_path data/raw/logs-gen9vgc2024regg.json   --out_dir data/parsed/gen9vgc2024regg

# Concatenate all parsed JSONL into one file
cat data/parsed/gen9vgc2024reggbo3/match_examples.jsonl \
    data/parsed/gen9vgc2025reggbo3/match_examples.jsonl \
    data/parsed/gen9vgc2024regg/match_examples.jsonl \
    > data/parsed/match_examples.jsonl
```

### Step 0c: Run rest of pipeline

```bash
# Canonicalize (normalize names, sort, dedup)
turnzero canonicalize --in_path data/parsed/match_examples.jsonl --out_dir data/canonical

# Cluster (union-find, ≥4/6 species overlap)
turnzero cluster --in_path data/canonical/match_examples.jsonl --out_dir data/clusters

# Split (Regime A + B)
turnzero split --in_path data/canonical/match_examples.jsonl \
    --clusters data/clusters/cluster_assignments.json \
    --out_dir data/splits --seed 42

# Assemble per-split JSONL
turnzero assemble --canonical_path data/canonical/match_examples.jsonl \
    --clusters data/clusters/cluster_assignments.json \
    --splits data/splits/splits.json \
    --out_dir data/assembled

# Stats + integrity validation
turnzero stats --data_dir data/assembled --validate
```

The full pipeline should take under 5 minutes. Verify `dataset_report.json` shows
`validation_passed: true` and check the new example counts (should be significantly
larger than the ladder-only 33K).

## What Exists (Week 1 — code done, data needs re-run)

**Code (all committed, ready to use):**
- `turnzero/schemas.py` — Pokemon, TeamSheet, MatchExample, Label, LabelQuality dataclasses
- `turnzero/action_space.py` — 90-way (lead2, back2) bijection, pre-computed lookup tables
- `turnzero/data/parser.py` — Showdown `|showteam|` extraction
- `turnzero/data/canonicalize.py` — CamelCase normalization, move/team sort, dedup
- `turnzero/splits/cluster.py` — union-find core clustering (4/6 species overlap)
- `turnzero/splits/split.py` — Regime A + B split generation
- `turnzero/data/assemble.py` — per-split JSONL assembly
- `turnzero/data/stats.py` — dataset report + integrity validation
- `turnzero/cli.py` — click CLI with parse/canonicalize/cluster/split/assemble/stats commands
- Full test suite in `tests/`

**Data artifacts (gitignored — must be regenerated on PC via Task 0):**
- Week 1 dev run used only ladder data (33K examples). Task 0 downloads the full
  BO3 dataset and re-runs the pipeline for a much larger, higher-quality training set.
- After Task 0 you'll have `data/assembled/regime_{a,b}/{train,val,test}.jsonl`

## Week 2 Deliverables (in order)

### Task 1: PyTorch Dataset + DataLoader

**Create `turnzero/data/dataset.py`**

- `VGCDataset(torch.utils.data.Dataset)` that reads assembled JSONL
- Builds vocabulary mappings on init: species→idx, item→idx, ability→idx, tera→idx, move→idx
- Each `__getitem__` returns a dict:
  - `team_a_pokemon`: (6, num_features) int tensor — indices into vocab for each field
  - `team_b_pokemon`: (6, num_features) int tensor
  - `action90_label`: int
  - `lead2_label`: int (0..14, for lead-only eval)
  - `bring4_observed`: bool
  - `is_mirror`: bool
- Vocab built from training split only; val/test use same vocab (UNK for unseen)
- Save vocab to `vocab.json` artifact for inference

**Key design**: each mon = (species_idx, item_idx, ability_idx, tera_idx, move0_idx, move1_idx, move2_idx, move3_idx) = 8 ints. Team = 6 mons. Input to model = 12 mons total (6 per side).

### Task 2: Baselines (owed from Week 1)

**Create `turnzero/models/baselines.py`**

1. **Popularity baseline**: `P(action90)` from training set frequencies. Also core-conditional `P(action90 | core_cluster_a, core_cluster_b)`.
2. **Logistic regression baseline**: sklearn `LogisticRegression` on one-hot OTS features (species, item, ability, tera, move presence per side). Multinomial → 90 logits.

**Create `turnzero/models/__init__.py`**

Both baselines should output 90-way probability vectors so they plug into the same eval harness.

### Task 3: Eval Harness

**Create `turnzero/eval/metrics.py`**

Compute all metrics from a (predictions, labels) pair:
- **Top-1, Top-3, Top-5 accuracy** (action90 — Tier 1 only for bring4; lead-2 on all)
- **NLL (log loss)** — mean -log p(y_true)
- **Brier score** — mean ||p - onehot(y)||^2
- **ECE** — binned expected calibration error
- Report each metric overall + mirror + non-mirror strata

**Create `turnzero/eval/plots.py`**

- **Reliability diagram**: accuracy vs confidence per bin + histogram
- **Top-k accuracy bar chart**: baselines vs model comparison
- Save to `outputs/plots/`

**Create `turnzero/eval/__init__.py`**

### Task 4: Transformer Set Model

**Create `turnzero/models/transformer.py`**

Architecture (from Bible Section 3, Option A):
- **Embedding layer**: `E = E_species + E_item + E_ability + E_tera + sum(E_move_i) + E_side`
  - E_side: learned embedding distinguishing team_a vs team_b tokens
  - All unknown fields → shared `UNK` embedding per field type
- **Transformer encoder**: L layers (start with L=4) of multi-head self-attention over 12 tokens
  - No positional encoding (permutation equivariant within each team)
  - d_model=128 or 256, nhead=4 or 8
  - Dropout in attention + FFN
- **Pooling**: mean-pool over 12 tokens (or learned CLS token)
- **Classification head**: MLP → 90 logits
- **Output**: 90-way softmax

### Task 5: Training Loop

**Create `turnzero/models/train.py`**

- AdamW optimizer, cosine LR schedule
- Batch size 256-512
- Mixed precision (torch.cuda.amp)
- Early stopping on val NLL (patience ~10-15 epochs)
- Label smoothing 0.02-0.05
- Weight decay
- Deterministic: seed everything (torch, numpy, python random, CUDA)
- Save best checkpoint + training curves (loss, val_nll per epoch)
- Log full config to `run_metadata.json` with git hash

**Add CLI commands:**
```
turnzero train --split_dir data/assembled/regime_a --config configs/transformer_base.yaml --out_dir outputs/runs/run_001
turnzero eval --model_ckpt outputs/runs/run_001/best.pt --test_split data/assembled/regime_a/test.jsonl --out_dir outputs/eval/run_001
```

### Task 6: Config System

**Create `configs/transformer_base.yaml`**

```yaml
model:
  d_model: 128
  n_layers: 4
  n_heads: 4
  d_ff: 512
  dropout: 0.1
  pool: mean

training:
  batch_size: 512
  lr: 3e-4
  weight_decay: 0.01
  label_smoothing: 0.03
  max_epochs: 100
  patience: 15
  seed: 42

data:
  regime: a
```

### Task 7: Paper-Ready Plotting Pipeline

- Stratified results table: overall / mirror / non-mirror
- Baseline vs model comparison
- Lead-2 accuracy (all examples) + action90 accuracy (Tier 1 only)
- Save tables as both JSON and LaTeX-ready format

## Module Layout After Week 2

```
turnzero/
  __init__.py
  schemas.py
  action_space.py
  cli.py
  data/
    __init__.py
    parser.py
    canonicalize.py
    assemble.py
    stats.py
    dataset.py          ← NEW
    io_utils.py
  splits/
    __init__.py
    cluster.py
    split.py
  models/
    __init__.py          ← NEW
    baselines.py         ← NEW
    transformer.py       ← NEW
    train.py             ← NEW
  eval/
    __init__.py          ← NEW
    metrics.py           ← NEW
    plots.py             ← NEW
configs/
  transformer_base.yaml  ← NEW
outputs/                 ← NEW (gitignored)
  runs/
  eval/
  plots/
```

## Order of Operations (suggested workflow with Claude)

0. "Download full BO3 dataset and re-run pipeline" → follow Task 0 steps above
1. "Build VGCDataset and DataLoader" → test it loads correctly
2. "Build popularity + logistic baselines" → get first numbers
3. "Build eval harness (metrics + plots)" → evaluate baselines, sanity check
4. "Build transformer model" → forward pass test
5. "Build training loop" → train on regime_a, evaluate
6. "Generate paper-ready plots" → baselines vs transformer comparison

Each step is independently testable. Don't move to the next until the current one works end-to-end.

## Sanity Check Targets

- Popularity baseline top-1 action90: ~1.5% (uniform would be 1.1%)
- Logistic regression top-1: ~5-10%
- Transformer top-1: should beat logistic by a solid margin
- Lead-2 top-1 accuracy will be higher than action90 (only 15 classes vs 90)
- NLL: uniform baseline = -log(1/90) ≈ 4.50 — everything should beat this
