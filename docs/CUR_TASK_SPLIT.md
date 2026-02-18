# Week 2 Parallel Task Split

## Dependency Graph

```
         ┌─────────────┐
         │  Assembled   │
         │  JSONL data  │
         └──────┬───────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌───────┐ ┌─────────┐ ┌─────────┐
│  1A   │ │   1B    │ │   1C    │
│Dataset│ │  Eval   │ │Transformer│
│+Loader│ │ Harness │ │  Model  │
└───┬───┘ └────┬────┘ └────┬────┘
    │          │           │
    │    ┌─────┴─────┐     │
    └───►│    2A     │◄────┘
         │ Baselines │
         │  + Eval   │
         └─────┬─────┘
               │
         ┌─────┴─────┐
         │    2B     │
         │ Training  │
         │Loop+Config│
         └─────┬─────┘
               │
         ┌─────┴─────┐
         │    3A     │
         │Paper Plots│
         └───────────┘
```

**Wave 1** — three fully independent agents (no shared files):
- **1A** creates `turnzero/data/dataset.py`
- **1B** creates `turnzero/eval/metrics.py` + `turnzero/eval/plots.py` + `turnzero/eval/__init__.py`
- **1C** creates `turnzero/models/transformer.py` + `turnzero/models/__init__.py`

**Wave 2** — two agents, after wave 1 merges:
- **2A** creates `turnzero/models/baselines.py`, evaluates baselines
- **2B** creates `turnzero/models/train.py` + `configs/transformer_base.yaml`, runs first training

**Wave 3** — one agent, after wave 2:
- **3A** paper-ready comparison plots + tables

---

## Shared Interfaces (ALL agents must respect these)

### JSONL row schema (the data on disk)

Each line in `data/assembled/regime_a/{train,val,test}.jsonl` is JSON:
```json
{
  "example_id": "926738a019fbac38",
  "match_group_id": "gen9vgc2024reggbo3-2191592264",
  "battle_id": "gen9vgc2024reggbo3-2191592264",
  "team_a": {
    "team_id": "...",
    "species_key": "...",
    "format_id": "gen9vgc2024reggbo3",
    "pokemon": [
      {"species": "Glimmora", "item": "Power Herb", "ability": "Toxic Debris",
       "tera_type": "Rock", "moves": ["Earth Power", "Meteor Beam", "Power Gem", "Spiky Shield"]},
      ... // 6 total
    ],
    "reconstruction_quality": {"fields_known": 42, "fields_total": 42, "source_method": "showteam_direct"}
  },
  "team_b": { ... same structure ... },
  "label": {"lead2_idx": [3, 4], "back2_idx": [0, 2], "action90_id": 73},
  "label_quality": {"bring4_observed": false, "notes": "..."},
  "format_id": "gen9vgc2024reggbo3",
  "metadata": {"source_dataset": "...", "timestamp_epoch": 1725094969},
  "split_keys": {
    "team_a_id": "...", "team_b_id": "...",
    "core_cluster_a": "cluster_0", "core_cluster_b": "cluster_3955",
    "is_mirror": false
  }
}
```

Each Pokemon has 8 categorical fields: `species, item, ability, tera_type, moves[0..3]`.

### Dataset __getitem__ return contract (1A defines, everyone consumes)

```python
{
    "team_a": torch.LongTensor,     # shape (6, 8) — 6 mons, 8 fields each
    "team_b": torch.LongTensor,     # shape (6, 8)
    "action90_label": int,           # 0..89
    "lead2_label": int,              # 0..14 (index into C(6,2) lead pairs)
    "bring4_observed": bool,
    "is_mirror": bool,
}
```

Field order per mon: `[species, item, ability, tera_type, move0, move1, move2, move3]`

Vocab object saved as `vocab.json`:
```json
{
  "species": {"Glimmora": 1, "Mamoswine": 2, ...},
  "item":    {"Power Herb": 1, ...},
  "ability": {"Toxic Debris": 1, ...},
  "tera_type": {"Rock": 1, ...},
  "move":    {"Earth Power": 1, ...}
}
```
Index 0 is reserved for `<UNK>` in every vocab. Index 1+ for known tokens.

### Model forward contract (1C defines, 2B consumes)

```python
# Constructor
model = OTSTransformer(vocab_sizes: dict[str, int], cfg: ModelConfig)
# vocab_sizes = {"species": N, "item": N, "ability": N, "tera_type": N, "move": N}

# Forward
logits = model(team_a, team_b)
# team_a: (batch, 6, 8) LongTensor
# team_b: (batch, 6, 8) LongTensor
# returns: (batch, 90) float logits
```

### Eval harness contract (1B defines, 2A/2B/3A consume)

```python
from turnzero.eval.metrics import compute_metrics

results = compute_metrics(
    probs,          # (N, 90) float numpy array — predicted probabilities
    action90_true,  # (N,) int numpy array — ground truth action90 ids
    lead2_true,     # (N,) int numpy array — ground truth lead2 ids (0..14)
    bring4_observed,# (N,) bool numpy array
    is_mirror,      # (N,) bool numpy array
)
# returns: dict with keys like "overall/top1_action90", "mirror/nll", etc.
```

Lead-2 probabilities are derived by marginalizing the 90-way probs (sum over all actions sharing each lead pair). The eval harness does this internally — callers always pass 90-way probs.

### Lead2 index mapping

Lead pair index (0..14) = position in `itertools.combinations(range(6), 2)`. Already in `turnzero/action_space.py`. The eval harness will import `ACTION_TABLE` from there to marginalize.

---

## Wave 1 Prompts

### Terminal 1A: Dataset + DataLoader

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126).

Read these files first to understand the project:
- docs/PROJECT_BIBLE.md (full spec)
- docs/WEEK2_PLAN.md (task breakdown)
- CLAUDE.md (current status + conventions)
- turnzero/schemas.py (data classes)
- turnzero/action_space.py (90-way action bijection)
- Look at 2-3 lines from data/assembled/regime_a/train.jsonl to see the JSON structure

YOUR TASK: Create turnzero/data/dataset.py — a PyTorch Dataset + DataLoader factory.

Requirements:

1. Class `Vocab`:
   - Built from a list of MatchExample dicts (training split only)
   - Creates token→index mappings for: species, item, ability, tera_type, move
   - Index 0 = "<UNK>" for all field types. Index 1+ for observed tokens
   - save(path) / load(path) methods writing vocab.json
   - vocab_sizes property → dict[str, int] giving total size (including UNK) for each field

2. Class `VGCDataset(torch.utils.data.Dataset)`:
   - __init__(jsonl_path: str, vocab: Vocab)
   - Loads all examples into memory on init (dataset fits in RAM)
   - __getitem__ returns a dict:
     - "team_a": LongTensor (6, 8) — species_idx, item_idx, ability_idx, tera_idx, move0..3_idx
     - "team_b": LongTensor (6, 8)
     - "action90_label": int (0..89)
     - "lead2_label": int (0..14, index into combinations(range(6),2))
     - "bring4_observed": bool
     - "is_mirror": bool
   - __len__ returns number of examples

3. Function `build_dataloaders(split_dir, batch_size, num_workers, vocab_path=None)`:
   - If vocab_path is None: build vocab from train split, save to split_dir/vocab.json
   - If vocab_path given: load existing vocab
   - Returns (train_loader, val_loader, test_loader, vocab)
   - Use pin_memory=True, drop_last=True for train only
   - Default num_workers=4

4. The lead2_label mapping: use itertools.combinations(range(6), 2) to create a
   lead_pair→index lookup (same ordering as action_space.py). Given label.lead2_idx
   = [i, j], look up index of tuple (i, j) in the combinations list.

Write a short test at the bottom (if __name__ == "__main__") that loads the regime_a
train split, builds vocab, prints vocab sizes, and iterates one batch printing shapes.

DO NOT touch any files other than:
- turnzero/data/dataset.py (create)

This is part of a parallel build. Other agents are simultaneously building
turnzero/eval/ and turnzero/models/. Do not create or modify those directories.
```

### Terminal 1B: Eval Harness

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126).

Read these files first to understand the project:
- docs/PROJECT_BIBLE.md (sections 4.3, 4.4 especially — metrics + selective prediction)
- docs/WEEK2_PLAN.md (Task 3 section)
- CLAUDE.md (current status + conventions)
- turnzero/action_space.py (ACTION_TABLE for marginalizing 90→15 lead probs)

YOUR TASK: Create the evaluation harness in turnzero/eval/.

Create these files:

1. turnzero/eval/__init__.py — just re-exports

2. turnzero/eval/metrics.py — all metric computation:

   def compute_metrics(probs, action90_true, lead2_true, bring4_observed, is_mirror):
       """
       Args:
           probs: (N, 90) numpy float array — predicted 90-way probabilities
           action90_true: (N,) numpy int array — ground truth action90 ids
           lead2_true: (N,) numpy int array — ground truth lead pair index (0..14)
           bring4_observed: (N,) numpy bool array
           is_mirror: (N,) numpy bool array
       Returns:
           dict with all metrics, keyed like "overall/top1_action90", "mirror/nll", etc.
       """

   Metrics to compute:
   - top1_action90, top3_action90, top5_action90 (only on Tier 1: bring4_observed==True)
   - top1_lead2, top3_lead2 (on ALL examples — leads always observable)
   - nll (negative log-likelihood, on action90 for Tier 1, on lead2 for all)
   - brier_score (multiclass Brier, same split as nll)
   - ece (Expected Calibration Error, 15 equal-width bins on confidence)

   Stratification: compute every metric on 3 subsets:
   - "overall": all examples (or all Tier 1 for action90 metrics)
   - "mirror": is_mirror==True subset
   - "non_mirror": is_mirror==False subset

   Lead-2 probs: marginalize 90-way probs by summing over all actions that share each
   lead pair. Use ACTION_TABLE from turnzero.action_space — ACTION_TABLE[i] = ((l0,l1),(b0,b1)),
   so the lead pair index for action i is the position of (l0,l1) in combinations(range(6),2).

   Helper: def _ece(probs_true_class, correct, n_bins=15) that bins by predicted
   confidence and returns weighted |accuracy - confidence| per bin.

3. turnzero/eval/plots.py — plotting functions:

   def reliability_diagram(probs, action90_true, out_path, title="", n_bins=15):
       """Two-panel plot: top = accuracy vs confidence, bottom = confidence histogram."""

   def topk_comparison_bar(results_dict, out_path):
       """Grouped bar chart: models on x-axis, top1/top3/top5 bars for each.
       results_dict: {"Popularity": metrics_dict, "Logistic": metrics_dict, "Transformer": metrics_dict}
       """

   def stratified_table(results_dict, out_path_json, out_path_latex=None):
       """Write a comparison table (overall/mirror/non_mirror x metric) as JSON and optionally LaTeX."""

   Use matplotlib with a clean style (no grid, legible fonts, tight_layout).
   Save figures at 300 DPI, PDF + PNG.

DO NOT touch any files other than:
- turnzero/eval/__init__.py (create)
- turnzero/eval/metrics.py (create)
- turnzero/eval/plots.py (create)

This is part of a parallel build. Other agents are simultaneously building
turnzero/data/dataset.py and turnzero/models/. Do not create or modify those.
```

### Terminal 1C: Transformer Model

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126, RTX 4080 Super with BF16).

Read these files first to understand the project:
- docs/PROJECT_BIBLE.md (section 3 — modeling spec, Option A)
- docs/WEEK2_PLAN.md (Task 4 section)
- CLAUDE.md (current status + conventions)
- turnzero/schemas.py (to understand the data)
- turnzero/action_space.py (90-way action space)

YOUR TASK: Create the Transformer set model in turnzero/models/.

Create these files:

1. turnzero/models/__init__.py — re-exports OTSTransformer

2. turnzero/models/transformer.py:

   @dataclass
   class ModelConfig:
       d_model: int = 128
       n_layers: int = 4
       n_heads: int = 4
       d_ff: int = 512
       dropout: float = 0.1
       pool: str = "mean"  # "mean" or "cls"

   class OTSTransformer(nn.Module):
       def __init__(self, vocab_sizes: dict[str, int], cfg: ModelConfig):
           """
           vocab_sizes: {"species": N, "item": N, "ability": N, "tera_type": N, "move": N}
           """

       def forward(self, team_a: Tensor, team_b: Tensor) -> Tensor:
           """
           team_a: (B, 6, 8) LongTensor — 6 mons, 8 fields each
           team_b: (B, 6, 8) LongTensor
           returns: (B, 90) logits
           """

   Architecture (from PROJECT_BIBLE Section 3, Option A):

   a) Embedding layer — one nn.Embedding per field type:
      - E_species(species_idx) + E_item(item_idx) + E_ability(ability_idx)
        + E_tera(tera_idx) + E_move(move0_idx) + E_move(move1_idx)
        + E_move(move2_idx) + E_move(move3_idx) + E_side
      - All 4 moves share the same E_move embedding table
      - E_side: learned embedding, 0=team_a, 1=team_b
      - Each embedding outputs d_model dimensions
      - Sum all to get per-mon embedding: (B, 12, d_model) for 6+6 mons

   b) Transformer encoder — L layers of standard TransformerEncoderLayer:
      - No positional encoding (permutation equivariant within each team)
      - d_model, n_heads, d_ff, dropout as per config
      - Use batch_first=True
      - Pre-norm (norm_first=True) for training stability

   c) Pooling:
      - "mean": mean over 12 token dimension → (B, d_model)
      - "cls": prepend learned [CLS] token, take CLS output → (B, d_model)

   d) Classification head:
      - Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, 90)

   Design notes:
   - Initialize embeddings with small std (0.02)
   - The model must be compatible with torch.compile() and BF16 autocast
   - Keep it clean and simple — no extra bells and whistles beyond the spec

   Add a smoke test at the bottom (if __name__ == "__main__") that:
   - Creates a model with small dummy vocab sizes
   - Runs a forward pass with random input
   - Prints output shape and parameter count

DO NOT touch any files other than:
- turnzero/models/__init__.py (create)
- turnzero/models/transformer.py (create)

This is part of a parallel build. Other agents are simultaneously building
turnzero/data/dataset.py and turnzero/eval/. Do not create or modify those.
```

---

## Wave 2 Prompts

### Terminal 2A: Baselines + Evaluation

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126).

Read these files first:
- CLAUDE.md
- docs/WEEK2_PLAN.md (Task 2 section)
- docs/PROJECT_BIBLE.md (section 3 — baseline specs)
- turnzero/data/dataset.py (just created — understand Vocab and VGCDataset)
- turnzero/eval/metrics.py (just created — understand compute_metrics interface)
- turnzero/eval/plots.py (just created — understand plot interfaces)
- turnzero/action_space.py

YOUR TASK: Build baselines, evaluate them, and produce first plots.

1. Create turnzero/models/baselines.py:

   class PopularityBaseline:
       """Global P(action90) from training set."""
       - fit(action90_labels: np.ndarray) — count frequencies, store as (90,) prob vector
       - predict(n: int) → (n, 90) numpy array — broadcast the same prob vector n times
       - Also: core-conditional variant P(action90 | core_cluster_a, core_cluster_b)
         with Laplace smoothing. fit_conditional(action90_labels, cluster_a, cluster_b).
         predict_conditional(cluster_a, cluster_b) → (90,) prob vector.

   class LogisticBaseline:
       """Multinomial logistic regression on one-hot OTS features."""
       - fit(X, y) where X is feature matrix, y is action90 labels
       - predict_proba(X) → (N, 90) prob array
       - Feature extraction: static method featurize(team_a_pokemon, team_b_pokemon)
         that creates sparse one-hot features for species, item, ability, tera, and
         move presence per side (bag of features per team, NOT per slot).
       - Use sklearn LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0)

2. Create a script or add CLI commands to:
   - Load regime_a train/test splits
   - Fit popularity baseline, evaluate with compute_metrics, print results
   - Fit logistic baseline, evaluate, print results
   - Generate reliability_diagram for each baseline
   - Save all results to outputs/baselines/

3. Update turnzero/models/__init__.py ONLY if needed to add baseline imports.
   The transformer model is already defined there by another agent — do not overwrite it.
   If __init__.py already exists, APPEND your imports, don't replace the file.

DO NOT modify:
- turnzero/data/dataset.py
- turnzero/eval/metrics.py or plots.py
- turnzero/models/transformer.py
```

### Terminal 2B: Training Loop + Config

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126, RTX 4080 Super 16GB, BF16).

Read these files first:
- CLAUDE.md (hardware notes especially)
- docs/WEEK2_PLAN.md (Task 5 + Task 6 sections)
- docs/PROJECT_BIBLE.md (section 3 — training recipe)
- turnzero/data/dataset.py (just created — understand build_dataloaders interface)
- turnzero/models/transformer.py (just created — understand OTSTransformer + ModelConfig)
- turnzero/eval/metrics.py (just created — understand compute_metrics interface)
- turnzero/action_space.py

YOUR TASK: Build the training loop and config system, then run a first training.

1. Create configs/transformer_base.yaml:
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
     num_workers: 4
   data:
     regime: a
     split_dir: data/assembled/regime_a

2. Create turnzero/models/train.py:
   - load_config(yaml_path) → dict
   - seed_everything(seed) — torch, numpy, random, CUDA deterministic
   - train_one_epoch(model, loader, optimizer, criterion, scaler, device) → avg_loss
   - validate(model, loader, criterion, device) → (avg_loss, probs, labels_dict)
     labels_dict has action90_true, lead2_true, bring4_observed, is_mirror
   - train(config) — full training loop:
     * Build dataloaders via build_dataloaders()
     * Build model via OTSTransformer(vocab.vocab_sizes, model_cfg)
     * torch.compile(model) for speed
     * AdamW optimizer + CosineAnnealingLR
     * Cross-entropy loss with label_smoothing
     * Mixed precision via torch.amp.GradScaler + autocast(dtype=torch.bfloat16)
     * Early stopping on val NLL (patience from config)
     * Save best checkpoint (model state_dict, vocab_sizes, config, epoch, val_nll)
     * Save training curves (train_loss, val_nll per epoch) as JSON
     * Log run_metadata.json with full config, seed, git hash, timestamp

3. Add CLI commands to turnzero/cli.py:
   - `turnzero train --config configs/transformer_base.yaml --out_dir outputs/runs/run_001`
   - `turnzero eval --model_ckpt outputs/runs/run_001/best.pt --test_split data/assembled/regime_a/test.jsonl --out_dir outputs/eval/run_001`
   Read cli.py first to match the existing click style.

4. After creating the code, run training on regime_a with the base config.
   Training should take roughly 10-30 min on the 4080 Super. Report val NLL and
   test top-1/top-3 accuracy when done.

Hardware utilization checklist:
- torch.compile() on the model
- BF16 autocast (not FP16 — this GPU has native BF16)
- pin_memory=True in dataloaders
- batch_size=512 (bump to 1024 if VRAM allows)
- num_workers=4

DO NOT modify:
- turnzero/data/dataset.py
- turnzero/eval/metrics.py or plots.py
- turnzero/models/transformer.py
```

---

## Wave 3 Prompt

### Terminal 3A: Paper-Ready Comparison Plots

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.

Read these files first:
- CLAUDE.md
- docs/PROJECT_BIBLE.md (section 4.3 — required plots)
- turnzero/eval/plots.py
- turnzero/eval/metrics.py
- Check outputs/baselines/ and outputs/eval/ for saved metric JSON files

YOUR TASK: Generate the Week 2 comparison plots and tables.

1. Load all saved metrics (popularity, logistic, transformer) from outputs/
2. Generate:
   - topk_comparison_bar: grouped bars for all 3 models
   - stratified_table: overall/mirror/non_mirror breakdown as JSON + LaTeX
   - reliability_diagram for the transformer model
   - Training curves plot from the training log JSON
3. Save everything to outputs/plots/week2/
4. Print a summary table to stdout comparing all models

DO NOT modify any source code. This is a read-only analysis task using
existing eval infrastructure.
```

---

## Merge Checklist

After each wave, before starting the next:

1. `git status` — confirm no conflicts (agents touched disjoint files)
2. Quick smoke test:
   ```bash
   source .venv/bin/activate
   python -c "from turnzero.data.dataset import VGCDataset, Vocab, build_dataloaders"
   python -c "from turnzero.eval.metrics import compute_metrics"
   python -c "from turnzero.models.transformer import OTSTransformer, ModelConfig"
   ```
3. Run any existing tests: `pytest tests/`
4. Commit the wave: `git add -A && git commit -m "week2 wave N: ..."`
