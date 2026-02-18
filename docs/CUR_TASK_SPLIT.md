# Week 4 Parallel Task Split

## Dependency Graph

```
          ┌────────────────┐
          │  Week 3 Done:  │
          │ ensemble+calib,│
          │ risk-coverage, │
          │ bootstrap CIs, │
          │ demo skeleton  │
          └───────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│  1A    │  │   1B     │  │   1C     │
│ Stress │  │ Retrieval│  │ Explain  │
│  Test  │  │  Index   │  │ Modules  │
│  Code  │  │  Code    │  │  Code    │
└───┬────┘  └────┬─────┘  └────┬─────┘
    │             │             │
    ▼             ▼             │
┌────────┐  ┌──────────┐       │
│  2A    │  │   2B     │       │
│Run Stress│ │Build Index│      │
│ Test    │  │(embed    │       │
│(~30min) │  │ train)   │       │
└───┬────┘  └────┬─────┘       │
    │             │             │
    └─────────────┼─────────────┘
                  │
            ┌─────┴─────┐
            │     3     │
            │ Integrate │
            │ + Figures  │
            │ + Paper   │
            │  Draft    │
            └───────────┘
```

**Wave 1** — three fully independent agents (code only, no GPU):
- **1A** creates `turnzero/eval/robustness.py` + `scripts/run_stress_test.py`
- **1B** creates `turnzero/tool/retrieval.py` (embedding extraction + index + evidence)
- **1C** creates `turnzero/tool/explain.py` + `turnzero/tool/lexicon.py` (marginals, role lexicon, feature sensitivity)

**Wave 2** — GPU-bound, can overlap:
- **2A** runs moves-hidden stress test (ensemble × 6 masking levels, ~30 min GPU)
- **2B** extracts train set embeddings + builds retrieval index (~5 min GPU)

**Wave 3** — integration + figures + paper (sequential, single agent):
- **3** integrates retrieval + explanations into demo tool, generates final paper figures, then writes paper draft

---

## Shared Interfaces (ALL agents must respect these)

### Existing interfaces (from Weeks 2-3 — do NOT change)

```python
# Model loading
ckpt = torch.load("best.pt", map_location=device, weights_only=False)
model = OTSTransformer(ckpt["vocab_sizes"], ModelConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model_state_dict"])

# Forward pass → logits (NOT probs)
logits = model(team_a, team_b)  # (B, 90) float

# Eval harness
from turnzero.eval.metrics import compute_metrics

# Ensemble prediction
from turnzero.uq.ensemble import ensemble_predict, load_ensemble_predictions

# Temperature scaling
from turnzero.uq.temperature import TemperatureScaler

# Dataset: team tensors are (6, 8) LongTensor
# Fields: [species=0, item=1, ability=2, tera=3, move0=4, move1=5, move2=6, move3=7]
# UNK = index 0 for all field types

# Demo tool
from turnzero.tool.coach import run_demo
```

### New interfaces for Week 4

#### Masking contract (1A defines, 2A consumes)

```python
# turnzero/eval/robustness.py

MASK_CONFIGS = {
    "none":            [],                    # baseline — no masking
    "moves_2":         [4, 5],                # hide 2 random moves per mon
    "moves_4":         [4, 5, 6, 7],          # hide all moves
    "items":           [1],                    # hide items
    "tera":            [3],                    # hide tera types
    "moves_items":     [1, 4, 5, 6, 7],       # hide moves + items
    "all_but_species": [1, 2, 3, 4, 5, 6, 7], # species only
}

def mask_batch(
    batch: dict,
    mask_config: str,
    rng: np.random.Generator | None = None,
) -> dict:
    """Clone batch and zero out specified field columns in team tensors.

    For 'moves_2', randomly select 2 of 4 move columns per mon using rng.
    For all others, zero out the specified columns entirely.
    """
```

Save as: `outputs/eval/stress_test.json`
```json
{
    "none": {"overall/top1_action90": 0.064, ...},
    "moves_2": {...},
    "moves_4": {...},
    ...
}
```

#### Embedding extraction contract (1B defines, 2B produces)

```python
# turnzero/tool/retrieval.py

def extract_embeddings(
    model: OTSTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Extract pooled (pre-head) representations.

    Replicates model forward up to mean-pool, skips self.head.

    Returns:
        "embeddings": (N, d_model) float32
        "action90_true": (N,) int
        "lead2_true": (N,) int
        "bring4_observed": (N,) bool
        "is_mirror": (N,) bool
    """
```

Save as: `outputs/retrieval/train_embeddings.npz`

#### Retrieval index contract (1B defines, 3A consumes)

```python
class RetrievalIndex:
    """Brute-force cosine similarity index over training embeddings."""

    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        ...

    def query(
        self,
        query_embedding: np.ndarray,  # (d_model,) single query
        k: int = 20,
    ) -> list[dict]:
        """Return top-k nearest neighbors with similarity scores."""

    def evidence_summary(self, neighbors: list[dict]) -> dict:
        """Aggregate action frequencies, top lead pairs, top bring mons."""

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalIndex": ...
```

#### Explanations contract (1C defines, 3A consumes)

```python
# turnzero/tool/explain.py

def compute_marginals(probs_90: np.ndarray) -> dict:
    """From (90,) or (N, 90) probs, extract per-mon marginals.

    Returns:
        "lead_probs": (6,) — P(mon i is in lead pair)
        "bring_probs": (6,) — P(mon i is in bring-4)
        "lead_pair_probs": (15,) — P(lead pair j)
    """

def feature_sensitivity(
    models: list,
    team_a: torch.Tensor,     # (1, 6, 8)
    team_b: torch.Tensor,     # (1, 6, 8)
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    """Mask each field type on team_b, measure KL from baseline.

    Returns: {"species": kl, "items": kl, "moves": kl, "ability": kl, "tera": kl}
    """
```

```python
# turnzero/tool/lexicon.py

ROLE_LEXICON: dict[str, set[str]]  # tag → set of moves/items/abilities

def annotate_team(team_dict: dict) -> list[dict]:
    """Return per-mon role annotations.

    Returns: [{"species": "Incineroar", "roles": ["fake_out", "intimidate", ...]}, ...]
    """
```

---

## Wave 1 Prompts

### Terminal 1A: Moves-Hidden Stress Test

```
You are working on the TurnZero project in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126).

Read these files first:
- CLAUDE.md
- docs/WEEK4_PLAN.md (Task 0)
- docs/PROJECT_BIBLE.md (Section 6.3 — moves-hidden stress test)
- turnzero/data/dataset.py (understand _encode_team, field layout)
- turnzero/uq/ensemble.py (ensemble_predict, _collect_probs)
- turnzero/eval/metrics.py (compute_metrics)
- turnzero/eval/plots.py (_save_fig, style conventions)

YOUR TASK: Create the stress test infrastructure.

1. Create turnzero/eval/robustness.py:

   MASK_CONFIGS dict mapping config names to field column indices.

   mask_batch(batch, mask_config, rng) → batch with specified fields zeroed out.
   - Input batch has "team_a" and "team_b" keys, each (B, 6, 8) LongTensor
   - Clone the tensors before modifying (don't mutate originals)
   - For "moves_2": randomly pick 2 of columns [4,5,6,7] per mon per example
   - For all others: zero out all specified columns
   - Zero = UNK index = 0

   run_stress_test(ckpt_paths, loader, device, temperature, mask_configs)
     → dict[str, dict] mapping mask config name to compute_metrics result
   - For each mask config: modify batches before forward pass, collect probs,
     compute metrics on the modified predictions
   - This is basically ensemble_predict but with masking injected per batch

   plot_stress_test(results, out_dir)
   - Line plot: x-axis = masking level (ordered by severity),
     y-axis = top-1, top-3, top-5 accuracy (action90)
   - Second plot: NLL + mean confidence vs masking level
   - Use _save_fig pattern from plots.py

2. Create scripts/run_stress_test.py:
   - Load ensemble checkpoints + temperature
   - Build test DataLoader
   - Run stress test across all masking configs
   - Save results JSON + plots
   - Print summary table

DO NOT touch any files other than:
- turnzero/eval/robustness.py (create)
- scripts/run_stress_test.py (create)
```

### Terminal 1B: Retrieval Index + Evidence

```
You are working on the TurnZero project in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126).

Read these files first:
- CLAUDE.md
- docs/WEEK4_PLAN.md (Task 1)
- docs/PROJECT_BIBLE.md (Section 5.1 — retrieval-based evidence)
- turnzero/models/transformer.py (understand forward pass, mean-pool location)
- turnzero/uq/ensemble.py (_load_model_from_ckpt, _collect_probs pattern)
- turnzero/data/dataset.py (VGCDataset, _encode_team)
- turnzero/data/io_utils.py (read_jsonl)
- turnzero/action_space.py (ACTION_TABLE, action90_to_lead_back)

YOUR TASK: Create the retrieval index module.

1. Create turnzero/tool/retrieval.py:

   extract_embeddings(model, loader, device) → dict
   - Run the forward pass but stop after mean-pool, before self.head
   - Use a forward hook on model.encoder to capture the output, then
     mean-pool manually. Or replicate the model's forward logic up to pool.
   - Return: embeddings (N, d_model), plus action90_true, lead2_true, etc.

   class RetrievalIndex:
     __init__(self, embeddings, metadata)
     - L2-normalize embeddings for cosine similarity
     - metadata: list of dicts with action90_id, lead2_idx, species_a, species_b, etc.

     query(self, query_embedding, k=20) → list[dict]
     - Cosine similarity via matrix multiply (brute force)
     - Return top-k neighbors with similarity score + metadata

     evidence_summary(self, neighbors) → dict
     - Count action90 frequencies in neighbors → top actions
     - Count lead pair frequencies → top lead pairs
     - Count per-mon bring frequency → which mons appear most
     - Return structured summary dict

     save(self, path) / load(cls, path) — np.savez / np.load

   Implementation notes:
   - The model uses mean-pool: pooled = encoder_output.mean(dim=1)
   - d_model = 128 (from config), so embeddings are 128-dim
   - Train set is ~246K examples. 246K × 128 × 4 bytes ≈ 126 MB — fits in RAM
   - Need metadata per example: action90_id, species names (for display)
   - Load species names from the raw JSONL examples in parallel

DO NOT touch any files other than:
- turnzero/tool/retrieval.py (create)
```

### Terminal 1C: Explanation Modules

```
You are working on the TurnZero project in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12).

Read these files first:
- CLAUDE.md
- docs/WEEK4_PLAN.md (Task 2)
- docs/PROJECT_BIBLE.md (Sections 5.2, 5.4, 5.5)
- turnzero/action_space.py (ACTION_TABLE — maps action90 to (lead2, back2))
- turnzero/eval/metrics.py (_marginalize_to_lead2, ACTION90_TO_LEAD2)
- turnzero/tool/coach.py (understand the demo output format)

YOUR TASK: Create explanation modules.

1. Create turnzero/tool/lexicon.py:

   ROLE_LEXICON: dict mapping role tags to sets of moves/items/abilities.
   Build a comprehensive lexicon for Gen 9 VGC:

   Roles to cover:
   - speed_control: Tailwind, Trick Room, Icy Wind, Electroweb, Scary Face, etc.
   - redirection: Follow Me, Rage Powder
   - fake_out: Fake Out
   - priority: Extreme Speed, Sucker Punch, Aqua Jet, Grassy Glide, etc.
   - spread: Heat Wave, Muddy Water, Rock Slide, Dazzling Gleam, Earthquake, etc.
   - protect: Protect, Detect, Wide Guard, Quick Guard
   - disruption: Taunt, Encore, Will-O-Wisp, Thunder Wave, Spore, Sleep Powder, etc.
   - weather_setter: Rain Dance, Sunny Day, Sandstorm, etc.
   - terrain_setter: Electric Terrain, Grassy Terrain, Psychic Terrain, Misty Terrain
   - setup: Swords Dance, Nasty Plot, Calm Mind, Dragon Dance, etc.
   - recovery: Recover, Roost, Moonlight, Synthesis, etc.
   - choice_item: Choice Scarf, Choice Band, Choice Specs
   - sash: Focus Sash
   - berry: Sitrus Berry, Lum Berry

   Also include ability-based roles:
   - intimidate: Intimidate
   - weather_ability: Drizzle, Drought, Sand Stream, Snow Warning
   - terrain_ability: Electric Surge, Grassy Surge, Psychic Surge, Misty Surge

   REVERSE_LEXICON: dict mapping individual move/item/ability → set of role tags

   annotate_team(team_dict) → list of per-mon annotation dicts
   - For each mon, collect all role tags from its moves, item, and ability
   - Return: [{"species": ..., "roles": [sorted list of tags]}, ...]

2. Create turnzero/tool/explain.py:

   compute_marginals(probs_90) → dict
   - Input: (90,) single example or (N, 90) batch
   - Build a (90, 6) binary matrix: LEAD_MASK[a, i] = 1 if mon i is in leads of action a
   - Build a (90, 6) binary matrix: BRING_MASK[a, i] = 1 if mon i is in bring-4 of action a
   - lead_probs = probs @ LEAD_MASK → (6,) or (N, 6)
   - bring_probs = probs @ BRING_MASK → (6,) or (N, 6)
   - Also return the 15-way lead pair probs (reuse _marginalize_to_lead2 logic)

   format_marginals(marginals, species_names) → str
   - Pretty-print: "P(Incineroar leads) = 62%, P(Rillaboom leads) = 48%, ..."
   - Sort by probability descending

   feature_sensitivity(models, team_a_tensor, team_b_tensor, temperature, device) → dict
   - Baseline: run ensemble forward → p_base (90,)
   - For each field group (species=[0], items=[1], ability=[2], tera=[3], moves=[4,5,6,7]):
     * Clone team_b, zero out those columns
     * Run ensemble forward → p_masked (90,)
     * Compute KL(p_base || p_masked) = sum(p_base * log(p_base / p_masked))
   - Return: {"species": kl, "items": kl, "ability": kl, "tera": kl, "moves": kl}
   - Higher KL = prediction depends more on that field type

DO NOT touch any files other than:
- turnzero/tool/lexicon.py (create)
- turnzero/tool/explain.py (create)
```

---

## Wave 2 Prompts

### Terminal 2A: Run Stress Test

```
You are working on the TurnZero project in /home/walter/CS229/turnzero.

Read CLAUDE.md and docs/WEEK4_PLAN.md first.

YOUR TASK: Run the moves-hidden stress test.

1. Run: .venv/bin/python scripts/run_stress_test.py

2. Verify outputs:
   - outputs/eval/stress_test.json exists with metrics for all masking levels
   - outputs/plots/week4/stress_test_degradation.{png,pdf}
   - outputs/plots/week4/stress_test_confidence.{png,pdf}

3. Print the summary table and verify:
   - Graceful degradation (not cliff) as more fields are hidden
   - Species-only should still beat random (1.1% top-1)
   - NLL should increase monotonically with masking severity

DO NOT modify any source code. This is a run + analysis task.
```

### Terminal 2B: Build Retrieval Index

```
You are working on the TurnZero project in /home/walter/CS229/turnzero.

Read CLAUDE.md and docs/WEEK4_PLAN.md first.

YOUR TASK: Extract training embeddings and build retrieval index.

1. Write a small runner script (scripts/build_retrieval_index.py) that:
   - Loads ensemble member 1 (or the single model)
   - Extracts pooled embeddings for all training examples
   - Also loads species names from train.jsonl for metadata
   - Builds a RetrievalIndex and saves it

2. Quick sanity: query a few test examples and verify:
   - Nearest neighbors have high species overlap
   - Evidence summary shows non-trivial action concentration

DO NOT modify any source code other than creating the runner script.
```

---

## Wave 3 Prompt

### Terminal 3: Integration + Figures + Paper

```
You are working on the TurnZero project in /home/walter/CS229/turnzero.

Read CLAUDE.md, docs/WEEK4_PLAN.md, and check all artifacts in outputs/.

YOUR TASK: Three sequential steps — do them in order.

Step 1 — Demo Integration:
- Update turnzero/tool/coach.py to import and use:
  * RetrievalIndex (query + evidence_summary)
  * compute_marginals + format_marginals
  * annotate_team (role lexicon)
  * feature_sensitivity
- Add --index_path option for pre-built retrieval index
- Full demo output should match the format in WEEK4_PLAN.md Task 3
- Smoke-test the updated demo before moving on

Step 2 — Final Paper Figures:
- Create scripts/run_final_figures.py
- Generate definitive paper figures to outputs/plots/paper/
- Include stress test plot, clean reliability, model comparison,
  risk-coverage, OOD comparison, uncertainty decomposition
- Run the script and verify all figures are generated

Step 3 — Paper Draft:
- Write the CS229 final project report for TurnZero
- Read all docs/ files and all outputs/ JSON artifacts
- Follow the outline in WEEK4_PLAN.md Task 5
- Save to docs/PAPER_DRAFT.md
- Include all concrete numbers from the evaluation artifacts
- Reference figures by their output paths
- Target length: ~6-8 pages (CS229 format)

Files you may create or modify:
- turnzero/tool/coach.py (update)
- turnzero/cli.py (update if needed for new demo options)
- scripts/run_final_figures.py (create)
- docs/PAPER_DRAFT.md (create)
```

---

## Merge Checklist

After each wave, before starting the next:

1. `git status` — confirm no conflicts
2. Quick smoke tests:
   ```bash
   # After Wave 1:
   python -c "from turnzero.eval.robustness import mask_batch, MASK_CONFIGS"
   python -c "from turnzero.tool.retrieval import RetrievalIndex, extract_embeddings"
   python -c "from turnzero.tool.explain import compute_marginals, feature_sensitivity"
   python -c "from turnzero.tool.lexicon import ROLE_LEXICON, annotate_team"

   # After Wave 2:
   ls outputs/eval/stress_test.json
   ls outputs/retrieval/train_embeddings.npz
   ls outputs/plots/week4/*.png

   # After Wave 3:
   turnzero demo --ensemble_dir outputs/runs \
     --calib outputs/calibration/run_001/temperature.json \
     --team_a "Incineroar,Rillaboom,Flutter Mane,Urshifu,Tornadus,Landorus" \
     --team_b "Calyrex,Amoonguss,Rillaboom,Incineroar,Entei,Iron Hands"
   ls outputs/plots/paper/*.png
   ```
3. Run existing tests: `pytest tests/`
4. Commit the wave

## Compute Budget

| Task | GPU Time | Notes |
|------|----------|-------|
| Stress test (6 configs × 5 members) | ~30 min | Ensemble inference × 6 |
| Embedding extraction (train set) | ~5 min | Single model, 246K examples |
| Feature sensitivity (per demo query) | ~10 sec | 5 field groups × 5 members |
| Final figures | ~1 min | Matplotlib only |
| **Total GPU** | **~40 min** | |

All fits easily on the RTX 4080S. **Bottleneck is the stress test (Wave 2A).**
Start 2B (embeddings) immediately while 2A runs.
