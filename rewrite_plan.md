# TurnZero Rewrite Plan: "Win-Stay, Lose-Explore"

**Instructions:** Read this file, then execute all steps in order. Commit after each major step. Stop and report when you reach a GPU-required step (marked with [GPU]).

## Project Location & Conventions
- **Project root:** The turnzero repo (wherever this file is)
- **Python:** Use the project's `.venv` with `PYTHONPATH=.`
- **Raw data:** `data/raw/logs-gen9vgc2025reggbo3.json`
- **Assembled data:** `data/assembled/regime_a/{train,val,test}.jsonl`
- **Existing outputs:** `outputs/eval/`, `outputs/plots/paper/`
- **Configs:** `configs/sequential/member_00{1..5}.yaml`
- **Commit style:** conventional commits (feat:/fix:/docs:)
- **Never amend commits**
- **Test command:** `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`

---

## New Thesis

In best-of-three competition, prior game outcomes create an information asymmetry: winners repeat leads (46% change rate), losers scramble (72% change rate). A sequential prediction model quantifies this: conditioning on game-1 outcome recovers +3.6pp on games 2-3, but the gain comes entirely from post-win predictions. Post-loss is genuinely irreducible.

## New Title

"Win-Stay, Lose-Explore: Asymmetric Adaptation in Best-of-Three Competitive Play"

---

## Step 1: Adaptation Effectiveness Analysis [NO GPU]

**Goal:** Determine if changing leads after a loss actually helps win game 2.

**File:** `scripts/run_adaptation_effectiveness.py`

**What to build:**
1. Extend the `Transition` dataclass from `scripts/run_bo3_adaptation.py` (or create a new one) to include `g2_winner_is_this_side: bool | None`
2. The `build_transitions()` function already has both `g1` and `g2` `GameInfo` objects in its inner loop (lines 298-344 of `run_bo3_adaptation.py`). The `GameInfo` dataclass already has `winner_side`. So just check `g2.winner_side == side`.
3. Compute a 2x2 table:

```
                    | Won G2    | Lost G2
-----------------------------------------
Lost G1 + Changed   |  count    |  count
Lost G1 + Stayed    |  count    |  count
Won G1 + Changed    |  count    |  count
Won G1 + Stayed     |  count    |  count
```

4. Chi-squared test + Wilson CIs on each cell's win rate
5. Also compute: among changers, did they switch **toward** opponent's g1 leads or **away**? (Use `g1.leads[opp_side]` vs `g2.leads[side]` overlap)
6. Save to `outputs/eval/adaptation_effectiveness.json`

**Key code pattern** — reuse from `run_bo3_adaptation.py`:
```python
from scripts.run_bo3_adaptation import (
    extract_bo3_linkage, parse_game_info, _wilson_ci,
    GameInfo, RAW_PATH, OUT_EVAL,
)
```

**`|player|` line format** for Elo (Step 3): `|player|p1|Name|avatar|rating` — rating is `parts[4]` if `len(parts) >= 5`.

**After running:** Print the 2x2 table. The result determines the paper's framing:
- Changers win more → "adaptation works but is unpredictable"
- Same rate → "adaptation is noise"
- Changers win less → "adaptation backfires under pressure"

---

## Step 2: Lead Transition Matrix Analysis [NO GPU]

**File:** `scripts/run_transition_analysis.py`

**What to build:**
1. For each transition, record `(g1_lead_pair_sorted, g2_lead_pair_sorted, prior_result)` where lead_pair_sorted is a canonical string like `"Calyrex-Ice|Incineroar"`
2. Find the top 15 most common lead pairs across all games
3. Build 15x15 transition matrices conditioned on outcome (won vs lost)
4. Compute conditional entropy: `H(g2_lead | g1_lead, outcome)` for won vs lost
5. Generate heatmap visualizations (2 panels: after-win, after-loss)
6. Save to `outputs/eval/transition_analysis.json`

**Lead pair extraction:** `GameInfo.leads[side]` is a `frozenset[str]` of 2 species. Sort and join with `|` for a canonical key.

**Expected finding:** After-win matrix should be diagonally concentrated (repeat). After-loss matrix should be dispersed (explore). Entropy for lost should be notably higher.

**Figure output:** `outputs/plots/paper/transition_heatmaps.{png,pdf}`

---

## Step 3: Elo-Stratified Adaptation Analysis [NO GPU]

**File:** `scripts/run_elo_analysis.py`

**What to build:**
1. Extract Elo ratings from raw logs. In `parse_game_info()`, the `|player|` line has format `|player|p1|Name|avatar|rating`. Rating is `parts[4]` when `len(parts) >= 5`. Modify `GameInfo` (or create a subclass) to store `ratings: dict[str, int]` (e.g., `{"p1": 1500, "p2": 1600}`).
2. Stratify transitions by Elo tier (e.g., terciles or quartiles of the rating distribution)
3. For each tier, compute: lead change rate after win, after loss, and the win-rate-if-changed vs stayed (from Step 1)
4. Test: do stronger players show more/less WSLS? More effective adaptation?
5. Save to `outputs/eval/elo_adaptation.json`

**Key risk:** Some `|player|` lines may not have a rating field (unrated games). Filter those out.

---

## Step 4: Review Results & Decide Framing [NO GPU]

After Steps 1-3, read the JSON outputs and decide:
- Which framing fits the adaptation effectiveness result?
- Is Elo stratification interesting enough for the paper?
- Print a summary of all findings.

Commit all analysis scripts and results.

---

## Step 5: Opponent-Context Sequential Model [GPU]

**Goal:** Test if opponent's g1 revealed species helps predict g2 leads (especially post-loss).

### 5a: Extend SequentialOTSTransformer

**File:** `turnzero/models/sequential_transformer.py`

Add opponent revealed species as additional context tokens:

```python
class OpponentContextConfig(SequentialConfig):
    n_opp_slots: int = 4  # max opponent revealed species

class OpponentContextTransformer(SequentialOTSTransformer):
    def __init__(self, vocab_sizes, cfg):
        super().__init__(vocab_sizes, cfg)
        # Embedding for opponent's revealed species (reuse species vocab)
        self.emb_opp_revealed = nn.Embedding(vocab_sizes["species"], cfg.d_model)
        # Learned "slot" embedding to distinguish opp slots from team tokens
        self.emb_opp_slot = nn.Embedding(cfg.n_opp_slots, cfg.d_model)
        nn.init.normal_(self.emb_opp_slot.weight, mean=0.0, std=0.02)

    def forward(self, team_a, team_b,
                prior_lead2_idx=None, prior_result=None, game_num=None,
                opp_revealed=None):
        # opp_revealed: (B, n_opp_slots) LongTensor, 0=padding
        # Build context token same as parent
        # Then add up to n_opp_slots tokens from opponent's revealed species
        # Prepend [ctx_token, opp_tok_1, ..., opp_tok_k, team_tokens...]
        ...
```

### 5b: Extend SequentialVGCDataset

**File:** `turnzero/data/sequential_dataset.py`

The `bo3_context.py` `build_context_lookup()` function already stores `"species6"` per side in the context table. But we need opponent's *revealed* species from g1, not full team. Two options:
- **Quick:** Use `GameInfo.revealed[opp_side]` — but `build_context_lookup()` currently only stores `prior_leads` and `prior_result`, not `revealed`. Extend it.
- Store `opp_revealed` in context table, then resolve species indices via vocab in the dataset.

### 5c: Create config

**File:** `configs/opponent_context/member_001.yaml` (through 005)

```yaml
model:
  arch: opponent_context
  d_model: 128
  n_layers: 4
  n_heads: 4
  d_ff: 512
  dropout: 0.1
  pool: mean
  n_opp_slots: 4

training:
  batch_size: 512
  lr: 3.0e-4
  weight_decay: 0.01
  label_smoothing: 0.03
  max_epochs: 100
  patience: 15
  seed: 42  # vary per member
  num_workers: 4

data:
  regime: a
  split_dir: data/assembled/regime_a
  context_path: data/assembled/regime_a/bo3_context_v2.json  # with opp revealed
```

### 5d: Wire into train.py

Add `elif arch == "opponent_context":` branch in `train.py` (around line 344). Follow the pattern of the `sequential` branch.

### 5e: Train + Evaluate

Train 5 members, evaluate stratified by outcome. Key question: does post-loss accuracy improve when model sees what opponent revealed in g1?

---

## Step 6: Separate Post-Win / Post-Loss Models [GPU]

**Goal:** Train sequential models on only post-win or only post-loss subsets to see if the architecture can find signal in pure post-loss data.

Create a filtered dataset wrapper or add a `filter_prior_result` option to `SequentialVGCDataset`. Train 5 members on each subset.

---

## Step 7: Generate All Figures [AFTER GPU]

**File:** `scripts/run_rewrite_figures.py`

Figures needed:
1. **Figure 1:** Lead change rate bar chart by condition (from `bo3_adaptation.json`) — already exists in `run_bo3_adaptation.py`, but regenerate with updated style
2. **Figure 2:** Transition heatmaps (from Step 2 results)
3. **Figure 3 (optional):** Per-team entropy vs accuracy scatter (data from existing `outputs/eval/`)
4. **Figure 4:** Base vs Sequential by game_num x prior_result (from `sequential_results.json` — already exists in `run_sequential_model.py`)
5. **NEW Figure:** Adaptation effectiveness bar chart (from Step 1)

Use `turnzero/eval/plots.py` helpers: `COLORS`, `_save_fig`, `setup_plotting`.

---

## Step 8: Write Paper [AFTER ALL RESULTS]

**File:** `paper/turnzero.tex` — full rewrite

### Structure (target: 7 pages + references)

**Title:** "Win-Stay, Lose-Explore: Asymmetric Adaptation in Best-of-Three Competitive Play"

- **S1 Introduction** (~1 page): Hook with adaptation asymmetry, ML as measurement instrument, key numbers
- **S2 Related Work** (~0.75 page): Win-stay/lose-shift (Nowak & Sigmund 1993), serial dependence (Walker & Wooders 2001), hot hand, esports drafting (BPCoach, JueWuDraft), Pokemon AI (VGC-Bench, Metamon)
- **S3 Data & Setup** (~0.75 page): 382K examples, 90-way action space, 53K linkable BO3 sets, split design
- **S4 The Adaptation Asymmetry** (~1.75 pages) — CORE:
  - S4.1 Winners Stay, Losers Shift: 72% vs 46% change rate, chi-squared
  - S4.2 Does Adaptation Work? (Step 1 results)
  - S4.3 Where Do Losers Go? (Step 2 transition matrices + entropy)
  - S4.4 Uniform Across Team Types (entropy tercile null, speed control null)
- **S5 Measuring via Sequential Prediction** (~1.5 pages):
  - S5.1 Base Model: convention not strategy (6.4% top-1, opponent ablation -1.3pp)
  - S5.2 Sequential Model: +3.55pp on G2-3, all from post-win
  - S5.3 Opponent Context Model Results (Step 5)
- **S6 Discussion** (~0.75 page): Why the asymmetry, limitations, beyond Pokemon
- **References**

### What gets cut from current paper
- Addendum (A1-A5) entirely
- JSD distributional evaluation table
- Reliability diagram (mention ECE in text)
- Top-k coverage curve (one sentence about k=17)
- Detailed calibration discussion
- Ensemble agreement figure (mention 16.6% vs 3.1% in text)

### What stays (condensed)
- Per-team predictability (commander 50% vs flexible 0%)
- Opponent ablation (-1.3pp)
- Capacity ceiling argument (text-only)

---

## Execution Order

1. **[NO GPU]** Step 1: `scripts/run_adaptation_effectiveness.py` — write, run, commit
2. **[NO GPU]** Step 2: `scripts/run_transition_analysis.py` — write, run, commit
3. **[NO GPU]** Step 3: `scripts/run_elo_analysis.py` — write, run, commit
4. **[NO GPU]** Step 4: Review results, print summary, commit
5. **[GPU]** Step 5: Opponent-context model — modify code, write configs, train
6. **[GPU]** Step 6: Post-win/post-loss separate models — train
7. **[AFTER GPU]** Step 7: Generate all figures
8. **[AFTER GPU]** Step 8: Write paper
9. **[AFTER GPU]** Compile paper (`pdflatex turnzero && bibtex turnzero && pdflatex turnzero && pdflatex turnzero`)

---

## Key File Reference

| File | Purpose |
|------|---------|
| `scripts/run_bo3_adaptation.py` | Existing BO3 adaptation analysis — has `GameInfo`, `Transition`, `extract_bo3_linkage()`, `parse_game_info()`, `build_transitions()` |
| `turnzero/data/bo3_context.py` | BO3 context builder — has `build_context_lookup()`, `_resolve_lead_pair_idx()` |
| `turnzero/models/sequential_transformer.py` | Sequential model — `SequentialOTSTransformer` with 3 context embeddings |
| `turnzero/models/transformer.py` | Base model — `OTSTransformer`, `ModelConfig` |
| `turnzero/models/train.py` | Training loop — handles `sequential` arch, auto-detects `prior_lead2_idx` in batch |
| `turnzero/data/sequential_dataset.py` | Sequential dataset — `SequentialVGCDataset`, `build_sequential_dataloaders()` |
| `turnzero/eval/plots.py` | Plotting helpers — `COLORS`, `_save_fig()`, `setup_plotting()` |
| `scripts/run_sequential_model.py` | Full sequential pipeline — context build + train + eval + figures |
| `configs/sequential/member_001.yaml` | Sequential config template |
| `paper/turnzero.tex` | Current paper (to be rewritten) |

## Key Data Structures

```python
# GameInfo (from run_bo3_adaptation.py)
@dataclass
class GameInfo:
    battle_id: str
    set_id: str
    game_num: int
    player_names: dict[str, str]     # {"p1": "Name", "p2": "Name"}
    winner_side: str | None          # "p1" or "p2" or None
    leads: dict[str, frozenset[str]] # {"p1": frozenset(species), ...}
    revealed: dict[str, frozenset[str]]
    species6: dict[str, frozenset[str]]
    species_key: dict[str, str]

# |player| line format: |player|p1|Name|avatar|rating
# parts[4] = rating (int), may not exist for unrated
```

## Critical Numbers from Existing Results
- 382K directed examples from 212K BO3 battles
- 53K linkable BO3 sets, ~136K game-to-game transitions
- Overall lead change rate: 59%
- After loss: 72% change, After win: 46% change
- Chi-squared: 10,126, p < 10^-6
- Base ensemble: 6.4% top-1, 15.5% top-3
- Sequential: +3.55pp on Games 2-3
- After wins: +6-8pp; after losses: <= 1pp
- Opponent ablation: -1.3pp
- Commander teams: 50% top-1; flexible teams: 0%
- Capacity ceiling: +34% params, same accuracy
- Entropy tercile null: similar change rates across terciles
- Speed control null: delta-H = 0.016 nats
