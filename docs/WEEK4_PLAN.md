# Week 4 Plan: Explanations + Robustness + Demo Polish + Paper

## Where We Stand

Weeks 1-3 delivered the full calibrated prediction pipeline:
- Data: 382K directed examples, 7826 clusters, Regime A/B splits with integrity checks
- Models: Popularity → Logistic → Transformer (1.16M) → 5-member Ensemble
- UQ: Temperature scaling (T=1.16), deep ensembles, risk-coverage, bootstrap CIs, OOD eval
- Demo: CLI coach tool with top-3 plans + abstention

**Current best (Ensemble, Regime A test, Tier 1):**

| Metric | Popularity | Logistic | Single | Ensemble |
|--------|-----------|----------|--------|----------|
| Action90 Top-1 | 1.3% | 4.0% | 5.5% | 6.4% |
| Action90 Top-3 | 3.9% | 10.1% | 14.0% | 15.5% |
| Lead-2 Top-1 | 7.1% | 14.0% | 18.3% | 19.8% |
| Lead-2 Top-3 | 21.8% | 33.8% | 41.0% | 43.2% |
| NLL (Action90) | 4.497 | 4.580 | 4.105 | 4.031 |
| ECE (Action90) | 0.001 | 0.059 | 0.016 | 0.011 |

OOD: entropy +0.17, MI +0.03, abstention 20% → 46%.

**What's still open from the Bible:**

From acceptance criteria (Section 8):
- [ ] Demo tool shows explanations + retrieval evidence
- [ ] Moves-hidden stress test implemented and reported

From Section 5 (Interpretability, all marked "mandatory"):
- [ ] 5.1 Retrieval-based evidence
- [ ] 5.2 Domain-legible cue extraction (OTS role lexicon)
- [ ] 5.3 Threats explanation (opponent lead marginals)
- [ ] 5.4 "Why these leads" (our lead/bring marginals)
- [ ] 5.5 Counterfactual / sensitivity (feature masking)

From Section 6 (Robustness):
- [ ] 6.3 Moves-hidden stress test

Plus: final paper writeup, definitive figure set, CLAUDE.md update.

**Why this week matters:** Weeks 1-3 proved the model works and the UQ is sound. Week 4
makes it *interpretable* and *robust*. Retrieval evidence transforms the demo from "trust
this number" to "here's what experts did in similar matchups." The stress test proves the
model degrades gracefully. Together they complete the story for the paper.

## Task Breakdown

### Task 0: Moves-Hidden Stress Test [~2-3 hours]

**Create `turnzero/eval/robustness.py`**

Test-time ablation: replace OTS fields with UNK tokens and measure performance degradation.
The model already has UNK embeddings (index 0 for every field type), so masking is trivial.

**Masking levels:**
- `moves_2`: randomly hide 2 of 4 moves per mon (partial knowledge)
- `moves_4`: hide all 4 moves per mon (species/item/ability/tera only)
- `items`: hide all items
- `tera`: hide all tera types
- `moves_4+items`: hide moves and items
- `all_but_species`: hide everything except species

**Implementation:**
```python
def mask_batch(batch: dict, mask_config: str, rng: np.random.Generator) -> dict:
    """Apply test-time masking to a batch. Replaces fields with UNK (0).

    Operates on the (B, 6, 8) team tensors. Field layout:
    [species=0, item=1, ability=2, tera=3, move0=4, move1=5, move2=6, move3=7]
    """
```

Run ensemble inference at each masking level, compute full metrics via `compute_metrics`.

**Outputs:**
- `outputs/eval/stress_test.json` — metrics at each masking level
- `outputs/plots/week4/stress_test_degradation.{png,pdf}` — line plot: top-1/3/5 and NLL vs masking level
- `outputs/plots/week4/stress_test_confidence.{png,pdf}` — mean confidence + abstention rate vs masking

**Acceptance:** Graceful degradation (not cliff). Species-only should still beat random.

### Task 1: Retrieval-Based Evidence [~2-3 hours]

**Create `turnzero/tool/retrieval.py`**

Build a retrieval index from the pooled transformer embeddings of training examples.
At query time, find similar matchups and report what experts did.

**Step 1: Extract embeddings**

Hook into the model after mean-pooling but before the classification head.
For each training example, store the 128-dim embedding + action90 label + metadata.

```python
def extract_embeddings(
    model: OTSTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Extract pooled representations (B, d_model) before the head."""
```

Need to modify the forward pass slightly (or use a forward hook) to get the
pooled vector. Use a `register_forward_hook` on the encoder's output, or
just replicate the embedding + encoder + pool steps and stop before `self.head`.

**Step 2: Build index**

For MVP, brute-force cosine similarity is fine (246K train vectors @ 128-dim fits in RAM
easily). Optional: FAISS for speed.

```python
class RetrievalIndex:
    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.metadata = metadata

    def query(self, query_emb: np.ndarray, k: int = 20,
              min_species_overlap: int = 0) -> list[dict]:
        """Return top-k most similar training matchups with evidence summary."""

    def evidence_summary(self, neighbors: list[dict]) -> dict:
        """Aggregate neighbors: action frequency table, top lead pairs, etc."""
```

**Step 3: Integrate into demo tool**

After running ensemble inference, also extract the query embedding, retrieve
neighbors, and print evidence summary.

**Outputs:**
- `outputs/retrieval/train_embeddings.npz` — embeddings + metadata for train set
- Evidence integrated into `turnzero demo` output

**Acceptance:** Retrieval returns matchups with high species overlap. Evidence summary
shows meaningful action concentration ("experts led X+Y in 42% of similar games").

### Task 2: Explanation Modules [~2 hours]

**Create `turnzero/tool/explain.py`**

Three lightweight explanation components:

**2a. Marginal extraction (Bible 5.4)**

From the 90-way distribution, compute:
- `P(mon i is led)` = sum over all actions where mon i is in the lead pair
- `P(mon i is brought)` = sum over all actions where mon i is in the bring set
- `P(lead pair (i,j))` = marginalize to 15-way lead distribution

These are just matrix multiplications against the action table. Already have
`_marginalize_to_lead2` for the 15-way; extend for per-mon marginals.

```python
def compute_marginals(probs_90: np.ndarray) -> dict:
    """From (90,) probs, extract per-mon lead/bring marginals."""
    # Returns: lead_probs (6,), bring_probs (6,), lead_pair_probs (15,)
```

**2b. OTS Role Lexicon (Bible 5.2)**

A deterministic dict mapping moves/items/abilities to domain-relevant tags.

```python
ROLE_LEXICON = {
    "speed_control": {"Tailwind", "Trick Room", "Icy Wind", "Electroweb", "Scary Face"},
    "redirection": {"Follow Me", "Rage Powder"},
    "fake_out": {"Fake Out"},
    "priority": {"Extreme Speed", "Sucker Punch", "Aqua Jet", "Mach Punch"},
    "spread": {"Heat Wave", "Muddy Water", "Rock Slide", "Dazzling Gleam", "Earthquake"},
    "protect": {"Protect", "Detect", "Wide Guard"},
    "disruption": {"Taunt", "Encore", "Will-O-Wisp", "Thunder Wave", "Spore", "Sleep Powder"},
    "weather": {"Rain Dance", "Sunny Day", "Sandstorm", "Snowscape"},
    # ... abilities and items too
}

def annotate_team(team_dict: dict) -> list[dict]:
    """Annotate each mon with role tags derived from its OTS."""
```

**2c. Feature Masking Sensitivity (Bible 5.5, Counterfactual 2)**

For a given matchup, mask one field type at a time on team_b and measure
the delta in predicted probabilities. Surface the biggest movers.

```python
def feature_sensitivity(
    model, team_a, team_b, vocab, temperature, ...
) -> dict[str, float]:
    """Mask each field type on team_b, measure KL-divergence from baseline probs."""
    # Returns: {"moves": 0.12, "items": 0.04, "tera": 0.08, ...}
```

**Outputs:**
- Marginals integrated into demo output: "P(Incineroar leads) = 62%"
- Role annotations on each mon
- Sensitivity report: "Prediction depends most on opponent's moves (+0.12 KL)"

### Task 3: Polished Demo Tool [~1-2 hours]

**Update `turnzero/tool/coach.py`**

Integrate Tasks 1-2 into the demo output. The full demo flow becomes:

```
turnzero demo --ensemble_dir ... --calib ... --team_a ... --team_b ...

═══════════════════════════════════════════════════════════
  TurnZero Coach — Turn-Zero Team Preview Advisor
═══════════════════════════════════════════════════════════

  Your Team (A):  Incineroar, Rillaboom, Flutter Mane, ...
  Opponent (B):   Calyrex, Amoonguss, Rillaboom, ...

  Ensemble: 5 members  |  Temperature: 1.158
  Confidence: 0.042  |  Entropy: 4.30  |  MI: 0.071

  Top-3 Recommended Plans:
  ─────────────────────────────────────────────────────────
  #1  (4.15%)  Lead: Tornadus + Landorus  |  Back: Incineroar + Rillaboom
  #2  (3.00%)  Lead: Urshifu + Landorus   |  Back: Incineroar + Tornadus
  #3  (2.94%)  Lead: Incineroar + Landorus  |  Back: Rillaboom + Tornadus

  Why These Leads:
    P(Landorus leads) = 72%    ← highest lead probability
    P(Tornadus leads) = 51%
    P(Incineroar leads) = 48%

  Key Opponent Cues:
    Calyrex: speed_control (Trick Room), spread, weather
    Amoonguss: redirection (Rage Powder), disruption (Spore)

  Sensitivity: Prediction depends most on opponent moves (+0.12 KL)

  Similar Matchups (k=10 from training set):
    Experts led Tornadus+Landorus in 38% of similar games
    Experts led Incineroar+Landorus in 22%
    Experts brought Rillaboom in 71% of similar games
```

If confidence < tau: replace plans section with scouting mode message + still
show retrieval evidence and role annotations.

**Also accept full OTS JSON for both teams** (not just species-only). The current
demo tool already supports `--team_a_ots` / `--team_b_ots` JSON paths.

### Task 4: Final Paper Figures + Tables [~1-2 hours]

**Create `scripts/run_final_figures.py`**

Generate the definitive figure set for the paper:

1. **Figure 1**: Model comparison bar chart (action-90 top-1/3/5, all 4 models)
   — already have this, just ensure it's the definitive version

2. **Figure 2**: Reliability diagram (ensemble, action-90)
   — clean single-model diagram for paper

3. **Figure 3**: Risk-coverage curves (both risk defs, 4 models)
   — already have, verify paper quality

4. **Figure 4**: Moves-hidden stress test degradation plot (NEW)
   — x-axis: masking level, y-axis: top-1/3 accuracy + NLL

5. **Figure 5**: OOD comparison (entropy shift + abstention shift)
   — already have bars, maybe simplify to 2-panel

6. **Figure 6**: Uncertainty decomposition (entropy vs MI scatter, colored by correctness)
   — NEW: shows what MI buys over raw entropy

7. **Table 1**: Full model comparison with CIs (already have LaTeX)

8. **Table 2**: Stress test results (NEW)

9. **Table 3**: OOD comparison (already have)

All saved to `outputs/plots/paper/` as the definitive set.

### Task 5: Paper Draft / Report [~3-4 hours]

**Structure (typical CS229 final project format):**

1. **Introduction** (~0.5 page)
   - Team preview as multi-modal classification
   - Why UQ matters for this task
   - Contributions: pipeline, model, UQ stack, retrieval evidence

2. **Related Work** (~0.5 page)
   - VGC-Bench (Angliss et al.)
   - Deep Ensembles (Lakshminarayanan et al.)
   - Temperature scaling (Guo et al.)
   - Selective prediction (Geifman & El-Yaniv)

3. **Data** (~1 page)
   - Source: Showdown protocol + |showteam|
   - 90-way action space
   - Label observability (bring4_observed)
   - Core clustering + split regimes
   - Dataset stats

4. **Methods** (~1.5 pages)
   - Model architecture (transformer, set-structured)
   - Baselines
   - UQ: ensembles + temperature scaling
   - Selective prediction + risk-coverage
   - Retrieval evidence

5. **Experiments** (~2 pages)
   - Main results table + bar chart
   - Calibration + reliability
   - Risk-coverage
   - OOD evaluation
   - Moves-hidden stress test
   - Mirror vs non-mirror analysis

6. **Demo** (~0.5 page)
   - Coach tool UX
   - Example outputs

7. **Discussion + Conclusion** (~0.5 page)
   - Multi-modality as the fundamental challenge
   - What UQ buys: knowing when not to recommend
   - Limitations: non-mirror difficulty, wide CIs per cluster
   - Future: conformal prediction, richer counterfactuals

### Task 6: Final Cleanup [~1 hour]

- Update CLAUDE.md with final Week 4 results
- Verify all outputs are reproducible
- Clean any dead code or unused imports
- Final `git status` + commit

## Compute Budget

| Task | GPU Time | Wall Time |
|------|----------|-----------|
| Task 0: Stress test (6 masking levels × ensemble) | ~30 min | 2-3 hours |
| Task 1: Embedding extraction (train set) | ~5 min | 2-3 hours |
| Task 2: Explanations (CPU only) | 0 | 2 hours |
| Task 3: Demo polish | ~2 min | 1-2 hours |
| Task 4: Final figures | ~1 min | 1-2 hours |
| Task 5: Paper draft | 0 | 3-4 hours |
| Task 6: Cleanup | 0 | 1 hour |

Total GPU: ~40 min. Total wall: ~12-17 hours including writing.
Bottleneck is writing (Task 5) and stress test compute (Task 0).

## Priority Triage

If time runs short:

**Non-negotiable (must ship):**
- Task 0 (stress test) — acceptance criterion
- Task 1 (retrieval) — MVP requirement, makes the demo real
- Task 4 (final figures) — need these for the paper
- Task 5 (paper draft) — the actual deliverable

**High priority:**
- Task 2 (explanations) — marginals are easy, lexicon is quick, sensitivity is nice
- Task 3 (demo polish) — ties everything together

**Can defer:**
- Full counterfactual response model (Bible 5.5 counterfactual 1)
- Conformal prediction sets (stretch)
- Threats explanation (Bible 5.3 — needs opponent-side prediction)

## End-of-Week 4 Definition of Done

- [ ] Moves-hidden stress test: 6+ masking levels evaluated, degradation plot produced
- [ ] Retrieval index built on train set; evidence shown in demo output
- [ ] Per-mon lead/bring marginals computed and displayed
- [ ] OTS role lexicon annotates opponent team in demo
- [ ] Feature sensitivity shows which cue types matter most
- [ ] Demo tool integrates all explanation components
- [ ] Definitive paper figure set in `outputs/plots/paper/`
- [ ] Paper draft complete (or structured outline with all results embedded)
- [ ] CLAUDE.md updated with final results and project status
- [ ] All acceptance criteria from Bible Section 8 satisfied

## Module Layout After Week 4

```
turnzero/
  tool/
    __init__.py
    coach.py              ← polished demo (Task 3)
    retrieval.py           ← embedding index + evidence (Task 1)
    explain.py             ← marginals + sensitivity (Task 2)
    lexicon.py             ← OTS role lexicon (Task 2)
  eval/
    robustness.py          ← moves-hidden stress test (Task 0)
scripts/
  run_stress_test.py       ← Task 0 runner
  run_final_figures.py     ← Task 4 runner
outputs/
  retrieval/               ← train embeddings index
  eval/stress_test.json    ← stress test results
  plots/
    week4/                 ← stress test + new plots
    paper/                 ← definitive paper figure set
```
