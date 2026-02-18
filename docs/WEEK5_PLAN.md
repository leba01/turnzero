# Week 5 Plan: Extensions Beyond MVP

## Where We Stand

Weeks 1-4 delivered the complete system: data pipeline → model → UQ stack →
explanations → robustness stress test → demo tool → paper figures. All Bible
acceptance criteria are satisfied. This is post-MVP work to strengthen the
paper's analysis and discussion sections.

**Current artifact status:**
- `outputs/ensemble/ensemble_predictions.npz` — per-example probs, entropy, MI, labels (40K test)
- `outputs/eval/stress_test.json` — 7 masking levels with full metrics
- `outputs/eval/ood_comparison.json` — Regime A vs B
- `outputs/eval/bootstrap_cis.json` — cluster-aware CIs (B=1000)
- `outputs/eval/risk_coverage.json` — AURC + operating points
- `outputs/retrieval/train_index.{npz,meta.json}` — 246K train embeddings
- `outputs/plots/paper/` — 7 publication-quality figures (14 files)

**What's missing for the paper:**
- No per-cluster breakdown (aggregate numbers mask enormous heterogeneity)
- No "team linearity" analysis (which teams are predictable vs flexible?)
- CLAUDE.md not updated with Week 4 final status
- Paper draft (owner: Walter)

## Task Breakdown

### Task 0: Per-Cluster Analysis + Team Linearity [~1 hour, no GPU]

**Create `scripts/run_cluster_analysis.py`**

This is the main extension. Group test predictions by `core_cluster_a` and
compute per-cluster metrics to reveal which team archetypes are predictable
and which aren't.

**Implementation:**

1. Load ensemble predictions from `outputs/ensemble/ensemble_predictions.npz`
   (probs, entropy, MI, action90_true, bring4_observed, is_mirror)

2. Load test JSONL to get `split_keys.core_cluster_a` per example.
   Join on index (both are ordered identically — shuffle=False, drop_last=False).

3. For each cluster with n ≥ 20 Tier 1 examples:
   - Top-1 / Top-3 accuracy (action90)
   - Mean entropy (= team linearity score; low entropy → telegraphed team)
   - Mean MI
   - Mean confidence
   - n_examples
   - Species composition (most common species in team_a for this cluster)

4. **Outputs:**
   - `outputs/eval/cluster_analysis.json` — per-cluster metrics dict
   - `outputs/plots/paper/cluster_entropy_vs_accuracy.{png,pdf}` — **money plot:**
     scatter of cluster mean entropy (x) vs cluster top-1 accuracy (y),
     point size = n_examples. Expect negative correlation.
   - `outputs/plots/paper/cluster_entropy_histogram.{png,pdf}` — distribution
     of cluster entropies, annotated with example species for extremes
   - Print to stdout: top-5 most linear teams (lowest entropy) with species
     and accuracy, top-5 most flexible teams (highest entropy), worst-5 and
     best-5 clusters by accuracy

**Acceptance:** The scatter plot shows a visible trend. The extreme clusters
are VGC-domain-plausible (e.g., Trick Room teams are linear, goodstuffs are
flexible).

### Task 1: Update CLAUDE.md [~10 min]

Update with Week 4 final status:
- Mark all Week 4 tasks as done
- Add final numbers
- Note Week 5 extensions

### Task 2: Paper Draft [owner: Walter, not automated]

Walter writes the paper using:
- `docs/PAPER_ANALYSIS.md` — story arc, key findings, numbers reference
- `docs/WEEK5_PLAN.md` — this doc
- `outputs/plots/paper/` — all figures
- `outputs/eval/cluster_analysis.json` — cluster analysis (from Task 0)

## Dependency Graph

```
    ┌──────────────┐
    │ Week 4 Done  │
    └──────┬───────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│  Task 0 │ │  Task 1  │
│ Cluster │ │ Update   │
│ Analysis│ │ CLAUDE.md│
│ (~1 hr) │ │ (~10min) │
└────┬────┘ └──────────┘
     │
     ▼
┌──────────┐
│  Task 2  │
│  Paper   │
│ (Walter) │
└──────────┘
```

**No parallelization needed.** Tasks 0 and 1 are both trivial single-agent
work (no GPU, pure numpy/IO). Task 0 is ~1 hour, Task 1 is ~10 minutes.
They touch completely different files so they *could* run in parallel, but
the overhead of spinning up two agents exceeds the time savings. Just run
them sequentially.

Task 2 depends on Task 0's output (cluster analysis goes in the paper) but
is human-driven.

## Compute Budget

| Task | GPU Time | Wall Time | Notes |
|------|----------|-----------|-------|
| Task 0: Cluster analysis | 0 | ~1 hour | Pure numpy over existing npz |
| Task 1: CLAUDE.md update | 0 | ~10 min | Text edit |
| Task 2: Paper draft | 0 | 3-4 hours | Walter writes |
| **Total** | **0** | **~5 hours** | No GPU needed |

## Files Created / Modified

```
scripts/run_cluster_analysis.py          ← Task 0 (create)
outputs/eval/cluster_analysis.json       ← Task 0 output
outputs/plots/paper/cluster_*.{png,pdf}  ← Task 0 figures
CLAUDE.md                                ← Task 1 (update)
docs/PAPER_DRAFT.md                      ← Task 2 (Walter creates)
```

## End-of-Week 5 Definition of Done

- [ ] Per-cluster analysis JSON + figures generated
- [ ] Scatter plot shows entropy-accuracy relationship
- [ ] Top/bottom cluster examples are VGC-plausible
- [ ] CLAUDE.md reflects final project status
- [ ] Paper draft complete (Walter)
