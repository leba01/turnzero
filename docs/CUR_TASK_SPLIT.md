# Week 5 Task Split

## Context

Week 4 is complete. All MVP acceptance criteria satisfied. Week 5 is
post-MVP extensions to strengthen the paper's analysis sections.

See `docs/WEEK5_PLAN.md` for full details.

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
└────┬────┘ └──────────┘
     │
     ▼
┌──────────┐
│  Task 2  │
│  Paper   │
│ (Walter) │
└──────────┘
```

## Execution Plan

**No waves / no parallelization.** Both tasks are small, no GPU, no
overlap benefits. Run sequentially as a single agent.

1. **Task 0** — Create + run `scripts/run_cluster_analysis.py`
   - Load `outputs/ensemble/ensemble_predictions.npz`
   - Load `data/assembled/regime_a/test.jsonl` for `core_cluster_a`
   - Group by cluster, compute per-cluster metrics
   - Save JSON + 2 figures to `outputs/plots/paper/`
   - Print summary table

2. **Task 1** — Update `CLAUDE.md` with final Week 4/5 status

3. **Task 2** — Walter writes the paper (not automated)

## Interfaces

Task 0 consumes existing artifacts only — no new contracts needed.

```python
# Input: existing ensemble predictions
ens = np.load("outputs/ensemble/ensemble_predictions.npz")
# Keys: probs (40083, 90), entropy (40083,), mi (40083,),
#        action90_true (40083,), bring4_observed (40083,)

# Input: cluster IDs from test JSONL
examples = list(read_jsonl("data/assembled/regime_a/test.jsonl"))
cluster_ids = [ex["split_keys"]["core_cluster_a"] for ex in examples]

# Output: per-cluster metrics
# outputs/eval/cluster_analysis.json
{
    "cluster_0": {
        "n_examples": 234,
        "n_tier1": 189,
        "top1_action90": 0.085,
        "top3_action90": 0.201,
        "mean_entropy": 3.92,
        "mean_mi": 0.054,
        "mean_confidence": 0.071,
        "top_species": ["Incineroar", "Rillaboom", "Flutter Mane", ...],
    },
    ...
}
```
