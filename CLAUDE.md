# CLAUDE.md

## Project: TurnZero

CS229 (Stanford ML) final project — a turn-zero OTS coach for Pokémon VGC Gen 9.
Predicts expert lead-2 + bring-4/back-2 as a joint 90-way action from two Open Team Sheets.

**Current phase**: Week 5 — Post-MVP extensions (paper analysis). All MVP criteria satisfied.

**Key references**:
- [docs/PROJECT_BIBLE.md](docs/PROJECT_BIBLE.md) — full spec (v4), schemas, contracts, acceptance criteria.
- [docs/PAPER_ANALYSIS.md](docs/PAPER_ANALYSIS.md) — story arc, key findings, numbers reference for paper.
- [docs/WEEK5_PLAN.md](docs/WEEK5_PLAN.md) — Week 5 extensions plan.
- [docs/CUR_TASK_SPLIT.md](docs/CUR_TASK_SPLIT.md) — current task breakdown.
- [docs/WEEK4_PLAN.md](docs/WEEK4_PLAN.md) — Week 4 build plan (completed).
- [docs/WEEK3_PLAN.md](docs/WEEK3_PLAN.md) — Week 3 build plan (completed).
- [docs/WEEK2_PLAN.md](docs/WEEK2_PLAN.md) — Week 2 build plan (completed).

## Week 1 Status (DONE)

Pipeline stages 1-7 complete. `dataset_report.json` passes all integrity checks.

## Task 0 Status (DONE)

Full dataset downloaded (BO3 2024 + BO3 2025 + Ladder 2024) and pipeline re-run.
- **Raw data**: 212,783 battles from 3 files (394KB + 1.37GB + 165MB)
- **Parsed**: 425,566 directed examples → 382,393 after dedup
- **Clusters**: 116,903 unique teams → 7,826 clusters
- **Regime A**: 246,762 train / 34,735 val / 40,083 test (321,580 total)
- **Regime B**: 346,798 train / 5,713 val / 14,708 test (367,219 total)
- ~80% bring4_observed, ~96.8% OTS completeness, all 90 actions in every split
- Integrity validation: **PASSED**

## Week 2 Status (DONE)

Full train-eval loop: Dataset → Baselines → Transformer → Training → Eval → Plots.
Single transformer (d=128, L=4, H=4, 1.16M params) beats both baselines.

**Best numbers (Regime A test, overall):**
- Action-90 Top-1/3/5: 5.5% / 14.0% / 20.7% (vs 1.3% / 3.9% / 6.3% popularity)
- Lead-2 Top-1/3: 18.3% / 41.0%
- NLL: 4.105 (vs 4.497 popularity, 4.580 logistic)
- ECE: 0.016 (well-calibrated for single model)

## Week 3 Status (DONE)

UQ stack complete: temperature scaling (T=1.158), deep ensembles (5 members),
risk-coverage curves, cluster-aware bootstrap CIs (B=1000), OOD evaluation (Regime B).

**Ensemble numbers (Regime A test, Tier 1):**
- Action-90 Top-1/3/5: 6.4% / 15.5% / 22.6% (5.8x random)
- Lead-2 Top-1/3: 19.8% / 43.2%
- NLL: 4.031, ECE: 0.011 (well-calibrated)
- AURC: 0.890 (top-1), 0.761 (top-3) — ensemble beats single on selective prediction

## Week 4 Status (DONE)

Explanations + robustness + demo polish. All MVP acceptance criteria satisfied.

- Stress test: 7 masking levels, graceful degradation (moves dominate signal)
- Retrieval index: 246K train embeddings (128d), brute-force cosine similarity
- Coach tool: marginals, role lexicon, feature sensitivity, retrieval evidence
- 7 publication-quality figures (14 files) in `outputs/plots/paper/`
- 180/180 tests passing

## Week 5 Status (IN PROGRESS)

Post-MVP extensions for paper analysis. See [docs/WEEK5_PLAN.md](docs/WEEK5_PLAN.md).

- [x] Task 0: Per-team analysis — 153 teams (species-6 grouping), entropy-accuracy
      correlation r = -0.561 (top-3). Commander teams 50% top-1, goodstuffs 0%.
      Figures: `cluster_entropy_vs_accuracy`, `cluster_entropy_histogram`.
- [x] Task 1: Update CLAUDE.md with final status
- [ ] Task 2: Paper draft (Walter)

### Multi-task loss ablation (code complete, training pending)

- **Problem**: ~20% of training examples (Tier 2, `bring4_observed=False`) have fabricated
  action-90 labels (back-2 filled with deterministic lowest-index heuristic). Model trains
  on these as ground truth.
- **Solution**: Multi-task loss — Tier 1: action-90 CE; Tier 2: lead-2 CE via marginalization
  `softmax(logits) @ margin_matrix` (90→15 probs, NOT in logit space).
- **Three loss modes** in `train.py`: `action90_all` (baseline), `multitask`, `tier1_only`.
- **Temperature scaling dropped** from final pipeline (T≈1.0, ensemble already calibrated).
- **Dead code cleaned up**: consolidated `_ece`, `_save_fig`, `_MARGIN_MATRIX`,
  `load_from_checkpoint` to single sources; removed unused imports.
- **15 ablation configs** ready: `configs/ablation_{a,b,c}/member_{001..005}.yaml`
- **Training**: `bash scripts/train_ablations.sh` (15 runs, ~2-3h on RTX 4080 Super)
- **Evaluation**: `python scripts/eval_ablations.py [--bootstrap]`
- **179/179 tests passing** after all changes.

## Repository Info

- **Remote**: https://github.com/leba01/turnzero.git
- **Branch**: main
- **Venv**: `.venv/` (Python 3.12, created with `--without-pip` + bootstrapped pip)

## Tech Stack

Python 3.12 · PyTorch 2.10+cu126 · pandas · scikit-learn · matplotlib · click (CLI)

## Hardware

- **CPU**: AMD Ryzen 7 7800X3D
- **GPU**: NVIDIA RTX 4080 Super (17.2 GB VRAM, BF16 supported)
- **Platform**: WSL2 on Windows, CUDA 13.0 driver
- Use `torch.compile()`, mixed precision (BF16), large batch sizes, `pin_memory=True`

## Canonical Schemas (summary)

- **Pokemon**: `{species, item, ability, tera_type, moves[4]}` — unknown fields use `"UNK"`.
- **TeamSheet**: 6 Pokemon + `team_id` (hash of sorted canonical keys) + `species_key` (sorted species hash for clustering) + `reconstruction_quality` metadata.
- **MatchExample** (directed, player-A perspective): `{team_a: TeamSheet, team_b: TeamSheet, label: {lead2_idx, back2_idx, action90_id}, label_quality: {bring4_observed}}` + split keys (`core_cluster_a/b`, `is_mirror`) + `match_group_id`.

## Data Reality

`|showteam|` lines provide **full OTS** (species, item, ability, tera, all 4 moves) for 100% of games. No cross-game reconstruction needed. Names arrive in CamelCase (`FakeOut`, `AssaultVest`) and are normalized during canonicalization.

## Pipeline Stages (strict evaluation order)

1. **download** — fetch raw JSON from HuggingFace `cameronangliss/vgc-battle-logs`; write `manifest.json`.
2. **parse** — extract full OTS from `|showteam|`, leads, bring-4 (`bring4_observed` flag) from Showdown protocol; emit two directed MatchExamples per game as JSONL.
3. **canonicalize** — CamelCase → display names, sort moves, sort mons by canonical key, compute stable hashes, dedup exact `(team_a, team_b, action90, format)` quadruples.
4. **cluster** — core-cluster teams (≥4/6 species overlap → connected components via union-find).
5. **split** — assign train/val/test under Regime A (hold out team_a variants within cluster) and Regime B (hold out entire clusters). Cross-split triple dedup + `match_group_id` integrity.
6. **assemble** — attach cluster IDs + split assignments, write per-split JSONL files.
7. **stats** — comprehensive dataset report + end-to-end integrity validation.
8. **train / calibrate / eval / demo** — model training, temp-scaling on val, paper-grade metrics, CLI coach tool.

## Key Integrity Rules

- Lead-2 is the primary supervised task (always fully observable).
- Bring-4/action90 metrics reported only on Tier 1 subset (`bring4_observed == True`).
- Same matchup with different expert actions is NOT a duplicate — preserves multi-modality.
- All `match_group_id` rows stay in one split; no `(team_a, team_b, action90)` triples cross splits.
- Mirror vs non-mirror stratification on all test metrics.

## Module Layout

- `turnzero/data/` — parser (`|showteam|` extraction), canonicalization, schemas, assembly, stats
- `turnzero/splits/` — clustering (union-find), split generation, validators
- `turnzero/models/` — baselines (frequency, logistic), Transformer set model, heads
- `turnzero/uq/` — ensembles, temperature scaling, abstention
- `turnzero/eval/` — metrics, plots, bootstrap CIs, stratified reporting
- `turnzero/tool/` — CLI coach demo (marginals, role lexicon, sensitivity, retrieval)

## Key Output Artifacts

- `outputs/ensemble/ensemble_predictions.npz` — 40K test: probs (90), entropy, MI, labels
- `outputs/eval/stress_test.json` — 7 masking levels with full metrics
- `outputs/eval/ood_comparison.json` — Regime A vs B
- `outputs/eval/bootstrap_cis.json` — cluster-aware CIs (B=1000)
- `outputs/eval/risk_coverage.json` — AURC + operating points
- `outputs/eval/cluster_analysis.json` — per-team metrics (153 teams)
- `outputs/retrieval/train_index.{npz,meta.json}` — 246K train embeddings
- `outputs/plots/paper/` — 9 publication-quality figures (18 files)
