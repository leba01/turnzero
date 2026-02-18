# CLAUDE.md

## Project: TurnZero

CS229 (Stanford ML) final project — a turn-zero OTS coach for Pokémon VGC Gen 9.
Predicts expert lead-2 + bring-4/back-2 as a joint 90-way action from two Open Team Sheets.

**Current phase**: Week 4 — Explanations + Robustness + Demo Polish + Paper

**Key references**:
- [docs/PROJECT_BIBLE.md](docs/PROJECT_BIBLE.md) — full spec (v4), schemas, contracts, acceptance criteria.
- [docs/WEEK4_PLAN.md](docs/WEEK4_PLAN.md) — detailed Week 4 build plan.
- [docs/CUR_TASK_SPLIT.md](docs/CUR_TASK_SPLIT.md) — current parallel task breakdown.
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

## Week 3 Progress

UQ stack + calibration + selective prediction. This is what turns "classifier" into
"calibrated decision system." See [docs/WEEK3_PLAN.md](docs/WEEK3_PLAN.md) and
[docs/CUR_TASK_SPLIT.md](docs/CUR_TASK_SPLIT.md) for the full task breakdown.

- [ ] Task 0: Temperature scaling (fit T on val, save artifact)
- [ ] Task 1: Deep ensembles (train 5 members, seeds 42/137/256/512/777)
- [ ] Task 2: Risk-coverage curves (both risk definitions)
- [ ] Task 3: Cluster-aware bootstrap CIs (B=1000)
- [ ] Task 4: Regime B (OOD) evaluation
- [ ] Task 5: Comprehensive Week 3 plots + tables
- [ ] Task 6: Demo tool skeleton (stretch)

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

## Module Layout (planned)

- `turnzero/data/` — parser (`|showteam|` extraction), canonicalization, schemas, assembly, stats
- `turnzero/splits/` — clustering (union-find), split generation, validators
- `turnzero/models/` — baselines (frequency, logistic), Transformer set model, heads
- `turnzero/uq/` — ensembles, temperature scaling, abstention
- `turnzero/eval/` — metrics, plots, bootstrap CIs, stratified reporting
- `turnzero/tool/` — CLI coach demo
