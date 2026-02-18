<p align="center">
  <picture>
    <img alt="TurnZero" src="https://img.shields.io/badge/TurnZero-CS229-b91c1c?style=for-the-badge&labelColor=1a1a2e" height="36"/>
  </picture>
</p>

<h3 align="center">A Turn-Zero OTS Coach for Pokemon VGC Gen 9</h3>

<p align="center">
  <em>Predicting expert team preview decisions with calibrated uncertainty</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/pytorch-2.1+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/tests-180_passing-2ea043?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"/>
</p>

---

## What is this?

In competitive Pokemon VGC, both players reveal their full 6-mon team sheets before each game. The first decision — **which 4 to bring and which 2 to lead** — happens before any moves are played. This is the "team preview" or "turn zero" problem.

**TurnZero** learns this decision from 212K tournament games. Given two Open Team Sheets, it predicts the expert's joint **(lead-2, back-2) plan** as one of **90 possible actions**, returns calibrated probabilities, and knows when to say "I don't know."

```
┌─────────────────────────────────────────────────────────┐
│  Your Team (OTS)         vs.       Opponent Team (OTS)  │
│   Pokemon  Item  Ability  Tera  Moves                   │
│  ×6 each side                                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  TurnZero Ensemble (5 members)                          │
│  ├── Calibrated 90-way distribution                     │
│  ├── Confidence check (abstain if uncertain)            │
│  └── Top-3 plans with explanations                      │
│                                                         │
│  Output                                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │  Plan 1 (12.3%): Lead Rillaboom + Flutter   │        │
│  │                   Bring Incineroar, Urshifu │        │
│  │  Plan 2  (9.1%): Lead Incineroar + Flutter  │        │
│  │  Plan 3  (7.8%): Lead Rillaboom + Urshifu   │        │
│  │                                              │        │
│  │  + Role annotations, sensitivity, retrieval  │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

## Key results

| Metric | Ensemble | Popularity baseline | Random |
|:---|:---:|:---:|:---:|
| **Action-90 Top-1** | 6.4% | 1.3% | 1.1% |
| **Action-90 Top-3** | 15.5% | 3.9% | 3.3% |
| **Lead-2 Top-1** | 19.8% | 6.0% | 6.7% |
| **NLL** | 4.031 | 4.497 | 4.500 |
| **ECE** | 0.011 | 0.065 | — |

> **The accuracy story, reframed:** The model's top-17 predictions cover the expert's actual choice 50% of the time (random needs 45). At 90% coverage, you need 54 of 90 actions — the model concentrates probability mass where it belongs.

- **AURC**: 0.890 (top-1), 0.761 (top-3) — selective prediction works
- **OOD detection**: Regime B (unseen team families) shows +0.06 entropy shift; the model knows what it hasn't seen
- **Robustness**: Moves carry the signal; hiding items/ability/tera costs ~0.5% top-3

## Repository structure

```
turnzero/
├── turnzero/                   # Main package
│   ├── action_space.py         # 90-way bijection: C(6,4)×C(4,2)
│   ├── schemas.py              # Pokemon, TeamSheet, MatchExample dataclasses
│   ├── cli.py                  # Click CLI (10 commands)
│   │
│   ├── data/                   # Data pipeline
│   │   ├── parser.py           #   |showteam| protocol extraction
│   │   ├── canonicalize.py     #   Name normalization, sort, dedup
│   │   ├── assemble.py         #   Attach splits + clusters → per-split JSONL
│   │   ├── dataset.py          #   PyTorch Dataset + Vocab
│   │   ├── stats.py            #   Integrity validation + dataset report
│   │   └── io_utils.py         #   JSONL streaming I/O
│   │
│   ├── splits/                 # Leakage-safe splitting
│   │   ├── cluster.py          #   Union-find core clustering (≥4/6 species)
│   │   └── split.py            #   Regime A (within-core) + Regime B (OOD)
│   │
│   ├── models/                 # Model zoo
│   │   ├── baselines.py        #   Popularity + multinomial logistic
│   │   ├── transformer.py      #   Permutation-equivariant set transformer
│   │   └── train.py            #   Training loop (AdamW, mixed precision, compile)
│   │
│   ├── uq/                     # Uncertainty quantification
│   │   ├── ensemble.py         #   Deep ensemble inference + entropy/MI
│   │   └── temperature.py      #   Post-hoc temperature scaling (val only)
│   │
│   ├── eval/                   # Evaluation + paper figures
│   │   ├── metrics.py          #   NLL, Brier, ECE, top-k, stratified
│   │   ├── plots.py            #   Reliability diagrams, model comparison
│   │   ├── bootstrap.py        #   Cluster-aware bootstrap CIs (B=1000)
│   │   ├── risk_coverage.py    #   AURC + abstention operating points
│   │   └── robustness.py       #   Feature masking stress test
│   │
│   └── tool/                   # Coach demo
│       ├── coach.py            #   Full demo pipeline (top-k + abstain)
│       ├── explain.py          #   Marginals, sensitivity analysis
│       ├── lexicon.py          #   OTS role annotations (speed control, etc.)
│       └── retrieval.py        #   Cosine similarity over 246K train embeddings
│
├── configs/                    # Model configs (YAML)
│   ├── transformer_base.yaml   #   d=128, L=4, H=4, 1.16M params
│   └── ensemble/               #   Per-member seed configs
│
├── scripts/                    # Standalone analysis scripts
│   ├── eval_baselines.py       #   Run both baselines, generate comparison plots
│   ├── train_ensemble.sh       #   Train all 5 ensemble members
│   ├── build_retrieval_index.py
│   ├── run_stress_test.py      #   7-level feature masking ablation
│   ├── run_final_figures.py    #   All 10 paper figures
│   ├── run_cluster_analysis.py #   Per-team entropy vs accuracy
│   └── run_supplementary_analysis.py  # Top-k curve, decomposition, speed control
│
├── tests/                      # 180 tests
│   ├── test_action_space.py    #   Bijection correctness
│   ├── test_parser.py          #   Showdown protocol edge cases
│   ├── test_canonicalize.py    #   Name normalization + dedup
│   ├── test_cluster.py         #   Union-find + connected components
│   ├── test_split.py           #   Leakage prevention assertions
│   ├── test_assemble.py        #   End-to-end pipeline integrity
│   ├── test_schemas.py         #   Schema validation
│   └── test_stats.py           #   Dataset report correctness
│
├── data/                       # Data artifacts (not in git)
│   ├── raw/                    #   HuggingFace JSON downloads
│   ├── parsed/                 #   Per-format JSONL
│   ├── canonical/              #   Deduplicated + normalized
│   ├── clusters/               #   cluster_assignments.json
│   ├── splits/                 #   splits.json
│   └── assembled/              #   Final per-split train/val/test JSONL + vocab
│       ├── regime_a/
│       └── regime_b/
│
├── outputs/                    # Model outputs (not in git)
│   ├── runs/                   #   Checkpoints (run_001, ensemble_001..005)
│   ├── ensemble/               #   Merged predictions (40K × 90 probs)
│   ├── calibration/            #   temperature.json
│   ├── retrieval/              #   246K embedding index
│   ├── baselines/              #   Baseline metrics + plots
│   ├── eval/                   #   All evaluation JSONs
│   └── plots/paper/            #   10 publication-quality figures (PDF + PNG)
│
└── docs/                       # Documentation
    ├── PROJECT_BIBLE.md        #   Original spec (v4)
    ├── TECHNICAL_COMPANION.md  #   Study guide: every decision explained
    ├── PAPER_ANALYSIS.md       #   Story arc + numbers reference for paper
    └── WEEK{2-5}_PLAN.md       #   Weekly build plans
```

## Quickstart

### Setup

```bash
# Clone
git clone https://github.com/leba01/turnzero.git
cd turnzero

# Create environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch pandas scikit-learn matplotlib pyyaml
```

### Run the full pipeline

Each stage reads the previous stage's output. Run them in order:

```bash
# 1. Download raw battle logs from HuggingFace
#    (manual: place JSON files in data/raw/)

# 2. Parse Showdown protocol → directed match examples
turnzero parse --raw_path data/raw/logs-gen9vgc2025reggbo3.json \
               --out_dir data/parsed/gen9vgc2025reggbo3

# 3. Canonicalize names, sort, dedup
turnzero canonicalize --in_path data/parsed/match_examples.jsonl \
                      --out_dir data/canonical

# 4. Core-cluster teams (≥4/6 species overlap, union-find)
turnzero cluster --in_path data/canonical/match_examples.jsonl \
                 --out_dir data/clusters

# 5. Generate train/val/test splits (Regime A + B)
turnzero split --in_path data/canonical/match_examples.jsonl \
               --clusters data/clusters/cluster_assignments.json \
               --out_dir data/splits

# 6. Assemble per-split JSONL with cluster IDs
turnzero assemble --canonical_path data/canonical/match_examples.jsonl \
                  --clusters data/clusters/cluster_assignments.json \
                  --splits data/splits/splits.json \
                  --out_dir data/assembled

# 7. Validate integrity
turnzero stats --data_dir data/assembled --validate
```

### Train

```bash
# Single model
turnzero train --config configs/transformer_base.yaml \
               --out_dir outputs/runs/run_001

# Full ensemble (5 members, different seeds)
bash scripts/train_ensemble.sh
```

### Evaluate

```bash
# Calibrate on validation set
turnzero calibrate --model_ckpt outputs/runs/ensemble_001/best.pt \
                   --val_split data/assembled/regime_a/val.jsonl \
                   --out_dir outputs/calibration/run_001

# Evaluate on test set
turnzero eval --model_ckpt outputs/runs/run_001/best.pt \
              --test_split data/assembled/regime_a/test.jsonl \
              --out_dir outputs/eval/run_001
```

### Coach demo

```bash
turnzero demo \
  --ensemble_dir outputs/runs \
  --calib outputs/calibration/run_001/temperature.json \
  --team_a "Rillaboom,Flutter Mane,Incineroar,Urshifu,Farigiraf,Ogerpon" \
  --team_b "Tornadus,Rillaboom,Incineroar,Urshifu-Rapid-Strike,Flutter Mane,Landorus" \
  --index_path outputs/retrieval/train_index
```

The demo outputs top-k plans with calibrated probabilities, role annotations, feature sensitivity ("your plan depends heavily on opponent having Trick Room"), and retrieval evidence from similar historical matchups. If confidence is below threshold, it abstains and switches to scouting report mode.

### Tests

```bash
pytest                # 180 tests, ~2s
```

## The model

A **permutation-equivariant set transformer** (1.16M params) over 12 Pokemon tokens:

```
Input: 6 mons (Team A) + 6 mons (Team B) = 12 tokens
  Each token: E_species + E_item + E_ability + E_tera + Σ E_moves + E_side

  → 4 layers self-attention (no positional encoding)
  → Mean pool
  → MLP head → 90 logits → softmax
```

**Position invariance by design**: canonical sort + mean pooling means the model cannot memorize Pokemon order. This avoids the [positional leakage trap](https://github.com/hspokemon/EliteFurretAI) that inflated other systems' reported accuracy from ~79% to 99.9%.

**Deep ensemble** (5 members): independent random inits, averaged probabilities. Gives predictive entropy + mutual information for free.

**Temperature scaling**: T=1.158 on validation — near identity, confirming the ensemble is already well-calibrated. We apply it anyway because it's free.

## The data

| | |
|:---|:---|
| **Source** | [cameronangliss/vgc-battle-logs](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs) (HuggingFace) |
| **Format** | Regulation G (Gen 9 VGC), BO3 tournaments + ladder |
| **Raw battles** | 212,783 |
| **Directed examples** | 382,393 (after dedup) |
| **Unique teams** | 116,903 → 7,826 core clusters |
| **OTS completeness** | 100% (via `\|showteam\|` protocol) |
| **Bring-4 observed** | ~80% (Tier 1 subset for action-90 eval) |

**Split design** — two regimes:
- **Regime A** *(main)*: Hold out Team A variants within each core cluster, let opponents float. Matches deployment: "I know my team; I face the open field."
- **Regime B** *(OOD)*: Hold out entire core clusters. Tests whether the model knows what it hasn't seen.

## Paper figures

All figures are generated by `scripts/run_final_figures.py` and live in `outputs/plots/paper/`.

| Figure | What it shows |
|:---|:---|
| `model_comparison` | Ensemble vs baselines across all metrics |
| `reliability_diagram` | ECE = 0.011 — well-calibrated |
| `risk_coverage_top1` | Selective prediction: abstain when uncertain |
| `risk_coverage_top3` | AURC = 0.761 for "expert in top-3" |
| `uncertainty_decomposition` | Entropy + MI: mirror vs non-mirror |
| `stress_test` | Moves dominate; hiding items/tera costs little |
| `ood_comparison` | Regime A vs B: entropy shift on unseen families |
| `topk_accuracy_curve` | k=17 for 50% coverage vs k=45 random |
| `cluster_entropy_vs_accuracy` | r = -0.561: linear teams are predictable |
| `cluster_entropy_histogram` | Distribution of team predictability |

## Documentation

| Doc | Purpose |
|:---|:---|
| [`PROJECT_BIBLE.md`](docs/PROJECT_BIBLE.md) | Original spec — schemas, contracts, acceptance criteria |
| [`TECHNICAL_COMPANION.md`](docs/TECHNICAL_COMPANION.md) | Study guide — every decision in Q&A format |
| [`PAPER_ANALYSIS.md`](docs/PAPER_ANALYSIS.md) | Paper narrative — key findings, related work, numbers |

## Citation

This project builds on the VGC-Bench dataset:

```bibtex
@inproceedings{angliss2026vgcbench,
  title={VGC-Bench: Evaluating and Advancing LLMs as Pokemon VGC Battling Agents},
  author={Angliss, Cameron and Luo, James and Wei, Xinpeng and Togelius, Julian},
  booktitle={AAMAS},
  year={2026}
}
```

## License

MIT

---

<p align="center">
  <sub>Stanford CS229 — Winter 2025</sub>
</p>
