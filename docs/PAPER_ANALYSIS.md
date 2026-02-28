# TurnZero — Paper Analysis Reference

Numbers and findings backing the paper. The paper itself is `paper/turnzero.tex`.

---

## Core Results (Regime A, Tier 1, n=32,328)

| Model | Params | Top-1 | Top-3 | NLL | ECE |
|-------|--------|-------|-------|-----|-----|
| Random | — | 1.1% | 3.3% | 4.500 | — |
| Popularity | 0 | 1.3% | 3.9% | 4.497 | 0.001 |
| Logistic | ~4K | 4.0% | 10.1% | 4.580 | 0.059 |
| Transformer | 1.16M | 5.5% | 14.0% | 4.105 | 0.016 |
| **Ensemble (5)** | **5.8M** | **6.4%** | **15.5%** | **4.031** | **0.012** |

Bootstrap 95% CIs: top-1 [2.6, 6.6], top-3 [6.8, 16.0], NLL [4.01, 4.43], ECE [0.002, 0.013]

## Top-k Coverage

k=10: 36.2% · k=17: 50% (vs k=45 random) · k=36: 75% · k=54: 90%

## Marginal Decomposition (Tier 1 only)

| Sub-decision | Classes | Top-1 | Top-3 |
|-------------|---------|-------|-------|
| Action-90 | 90 | 6.4% | 15.5% |
| Bring-4 | 15 | 17.8% | 43.7% |
| Lead-2 | 15 | 18.7% | 41.8% |
| Lead arrangement \| correct bring-4 | 6 | 32.1% | — |

## UQ Numbers

- Adaptive ECE: 0.012 (equal-mass binning, Nixon et al. 2019)
- AURC: 0.890 (top-1), 0.761 (top-3); E-AURC: 0.452 (top-1), 0.404 (top-3)
- Temperature scaling: T = 1.158 (dropped — near-identity)
- OOD AUROC: 0.80 (MI-based), 0.70 (entropy-based)
- OOD abstention: 20% → 46%; entropy shift +0.168 nats
- OOD calibration: ECE 0.012 → 0.076
- Regime B accuracy (non-abstained): 11.7% top-1 (selection effect)

## Ensemble Agreement

| Members agreeing | % of data | Top-1 | Top-3 |
|-----------------|-----------|-------|-------|
| All 5 | 5.8% | 16.6% | 29.7% |
| 4 of 5 | 19.4% | 10.0% | 22.2% |
| 3 of 5 | 26.9% | 6.3% | 16.6% |
| 2 of 5 | 29.3% | 4.2% | 11.4% |
| All differ | 18.7% | 3.1% | 9.0% |

## Confidence Distribution

Mean: 5.3% · Median: 4.5% · Max: 38.1% · >10%: 6.0% · >20%: 0.8%

## Feature Importance (Stress Test)

| Masked | Top-1 | Top-3 | NLL |
|--------|-------|-------|-----|
| None | 6.4 | 15.5 | 4.031 |
| Items | 4.5 | 12.2 | 4.188 |
| Tera type | 5.1 | 12.2 | 4.177 |
| 2/4 moves | 3.1 | 8.1 | 4.388 |
| All moves | 1.0 | 3.2 | 4.714 |
| All moves + items | 1.2 | 3.7 | 4.636 |
| All except species | 1.2 | 3.4 | 4.672 |

## Mirror vs Non-Mirror

| Stratum | Top-1 | Top-3 | N | % |
|---------|-------|-------|---|---|
| Mirror | 6.8% | 16.4% | 28,443 | 87.9% |
| Non-mirror | 3.8% | 8.9% | 3,885 | 12.1% |

## Per-Team Predictability (153 teams, n≥20 Tier 1)

- Entropy range: [3.10, 4.41], mean 4.034 ± 0.212
- Entropy vs top-3: r = -0.561; entropy vs top-1: r = -0.353
- Mode frequency vs accuracy: r = 0.55
- Speed control delta: 0.016 nats (not significant)
- 22/153 teams have 0% top-1; 47/153 exceed 10%

## BO3 Adaptation

- 53K linkable sets, 136K game-to-game transitions
- Lead changes: 59% overall, 72% after loss, 46% after win (χ²=10,126)
- Bring-4 changes: 52%

## Label Quality Ablation

| Loss mode | Top-1 | Top-3 | NLL |
|-----------|-------|-------|-----|
| action90_all (baseline) | 6.4% | 15.5% | 4.022 |
| multitask (marginalize Tier 2) | 6.5% | 15.8% | 4.017 |
| tier1_only | 5.9% | 15.1% | 4.047 |

## Hierarchical Ablation

| Metric | Flat | Hierarchical | Delta |
|--------|------|-------------|-------|
| Top-1 | 6.4% | 6.3% | -0.1pp |
| Top-3 | 15.5% | 15.5% | 0.0pp |
| NLL | 4.022 | 4.023 | +0.001 |
| Params | 1.16M | 1.56M | +34% |

## Related Work Summary

| Aspect | VGC-Bench | EliteFurretAI | Carli 2025 | TurnZero |
|--------|-----------|---------------|------------|----------|
| Target | Full-game RL | Full-game + preview | Lead-2 only | 90-way joint |
| Preview eval | Not isolated | 79% post-fix | ~5K, no metrics | Dedicated |
| UQ | None | None | None | Standard (ensemble, calibration, OOD) |
| Position invariance | N/A | Augmented (post-fix) | N/A | By construction |
| Problem characterization | No | No | No | Per-team predictability, multi-modality ceiling |

## Dataset Stats

- 212K battles → 425K directed → 382K after dedup
- 7,826 clusters via union-find (≥4/6 overlap)
- Regime A: 247K / 35K / 40K
- ~80% Tier 1 (bring-4 observed)
- 87.9% mirror matches in test
- All 90 actions in every split
