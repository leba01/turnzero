# Project Bible — OTS Team Preview Coach, Pokémon VGC Gen 9

## Executive Summary

Turn-zero coach: given two Open Team Sheets (6v6 with species/item/ability/tera/moves), predict the **human expert preview decision** — **lead-2 + bring-4/back-2** as a joint 90-way action.

Pipeline: raw Showdown logs → parsed OTS (via `|showteam|`) → canonical dataset → leakage-safe cluster splits → models → calibrated probabilities → evaluation → coach demo.

Key differentiator: designed around **proper scoring rules**, **calibration**, **confidence intervals**, and **selective prediction** — not just top-1 accuracy.

## Data Source

- **Dataset**: `cameronangliss/vgc-battle-logs` (HuggingFace), VGC-Bench (Angliss et al.)
- **Volume**: 212K battles → 425K directed → 382K after dedup
- **Format**: Raw Showdown protocol, `|showteam|` provides 100% OTS coverage
- **Labels**: Lead-2 always observable; bring-4 observable in ~80% (Tier 1)

## Canonical Schemas

```yaml
Pokemon:
  species: string
  item: string | "UNK"
  ability: string | "UNK"
  tera_type: string | "UNK"
  moves: [string | "UNK"] × 4  # sorted alphabetically, UNK last

TeamSheet:
  team_id: string              # hash of canonicalized content
  species_key: string          # hash of sorted species set
  reconstruction_quality: {fields_known, fields_total, source_method}
  pokemon: [Pokemon] × 6      # sorted by canonical key

MatchExample:
  example_id: string
  match_group_id: string       # leakage-control group
  team_a: TeamSheet
  team_b: TeamSheet
  label: {lead2_idx, back2_idx, action90_id}
  label_quality: {bring4_observed: bool}
```

## Pipeline Stages

1. **download** → raw JSON from HuggingFace
2. **parse** → extract OTS from `|showteam|`, leads from first `|switch|`, bring-4 from all switches
3. **canonicalize** → name normalization, move/team sort, stable hashing, dedup
4. **cluster** → species-6 overlap ≥4/6, union-find connected components (7,826 clusters)
5. **split** → Regime A (team-A variants held out within cluster) + Regime B (entire clusters held out)
6. **assemble** → per-split JSONL files
7. **stats** → integrity validation
8. **train/calibrate/eval/demo**

## Key Design Rules

- **Dedup**: on `(team_a_id, team_b_id, action90_id, format)` — same matchup + different actions is NOT a duplicate
- **Split leakage**: no `(team_a, team_b, action90)` triples cross splits; all `match_group_id` rows stay in one split
- **Regime A**: hold out Team A variants; opponents unrestricted (matches deployment)
- **Regime B**: hold out entire clusters for OOD evaluation
- **Tier 1/2**: action-90 metrics on Tier 1 only; lead-2 on all examples
- **Mirror stratification**: report all test metrics split by mirror vs non-mirror

## Model Architecture

**Transformer set model**: 12 Pokemon tokens (6+6), L=4 layers, d=128, H=4 heads, 1.16M params.
- Token: `E_species + E_item + E_ability + E_tera + Σ E_move + E_side`
- Mean pool → MLP → 90 logits → softmax
- Position-invariant by design (canonical sort + mean pool)

**Ensemble**: 5 independently trained members, averaged softmax.

## UQ Stack

- **Deep ensembles** (5 members): predictive entropy, mutual information, confidence
- **Temperature scaling**: T=1.158 (near-identity, dropped — ensemble already calibrated)
- **Selective prediction**: risk-coverage curves, AURC, operating points
- **OOD detection**: MI-based AUROC 0.80; abstention doubles on Regime B
- **Bootstrap CIs**: cluster-aware, B=1000

## Results Summary

| Model | Top-1 | Top-3 | NLL | ECE |
|-------|-------|-------|-----|-----|
| Random | 1.1% | 3.3% | 4.500 | — |
| Popularity | 1.3% | 3.9% | 4.497 | 0.001 |
| Logistic | 4.0% | 10.1% | 4.580 | 0.059 |
| Transformer | 5.5% | 14.0% | 4.105 | 0.016 |
| **Ensemble (5)** | **6.4%** | **15.5%** | **4.031** | **0.012** |

See `docs/PAPER_ANALYSIS.md` for full numbers.

## Interview Talk Track

- **Pitch**: Turn-zero OTS coach that recommends lead+bring plans with calibrated confidence and evidence from similar games.
- **Data**: `|showteam|` gives 100% OTS; bring-4 labels ~80% observable, flagged per-example. Designed a full reconstruction pipeline first, discovered it wasn't needed — architecture still supports UNK gracefully.
- **Why UQ**: Experts genuinely disagree (59% lead change rate in BO3). A system that says "I'm 6% confident" is more useful than one that guesses wrong confidently.
- **Split design**: "Pilot my team vs the field" — hold out Team A variants within clusters, opponents float. Matches deployment, avoids data waste.
- **Leakage**: Dedup on `(matchup, action)` triples, not matchup pairs — different expert decisions on the same matchup is signal.
- **Calibration**: ECE 0.012 (adaptive equal-mass). Temperature scaling barely moved (T≈1.0) — ensemble averaging is the calibration mechanism.
- **Abstention**: Doubles on OOD data. MI-based AUROC 0.80.
- **Per-team variation**: Entropy r=-0.56 with accuracy. Commander teams hit 50% top-1 (mechanical constraint), flexible teams 0%.
- **Architecture negative result**: Hierarchical dual encoder (1.56M params, +34%) achieved ±0.1pp. Noise ceiling is the constraint, not model capacity.
- **Key insight**: Teams agree on what to bring (52% stable) but disagree about who leads (59% change rate). The model captures this: bring-4 marginal at 17.8% vs lead arrangement at 32.1% given correct bring-4.
