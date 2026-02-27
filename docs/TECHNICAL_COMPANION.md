# Technical Companion — TurnZero Study Guide

Q&A walkthrough of technical decisions. Read alongside the Project Bible (the *what*),
PAPER_ANALYSIS (the *numbers*), and DECISIONS.md (the *rationale*).

---

## 1. Why this problem?

Team preview — which 4 of 6 to bring and which 2 to lead — is a clean supervised problem
(one decision per game, fixed input, observable labels) that's unsolved as an isolated
prediction target. VGC-Bench defines the 90-way action space but never evaluates preview
standalone. Experts genuinely disagree on the "right" play, making UQ the natural contribution.

## 2. Why 90-way joint classification?

C(6,4) × C(4,2) = 90 joint actions. Joint prediction is tractable at this scale and avoids
the dependency issues of factored bring-4 → lead-2 models. Independent per-Pokemon predictions
would violate the constraint that exactly 4 are brought and 2 lead.

## 3. Why a transformer?

The input is two sets of 6 Pokemon (categorical features, relational structure). Self-attention
over 12 concatenated tokens naturally captures pairwise interactions. Mean pooling gives
permutation invariance — no position leakage possible (unlike EliteFurretAI's 88.6%
"determinism" from slot-order encoding). d=128, L=4, H=4, 1.16M params.

## 4. Why deep ensembles for UQ?

5 independently trained models (different seeds) averaged at test time. Gives both epistemic
uncertainty (MI = disagreement between members) and aleatoric uncertainty (entropy - MI).
MC Dropout empirically underperforms (Ovadia et al. 2019). Single models can't decompose
uncertainty types. 5 members is the literature standard; 5× training cost is trivial (~25 min).

**Result**: top-1 5.5% → 6.4%, AURC 0.905 → 0.890. Near-calibrated before temperature
scaling (T=1.158 ≈ 1.0).

## 5. Why temperature scaling?

Single scalar T on validation logits — simplest post-hoc calibration. With 90 classes and 35K
val examples, Platt scaling (180 params) and isotonic regression risk overfitting.

**Result**: T=1.158, barely moved. Ensemble averaging IS the calibration mechanism.

## 6. How do we measure calibration?

**Adaptive ECE** (equal-mass binning, Nixon et al. 2019) = 0.012. Equal-width binning on a
90-class problem with max confidence ~38% concentrates examples in low bins, making results
misleadingly small. Equal-mass distributes evenly — more rigorous.

## 7. Why selective prediction?

Risk-coverage curves: rank by confidence, trade coverage for accuracy. AURC = 0.890, E-AURC
= 0.452. At 80% coverage, top-3 improves from 15.5% → 18.1%. On OOD data, abstention jumps
from 20% → 46% automatically.

## 8. Why OOD detection via Regime B?

Hold out entire species clusters (not random examples). Simulates genuinely novel teams.
MI-based AUROC = 0.80 (entropy = 0.70). MI is better because it isolates epistemic
uncertainty. Accuracy on non-abstained OOD is 11.7% (selection effect — confident predictions
on unusual teams tend to be right). ECE degrades 0.012 → 0.076.

## 9. Why cluster-aware bootstrap CIs?

Test examples within a cluster are correlated. Standard bootstrap underestimates variance.
Cluster-aware resampling gives wider, more honest CIs: top-1 [2.6%, 6.6%]. The wide range
reflects heterogeneity across team archetypes.

## 10. Why entropy as a team predictability score?

Species-6 grouping (exact sorted 6-species tuple) gives 10,376 unique compositions.
153 teams with ≥20 Tier 1 examples. Entropy range [3.10, 4.41], r = -0.561 with top-3.

Most predictable: Dondozo commander (H=3.10, 50% top-1) — Commander ability mechanically
forces a lead pair. Least predictable: diverse goodstuffs (H=4.41, 0% top-1).

Species-6 grouping instead of core clusters because union-find's transitive closure creates
a mega-cluster (91% of data) via hub species (Incineroar 37%, Urshifu 38%).

## 11. Why a stress test?

Masking OTS fields at test time = causal feature importance. Moves carry ~70% of signal
(all moves hidden: top-3 drops 15.5% → 3.2%). Species alone barely beats popularity.
Moves reveal strategy (Trick Room, Tailwind); species only reveals composition.

## 12. Why retrieval-based evidence?

128-dim pooled representation → 246K training embeddings → cosine similarity KNN.
Brute-force exact search (not FAISS) — 246K × 128 is instant. Grounded in real expert
decisions, more interpretable than attention heatmaps.

## 13. Why this data pipeline?

- `|showteam|` gives 100% OTS — no cross-game reconstruction needed
- Dedup on `(team_a, team_b, action90, format)` — different actions = different signal
- Tier 1/2 stratification: action-90 only on fully observed bring-4
- `match_group_id` integrity: both directed examples from one game stay in same split

## 14. Why these baselines?

Popularity (1.3%) → linear signal exists? Logistic (4.0%) → yes, but badly calibrated
(NLL 4.580 > popularity 4.497, Brier > 1.0 on non-mirror). Transformer + ensemble needed
for both accuracy and calibration.

## 15. Position invariance

Canonical sort + mean pool = position-invariant by construction. No augmentation needed.
Avoids EliteFurretAI's positional leakage entirely.

## 16. Top-k coverage reframing

k=17 for 50% (vs k=45 random). Model concentrates probability on the right neighborhood.
Bring-4 marginal: 17.8% top-1. Lead-2 marginal: 18.7%. Lead arrangement | correct bring-4:
32.1% (only 2× random for 6-way). Teams agree on roster, disagree on leads.

## 17. Speed control hypothesis (negative result)

Trick Room teams: H=4.025 vs non-TR: H=4.041. Delta 0.016 nats — null. Real predictability
driver is mechanical constraints (Commander), not speed control.

## 18. Architecture ablation (negative result)

Hierarchical dual encoder (1.56M params, +34%): top-1 6.3% vs 6.4%, NLL 4.023 vs 4.022.
Within noise. Noise ceiling is the binding constraint, not model capacity. Using the vanilla
transformer makes the UQ contribution cleaner.

## 19. BO3 adaptation: where multi-modality comes from

53K linkable BO3 sets, 136K transitions. Lead changes: 59% overall, 72% after loss, 46%
after win (χ²=10,126). Bring-4 changes: 52%. Same player, same matchup, different choices —
driven by within-set adaptation the model can't observe. Principled floor on irreducible noise.

---

## Glossary

| Term | Definition | Value |
|------|-----------|-------|
| **ECE** | Weighted avg gap between predicted confidence and actual accuracy | 0.012 |
| **NLL** | -log p(y_true\|x), lower = better | 4.031 |
| **AURC** | Area under risk-coverage curve, lower = better | 0.890 |
| **E-AURC** | AURC minus optimal AURC, isolates ranking quality | 0.452 |
| **OOD AUROC** | ID vs OOD discrimination, higher = better | 0.80 |
| **MI** | Ensemble mutual information (epistemic uncertainty) | ~0.05 |
| **Tier 1** | Examples with fully observed bring-4 | ~80% |
| **Action-90** | Joint (lead-2, back-2) space: C(6,4)×C(4,2) | 90 |
| **Regime A** | Hold out team variants within clusters | 247K/35K/40K |
| **Regime B** | Hold out entire clusters (OOD) | 347K/6K/15K |

## Elevator Pitch

"We predict which 4 of 6 Pokemon to bring and which 2 to lead in VGC — a 90-class problem
where experts genuinely disagree. Top-1 is 6.4%, but our top-17 captures the right action
50% of the time. The real contribution is the UQ stack: calibrated probabilities (ECE 0.012),
selective prediction that doubles abstention on novel teams, and per-team predictability
scores revealing that mechanical constraints (Commander), not strategic choices, drive the
most predictable teams. No prior work treats VGC team preview as a standalone prediction
problem with uncertainty quantification."
