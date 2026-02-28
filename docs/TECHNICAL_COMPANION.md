# Technical Companion — TurnZero Study Guide

Q&A walkthrough of technical decisions. Read alongside the Project Bible (the *what*),
PAPER_ANALYSIS (the *numbers*), and DECISIONS.md (the *rationale*).

**What this project actually contributes** (vs. standard techniques applied):
1. **Problem characterization**: per-team predictability driven by mechanical constraints, not strategic reasoning (speed control null: ΔH = 0.016 nats)
2. **Multi-modality ceiling**: within-set adaptation (59% lead changes, 72% after loss) + architecture ablation (±0.3pp at +34% params) prove the binding constraint is irreducible label noise, not model capacity
3. **Leakage prevention by construction**: canonical sort + mean pool avoids EliteFurretAI's 99.9% → 79% collapse without augmentation
4. **Problem isolation**: first to treat VGC team preview as a standalone 90-way prediction problem with proper evaluation

The model (transformer) and UQ stack (deep ensembles, calibration, selective prediction, OOD detection) are standard techniques correctly applied — not the contribution.

---

## 1. Why this problem?

Team preview — which 4 of 6 to bring and which 2 to lead — is a clean supervised problem
(one decision per game, fixed input, observable labels) that's unsolved as an isolated
prediction target. VGC-Bench defines the 90-way action space but never evaluates preview
standalone. Experts genuinely disagree on the "right" play, making the characterization
of *why* they disagree and *where* prediction is possible the natural contribution.

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

*Standard technique, not a contribution — applied because it's the right tool.*

5 independently trained models (different seeds) averaged at test time. Gives both epistemic
uncertainty (MI = disagreement between members) and aleatoric uncertainty (entropy - MI).
MC Dropout empirically underperforms (Ovadia et al. 2019). Single models can't decompose
uncertainty types. 5 members is the literature standard; 5× training cost is trivial (~25 min).

**Result**: top-1 5.5% → 6.4%, AURC 0.905 → 0.890. Near-calibrated before temperature
scaling (T=1.158 ≈ 1.0).

## 5. Why temperature scaling?

*Standard technique — dropped because unnecessary.*

Single scalar T on validation logits — simplest post-hoc calibration. With 90 classes and 35K
val examples, Platt scaling (180 params) and isotonic regression risk overfitting.

**Result**: T=1.158, barely moved. Ensemble averaging IS the calibration mechanism.

## 6. How do we measure calibration?

*Standard metric — chosen carefully for this problem's characteristics.*

**Adaptive ECE** (equal-mass binning, Nixon et al. 2019) = 0.012. Equal-width binning on a
90-class problem with max confidence ~38% concentrates examples in low bins, making results
misleadingly small. Equal-mass distributes evenly — more rigorous.

**Caveat**: median confidence is 4.5%, only 6% of predictions exceed 10%. The model is
well-calibrated but rarely confident — calibration matters most for the tail of predictions
where the model concentrates mass on a few actions.

## 7. Why selective prediction?

*Standard technique.*

Risk-coverage curves: rank by confidence, trade coverage for accuracy. AURC = 0.890, E-AURC
= 0.452. At 80% coverage, top-3 improves from 15.5% → 18.1%. On OOD data, abstention jumps
from 20% → 46% automatically.

## 8. Why OOD detection via Regime B?

*Standard technique (MI-based), but the split design is ours.*

Hold out entire species clusters (not random examples). Simulates genuinely novel teams.
MI-based AUROC = 0.80 (entropy = 0.70). MI is better because it isolates epistemic
uncertainty. Accuracy on non-abstained OOD is 11.7% (selection effect — confident predictions
on unusual teams tend to be right). ECE degrades 0.012 → 0.076.

## 9. Why cluster-aware bootstrap CIs?

Test examples within a cluster are correlated. Standard bootstrap underestimates variance.
Cluster-aware resampling gives wider, more honest CIs: top-1 [2.6%, 6.6%]. The wide range
reflects heterogeneity across team archetypes.

## 10. Why entropy as a team predictability score?

**This is a core finding, not just a metric choice.**

Species-6 grouping (exact sorted 6-species tuple) gives 10,376 unique compositions.
153 teams with ≥20 Tier 1 examples. Entropy range [3.10, 4.41], r = -0.561 with top-3.

Most predictable: Dondozo commander (H=3.10, 50% top-1) — Commander ability mechanically
forces a lead pair. Least predictable: diverse goodstuffs (H=4.41, 0% top-1).

The key finding: predictability is driven by **game-rule constraints** (Commander ability
forcing a lead pair), not **strategic reasoning**. Speed control (Trick Room vs Tailwind)
— the obvious hypothesis — is a null result (ΔH = 0.016 nats).

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

## 16. Top-k coverage

k=17 for 50% (vs k=45 random) — narrows the space from 90 to 17 actions for half-coverage.
Bring-4 marginal: 17.8% top-1. Lead-2 marginal: 18.7%. Lead arrangement | correct bring-4:
32.1% (only 2× random for 6-way). Teams agree on roster, disagree on leads.

## 17. Speed control hypothesis (negative result) ★

**Core finding.** Trick Room teams: H=4.025 vs non-TR: H=4.041. Delta 0.016 nats — null.
This kills the obvious alternative hypothesis that strategic clarity (having a clear game plan)
drives predictability. The actual driver is game-mechanical constraints (Commander), not
strategic reasoning. This is non-obvious — a VGC player would intuitively expect Trick Room
teams to be more predictable because they have a clear "set up TR → sweep" plan.

## 18. Architecture ablation (negative result) ★

**Core evidence for the multi-modality ceiling.** Hierarchical dual encoder (1.56M params,
+34%): top-1 6.3% vs 6.4%, NLL 4.023 vs 4.022. Within noise. This is not a negative result
about the architecture — it's positive evidence that the binding constraint is irreducible
label noise from within-set adaptation, not model capacity. Combined with the BO3 analysis
(§19), this establishes a principled ceiling.

## 19. BO3 adaptation: where multi-modality comes from ★

**Core evidence for the multi-modality ceiling.** 53K linkable BO3 sets, 136K transitions.
Lead changes: 59% overall, 72% after loss, 46% after win (χ²=10,126). Bring-4 changes: 52%.
Same player, same matchup, different choices — driven by within-set adaptation to information
revealed in prior games that no turn-zero model can observe. This isn't "players change leads
after losses" (obvious) — it's a quantified, empirically-backed explanation for *why* the
accuracy ceiling exists and why adding model capacity won't help.

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

"We isolate VGC team preview as a standalone 90-way prediction problem — nobody else has —
and characterize *why* it's hard and *where* prediction is possible. The aggregate 6.4% top-1
masks a heterogeneous landscape: mechanically constrained teams (Commander ability) reach
50% top-1 while flexible teams reach 0%, and this split is driven by game-rule constraints,
not strategic reasoning — speed control is a null result. Within-set adaptation (59% lead
changes between games, 72% after a loss) creates irreducible label noise that no turn-zero
model can resolve, confirmed by an architecture ablation showing +34% parameters buys ±0.3pp.
The model and UQ stack (transformer ensemble, calibration, OOD detection) are standard
techniques applied to support the analysis — the contribution is the problem characterization."
