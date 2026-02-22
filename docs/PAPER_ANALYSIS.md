# TurnZero Paper Analysis — Key Findings & Story Arc

## The Story in One Paragraph

We frame VGC turn-zero team selection — choosing which 4 of 6 Pokemon to bring
and which 2 to lead, given two Open Team Sheets — as a 90-way classification
problem and show that **uncertainty quantification matters more than raw
accuracy** in this domain. Top-1 accuracy is modest (6.4%) because the task is
intrinsically multi-modal: experts genuinely disagree on the "right" play. But a
lightweight transformer ensemble (1.16M params, 5 members) with temperature
scaling, selective prediction, and entropy-based team linearity scoring produces
a system that (a) is well-calibrated (ECE = 0.011), (b) knows when it doesn't
know (OOD abstention doubles from 20% to 46%), and (c) reveals which team
archetypes are predictable (Dondozo commander: 50% top-1) vs. which are not
(goodstuffs: 0% top-1). No prior work isolates the team preview decision as a
first-class prediction problem with UQ — VGC-Bench subsumes it into full-game
RL, and the only other lead prediction paper (Carli 2025) uses LSA on 1,174
logs with no uncertainty estimates. The real contribution is not the classifier
but the calibrated decision system around it.

## The Thesis (for framing the paper)

The paper is NOT "we built a good predictor." Raw accuracy is modest and we
don't pretend otherwise. The paper IS:

> In a 90-class domain with intrinsic label noise (expert disagreement), the UQ
> infrastructure matters more than the model. A small transformer ensemble with
> proper calibration, selective prediction, and entropy-based team linearity
> scoring creates a decision support system that honestly communicates what it
> knows and doesn't know. Per-team analysis reveals that aggregate accuracy
> masks a bimodal distribution — some teams are 50% predictable, others are 0%
> — and ensemble entropy cleanly separates them (r = -0.56).

The three money figures:
1. **Reliability diagram** — calibration is excellent (ECE = 0.011)
2. **Risk-coverage curves** — selective prediction works (AURC improvement)
3. **Entropy vs accuracy scatter** — team linearity is real and domain-plausible

Everything else supports these three.

---

## Related Work

### Full Battle Agents (team preview as a sub-decision)

**VGC-Bench** (Angliss et al., AAMAS 2026) is the closest prior work. They
define the same 90-way team preview action space (C(6,2) leads × C(4,2) backs)
and train transformer-based policies on 705K OTS battle logs from Pokemon
Showdown — the same data source we use. However, VGC-Bench treats team preview
as one incidental sub-action in a full-game RL pipeline (behavior cloning →
self-play/fictitious play/double oracle). Team preview accuracy is never
isolated or evaluated separately. Their focus is "can RL agents generalize
across teams?" (answer: performance degrades from ~79% to ~40% win rate as team
diversity increases). They report no uncertainty quantification — no
calibration, no selective prediction, no entropy analysis. We use VGC-Bench's
raw data but ask a fundamentally different question: not "how well can an agent
play?" but "when can we predict the team selection, and when is it genuinely
uncertain?"

**EliteFurretAI** (Simpson, GitHub) builds a full battle agent with a ~135M
parameter model that includes a team preview head. Their initial team preview
model reported 99.9% top-1 accuracy, but the author identified this as
**positional memorization**: Showdown logs store Pokemon in a fixed order, the
action label references slot positions, and 88.6% of unique full-state vectors
(species + moves + items + abilities, *in positional order*) mapped to exactly
one action. The author explicitly called it "just creating a lookup table" and
created `diagnose_teampreview_leakage.py` to detect the issue. After fixing via
random team-order augmentation, accuracy dropped to 79% top-1 / 95% top-3 /
99% top-5 — realistic but not directly comparable to our numbers (different
format, different elo filter, no UQ). Our architecture avoids this leakage by
design: we sort mons by canonical key and use mean pooling (position-invariant).
EliteFurretAI never asks which teams are predictable vs. not, and provides no
uncertainty estimates.

**Metamon** (Grigsby et al., RL Conference 2025) trains a causal transformer
(15M–200M params) via behavior cloning + offline RL on 475K human Singles
battles. Achieves top-10% human ranking in Gen 1 OU. Singles format only — no
team preview decision, no VGC, no OTS.

**PokeLLMon** (Hu et al., 2024) uses LLM in-context learning for Singles
battle decisions. 49% ladder win rate. No team selection, no VGC.

**PokeChamp** (Karten et al., ICML 2025 Spotlight) applies minimax tree search
with LLM-based action sampling to Singles. 76% win rate vs best LLM bot. No
team selection, no VGC.

**Sarantinos (2022)** built a Pokemon battle AI that peaked at 33rd globally
using MinMax variants with uncertainty handling. Focused entirely on in-battle
decisions, not team preview.

### Lead Prediction (direct competitors)

**Carli (2025)** — "Predicting Competitive Pokemon VGC Leads Using Latent
Semantic Analysis" (Journal of Geek Studies). The only other work that
specifically targets lead prediction. Predicts opponent lead-2 only (not
bring-4) using SVD-based similarity matching on team compositions represented
as bag-of-words strings. Trained on ~5K Showdown logs, evaluated against 8
manually-verified NAIC 2025 bracket games. No formal metrics beyond "hard
prediction" (both leads correct) and "soft prediction" (partial credit). No
moves/items/abilities as features. No uncertainty quantification. Criticized on
Hacker News for potentially being equivalent to simple heuristics. Our work
differs in every dimension: supervised 90-way classification (not similarity
matching), full OTS features (not species-only), 382K examples (not 5K),
complete UQ stack, and systematic evaluation.

### Win/Outcome Prediction (different task)

Several projects predict match *winner* from team composition:

- **Sangalli (Trustfull blog)**: Random Forest on ~10K VGC matches, achieved
  ~51% accuracy (essentially chance). Noted that incorporating movesets and
  items might help.
- **Pokemon Battle Predictor** (aed3, Smogon): 84 TensorFlow.js neural nets
  across formats, predicting win probability + opponent moves/switches during
  battle. 75-95% accuracy on in-battle predictions. Singles focused.
- Various Kaggle/student projects (including Stanford CS230 2022) predict
  win/loss from Pokemon stats. None address team selection.

These predict a different target (win/loss or in-battle decisions) and are not
directly comparable.

### Team Building (different task)

- **Reis et al. (IEEE CoG 2021, Trans. Games 2023)**: Adversarial framework
  for team *construction* (choosing which 6 Pokemon to put on a roster), not
  team *selection* from a fixed 6.
- **PokeHit (2025)**: LLM + MCTS for team formation. Same distinction — they
  build rosters, we select from them.
- **AI Team Builder** (aiteambuilder.com): Recommendation tool for team
  construction, not selection.

### Community Tools (non-ML)

- **Pikalytics**: Usage stats, cores, matchup data. Descriptive, not predictive.
- **VGC Helper**: Damage calculator + team builder. Heuristic, no ML.
- No ML-based team selection tools exist in the VGC community. Twitter/X
  search found only manual stat accounts (@VGCdata, @vgcstats, @Pikalytics).

### Gap Summary

| Aspect | VGC-Bench | EliteFurretAI | Carli 2025 | TurnZero |
|--------|-----------|---------------|------------|----------|
| Prediction target | Full-game policy | Full-game + preview head | Lead-2 only | Lead-2 + bring-4 (90-way) |
| Team preview eval | Not isolated | 79% post-fix (no UQ) | ~5K logs, no formal metrics | Dedicated, stratified |
| UQ | None | None | None | **Full stack** |
| Calibration | None | None | None | **ECE = 0.011** |
| Selective prediction | None | None | None | **AURC, risk-coverage** |
| OOD detection | None | None | None | **Regime B, entropy shift** |
| Per-team analysis | None | None | None | **153 teams, r = -0.56** |
| Feature importance | None | None | None | **7-level stress test** |
| Explainability | Win/loss | None | Cosine similarity | **Marginals, retrieval, roles** |
| Position invariance | N/A | Fixed (leakage) → augmented | N/A | **By design** (canonical sort + mean pool) |

**No prior work frames VGC team preview as a standalone classification problem
with uncertainty quantification.** This is the contribution.

---

## The Big Punches

### 1. The model learns real signal despite extreme multi-modality

**Action-90 top-1 accuracy: 6.4% (ensemble) vs 1.1% random.**

That looks terrible in isolation. The better framing: **the top-k coverage
curve shows you need k=17 predictions to reach 50% coverage** (vs k=45 for
random). The model concentrates probability mass on the right neighborhood of
the action space. At k=10, coverage is already 36.2%.

**The marginal decomposition tells the richer story:**

| Sub-decision | Classes | Top-1 | Top-3 |
|-------------|---------|-------|-------|
| Action-90 (joint) | 90 | 6.4% | 15.5% |
| Bring-4 (which 4 to bring) | 15 | 17.8% | 43.7% |
| Lead-2 (which 2 to lead) | 15 | 18.7% | 41.8% |
| Lead arrangement given correct bring-4 | 6 | 32.1% | — |

Bring-4 and lead-2 marginals are nearly identical (~18% top-1, ~43% top-3).
The harder part is the conditional: even when the model correctly identifies
which 4 to bring, it only picks the right lead pair 32.1% of the time (2x
the 16.7% random baseline for a 6-way choice). **Teams agree on what to bring
but disagree about who goes in front.**

**Paper angle:** Lead the results with the top-k curve and the decomposition
table, not raw top-1. The k=17 for 50% coverage quantifies multi-modality
directly, and the decomposition shows exactly where the model succeeds
(identifying the bring-4) vs. struggles (lead arrangement).

### 2. Calibration is excellent — the probabilities mean what they say

**ECE = 0.011 (action-90), 0.023 (lead-2) after temperature scaling.**

This is remarkably well-calibrated for a 90-class problem. Temperature scaling barely moved T from 1.0 to 1.158 — the ensemble was already near-calibrated. For comparison, the logistic baseline has ECE = 0.059 and the single transformer has 0.016.

**Paper angle:** The reliability diagram is your showcase figure. A well-calibrated model with 6.4% top-1 is more useful than a 10% model that's badly calibrated, because users can trust the confidence scores.

### 3. The ensemble knows when it doesn't know (OOD detection works)

**Regime B (held-out clusters) results are the headline surprise:**
- Abstention rate jumps from 20% → 46% (more than doubles)
- Entropy +0.17, MI +0.03
- The model automatically becomes more cautious on novel teams

**But here's the unexpected part:** Regime B *accuracy is higher* than Regime A (11.7% vs 6.4% top-1). This seems paradoxical but makes sense — Regime B's held-out clusters are rarer/more distinctive teams, so when the model IS confident about them, it's often right. The experts using these unusual teams may also be more predictable (fewer viable strategies with niche compositions).

**Calibration caveat:** Regime B ECE = 0.076 (vs 0.011 for Regime A). The ensemble is notably less calibrated on OOD data — a genuine limitation. The abstention mechanism partially compensates (higher-confidence predictions are still reasonably calibrated), but report this honestly.

**Paper angle:** This is your "the model does the right thing" story. On novel matchups, it abstains more AND is more accurate when it does predict. That's exactly what you want from a decision support system. Acknowledge ECE degradation on OOD as a limitation.

### 4. Moves are king — the stress test proves feature importance

Degradation from baseline by masking level:

| Masked | Top-1 | Top-3 | NLL Delta |
|--------|-------|-------|-----------|
| Tera only | -1.3pp | -3.3pp | +0.15 |
| Items only | -1.9pp | -3.3pp | +0.16 |
| 2/4 moves | -3.3pp | -7.4pp | +0.36 |
| All moves | -5.4pp | -12.3pp | +0.68 |
| All but species | -5.2pp | -12.1pp | +0.64 |

**Key insight:** Moves account for the majority of the signal. Hiding all moves drops top-3 from 15.5% to 3.2% — almost back to the popularity baseline's 3.9%. But species alone (all_but_species) still beats random, confirming the model learns team-composition-level patterns too.

**Unexpected:** `moves_4+items` is very slightly *better* than `moves_4` alone (3.7% vs 3.2% top-3). This might be noise, or removing items alongside moves might reduce a confounding interaction. Worth a footnote.

**Also unexpected:** Tera and items contribute roughly equally despite items having much more vocabulary diversity (193 items vs 20 tera types). Tera type matters more per-bit-of-information than you'd naively expect.

### 5. The mirror/non-mirror gap is the elephant in the room

| Stratum | Ensemble Top-1 | Ensemble Top-3 | N |
|---------|---------------|---------------|---|
| Mirror | 6.8% | 16.4% | 28,443 |
| Non-mirror | 3.8% | 8.9% | 3,885 |
| Overall | 6.4% | 15.5% | 32,328 |

Non-mirror matchups are nearly 2x harder. This makes sense — mirror matches are common team archetypes where conventional wisdom applies; non-mirror matchups are novel and require creative adaptation.

**Paper angle:** This is a fundamental limitation worth discussing honestly. The model is a creature of convention — it learns what experts *typically* do, not what they *should* do in novel situations.

---

## Surprising / Counter-Intuitive Findings

### S1. Logistic regression has *worse* NLL than popularity

Logistic NLL = 4.580 vs Popularity NLL = 4.497. The logistic baseline overfits to training patterns and produces overconfident wrong predictions on test. This validates the transformer + ensemble approach — you need both a flexible model AND proper calibration.

**Even worse on non-mirror:** Logistic NLL = 5.481 (!) and Brier = 1.048 (>1.0, which means it's worse than predicting uniform). The logistic baseline actively *hurts* on novel matchups.

### S2. The bootstrap CIs are wide — this task has high variance

Ensemble top-1 CI: [2.6%, 6.6%]. That's a 4pp range. The cluster-aware bootstrap reveals that performance varies dramatically by team composition cluster. Some clusters the model nails, others it's lost.

**Bootstrap mean vs point estimate discrepancy:** The bootstrap mean may differ slightly from the point estimate (e.g., 6.4% point vs ~6.2% bootstrap mean for top-1). This is expected because cluster-aware resampling changes the effective weighting across clusters — larger clusters get proportionally more weight in point estimates but are sampled uniformly in bootstrap. Report both.

**The mirror CIs are wild:** Mirror top-1 CI has a *lower bound of 0.0%*. This means there exist team clusters where the model gets zero top-1 accuracy. The 6.8% mean masks enormous heterogeneity.

### S3. Regime B is better on accuracy — the selection effect

As noted above, Regime B top-1 = 11.7% vs Regime A = 6.4%. This isn't the model being better on OOD data — it's that the OOD examples happen to be more predictable (rare team compositions that have fewer viable strategies). The model's higher abstention rate (46% vs 20%) means it's only predicting on the subset it's confident about.

**Paper angle:** This is a nice illustration of selective prediction at work. The system knows its limitations.

### S4. Temperature scaling does almost nothing — dropped from pipeline

T = 1.158 (close to 1.0). Val NLL improvement: 4.063 → 4.055 (tiny). Val ECE: 0.011 → 0.003. The ensemble is already well-calibrated before temperature scaling. This suggests the 5-member ensemble's probability averaging is doing most of the calibration work.

**Decision:** Temperature scaling has been **dropped from the final pipeline** since T≈1.0 adds no meaningful benefit. The `TemperatureScaler` class is retained for documentation and experimentation, but `ensemble_predict()` and the coach demo now use raw softmax directly.

**Paper angle:** This validates deep ensembles as a strong calibration mechanism in their own right, not just a way to get uncertainty estimates. Framed as: "we tried temperature scaling; the near-identity T confirms the ensemble is already well-calibrated, so we omit it."

### S5. Confidence values are universally low

Mean confidence = 5.3%, max in test set = 38.1%. Nobody gets confident predictions. This is actually correct behavior for a 90-class problem with high intrinsic uncertainty — the model is saying "there are many plausible plays here." Even the most extreme case only puts 38% on a single action.

### S6. Core clustering produces a mega-cluster (small-world network)

Our core clustering algorithm (≥4/6 species overlap → connected components via union-find) puts **91% of test data into a single cluster** (36,587 of 40,083 examples). This initially looked like a parser bug, but investigation confirmed it's a genuine small-world network phenomenon in the VGC metagame.

**The mechanism:** Hub species (Incineroar 37%, Urshifu-Rapid-Strike 38%, Rillaboom 29%, Flutter Mane 26%) appear on so many teams that transitive closure through 4/6 overlap connects nearly everything. Only 1.8% of random team pairs share ≥4 species directly, but the union-find bridges through shared hub mons create one giant connected component.

**Resolution:** We abandoned `core_cluster_a` for per-team analysis and instead grouped by exact species-6 composition (sorted 6-species tuple, ignoring moves/items/abilities). This yields **10,376 unique compositions** in the test set, with **153 having ≥20 Tier 1 examples** (covering 33% of Tier 1 data). This granularity reveals the heterogeneity that the aggregate numbers mask.

**Paper angle:** Worth mentioning in the data/methods section as a lesson in clustering design. The 4/6 overlap threshold is too coarse for a meta dominated by a handful of popular species. Species-6 composition is the right granularity for "what team archetype is this?"

### S7. Per-team predictability varies enormously (entropy as linearity)

Grouping by species-6 composition and computing mean ensemble entropy per team reveals a **1.3-nat spread** in team "linearity" (entropy range: [3.099, 4.406], mean 4.034 ± 0.212).

**The correlation is real:** entropy vs top-3 accuracy: r = -0.561. Teams with lower entropy (more predictable game plans) are significantly easier to model. Entropy vs top-1: r = -0.353 (weaker but still significant).

**Domain-plausible extremes:**
- **Most linear (H=3.10):** Calyrex-Shadow / Dondozo / Ogerpon-Cornerstone / Roaring Moon — 50% top-1, 68% top-3. The Dondozo "commander" archetype has an extremely telegraphed game plan.
- **Most linear #2 (H=3.21):** Calyrex-Shadow / Dondozo / Indeedee-F / Tatsugiri-Droopy — 27% top-1, 50% top-3. Another commander variant.
- **Most flexible (H=4.41):** Ceruledge / Garchomp / Samurott-Hisui / Staraptor — 0% top-1, 10% top-3. Diverse goodstuffs with many viable lead configurations.

**Key stats:**
- 22/153 teams (14%) have 0% top-1 accuracy — the model is completely lost
- 47/153 teams (31%) exceed 10% top-1 — well above the 6.4% aggregate
- Median top-1: 7.7%, median top-3: 17.2% (both above aggregate, since high-n teams pull the aggregate down)

**Paper angle:** This is the centerpiece Week 5 extension. The aggregate 6.4% top-1 is a misleading average over a bimodal distribution — some teams are 50% predictable, others are 0%. The entropy-accuracy scatter is a publication-quality figure that tells the whole story.

### S8. Speed control does NOT explain team linearity (negative result)

We hypothesized that speed control mode (Trick Room vs Tailwind) would be the
main driver of team linearity — Trick Room teams must lead the setter, so their
leads should be more constrained. **The data says otherwise.**

- Trick Room teams (66/153): mean entropy 4.025 ± 0.209
- Non-Trick Room teams (87/153): mean entropy 4.041 ± 0.214
- **Delta: 0.016 nats — effectively zero.**

The real linearity driver is **mechanical constraints**, not speed control. The
most linear teams are Dondozo commander archetypes (H=3.10), where the
Commander ability requires Dondozo + Tatsugiri/Maushold to lead together —
it's a hardcoded game mechanic, not a strategic choice. Trick Room teams
actually have substantial lead flexibility because many mons on the team can
function in or out of Trick Room.

**Paper angle:** Worth a sentence in the discussion. "While speed control mode
intuitively constrains lead selection, we find no significant entropy
difference (Δ = 0.016 nats). Team linearity is instead driven by mechanical
constraints like the Commander ability."

---

## Ablation Study: Multi-Task Loss for Tier 2 Labels

### Motivation

~20% of training examples (Tier 2, `bring4_observed=False`) have **fabricated action-90
labels** — the parser fills back-2 slots with a deterministic lowest-index heuristic since
only leads are observed. The model trains on these as ground truth. While evaluation
correctly restricts to Tier 1, the training signal is contaminated.

### Three Loss Modes

| Mode | Label | Tier 1 Loss | Tier 2 Loss | Train Data |
|------|-------|-------------|-------------|------------|
| `action90_all` | A (baseline) | Action-90 CE | Action-90 CE (fabricated) | Full |
| `multitask` | B | Action-90 CE | Lead-2 CE via marginalization | Full |
| `tier1_only` | C | Action-90 CE | — (dropped) | Tier 1 only |

### Multi-task loss math

For Tier 2 examples, marginalize 90-way action probs to 15-way lead-2 probs:
```
probs_90 = softmax(logits)         # (n, 90) — in probability space
lead2_probs = probs_90 @ M         # (n, 15) — M is (90, 15) binary margin matrix
loss_t2 = NLL(log(lead2_probs), lead2_true)
```

**Critical**: Marginalization happens in probability space, NOT logit space.
`softmax(logits) @ M ≠ softmax(logits @ M)`.

Combined loss weighted by batch proportion: `(n_tier1/N)*L1 + (n_tier2/N)*L2`.

### Configs & Training

- 15 configs: `configs/ablation_{a,b,c}/member_{001..005}.yaml`
- Seeds: 42, 137, 256, 512, 777 (same as existing ensemble)
- Train: `bash scripts/train_ablations.sh`
- Eval: `python scripts/eval_ablations.py [--bootstrap]`

### Expected Outcomes

- Ablation B (multitask) should improve lead-2 metrics since Tier 2 now trains on
  correct lead-2 labels instead of fabricated action-90 labels.
- Ablation C (tier1_only) loses ~20% of training data but avoids all label noise.
- If B and C are close, Tier 2 examples contribute useful lead-2 signal.
- If A ≈ B, the fabricated Tier 2 labels are not hurting much (noise is tolerable).

### Results

*To be filled after training completes.*

---

## Numbers for the Paper — Quick Reference

### Main Results Table

| Model | Params | Top-1 (A90) | Top-3 (A90) | Top-1 (L2) | Top-3 (L2) | NLL (A90) | ECE (A90) |
|-------|--------|-------------|-------------|------------|------------|-----------|-----------|
| Popularity | 0 | 1.3% | 3.9% | 7.1% | 21.8% | 4.497 | 0.001 |
| Logistic | ~4K | 4.0% | 10.1% | 14.0% | 33.8% | 4.580 | 0.059 |
| Transformer | 1.16M | 5.5% | 14.0% | 18.3% | 41.0% | 4.105 | 0.016 |
| Ensemble (5) | 5.8M | 6.4% | 15.5% | 19.8% | 43.2% | 4.031 | 0.011 |

### Dataset Stats
- 382K directed examples from 212K battles
- 7,826 team clusters via core-clustering
- Regime A: 247K train / 35K val / 40K test
- 80.7% bring-4 observed (Tier 1)
- 87.9% mirror matches in test

### Architecture
- d=128, L=4, H=4, d_ff=512, dropout=0.1
- 1.16M params per member, 5 members
- Mean pooling over 12 tokens (6 mons × 2 teams)
- 8 fields per mon: species, item, ability, tera, 4 moves

### Key UQ Numbers
- Temperature: T = 1.158 (near-identity; dropped from final pipeline)
- AURC (top-1): 0.890 ensemble vs 0.905 single
- AURC (top-3): 0.761 ensemble vs 0.791 single
- OOD entropy shift: +0.168 nats
- OOD abstention: 20% → 46%

### Top-k Coverage Milestones
- 50% coverage at k = 17 (vs k = 45 for random)
- 75% coverage at k = 36
- 90% coverage at k = 54

### Decomposition (all Tier 1, for apples-to-apples comparison)
- Bring-4 (15-way): 17.8% top-1, 43.7% top-3
- Lead-2 (15-way): 18.7% top-1, 41.8% top-3
- Lead arrangement | correct bring-4 (6-way): 32.1% (2x random)
- Note: previously reported lead-2 as 19.8%/43.2% — that was on all 40K
  examples (lead-2 is always observable). The 18.7% is Tier 1 only.

### Per-Team Linearity
- 153 teams with n ≥ 20 Tier 1 examples (33% coverage)
- Entropy range: [3.099, 4.406], mean 4.034 ± 0.212
- r = -0.353 (entropy vs top-1), r = -0.561 (entropy vs top-3)
- Speed control (TR vs non-TR) entropy delta: 0.016 nats (not significant)

---

## Suggested Paper Structure

1. **Intro** — Motivate with "90-class problem where experts disagree." The
   43% lead-2 top-3 is the hook. State the thesis: UQ matters more than raw
   accuracy in this domain.

2. **Related Work** — Use the section above. Key points: VGC-Bench defines the
   action space but never isolates team preview; Carli 2025 is the only direct
   lead prediction work but is toy-scale with no UQ; all other Pokemon ML is
   Singles format or in-battle decisions. Position our contribution as the
   first to treat team preview as a first-class prediction problem with UQ.

3. **Data** — 382K examples, 90-way label space, bring-4 observability. Note
   the small-world clustering finding (S6) and why species-6 composition is
   the right granularity. Mirror/non-mirror split.

4. **Methods** — Architecture (1.16M params, position-invariant by design),
   baselines, ensemble, temperature scaling. Frame the UQ stack as the main
   methodological contribution: deep ensembles → calibration → selective
   prediction → entropy-based team linearity.

5. **Results**:
   - Table 1: Main comparison (4 models × 7 metrics + bring-4/lead-2 decomposition)
   - Figure 1: Top-k coverage curve (the multi-modality story — k=17 for 50%)
   - Figure 2: Reliability diagram (the calibration story — money figure #1)
   - Figure 3: Risk-coverage curves (selective prediction — money figure #2)
   - Figure 4: Per-team entropy vs accuracy scatter (heterogeneity — money figure #3)
   - Table 2: Stress test (feature importance — moves dominate)
   - Figure 5: Stress test degradation plot
   - Table 3: OOD comparison (Regime A vs B — abstention doubles)
   - Figure 6: Team entropy histogram (annotated with linear vs flexible extremes)

6. **Discussion** — Per-team heterogeneity (6.4% aggregate masks bimodal
   distribution), bring-4 vs lead-2 (teams agree on what to bring, disagree on
   who leads), mirror/non-mirror gap (model learns convention not strategy),
   speed control negative result (Commander mechanics, not TR, drive linearity),
   wide CIs (honest about variance), position invariance vs EliteFurretAI's
   leakage. Future work: conformal prediction sets, opponent-conditional
   analysis (does team_b matter for linear teams?), cluster-specific models.

---

## Project Retrospective

### What went well
- **The pipeline is rock-solid.** From raw Showdown logs to calibrated ensemble predictions with full integrity checks at every stage. No data leakage, proper cluster-aware splits, clear Tier 1/Tier 2 delineation.
- **UQ stack is the real product.** Temperature scaling + deep ensembles + risk-coverage + bootstrap CIs + OOD detection. This is a complete uncertainty quantification story that's rare for a course project.
- **The model is right-sized.** 1.16M params, trains in ~5 minutes per member, 23 epochs to convergence. No massive compute needed, reproducible on consumer hardware.
- **Explanations are grounded.** Marginals come from the model's own distribution. Retrieval evidence comes from real training examples. Role lexicon is deterministic domain knowledge. Feature sensitivity is causal (mask and measure). No hand-waving.

### What was harder than expected
- **The action-90 space is genuinely hard.** The multi-modality ceiling means top-1 will always look bad. We spent more time than expected figuring out the right way to evaluate and present this.
- **Non-mirror matchups are a different problem.** The 2x difficulty gap suggests the model is mostly learning "what do people usually do with this archetype" rather than genuine strategic reasoning.
- **Bootstrap CIs are wide.** Cluster-level variance is high. This is honest but means some of our point estimates are less reliable than they look.

### What we'd do differently
- Start with the lead-2 framing earlier (it's more interpretable and our numbers are stronger)
- Build the retrieval index from day 1 — it's arguably the most useful component for a human user

---

## Stretch Features (If Time Allows)

### Stretch 1: Conformal Prediction Sets
Instead of "top-3 plans," produce prediction sets with guaranteed coverage: "with 90% probability, the expert's choice is in this set." The set size adapts to uncertainty — confident matchups get small sets (2-3 actions), uncertain ones get larger sets (10+). This is a direct upgrade over fixed top-k.

**Implementation:** ~2-3 hours. Use split conformal prediction with the validation set to calibrate quantiles. The infrastructure (ensemble probs, val set, metrics) already exists. Would produce:
- `turnzero/uq/conformal.py`
- A new `--conformal` flag in the demo
- One new figure (coverage vs set size)

**Paper value:** High. Conformal prediction is trending in ML and directly connects to the "knowing what you don't know" narrative.

### Stretch 2: Attention Visualization / Token Attribution
Extract attention weights from the transformer encoder and visualize which opponent Pokemon the model "looks at" when making predictions. This would give per-matchup explanations like "the model focused on opponent's Calyrex and Amoonguss."

**Implementation:** ~1-2 hours. Hook into `nn.TransformerEncoder` attention weights, aggregate across heads/layers, produce a 6×6 heatmap (our mons × their mons).

**Paper value:** Medium. Visually compelling for the demo section but attention ≠ attribution (the usual caveat).

### Stretch 3: Per-Cluster Performance Dashboard — DONE (Week 5)
Implemented as per-team analysis using species-6 composition grouping (not core_cluster_a, which produces a mega-cluster — see S6). 153 teams with ≥20 Tier 1 examples analyzed. Entropy-accuracy correlation r = -0.561 (top-3). Commander teams hit 50% top-1; goodstuffs teams hit 0%. See `scripts/run_cluster_analysis.py`, `outputs/eval/cluster_analysis.json`, and `outputs/plots/paper/cluster_*.{png,pdf}`.

### Stretch 4: Opponent Dependence Analysis (future work)
For linear teams (low entropy), does the opponent's OTS even matter? Mask
team_b entirely and compare accuracy — if it doesn't drop, linear teams are
opponent-invariant (they always execute the same game plan regardless of
matchup). This would confirm the "creature of convention" interpretation and
quantify when opponent modeling adds value. Not implemented — noted as future
work in the paper.
