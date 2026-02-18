# Technical Companion — TurnZero Study Guide

A question-driven walkthrough of every technical decision in the project.
Read this alongside the PROJECT_BIBLE (the *what*) and PAPER_ANALYSIS (the
*results*). This doc is the *why*.

---

## 1. Why this problem?

**Question:** What's an interesting ML problem in competitive Pokemon?

Most prior work predicts battle *outcomes* (win/loss) or trains full battle
*agents* (turn-by-turn move selection). We picked the **team preview decision**
— which 4 of 6 Pokemon to bring and which 2 to lead — because:

1. **It's a clean supervised learning problem.** One decision per game, fully
   observable labels (we can see what the player actually chose), fixed input
   (two team sheets). No sequential decision-making, no reward shaping, no
   environment simulation.

2. **It's unsolved.** VGC-Bench (Angliss et al., AAMAS 2026) defines the same
   90-way action space but never isolates or evaluates team preview — it's just
   one sub-action in a full-game RL pipeline. The only direct competitor (Carli
   2025) uses LSA on 1,174 logs with no uncertainty estimates.

3. **Uncertainty is the interesting part.** Experts genuinely disagree on the
   "right" play. A system that says "I'm 6% confident" is more useful than one
   that says "definitely this" and is wrong. This makes UQ the natural core
   contribution, not raw accuracy.

**Why not build a full battle agent?** That's a reinforcement learning problem
requiring environment simulation, reward design, and orders of magnitude more
compute. VGC-Bench needed months and a research team. Team preview as supervised
classification is tractable for a course project and the UQ story is novel.

**Why not predict win/loss instead?** Win prediction from team preview is ~51%
accuracy (Sangalli, Trustfull blog) — essentially random. The team sheets don't
contain enough information to predict who wins because in-game execution
dominates. Team *selection* is a more tractable prediction target because the
input (OTS) directly determines the output (what the player chose).

---

## 2. Why 90-way joint classification?

**Question:** How do we formalize "pick 4 of 6 to bring + pick 2 of those 4 to
lead" as a prediction target?

The action space decomposes as:
- C(6,4) = 15 ways to choose which 4 to bring
- C(4,2) = 6 ways to split those 4 into 2 leads + 2 backs
- Total: 15 × 6 = **90 joint actions**

We predict the joint action directly (single 90-way softmax) rather than
factoring into separate bring-4 and lead-2 predictions.

**Why joint, not factored?** Because the sub-decisions are correlated. The
choice of who to bring constrains who can lead, and vice versa. A factored
model (predict bring-4 independently, then predict leads given bring-4) would
need to handle this dependency explicitly and would lose information about the
joint distribution. With only 90 classes, joint prediction is tractable.

**Why not predict each Pokemon independently?** (e.g., 6 binary "bring or not"
predictions) Because the outputs are constrained — you must bring exactly 4 and
lead exactly 2. Independent predictions would violate these constraints and
you'd need post-hoc correction. The joint formulation handles constraints
naturally.

**Isn't 90 classes a lot?** Not really. ImageNet has 1,000 classes. The real
challenge isn't the number of classes but the **multi-modality** — multiple
actions are simultaneously "correct" for the same input because experts
disagree. This is fundamentally different from a 90-class problem where each
input has one right answer.

---

## 3. Why a transformer?

**Question:** What model architecture should we use for two team sheets → action?

Each team sheet is a *set* of 6 Pokemon, each described by 8 categorical fields
(species, item, ability, tera type, 4 moves). The input is inherently:
- **Set-structured**: Pokemon order within a team shouldn't matter
- **Categorical**: all features are discrete tokens, not continuous values
- **Relational**: the interaction between Pokemon matters (e.g., Trick Room
  setter + slow attacker)

A **transformer encoder** naturally handles all three:
- Self-attention captures pairwise interactions between all 12 Pokemon
- Learned embeddings handle categorical features
- Mean pooling over tokens gives a permutation-invariant representation

**Why not a simple MLP?** You'd need to flatten 12 Pokemon × 8 fields = 96
features into a fixed vector. This loses the set structure — the model would
be sensitive to Pokemon ordering, which is meaningless. EliteFurretAI ran into
exactly this problem: their positional encoding caused 88.6% "determinism" that
was actually data leakage from team ordering.

**Why not a GNN (graph neural network)?** Pokemon on a team don't have a
natural graph structure (no edges). You'd be inventing an adjacency matrix.
A transformer's fully-connected attention is the special case of a GNN where
every node attends to every other node — it's strictly more general.

**Why not an RNN/LSTM?** RNNs process sequences with a fixed order. Pokemon
sets have no natural order. You'd need to either sort them (losing information
about which orderings the model should be invariant to) or use order-agnostic
pooling (which is what a transformer with mean pooling already does, but
better).

**Architecture specifics:** d=128, L=4, H=4, d_ff=512, dropout=0.1. Total:
1.16M parameters. This is deliberately small — the dataset (247K train
examples) is not large enough to train a huge model without overfitting. The
model converges in ~23 epochs (~5 min per member on an RTX 4080 Super).

**Why mean pooling, not a [CLS] token?** Mean pooling is naturally
permutation-invariant — swapping two Pokemon in the input doesn't change the
output. A CLS token can learn to be invariant, but it's not guaranteed, and
it adds a learnable parameter that serves no purpose when you already have
12 informative tokens. We verified this is correct by noting that our model
has zero positional leakage (unlike EliteFurretAI).

---

## 4. Why deep ensembles for UQ?

**Question:** How do we get uncertainty estimates from a neural network?

A single neural network outputs a point estimate (softmax probabilities) but
has no way to express "I'm unsure about my own uncertainty." Deep ensembles
(Lakshminarayanan et al., 2017) train M independent models with different
random seeds and average their predictions.

**How it works:**
- Train 5 models with seeds 42, 137, 256, 512, 777
- Each sees the same data but different weight initialization + dropout masks
- At test time: average the 5 softmax outputs → ensemble prediction
- **Predictive entropy** H[ŷ|x] = uncertainty about the prediction
- **Mutual information** MI = H[ŷ|x] - E[H[ŷ|x,θ]] = how much the models
  *disagree* (epistemic uncertainty)
- Entropy - MI = aleatoric uncertainty (irreducible noise in the data)

**Why not MC Dropout?** MC Dropout (Gal & Ghahramani, 2016) approximates
Bayesian inference by running the same model multiple times with dropout
enabled at test time. It's cheaper (one model, not five) but empirically
produces worse uncertainty estimates than deep ensembles. The landmark
comparison (Ovadia et al., 2019) showed deep ensembles consistently dominate
MC Dropout on calibration and OOD detection across multiple benchmarks.

**Why not a Bayesian neural network?** Bayesian NNs place distributions over
weights and do exact (or approximate) posterior inference. In theory this is
the right thing to do. In practice, variational inference for neural nets
(Blundell et al., 2015) is hard to tune, scales poorly, and often
underperforms deep ensembles on calibration. Deep ensembles are the pragmatic
standard for neural network UQ.

**Why not a single model with temperature scaling?** Temperature scaling
calibrates the *average* confidence but doesn't give you uncertainty
*decomposition*. With a single model, you can't distinguish "this input is
inherently ambiguous" (high aleatoric uncertainty) from "this input is unlike
anything in training" (high epistemic uncertainty). Ensembles give you both
via the entropy/MI decomposition.

**Why 5 members?** Empirically, 3-10 members capture most of the diversity
benefit. Going from 1→3 is a big improvement; 3→5 is moderate; 5→10 is
marginal. Five is the standard in the literature (Lakshminarayanan et al.
used 5). We chose 5 because it's standard and our models are small enough
that 5× cost is trivial (~25 min total training).

**What we found:** The ensemble improved top-1 from 5.5% (single) to 6.4%,
and more importantly, improved AURC from 0.905 to 0.890 (top-1). The
uncertainty estimates were already near-calibrated before temperature scaling
(T=1.158, barely moved from 1.0), confirming that ensemble averaging is
itself a strong calibration mechanism.

---

## 5. Why temperature scaling for calibration?

**Question:** Are the model's predicted probabilities trustworthy? When it says
"8% chance of action X," does that action actually happen 8% of the time?

**What calibration means:** A model is calibrated if, among all predictions
where it assigns probability p to the correct class, the correct class actually
occurs with frequency p. Miscalibrated models are dangerous — an overconfident
wrong prediction is worse than an uncertain one because users trust it.

**How temperature scaling works:** Learn a single scalar T > 0 on the
validation set. At test time, divide logits by T before softmax:

    p(y|x) = softmax(z / T)

- T > 1 → softens probabilities (reduces overconfidence)
- T < 1 → sharpens probabilities (increases confidence)
- T = 1 → no change

Fit T by minimizing NLL on the validation set (convex optimization, global
optimum guaranteed). This is the simplest possible post-hoc calibration method.

**Why not Platt scaling?** Platt scaling learns a linear transformation of
logits (a, b parameters per class). With 90 classes, that's 180 parameters to
fit on the validation set. Temperature scaling has 1 parameter. For
high-dimensional classification, temperature scaling is the standard (Guo et
al., 2017 "On Calibration of Modern Neural Networks").

**Why not isotonic regression?** Isotonic regression is non-parametric — it
fits a monotonic function from predicted probability to true probability. It's
more flexible than temperature scaling but requires binning, doesn't scale
well to many classes, and can overfit on small calibration sets.

**Why not histogram binning?** Same flexibility/overfitting tradeoff. With 90
classes and only 35K validation examples, we'd have ~389 examples per class
on average — not enough for reliable binning.

**What we found:** T = 1.158 (barely above 1.0). The ensemble was already
near-calibrated. Val ECE improved from 0.011 to 0.003 (tiny). This tells us
the ensemble's probability averaging is doing most of the calibration work —
temperature scaling is a safety net, not a rescue. This is itself a finding
worth reporting: deep ensembles are a strong calibration mechanism in their
own right.

---

## 6. How do we measure calibration? (ECE)

**Question:** How do we quantify whether the model's probabilities are
trustworthy?

**Expected Calibration Error (ECE):** Partition predictions into B bins by
predicted confidence. In each bin, compare the average predicted confidence
to the actual accuracy. ECE is the weighted average of these gaps:

    ECE = Σ (n_b / N) × |accuracy_b - confidence_b|

Intuitively: if the model puts 70% confidence on a class and that class
actually occurs 70% of the time, the bin contributes zero error. If it occurs
50% of the time, the gap is 20%.

**Our ECE = 0.011.** This is remarkably low for a 90-class problem. For
context, a well-calibrated ResNet on ImageNet (1000 classes) typically has ECE
~0.03-0.05 *after* temperature scaling. Our ensemble achieves 0.011 before
temperature scaling even does much.

**Caveat with ECE:** When maximum confidence is low (ours is 5.3% mean, 38.1%
max), most predictions fall in the lowest bins where accuracy is naturally
close to confidence. This makes low ECE partly an artifact of the model never
being confident. The reliability diagram is a more honest visualization —
it shows the calibration curve across all confidence levels.

**Why not Brier score?** Brier score = E[(p - y)²] measures both calibration
AND sharpness (how concentrated the predictions are). It penalizes a model for
being "correctly uncertain" — if the true distribution is uniform over 10
actions, a model that predicts uniform gets penalized more than one that
confidently picks the right action. ECE isolates calibration from sharpness,
which is what we care about. (We do report Brier score separately.)

---

## 7. Why selective prediction? (Risk-coverage)

**Question:** If the model is uncertain, should it abstain rather than guess?

**Selective prediction** means the model can say "I don't know" and refuse to
predict. This is valuable in decision support — a VGC player would rather hear
"this matchup is too uncertain for me to help" than get a bad recommendation.

**How it works:** Rank test examples by model confidence (highest → lowest).
Plot *coverage* (fraction of examples the model predicts on) vs *risk* (error
rate on those predictions). As you lower the confidence threshold, you predict
on fewer examples but with higher accuracy.

**AURC (Area Under the Risk-Coverage curve):** A single number summarizing
selective prediction quality. Lower AURC = better. A model with good
uncertainty estimates will have low risk at high coverage (it's accurate when
confident and abstains when uncertain). A randomly uncertain model will have
a flat risk-coverage curve.

**Our AURC:** 0.890 (ensemble) vs 0.905 (single transformer). The ensemble's
uncertainty estimates are better at identifying which predictions to trust.
This is the practical UQ payoff — not just better probabilities, but better
*selective* predictions.

**Why not just use a confidence threshold?** That's what selective prediction
IS — but AURC evaluates it across all possible thresholds simultaneously, so
you don't have to choose one. The risk-coverage curve shows the full tradeoff.

**Operating point example:** At our default 80% coverage (predict on 80% of
examples, abstain on 20%), the ensemble's top-3 accuracy is 18.1% vs 15.5%
without abstention. By refusing to predict on the hardest 20%, accuracy
improves by 2.6pp. On OOD data (Regime B), abstention rate automatically
jumps to 46% — the model knows it's out of distribution and becomes more
cautious without being told to.

---

## 8. Why OOD detection via Regime B?

**Question:** Does the model know when it encounters team compositions it
hasn't seen in training?

**The setup:** We create two data split regimes:
- **Regime A** (standard): Hold out team variants within each cluster.
  Train/val/test share the same cluster distribution.
- **Regime B** (OOD): Hold out *entire clusters*. Test set contains team
  compositions the model has never seen during training.

This is stricter than a random split — it simulates what happens when a
genuinely novel team appears at a tournament.

**Why cluster-level holdout, not random holdout?** With random holdout, the
same team composition (e.g., Incineroar/Rillaboom/Flutter Mane/...) appears in
both train and test. The model has seen this archetype before, just not this
exact game. Cluster-level holdout ensures the model faces truly novel
compositions.

**What we found:**
- Entropy increases by +0.168 nats on Regime B vs A → model is more uncertain
- Mutual information increases by +0.03 → models disagree more
- Abstention rate doubles: 20% → 46%
- But accuracy on non-abstained predictions is *higher*: 11.7% vs 6.4% top-1

The last finding seems paradoxical. The explanation: Regime B's held-out
clusters tend to be rare/distinctive teams. When the model IS confident about
them (the 54% it doesn't abstain on), it's often right because unusual teams
have fewer viable strategies. This is selective prediction working as intended:
abstain on the uncertain cases, be right on the rest.

---

## 9. Why bootstrap CIs? Why cluster-aware?

**Question:** How confident should we be in our performance numbers?

A point estimate like "6.4% top-1 accuracy" is meaningless without error bars.
The standard approach: **bootstrap resampling** — resample the test set with
replacement B=1,000 times, compute the metric on each resample, take the
2.5th and 97.5th percentiles as a 95% confidence interval.

**Why cluster-aware, not standard bootstrap?** Standard bootstrap assumes
i.i.d. data. Our test examples are NOT i.i.d. — examples from the same team
cluster are correlated (similar teams → similar predictions). Standard
bootstrap underestimates variance because it breaks these correlations.

**Cluster-aware bootstrap:** Instead of resampling individual examples,
resample *clusters* with replacement. All examples in a resampled cluster come
along together, preserving within-cluster correlation. This gives wider
(more honest) CIs.

**What we found:** Ensemble top-1 CI: [2.6%, 6.6%]. That's a 4pp range —
much wider than a standard bootstrap would give. The wide CIs are a feature,
not a bug: they honestly reflect that performance varies enormously by team
archetype. Some clusters the model nails (50% top-1), others it scores 0%.

**Why not analytic confidence intervals?** For complex metrics like top-k
accuracy over a multi-modal distribution with cluster structure, there's no
clean formula. Bootstrap is the standard nonparametric approach.

---

## 10. Why entropy as a team linearity score?

**Question:** Which team archetypes are predictable and which aren't?

The aggregate 6.4% top-1 accuracy masks enormous heterogeneity. We want to
identify which teams are "linear" (one obvious game plan) vs "flexible" (many
viable strategies).

**The approach:** Group test examples by species-6 composition (exact sorted
6-species tuple). For each team with ≥20 Tier 1 examples, compute the mean
ensemble entropy over all examples of that team.

**Why entropy?** Entropy H = -Σ p(a) log p(a) measures the "flatness" of the
probability distribution. If the model puts 50% on one action and spreads the
rest, entropy is low → the team is predictable. If the model spreads
probability roughly equally across many actions, entropy is high → the team is
flexible. This is a model-derived measure — it reflects what the *model*
learned about the team, not a hand-coded heuristic.

**Why not conformal prediction set size?** Conformal prediction produces
prediction sets with guaranteed coverage (e.g., "the expert's choice is in
this set with 90% probability"). Set size adapts to uncertainty — small sets
for easy cases, large sets for hard ones. This would work as a linearity
measure, BUT: with mean confidence of 5.3%, the 90% coverage sets would
contain 20-30 actions (out of 90), making them uninformative. Entropy captures
the same information continuously without needing a coverage threshold.

**Why species-6 grouping, not the core clustering?** Our core clustering
algorithm (≥4/6 species overlap → connected components via union-find) puts
91% of test data into a single mega-cluster. This is a small-world network
effect: hub species (Incineroar 37%, Urshifu 38%) appear on so many teams that
transitive closure connects nearly everything. Only 1.8% of random pairs share
4+ species directly, but union-find bridges through hubs. Species-6 grouping
(exact sorted species tuple) gives 10,376 unique compositions with meaningful
granularity.

**What we found:**
- 153 teams with ≥20 Tier 1 examples
- Entropy range: [3.099, 4.406], mean 4.034 ± 0.212
- r = -0.561 between entropy and top-3 accuracy (strong negative correlation)
- Most linear: Dondozo commander teams (H=3.10, 50% top-1) — the Commander
  ability requires Dondozo + Tatsugiri to lead together, mechanically
  constraining the decision
- Most flexible: Diverse goodstuffs teams (H=4.41, 0% top-1) — many viable
  lead configurations, no single dominant strategy

---

## 11. Why a stress test? Why masking?

**Question:** Which OTS features matter most for prediction?

**Feature importance via masking:** Replace specific fields with UNK tokens at
test time and measure accuracy degradation. This is a causal intervention —
if masking moves drops accuracy by 12pp, moves causally contribute 12pp of
signal. This is stronger than correlation-based importance (e.g., SHAP) because
it directly measures what happens when the information is removed.

**Masking levels (progressive):**

| Level | What's hidden | Top-3 Δ | Interpretation |
|-------|-------------|---------|----------------|
| baseline | nothing | — | Full OTS information |
| tera | tera type | -3.3pp | Tera reveals intended strategy |
| items | held items | -3.3pp | Items reveal role (Choice Scarf = fast, Sash = glass cannon) |
| moves_2 | 2 of 4 moves | -7.4pp | Partial moveset still informative |
| moves_4 | all 4 moves | -12.3pp | Moves are the dominant signal |
| all_but_species | everything except species | -12.1pp | Species alone barely beats popularity |

**Key insight:** Moves account for the majority of the signal. Without moves,
the model drops to near-popularity-baseline performance. This makes domain
sense — movesets reveal the team's *strategy* (Trick Room, Tailwind, hyper
offense, etc.), while species alone only reveal the team *composition*.

**Why not gradient-based attribution (saliency maps)?** Gradient-based methods
tell you what the model is *looking at*, not what *matters*. A model might have
high gradient on a feature it uses as a shortcut, not one that's genuinely
informative. Masking is a direct causal test. The downside is it's coarser
(field-level, not token-level), but for our purposes, field-level importance is
exactly what we want to report.

**Why not SHAP?** SHAP values compute Shapley values — the average marginal
contribution of each feature across all possible coalitions. This is
theoretically elegant but computationally expensive (2^n coalitions) and
requires approximations for neural networks (KernelSHAP, DeepSHAP). Our
masking approach is faster, simpler, and gives cleaner results for the
question we're asking.

---

## 12. Why retrieval-based evidence?

**Question:** How do we make the model's predictions *trustworthy* to a human
user?

A model that says "lead Incineroar + Rillaboom with 8% confidence" is not
useful unless the user understands *why*. We provide retrieval evidence: "here
are 10 training examples with similar team matchups, and here's what those
experts actually did."

**How it works:**
1. Extract the 128-dim pooled representation from the transformer encoder
   (before the classification head)
2. Build an index of all 246K training example embeddings
3. At query time, find the K nearest neighbors by cosine similarity
4. Show the user: "In similar matchups, experts chose action X (40%), action Y
   (25%), action Z (15%)..."

**Why cosine similarity, not Euclidean distance?** In high-dimensional spaces,
cosine similarity measures *directional* alignment (are the representations
pointing the same way?) rather than *magnitude* (how far apart are they?).
This is more meaningful for learned representations where the norm can vary
for reasons unrelated to semantic similarity.

**Why brute-force search, not FAISS or ANN?** With only 246K vectors of
dimension 128, brute-force cosine similarity takes milliseconds. Approximate
nearest neighbor (ANN) libraries like FAISS add complexity and approximation
error. At our scale, exact search is fast enough.

**Why not attention visualization instead?** Attention weights show which
tokens the model "looks at" but attention ≠ attribution (Jain & Wallace, 2019).
A token can have high attention weight without actually influencing the
prediction. Retrieval evidence is grounded in *real data* — it shows what
actual experts did in similar situations, which is more interpretable and
trustworthy than a heatmap.

---

## 13. Why this data pipeline?

**Question:** How do we go from raw Showdown battle logs to clean training
data?

**Key design decisions:**

**Why `|showteam|` instead of cross-game reconstruction?** Showdown's OTS
format includes a `|showteam|` protocol line with the FULL team sheet (all 6
Pokemon, all moves, all items, all abilities, tera type). This is 100%
complete and 100% reliable. Earlier versions of the project planned to
reconstruct team sheets by aggregating information across multiple games
(seeing which moves a Pokemon uses, which items it consumes). This is
unnecessary with `|showteam|` and would introduce noise from partial
observation.

**Why dedup on (team_a, team_b, action90, format) quadruples?** The same
matchup with the same expert action appearing twice is a true duplicate (same
game processed twice, or two games that are indistinguishable). But the same
matchup with *different* expert actions is NOT a duplicate — it represents
genuine strategic disagreement and preserving it is essential for learning the
multi-modal action distribution.

**Why Tier 1 / Tier 2 stratification?** Bring-4 is only observable when we
can identify all 4 Pokemon the player brought (from switch events in the game
log). In ~80% of games, all 4 brought Pokemon appear; in the remaining ~20%,
some never entered the field. We call the former "Tier 1" and only evaluate
action-90 and bring-4 metrics on Tier 1. Lead-2 is always observable (the
first two Pokemon to enter are always visible), so lead-2 metrics use all
examples.

**Why match_group_id integrity?** Each battle produces two directed examples
(one from each player's perspective). These MUST stay in the same split — if
player A's perspective is in train and player B's perspective is in test, the
model could learn to "cheat" by memorizing the opponent's choice from
training.

---

## 14. Why these baselines?

**Question:** How do we establish that the transformer is actually learning
something useful?

**Popularity baseline (0 parameters):** For each action, compute its frequency
in the training set. Predict the same distribution for every input. This is
the "bet on the most common play" strategy. Top-1: 1.3%, Top-3: 3.9%.

**Conditional popularity baseline (0 parameters):** Same idea, but conditioned
on the mirror/non-mirror stratum. Slightly better because mirror matches have
different action distributions than non-mirror matches.

**Logistic regression (~4K parameters):** Flatten the categorical features into
a bag-of-tokens representation, fit logistic regression (multinomial). This is
the "is there a linear signal?" check. Top-1: 4.0%, Top-3: 10.1%.

**Surprising finding:** Logistic regression has WORSE NLL than popularity
(4.580 vs 4.497). It achieves higher accuracy but its probabilities are badly
calibrated — it's overconfident on wrong predictions. On non-mirror matchups,
its Brier score exceeds 1.0, meaning it's *worse than predicting uniform*.
This validates the transformer + ensemble approach: you need both a flexible
model AND proper calibration.

**Why not random forest / XGBoost?** These are strong for tabular data but
struggle with the set structure of our input. You'd need to hand-engineer
features (e.g., "does team A have a Trick Room user?") rather than learning
representations. The transformer learns its own features from raw tokens.

**Why not a larger transformer / more data?** We explored this question and
concluded the ceiling is label noise (expert disagreement), not model capacity.
When top-1 accuracy is 6.4% and the bootstrap CI is [2.6%, 6.6%], a larger
model would overfit to training conventions rather than learn generalizable
strategy. The 1.16M model is a feature, not a limitation — it's reproducible
on consumer hardware in minutes.

---

## 15. Position invariance: a design decision that paid off

**Question:** Should the model care about the *order* of Pokemon within a team?

**No.** A team of {Incineroar, Rillaboom, Flutter Mane, ...} is the same team
regardless of which Pokemon is listed first. We enforce position invariance
through two mechanisms:

1. **Canonical sorting:** Before encoding, sort Pokemon by a canonical key
   (species name → item → ability → moves). This ensures the same team always
   produces the same input representation.

2. **Mean pooling:** The transformer processes 12 tokens (6 per team), then
   mean-pools over tokens. Mean pooling is invariant to token order by
   definition — swapping two tokens doesn't change the mean.

**Why this matters — the EliteFurretAI leakage:** EliteFurretAI encoded
Pokemon positionally (slot 1 features, slot 2 features, ...) and labeled
actions by slot positions ("bring slots 1,3,4,5"). Because Showdown logs
store teams in a fixed order, the model memorized "this positional pattern →
this action label" and achieved 99.9% accuracy. The author diagnosed this as
"just creating a lookup table" and fixed it with random order augmentation
(accuracy dropped to 79%). Our architecture avoids this entirely by design —
no augmentation needed, no leakage possible.

---

## 16. The top-k coverage curve: reframing accuracy

**Question:** How do we communicate model performance when top-1 accuracy
looks bad?

6.4% top-1 sounds terrible. But this is a problem where experts disagree — the
"right" answer is not unique. The top-k coverage curve plots accuracy as a
function of how many predictions you're allowed to make.

**Key milestones:**
- k=1: 6.4% (one guess)
- k=3: 15.5% (three guesses)
- k=10: 36.2% (ten guesses)
- k=17: 50% (seventeen guesses to reach coinflip)
- k=45 would give 50% for a random model

**The insight:** k=17 for 50% coverage (vs k=45 for random) means the model
concentrates probability on the right **neighborhood** of the action space.
It may not nail the exact action, but it correctly identifies the cluster of
~17 plausible actions. This is much more useful than it sounds — a VGC player
can scan 17 options quickly and pick the one that fits their read of the
opponent.

**The bring-4 vs lead-2 decomposition adds depth:**
- Bring-4 (15-way): 17.8% top-1, 43.7% top-3
- Lead-2 (15-way): 18.7% top-1, 41.8% top-3
- Lead arrangement given correct bring-4 (6-way): 32.1%

Both marginals are around 18% top-1, meaning the model is roughly equally good
at predicting what to bring vs. who to lead. But the conditional (32.1%) is
only 2x random for a 6-way choice, revealing that the harder part is the lead
arrangement — teams agree on what to bring but disagree about who goes in
front.

---

## 17. The speed control hypothesis (a negative result)

**Question:** Do teams with speed control moves (Trick Room, Tailwind) have
more predictable leads?

**The hypothesis:** Trick Room teams must lead the Trick Room setter, so their
leads should be more constrained → lower entropy → higher accuracy.

**The result:** Trick Room teams (66/153) have mean entropy 4.025 vs non-TR
teams (87/153) at 4.041. **Delta: 0.016 nats — essentially zero.** The
hypothesis is busted.

**Why?** The real linearity driver is **mechanical constraints**, not speed
control. The most linear teams are Dondozo commander archetypes (H=3.10),
where the Commander ability requires Dondozo + Tatsugiri to lead together —
this is a hardcoded game mechanic, not a strategic choice. Trick Room teams
actually have substantial lead flexibility because different TR setters
(Indeedee-F, Hatterene, Porygon2) can lead with different partners depending
on the matchup.

**Why report a negative result?** It shows intellectual honesty and prevents
future researchers from pursuing a dead end. It also reveals something genuine
about VGC strategy: speed control is flexible enough that having Trick Room
doesn't determine your leads the way a mechanical constraint like Commander
does.

---

## Glossary of Key Terms

| Term | Definition | Our value |
|------|-----------|-----------|
| **ECE** | Expected Calibration Error: weighted average gap between predicted confidence and actual accuracy across bins | 0.011 |
| **NLL** | Negative Log-Likelihood: -log p(y_true\|x). Lower = the model assigns higher probability to the correct answer | 4.031 |
| **Brier Score** | E[(p - y)²]. Combines calibration + sharpness. Lower = better. >1.0 means worse than uniform | 0.978 (ens) |
| **AURC** | Area Under the Risk-Coverage curve. Lower = better selective prediction | 0.890 |
| **Entropy** | H = -Σ p log p. Measures uncertainty of a distribution. Higher = more uncertain | 4.034 mean |
| **MI** | Mutual Information between prediction and model parameters. Measures epistemic (model) uncertainty | ~0.05 mean |
| **Temperature** | Scalar T dividing logits before softmax. T>1 softens, T<1 sharpens | 1.158 |
| **Tier 1** | Examples where all 4 brought Pokemon are observable (bring4_observed=True) | 80.7% |
| **Action-90** | The joint (lead-2, back-2) decision space. C(6,4)×C(4,2) = 90 | — |
| **Regime A** | Standard split (hold out team variants within cluster) | 247K/35K/40K |
| **Regime B** | OOD split (hold out entire clusters) | 347K/6K/15K |

---

## The 30-Second Elevator Pitch

"We predict which 4 of 6 Pokemon to bring and which 2 to lead in VGC from two
Open Team Sheets — a 90-class problem where experts genuinely disagree on the
right play. Raw accuracy is 6.4% top-1, which sounds bad until you realize
experts disagree with *each other* and our top-17 predictions capture the
right action 50% of the time. The real contribution isn't the classifier —
it's the uncertainty quantification stack around it. Deep ensembles give us
calibrated probabilities (ECE 0.011), selective prediction (abstention doubles
on novel teams), and per-team linearity scores that reveal which archetypes
are predictable (Dondozo commander: 50% top-1) and which aren't (goodstuffs:
0%). No prior work treats VGC team preview as a standalone prediction problem
with UQ."
