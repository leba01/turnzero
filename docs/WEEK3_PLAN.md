# Week 3 Plan: UQ Stack + Calibration + Selective Prediction

## Where We Stand

Week 2 delivered the full train-eval loop: transformer (1.16M params, d=128/L=4/H=4),
baselines (popularity, logistic), eval harness with stratified metrics, paper-ready plots.

**Current best numbers (Transformer, Regime A test):**

| Metric | Overall | Mirror | Non-Mirror |
|--------|---------|--------|------------|
| Action-90 Top-1 | 5.5% | 6.0% | 2.1% |
| Action-90 Top-3 | 14.0% | 14.9% | 6.9% |
| Lead-2 Top-1 | 18.3% | 19.3% | 11.2% |
| Lead-2 Top-3 | 41.0% | 42.5% | 30.5% |
| Action-90 NLL | 4.105 | 4.053 | 4.485 |
| ECE (action90) | 0.016 | 0.014 | 0.035 |

**What's missing for MVP** (Bible Section 7.5):
- Calibration + abstention (temperature scaling, ensemble averaging)
- Risk-coverage curves (both risk definitions)
- Bootstrap CIs on all metrics
- Demo tool (top-3 plans, abstain/scouting mode)

**Why this week matters:** Without UQ, we have "we trained a classifier." With it, we
have "a calibrated decision-support system that knows when to abstain." That's the
entire thesis — team preview is multi-modal, and the right question is whether the
model's top-3 includes the expert's choice AND whether it knows when it doesn't.

## Task Breakdown

### Task 0: Temperature Scaling [~1 hour]

**Create `turnzero/uq/temperature.py`**

Fit a single scalar T > 0 on the validation set to minimize NLL.
- Load best checkpoint, run inference on val split to get logits (not probs)
- Optimize T via L-BFGS or grid search (T typically lands in 1.0-3.0 range)
- Save `temperature.json` artifact: `{T, val_nll_before, val_nll_after, val_ece_before, val_ece_after}`
- At inference time: `calibrated_probs = softmax(logits / T)`

This is the minimal calibration step. Fast, well-understood, big ECE improvement.

**Add CLI command:**
```bash
turnzero calibrate \
  --model_ckpt outputs/runs/run_001/best.pt \
  --val_split data/assembled/regime_a/val.jsonl \
  --out_dir outputs/calibration/run_001
```

**Acceptance:** Val ECE drops meaningfully. Reliability diagram tightens toward diagonal.

### Task 1: Deep Ensembles (5 members) [~2-3 hours compute]

**Create `turnzero/uq/ensemble.py`**

Train 5 independently-initialized transformers (same architecture, different seeds).

**Create `configs/ensemble/` directory with 5 configs:**
```yaml
# member_001.yaml through member_005.yaml
# Identical architecture, different training.seed: 42, 137, 256, 512, 777
```

**Training:**
```bash
for i in 001 002 003 004 005; do
  turnzero train \
    --split_dir data/assembled/regime_a \
    --config configs/ensemble/member_${i}.yaml \
    --out_dir outputs/runs/ensemble_${i}
done
```

These can run sequentially (each takes ~5-10 min on RTX 4080S) or we can batch them.

**Ensemble inference:**
- Load all 5 checkpoints
- For each test example: average softmax probabilities across members
- `p_bar(y|x) = (1/5) * sum_m p_m(y|x)`

**Uncertainty decomposition:**
- Predictive entropy: `H(p_bar)` — total uncertainty
- Mean member entropy: `(1/5) * sum_m H(p_m)` — aleatoric proxy
- Mutual information: `H(p_bar) - mean_member_H` — epistemic proxy (high = members disagree)

**Save:** `outputs/ensemble/ensemble_predictions.npz` with probs, member_probs, entropy, MI arrays.

**Acceptance:** Ensemble top-3 >= single model top-3. Ensemble ECE <= single model ECE.

### Task 2: Risk-Coverage Curves [~1 hour]

**Create `turnzero/eval/risk_coverage.py`**

Sweep confidence threshold tau from 0 to 1 and compute:
- Coverage = fraction of examples where conf >= tau (not abstained)
- Risk Def 1: `1 - top1_accuracy` on non-abstained subset
- Risk Def 2: `P(expert not in top-3)` on non-abstained subset — this is the product-aligned metric

**Outputs:**
- Risk-coverage curve plot (both risk definitions, with baselines for reference)
- AURC (area under risk-coverage curve) — lower is better
- Operating point table: risk @ 95%, 80%, 60% coverage

**Plots:**
```
outputs/plots/week3/risk_coverage_top1.{png,pdf}
outputs/plots/week3/risk_coverage_top3.{png,pdf}
outputs/plots/week3/operating_points.json
```

**Acceptance:** Risk decreases as coverage decreases (confidence is informative).
Top-3 risk at 80% coverage should be meaningfully lower than at 100% coverage.

### Task 3: Cluster-Aware Bootstrap CIs [~30 min code, ~10 min compute]

**Create `turnzero/eval/bootstrap.py`**

Cluster-aware bootstrap (Bible Section 6.4):
- Sample `core_cluster_a` groups with replacement (not individual examples)
- Recompute all metrics on resampled set
- B=1000 iterations
- Report 95% CIs (2.5th and 97.5th percentiles)

**Compute CIs for:** NLL, Brier, ECE, Top-1, Top-3, abstention rate, AURC.

**Save:** `outputs/eval/bootstrap_cis.json`

**Acceptance:** CIs are reasonable widths. No metric's CI crosses a baseline's point estimate
in a way that would undermine the story.

### Task 4: Regime B (OOD) Evaluation [~30 min]

**Run the full eval pipeline on Regime B test set.**

The claim from the bible: "calibrated confidence drops and abstention rises OOD."

- Train a model on Regime B train split (or reuse Regime A model and just eval on Regime B test — the bible is slightly ambiguous here, but the cleaner version is: train on Regime B train, eval on Regime B test)
- Compare: predictive entropy, MI, abstention rate at the tau chosen on within-core val
- Key table: within-core vs OOD metrics side by side

This is a supporting result, not the main story. Don't over-invest.

### Task 5: Comprehensive Week 3 Plots + Tables [~1 hour]

Generate the full paper-grade figure set:
- **Reliability diagrams**: single model vs ensemble, pre vs post temp-scaling
- **Risk-coverage curves**: both risk definitions, ensemble vs single model
- **Comparison table**: all models (pop, logistic, single transformer, ensemble) with CIs
- **Uncertainty histograms**: confidence distribution, entropy distribution, MI distribution
- **Within-core vs OOD comparison table**

All saved to `outputs/plots/week3/`.

### Task 6 (Stretch): Demo Tool Skeleton [~1-2 hours]

**Create `turnzero/tool/coach.py`**

Minimal CLI:
```bash
turnzero demo \
  --model_ckpt outputs/ensemble/ \
  --calib outputs/calibration/run_001/temperature.json \
  --team_a "Incineroar,Rillaboom,Flutter Mane,..." \
  --team_b "Urshifu,Amoonguss,Tornadus,..."
```

Output: top-3 action plans with calibrated probabilities.
If conf < tau: print "Low confidence — consider scouting mode."

This is the minimum viable demo. Retrieval evidence and explanations are Week 4.

## Compute Budget

| Task | GPU Time | Wall Time |
|------|----------|-----------|
| Task 0: Temp scaling | ~2 min | 30 min |
| Task 1: 5 ensemble members | ~50 min | 2-3 hours |
| Task 2: Risk-coverage | ~5 min | 1 hour |
| Task 3: Bootstrap CIs | ~15 min | 1 hour |
| Task 4: Regime B eval | ~10 min | 30 min |
| Task 5: Plots | ~5 min | 1 hour |
| Task 6: Demo skeleton | 0 | 1-2 hours |

Total GPU: ~1.5 hours. Total wall: ~7-9 hours including coding.
The RTX 4080S handles this easily. The bottleneck is ensemble training (5 sequential runs).

## End-of-Week 3 Definition of Done

- [ ] Temperature scaling artifact saved; reliability diagram shows improvement
- [ ] 5-member ensemble trained and evaluated; ensemble beats single model
- [ ] Risk-coverage curves for both risk definitions; AURC numbers reported
- [ ] Bootstrap 95% CIs on all headline metrics
- [ ] Regime B OOD eval showing increased uncertainty
- [ ] All plots paper-ready in `outputs/plots/week3/`
- [ ] CLAUDE.md updated with Week 3 results
- [ ] **MVP stats story locked** — every number in the paper is defensible

## Priority Triage

If time runs short, the **non-negotiable core** is Tasks 0-2 (temp scaling, ensembles,
risk-coverage). That's what transforms the project from "classifier" to "calibrated
decision system." Tasks 3-4 (bootstrap CIs, OOD) strengthen the paper but can be
done in Week 4 if needed. Task 6 (demo) is nice-to-have for Week 3.

## Module Layout After Week 3

```
turnzero/
  uq/                      <- NEW
    __init__.py
    temperature.py          <- Task 0
    ensemble.py             <- Task 1
  eval/
    risk_coverage.py        <- Task 2
    bootstrap.py            <- Task 3
  tool/                     <- NEW (stretch)
    __init__.py
    coach.py                <- Task 6
configs/
  ensemble/                 <- NEW
    member_001.yaml
    ...
    member_005.yaml
outputs/
  calibration/              <- NEW
  ensemble/                 <- NEW
  plots/week3/              <- NEW
```
