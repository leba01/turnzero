# Week 3 Parallel Task Split

## Dependency Graph

```
          ┌────────────────┐
          │  Week 2 Done:  │
          │ best.pt, vocab,│
          │ eval harness,  │
          │ regime_a splits│
          └───────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│  1A    │  │   1B     │  │   1C     │
│  Temp  │  │ Risk-Cov │  │ Ensemble │
│ Scale  │  │ +Bootstrap│  │  Infra  │
│ Module │  │ (code)   │  │ +Configs │
└───┬────┘  └────┬─────┘  └────┬─────┘
    │             │             │
    │         (code ready,      │
    │          needs probs)     │
    ▼             │             ▼
┌────────┐        │       ┌──────────┐
│  2A    │        │       │   2B     │
│Run Temp│        │       │ Train 5  │
│ Scale  │        │       │ Members  │
│(2 min) │        │       │(~50 min) │
└───┬────┘        │       └────┬─────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
            ┌─────┴─────┐
            │    3A     │
            │ Ensemble  │
            │ Eval +    │
            │ Risk-Cov +│
            │ Bootstrap │
            │ + OOD     │
            └─────┬─────┘
                  │
            ┌─────┴─────┐
            │    3B     │
            │  Week 3   │
            │  Plots +  │
            │  Demo     │
            └───────────┘
```

**Wave 1** — three fully independent agents (code only, no GPU):
- **1A** creates `turnzero/uq/temperature.py` + `turnzero/uq/__init__.py` + CLI `calibrate` command
- **1B** creates `turnzero/eval/risk_coverage.py` + `turnzero/eval/bootstrap.py`
- **1C** creates `turnzero/uq/ensemble.py` + `configs/ensemble/member_{001..005}.yaml`

**Wave 2** — GPU-bound, can overlap (temp scaling finishes in minutes):
- **2A** runs temperature scaling on existing best.pt
- **2B** trains 5 ensemble members (sequential GPU, ~10 min each)

**Wave 3** — evaluation + final outputs:
- **3A** ensemble eval + risk-coverage + bootstrap CIs + OOD eval
- **3B** comprehensive plots + demo tool skeleton

---

## Shared Interfaces (ALL agents must respect these)

### Existing interfaces (from Week 2 — do NOT change)

```python
# Model loading (turnzero/models/train.py)
ckpt = torch.load("best.pt", map_location=device, weights_only=False)
# ckpt keys: model_state_dict, vocab_sizes, model_config, config, epoch, val_nll

model = OTSTransformer(vocab_sizes, ModelConfig(**model_config))
model.load_state_dict(ckpt["model_state_dict"])

# Forward pass → logits (NOT probs)
logits = model(team_a, team_b)  # (B, 90) float

# Eval harness
from turnzero.eval.metrics import compute_metrics
results = compute_metrics(probs, action90_true, lead2_true, bring4_observed, is_mirror)
```

### New interfaces for Week 3

#### Temperature scaling artifact (1A defines, 2A produces, 3A consumes)

```json
// outputs/calibration/run_001/temperature.json
{
  "T": 1.87,
  "method": "lbfgs_nll",
  "val_nll_before": 4.1096,
  "val_nll_after": 4.0523,
  "val_ece_before": 0.0162,
  "val_ece_after": 0.0041,
  "model_ckpt": "outputs/runs/run_001/best.pt"
}
```

Usage: `calibrated_probs = softmax(logits / T)`

#### Ensemble prediction contract (1C defines, 2B produces, 3A consumes)

```python
# turnzero/uq/ensemble.py

def ensemble_predict(
    ckpt_paths: list[str],    # 5 checkpoint paths
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,  # apply after averaging logits
) -> dict:
    """
    Returns:
        "probs": (N, 90) float — averaged calibrated probs
        "member_probs": (M, N, 90) float — individual member probs
        "entropy": (N,) float — H(p_bar), total uncertainty
        "mi": (N,) float — mutual information (epistemic)
        "confidence": (N,) float — max p_bar per example
        "action90_true": (N,) int
        "lead2_true": (N,) int
        "bring4_observed": (N,) bool
        "is_mirror": (N,) bool
    """
```

Save as: `outputs/ensemble/ensemble_predictions.npz`

#### Risk-coverage contract (1B defines, 3A produces)

```python
# turnzero/eval/risk_coverage.py

def risk_coverage_curve(
    probs: np.ndarray,       # (N, 90) or (N, 15)
    labels: np.ndarray,      # (N,) int
    k: int = 1,              # top-k for risk definition
    n_thresholds: int = 200,
) -> dict:
    """
    Returns:
        "coverage": (T,) float — fraction not abstained at each threshold
        "risk": (T,) float — 1 - topk_acc on non-abstained
        "thresholds": (T,) float
        "aurc": float — area under risk-coverage curve
        "operating_points": {
            "95": {"threshold": ..., "risk": ..., "coverage": 0.95},
            "80": {...},
            "60": {...},
        }
    """
```

#### Bootstrap CI contract (1B defines, 3A produces)

```python
# turnzero/eval/bootstrap.py

def cluster_bootstrap_ci(
    probs: np.ndarray,
    action90_true: np.ndarray,
    lead2_true: np.ndarray,
    bring4_observed: np.ndarray,
    is_mirror: np.ndarray,
    cluster_ids: np.ndarray,   # core_cluster_a per example
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Returns: {
        "overall/top1_action90": {"mean": ..., "lo": ..., "hi": ..., "std": ...},
        "overall/top3_action90": {...},
        ...
    }
    """
```

Save as: `outputs/eval/bootstrap_cis.json`

---

## Wave 1 Prompts

### Terminal 1A: Temperature Scaling Module

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126).

Read these files first:
- CLAUDE.md
- docs/WEEK3_PLAN.md (Task 0)
- docs/PROJECT_BIBLE.md (section 4.2 — calibration spec)
- turnzero/models/train.py (understand validate() and checkpoint format)
- turnzero/models/transformer.py (ModelConfig)
- turnzero/eval/metrics.py (compute_metrics, _ece)
- turnzero/cli.py (understand click CLI style)

YOUR TASK: Create the temperature scaling module and CLI command.

1. Create turnzero/uq/__init__.py (empty or re-exports)

2. Create turnzero/uq/temperature.py:

   class TemperatureScaler:
       """Post-hoc temperature scaling (Guo et al., 2017).

       Fits a single scalar T > 0 on validation logits to minimize NLL.
       At inference: calibrated_probs = softmax(logits / T)
       """

       def __init__(self):
           self.T: float = 1.0

       def fit(self, logits: np.ndarray, labels: np.ndarray) -> dict:
           """Fit T on validation set using scipy.optimize.minimize (L-BFGS-B).

           Args:
               logits: (N, 90) raw logits from model
               labels: (N,) ground truth action90 ids

           Returns:
               dict with T, val_nll_before, val_nll_after, val_ece_before, val_ece_after
           """

       def calibrate(self, logits: np.ndarray) -> np.ndarray:
           """Apply temperature scaling: softmax(logits / T).

           Returns: (N, 90) calibrated probabilities.
           """

       def save(self, path: str) -> None:
           """Save temperature.json artifact."""

       @classmethod
       def load(cls, path: str) -> "TemperatureScaler":
           """Load from temperature.json."""

   Implementation notes:
   - Optimize T via scipy.optimize.minimize_scalar or L-BFGS-B on a single param
   - Objective: mean NLL = mean(-log softmax(logits / T)[y_true])
   - Bounds: T in [0.01, 50.0] (must be positive)
   - Compute ECE before and after using the _ece helper logic from eval/metrics.py
   - T typically lands in range [1.0, 3.0] for neural nets

3. Also create a helper function in temperature.py:

   def collect_logits(model, loader, device) -> tuple[np.ndarray, dict]:
       """Run inference, return raw logits (N, 90) and labels dict.

       Like validate() in train.py but returns logits instead of probs.
       """

4. Add CLI command to turnzero/cli.py:

   @cli.command("calibrate")
   @click.option("--model_ckpt", required=True, ...)
   @click.option("--val_split", required=True, ...)
   @click.option("--out_dir", required=True, ...)
   def calibrate_cmd(...):
       """Fit temperature scaling on validation set."""

DO NOT touch any files other than:
- turnzero/uq/__init__.py (create)
- turnzero/uq/temperature.py (create)
- turnzero/cli.py (add calibrate command only — do not modify existing commands)
```

### Terminal 1B: Risk-Coverage + Bootstrap

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12).

Read these files first:
- CLAUDE.md
- docs/WEEK3_PLAN.md (Tasks 2 + 3)
- docs/PROJECT_BIBLE.md (sections 4.4 — selective prediction, 6.4 — bootstrap CIs)
- turnzero/eval/metrics.py (understand compute_metrics, _topk_accuracy, _nll, _ece)
- turnzero/eval/plots.py (understand _save_fig pattern and style conventions)

YOUR TASK: Create risk-coverage curves and cluster-aware bootstrap CIs.

1. Create turnzero/eval/risk_coverage.py:

   def risk_coverage_curve(
       probs: np.ndarray,       # (N, C) predicted probabilities
       labels: np.ndarray,      # (N,) ground truth
       k: int = 1,              # 1 for top-1 risk, 3 for top-3 risk
       n_thresholds: int = 200,
   ) -> dict:
       """Sweep confidence threshold and compute coverage vs risk.

       Confidence = max_y p(y|x).
       Coverage = fraction where confidence >= threshold.
       Risk = 1 - top_k_accuracy on the non-abstained subset.

       Returns dict with:
           coverage: (T,) array
           risk: (T,) array
           thresholds: (T,) array
           aurc: float (area under risk-coverage curve, trapezoidal)
           operating_points: dict with keys "95", "80", "60" giving
               the threshold, risk, and exact coverage at those levels.
       """

   def plot_risk_coverage(
       curves: dict[str, dict],  # {"Model A": rc_dict, "Model B": rc_dict}
       out_path: str | Path,
       title: str = "",
       risk_label: str = "Risk (1 - Top-1 Acc)",
   ) -> None:
       """Multi-model risk-coverage plot.

       One line per model. X-axis = coverage (1.0 to 0.0, reversed).
       Y-axis = risk. Include AURC in legend labels.
       Save PNG + PDF at 300 DPI (use _save_fig pattern from plots.py).
       """

2. Create turnzero/eval/bootstrap.py:

   def cluster_bootstrap_ci(
       probs: np.ndarray,          # (N, 90)
       action90_true: np.ndarray,  # (N,)
       lead2_true: np.ndarray,     # (N,)
       bring4_observed: np.ndarray,# (N,) bool
       is_mirror: np.ndarray,      # (N,) bool
       cluster_ids: np.ndarray,    # (N,) str or int — core_cluster_a per example
       n_bootstrap: int = 1000,
       ci_level: float = 0.95,
       seed: int = 42,
   ) -> dict[str, dict[str, float]]:
       """Cluster-aware bootstrap CIs for all metrics.

       Resampling strategy:
       - Get unique cluster IDs
       - For each bootstrap iteration:
         * Sample cluster IDs with replacement
         * Gather all examples belonging to sampled clusters
         * Compute all metrics via compute_metrics()
       - Report percentile CIs

       Returns: {
           "overall/top1_action90": {"mean": ..., "lo": ..., "hi": ..., "std": ...},
           ...
       }
       """

   Notes:
   - Import compute_metrics from turnzero.eval.metrics
   - B=1000 iterations, percentile method (alpha/2 and 1-alpha/2)
   - This will be ~10 min compute for 1000 iterations — print progress every 100
   - Use numpy random Generator with seed for reproducibility

DO NOT touch any files other than:
- turnzero/eval/risk_coverage.py (create)
- turnzero/eval/bootstrap.py (create)
```

### Terminal 1C: Ensemble Infrastructure + Configs

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.
The venv is at .venv/ (Python 3.12, PyTorch 2.10+cu126, RTX 4080 Super).

Read these files first:
- CLAUDE.md
- docs/WEEK3_PLAN.md (Task 1)
- docs/PROJECT_BIBLE.md (section 4.1 — deep ensembles)
- turnzero/models/train.py (understand train(), validate(), checkpoint format)
- turnzero/models/transformer.py (OTSTransformer, ModelConfig)
- turnzero/data/dataset.py (build_dataloaders, Vocab)
- configs/transformer_base.yaml

YOUR TASK: Create ensemble training infrastructure and prediction code.

1. Create configs/ensemble/ directory with 5 YAML configs:
   - member_001.yaml through member_005.yaml
   - Identical to transformer_base.yaml EXCEPT training.seed:
     001→42, 002→137, 003→256, 004→512, 005→777
   - All other hyperparameters identical

2. Create turnzero/uq/ensemble.py:

   def ensemble_predict(
       ckpt_paths: list[str | Path],
       loader: DataLoader,
       device: torch.device,
       temperature: float = 1.0,
   ) -> dict[str, np.ndarray]:
       """Load M checkpoints, run inference, return averaged predictions.

       For each checkpoint:
       - Load model from ckpt
       - Run forward pass on all batches
       - Collect logits

       Then:
       - Average softmax(logits_m / T) across M members
       - Compute uncertainty decomposition

       Returns dict with keys:
           "probs": (N, 90) — averaged calibrated probs p_bar
           "member_probs": (M, N, 90) — per-member probs
           "entropy": (N,) — H(p_bar), predictive entropy
           "member_entropy": (N,) — mean H(p_m), aleatoric proxy
           "mi": (N,) — H(p_bar) - mean H(p_m), epistemic (mutual info)
           "confidence": (N,) — max p_bar per example
           "action90_true": (N,) int
           "lead2_true": (N,) int
           "bring4_observed": (N,) bool
           "is_mirror": (N,) bool
       """

   def save_ensemble_predictions(preds: dict, out_path: str | Path):
       """Save to .npz for later analysis."""
       np.savez(out_path, **preds)

   def load_ensemble_predictions(path: str | Path) -> dict:
       """Load from .npz."""

   Implementation notes:
   - Load each checkpoint exactly like evaluate_checkpoint does in train.py
   - Use @torch.no_grad() and model.eval()
   - Process one model at a time to avoid OOM (don't load all 5 simultaneously)
   - Entropy: H(p) = -sum(p * log(p + eps))
   - Print progress: "Ensemble member 1/5 ..."

3. Create a helper script scripts/train_ensemble.sh:
   ```bash
   #!/bin/bash
   # Train all 5 ensemble members sequentially
   set -e
   for i in 001 002 003 004 005; do
     echo "=== Training ensemble member $i ==="
     turnzero train \
       --config configs/ensemble/member_${i}.yaml \
       --out_dir outputs/runs/ensemble_${i}
   done
   echo "=== All 5 members trained ==="
   ```

DO NOT touch any files other than:
- configs/ensemble/member_001.yaml through member_005.yaml (create)
- turnzero/uq/ensemble.py (create)
- scripts/train_ensemble.sh (create)

Do NOT modify turnzero/uq/__init__.py (another agent creates it).
Do NOT modify turnzero/models/train.py.
```

---

## Wave 2 Prompts

### Terminal 2A: Run Temperature Scaling

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.

Read CLAUDE.md and docs/WEEK3_PLAN.md first.

YOUR TASK: Run temperature scaling on the existing single model.

1. Run the calibrate CLI command:
   turnzero calibrate \
     --model_ckpt outputs/runs/run_001/best.pt \
     --val_split data/assembled/regime_a/val.jsonl \
     --out_dir outputs/calibration/run_001

2. Verify the output:
   - Check outputs/calibration/run_001/temperature.json exists
   - Print the fitted T value and before/after NLL + ECE
   - Sanity check: T should be > 1.0 (neural nets are typically overconfident)

3. Generate a before/after reliability diagram:
   - Run inference on test set with and without temperature scaling
   - Plot both side by side or overlay
   - Save to outputs/plots/week3/reliability_temp_comparison.{png,pdf}

DO NOT modify any source code. This is a run + analysis task.
```

### Terminal 2B: Train Ensemble Members

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.

Read CLAUDE.md and docs/WEEK3_PLAN.md first.

YOUR TASK: Train all 5 ensemble members.

1. Run the ensemble training script:
   bash scripts/train_ensemble.sh

   This trains 5 identical-architecture transformers with different seeds.
   Each member takes ~10 min on the RTX 4080S. Total ~50 min.

2. After all 5 are trained, verify:
   - All 5 checkpoints exist in outputs/runs/ensemble_{001..005}/best.pt
   - Print each member's best_val_nll from run_metadata.json
   - Confirm they have similar but not identical val NLL (diversity check)

3. Quick sanity: run ensemble_predict on a small batch to verify averaging works.

DO NOT modify any source code. This is a compute task.
```

---

## Wave 3 Prompts

### Terminal 3A: Ensemble Eval + Risk-Coverage + Bootstrap + OOD

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.

Read CLAUDE.md, docs/WEEK3_PLAN.md, and check all artifacts in outputs/.

YOUR TASK: Run all Week 3 evaluations.

1. Ensemble evaluation on Regime A test:
   - Load 5 checkpoints + temperature from outputs/calibration/
   - Run ensemble_predict on test set
   - Compute metrics via compute_metrics
   - Save to outputs/ensemble/test_metrics.json
   - Save predictions to outputs/ensemble/ensemble_predictions.npz

2. Risk-coverage curves (both risk definitions):
   - Risk 1: 1 - top1_accuracy (action90, Tier 1)
   - Risk 2: P(expert not in top-3) (action90, Tier 1)
   - Compute for: popularity, logistic, single transformer, ensemble
   - Save curves + operating points to outputs/eval/risk_coverage.json
   - Plot to outputs/plots/week3/risk_coverage_top1.{png,pdf}
   - Plot to outputs/plots/week3/risk_coverage_top3.{png,pdf}

3. Bootstrap CIs:
   - Run cluster_bootstrap_ci on ensemble predictions
   - B=1000, seed=42
   - Save to outputs/eval/bootstrap_cis.json

4. Regime B OOD evaluation:
   - Run ensemble on regime_b/test.jsonl
   - Compare: entropy, MI, abstention rate vs Regime A
   - Save comparison to outputs/eval/ood_comparison.json

5. Print comprehensive summary to stdout.

DO NOT modify any source code. This is a run + analysis task.
```

### Terminal 3B: Week 3 Plots + Demo Skeleton

```
You are working on the TurnZero project (CS229 final project) in /home/walter/CS229/turnzero.

Read CLAUDE.md, docs/WEEK3_PLAN.md, and check all artifacts in outputs/.

YOUR TASK: Generate all paper-ready Week 3 figures and build the demo skeleton.

Part 1 — Plots (save all to outputs/plots/week3/):

1. Reliability diagrams:
   - Single model vs ensemble (side by side or overlay)
   - Pre vs post temperature scaling
   - Both action-90 and lead-2

2. Risk-coverage curves (from saved data in outputs/eval/):
   - Top-1 risk: all models on one plot
   - Top-3 risk: all models on one plot

3. Updated comparison table:
   - All models: popularity, logistic, single transformer, ensemble
   - With bootstrap CIs: "18.3% [17.8, 18.9]" format
   - JSON + LaTeX

4. Uncertainty histograms:
   - Confidence distribution (ensemble)
   - Predictive entropy distribution
   - MI distribution (epistemic)
   - Mirror vs non-mirror overlay

5. Within-core vs OOD comparison:
   - Side-by-side bar chart or table

Part 2 — Demo Tool (stretch):

Create turnzero/tool/__init__.py and turnzero/tool/coach.py:
- Accept two team sheets as CLI input (species list or paste format)
- Load ensemble + temperature
- Output top-3 plans with calibrated probabilities
- If max_confidence < tau: print abstention message

Add CLI command: turnzero demo --ensemble_dir ... --calib ... --team_a ... --team_b ...

DO NOT modify any existing source code. Only create new files.
```

---

## Merge Checklist

After each wave, before starting the next:

1. `git status` — confirm no conflicts (agents touch disjoint files)
2. Quick smoke tests:
   ```bash
   # After Wave 1:
   python -c "from turnzero.uq.temperature import TemperatureScaler"
   python -c "from turnzero.uq.ensemble import ensemble_predict"
   python -c "from turnzero.eval.risk_coverage import risk_coverage_curve"
   python -c "from turnzero.eval.bootstrap import cluster_bootstrap_ci"

   # After Wave 2:
   ls outputs/calibration/run_001/temperature.json
   ls outputs/runs/ensemble_{001..005}/best.pt

   # After Wave 3:
   ls outputs/ensemble/ensemble_predictions.npz
   ls outputs/eval/bootstrap_cis.json
   ls outputs/plots/week3/*.png
   ```
3. Run existing tests: `pytest tests/`
4. Commit the wave: `git add -A && git commit -m "week3 wave N: ..."`

## Compute Budget

| Task | GPU Time | Notes |
|------|----------|-------|
| Temp scaling | ~2 min | L-BFGS on val logits |
| 5 ensemble members | ~50 min | Sequential, ~10 min each |
| Ensemble inference | ~5 min | 5x forward passes on test |
| Bootstrap (B=1000) | ~15 min | CPU-bound, parallelizable |
| OOD eval | ~2 min | Single forward pass |
| Plots | ~1 min | matplotlib only |
| **Total GPU** | **~1.5 hours** | |

The RTX 4080S handles this easily. **Bottleneck is ensemble training (Wave 2B).**
Start 2A (temp scaling) immediately while 2B trains — it finishes in minutes.
