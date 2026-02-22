# Laptop Claude Memo — Feb 21, 2026

Strategic analysis session from Lucas's laptop. PC Claude: read this and factor it into future work.

---

## 1. Should we build a fancier model?

**No.** The current 4-layer transformer (1.16M params) is the right choice. Reasons:

- The paper's story is "UQ infrastructure matters more than the model." A fancier architecture undermines that thesis without changing the fundamental problem (experts disagree, action space is huge).
- Even if a bigger model gets 8% top-1 instead of 6.4%, the UQ argument is identical. Days of work for marginal gains.
- WrestlingMamba already covers the "novel architecture" angle on Lucas's resume. TurnZero is more valuable as an ML engineering / evaluation rigor showcase.
- The model section is intentionally thin and honest — bolting on a half-baked architecture would look worse than the clean framing.

## 2. Ablation null result — INCLUDE IT

The multi-task loss ablation was run (3 loss modes x 5 seeds = 15 runs). Results: ~1% differences across modes, not significant given the wide bootstrap CIs (top-1: [2.6%, 6.6%]).

**Include this as a finding.** Here's why:
- A null result IS a result. "We designed a principled marginalized loss for fabricated labels, tested it, and it didn't matter" is interesting — it means the model is robust to ~20% label noise.
- It closes the loop on Tier 1/Tier 2, which is described in the data section but never addressed in methods/results. A careful reader would notice the gap.
- It shows methodological rigor — identified problem, designed solution, tested it, honestly reported null result.
- One paragraph + small table. Does not dilute the story.

**Recommended addition to turnzero.tex:** A subsection in Results (5.6 or similar) titled "Label Quality Ablation" with a 3-row table (action90_all, multitask, tier1_only) showing top-1, top-3, NLL for each. One paragraph interpreting the null result.

## 3. Conformal prediction — correctly rejected

This was already decided and documented in `docs/TECHNICAL_COMPANION.md` lines 379-385:

> With mean confidence of 5.3%, the 90% coverage sets would contain 20-30 actions (out of 90), making them uninformative. Entropy captures the same information continuously without needing a coverage threshold.

The k=17 for 50% coverage (top-k analysis) already demonstrates the key point. Conformal would add formalism but the sets would be too large to be practically useful. Correct call to drop it.

## 4. What WOULD move the needle (recommended next steps)

### High priority
1. **Streamlit demo** (~half day). Single biggest portfolio impact item. A web demo interviewers can click through in 30 seconds is worth more than any model improvement. Wrap the existing coach tool (marginals, role lexicon, sensitivity, retrieval) in a simple UI.
2. **Fold ablation results into paper** (30 min). One paragraph + small table in the Results section.
3. **Feature ablation table** — the stress test results are currently buried in a text paragraph in Section 5.5. A small table makes them skimmable and adds visual weight to the paper.

### Medium priority
4. **Fix 382K/212K wording in milestone.tex** — currently says "I extract 382K directed examples from 212K battles" which implies 382K = 212K * 2 (it doesn't, because of dedup). Should say "I parse 212K battles into 425K directed examples, then deduplicate to 382K."
5. **Save memory notes on PC** — the memory directory is empty on the laptop. Memories are per-machine, not synced to account. PC Claude should write memories from the full project history.

### Low priority / later
6. **Attention visualization** (~1-2h, noted as stretch in WEEK5_PLAN). Would add to the model interpretability story but isn't essential.

## 5. Project assessment for hiring (honest)

### Strengths that hiring managers notice
- **179 tests.** Top 5% of course projects. Signals "ships reliable code."
- **UQ framing.** Recognizing top-1 accuracy is the wrong metric and building the whole evaluation around calibration, selective prediction, and entropy. Shows ML maturity.
- **Data pipeline rigor.** 7-stage pipeline with leakage prevention, cluster-aware splits, label quality tiering. Not a Kaggle download.
- **Code quality.** Clean modules, typed Python 3.12, Click CLI, dataclasses. Looks like a teammate, not a researcher.
- **GPU engineering literacy.** BF16, torch.compile, pin_memory, GradScaler. Real ML workload experience.

### Weaknesses to be ready for
- **6.4% top-1 is a hard sell in 30 seconds.** Lead with lead-2 numbers (19.8% top-1) and the UQ story. Frame as "5.8x better than random on a 90-way problem where experts regularly disagree."
- **Model is vanilla.** 4-layer transformer with mean pooling, no architectural novelty. Paper correctly frames UQ as the contribution, but some interviewers will probe.
- **No deployment.** CLI only. The Streamlit demo (item 1 above) fixes this.
- **Scale is modest.** 382K examples, 1.16M params, trains in minutes. Doesn't demonstrate large-scale ML.
- **AI assistance question.** CLAUDE.md is an AI-readable project spec. Be ready to discuss any component in depth without the spec.

### Resume positioning
- TurnZero = "I do ML engineering with evaluation rigor and stats"
- WrestlingMamba = "I can do novel architectures"
- Nuha = "I can do hardware-adjacent ML"
- These three are non-overlapping — keep them distinct.

## 6. Target companies (by fit to Lucas's profile)

### Tier 1 — Strong signal match
| Company | Location | Why | Valuation |
|---------|----------|-----|-----------|
| Anduril | Costa Mesa/SF | Sensor fusion (Nuha) + UQ (TurnZero) + PyTorch | $14B+ |
| Hebbia | NYC | BCG + Stanford + deep ML (not prompt eng) | $700M |
| Modal Labs | SF | Full-pipeline builder + dev tools (MCP at Composite) | ~$2.5B |
| Physical Intelligence | SF | Sensor fusion + transformers + state-space models | $5.6B |

### Tier 2 — Good fit
| Company | Location | Why | Valuation |
|---------|----------|-----|-----------|
| Shield AI | San Diego/SF | UQ for autonomous flight | $5.6B |
| Atomic Semi | SF | Hardware-AI crossover, Jim Keller | ~$100M |
| Harvey AI | SF/NYC | BCG credibility + UQ for legal AI | $8B |
| Skydio | Bay Area | CV + autonomy + sensor ML | $2.2B |
| Bayesline | NYC | Probabilistic ML is their product | Early stage, YC |
| Scale AI | SF | Data pipeline expertise | $14B+ |

### Tier 3 — Worth a look
- Patronus AI (SF) — calibration/reliability as a product
- Elicit (SF) — calibrated AI for science
- Glean (Palo Alto) — retrieval + embeddings, $7.2B
- Inworld AI (Mountain View) — game AI

### Rarest skill combination
Hardware-adjacent ML (Nuha) + uncertainty quantification (TurnZero) + consulting communication (BCG). Very few Stanford MS grads have all three. Defense tech and robotics companies value this the most.

## 7. Market context — competitive gaming AI

- **NeurIPS 2025** ran a full Pokemon AI competition (PokeAgent Challenge)
- **ICML 2025 Spotlight**: PokeChamp (Pokemon battle AI)
- **Hacker News front page**: Carli's lead prediction paper (the closest prior work to TurnZero)
- **GTO Wizard** (poker) proves calibrated decision support for imperfect-information games is a real product category
- AI esports coaching market: $412M (2024) → $1.8B projected (2033)
- TurnZero fills a unique gap: no one else does supervised, transformer-based, joint lead+bring prediction from OTS with UQ

---

*Written by laptop Claude, Feb 21 2026. PC Claude: feel free to update, challenge, or extend.*
