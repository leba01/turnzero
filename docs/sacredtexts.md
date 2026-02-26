# Project Bible v4 (FINAL) — OTS Team Preview Coach, Pokémon VGC Gen 9

## 0) EXEC SUMMARY

- We build a **turn-zero coach**: given two Open Team Sheets (6v6 with species/item/ability/tera/moves), it predicts the **human expert preview decision**: **lead-2 + bring-4/back-2** as a single joint action among **90** possibilities.
- The core deliverable is a **reproducible ML pipeline + CLI** that goes from raw Showdown logs → parsed OTS (via `|showteam|`) → canonical dataset → leakage-safe cluster splits → models → calibrated probabilities → paper-grade evaluation → a demo coach tool.
- What makes it "stats cracked": the system is designed around **proper scoring rules** (NLL, Brier), **calibration plots**, **confidence intervals**, and **selective prediction** (abstain when not confident), rather than only top-1 accuracy.
- Evaluation reflects the real deployment goal: **"I'm piloting my team against the open field"** — hold out Team A variants within core clusters, let opponents float freely. Species-only core clusters (≥4/6 overlap; connected components) measure generalization across team families, not sheet memorization.
- We support two test regimes:
  - **Within-core generalization (main story)**: hold out Team A sheet variants within a cluster; opponents unrestricted.
  - **Out-of-core (OOD)**: hold out entire core clusters; abstention should increase.
- All metrics reported with **mirror vs non-mirror stratification** (where `CoreCluster(A) == CoreCluster(B)`).
- The modeling approach is set-structured and OTS-native: **permutation-equivariant Transformer** over 12 mon tokens with **cross-team attention** + side embeddings.
- Uncertainty stack: **deep ensembles** (5 members) + **temperature scaling** (val only) + **risk–coverage curves** for abstention, including "expert not in top-3" as a binary risk.
- "Coach tool" UX: returns **top 3 plans** with calibrated probabilities; if confidence is low it **abstains** and enters **scouting report mode**.
- Explanations are **domain-legible**: threats, why-these-leads (tied to OTS cues), counterfactual plans for opponent likely leads, and retrieval evidence from similar historical matchups.
- Robustness: **moves-hidden test-time ablation** + optional train-time random masking. UNK embeddings are available for graceful degradation when OTS fields are missing, though the primary dataset provides full OTS via `|showteam|`.
- **Data reality**: the raw dataset contains `|showteam|` protocol lines that provide **full OTS** (species, item, ability, tera type, all 4 moves) for 100% of games — verified empirically. No cross-game reconstruction is needed. Bring-4 labels remain **partially observable** (~85% fully observed); lead-2 is always fully observable and serves as the primary supervised task. The pipeline architecture supports UNK embeddings and reconstruction for future datasets without `|showteam|`, but this is not load-bearing for the current dataset.
- Feasibility: ~1 month on a single strong GPU (RTX 4080 class). Architectures stay small; prioritize ensemble size over model size.
- Engineering contracts are explicit: schemas, dedup rules, split correctness, saved artifacts, CLI behavior, reproducibility, and paper-ready outputs.

## 1) SOURCE RESEARCH

### Primary dataset: `cameronangliss/vgc-battle-logs` (VERIFIED)

- **Location**: https://huggingface.co/datasets/cameronangliss/vgc-battle-logs
- **Associated paper**: VGC-Bench (arXiv:2506.10326) — Cameron Angliss et al.
- **Associated code**: https://github.com/cameronangliss/vgc-bench
- **Volume**: 705,487 scraped logs across 16 Gen 9 VGC formats, all filtered for OTS. ~5.11 GB total.
- **Format**: JSON files keyed by battle-id → `(epoch_seconds, raw_battle_log)`. The battle log is **raw Pokémon Showdown protocol text** (pipe-delimited messages).

**Available regulation files (verified):**
- `logs-gen9vgc2024regg.json` (165 MB) — Regulation G ladder
- `logs-gen9vgc2024reggbo3.json` (394 KB) — Regulation G BO3 tournaments
- `logs-gen9vgc2025reggbo3.json` (1.37 GB) — 2025 Reg G BO3
- Plus Reg F, H, I, J variants in both ladder and BO3 formats
- **Primary training set**: Reg G BO3 files (`reggbo3`) for tournament-quality data; supplement with Reg G ladder for volume. **Quality filter for ladder data:** apply a minimum Elo/rating threshold (e.g., ≥1500) and tag each example with `source_quality: "bo3_tournament" | "ladder_filtered"` so metrics can be stratified by data provenance. This defends the "expert imitation" framing.

### What the raw logs contain (VERIFIED)

The raw Showdown protocol exposes:
- **`|showteam|PLAYER|POKEMON_DATA`** — **full OTS** for each side: species, item, ability, all 4 moves, tera type, EVs/IVs. Present in 100% of games in the OTS-filtered dataset. This is the primary source for team sheet features. Format: mons separated by `]`, fields separated by `|` within each mon.
- `|poke|PLAYER|DETAILS|ITEM` — species + item flag for all 6 mons (team preview phase). Redundant with `|showteam|` but useful as a cross-check.
- `|switch|PLAYER|POKEMON|HP` — reveals leads and all subsequent switches
- `|move|PLAYER|MOVE|TARGET` — moves actually used in the game
- `|-ability|POKEMON|ABILITY` — abilities when they visibly trigger
- `|-terastallize|POKEMON|TYPE` — tera type only if player terastallizes
- `|-item|` / `|-enditem|` — items only when consumed/activated

**Key finding**: because `|showteam|` provides complete OTS, the reconstruction pipeline (cross-game aggregation, VGCPastes matching) documented in earlier versions of this bible is **not needed** for this dataset. The architecture retains UNK embedding support for future datasets that may lack `|showteam|`.

> **For datasets without `|showteam|`:** the protocol messages after `|poke|` only reveal partial OTS — moves used, abilities triggered, items consumed, tera if activated. A reconstruction strategy (cross-game aggregation by species-key, VGCPastes matching, top-4-by-frequency move selection) would be needed. See git history for the full reconstruction spec.

### Extractability of labels (VERIFIED)

- **Leads (lead-2)**: ✅ Always extractable from first `|switch|` messages at game start. Fully observable.
- **Bring-4**: ⚠️ **Partially observable.** Collected by tracking all unique mons that appear in any `|switch|` or `|drag|` message per side. However, this only recovers mons that *actually entered the field*. If a game ends quickly (e.g., 3-turn sweep), a back mon that was brought but never switched in is indistinguishable from a benched mon. Full bring-4 is only reliably known when `num_revealed == 4` for that side. The VGC-Bench codebase confirms this limitation — their `logs2trajs.py` randomly samples back choices during team preview and patches them post-hoc using revealed data, acknowledging that the full bring-4 is not consistently recoverable.
- **Back-2**: ⚠️ Derived as `bring-4 minus lead-2` — inherits the partial observability of bring-4.

See Section 2.2 "Label Observability Policy" for how this is handled in training and evaluation.

### Supplementary data source: VGCPastes (optional, for future use)

- The VGC-Bench repo includes `scrape_teams.py` for collecting tournament teams with full sets from VGCPastes.
- Not needed for the current dataset (full OTS available via `|showteam|`), but useful for: (a) cross-referencing team builds with tournament provenance, (b) future datasets without `|showteam|`.

### Method citations for paper writeup

- **Deep Ensembles**: Lakshminarayanan et al., 2017. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS.
- **Temperature scaling**: Guo et al., 2017. "On Calibration of Modern Neural Networks." ICML. (arXiv:1706.04599)
- **Selective prediction / risk–coverage**: Geifman & El-Yaniv, 2017. "Selective Classification for Deep Neural Networks." NeurIPS.
- **DeepSets**: Zaheer et al., 2017. "Deep Sets." NeurIPS.
- **Set Transformer**: Lee et al., 2019. "Set Transformer: A Framework for Attention-based Permutation-Invariant Input." ICML.
- **Conformal prediction**: Angelopoulos & Bates, 2021. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification."
- **Proper scoring rules**: Brier, 1950; NLL / log loss (standard in calibration literature).
- **Group-aware CV**: scikit-learn `GroupKFold` — standard practice for clustered/hierarchical data splits.

## 2) DATA + PIPELINE SPEC

### 2.1 Canonical data schema

#### TeamSheet
```yaml
TeamSheet:
  team_id: string            # stable hash of canonicalized content
  species_key: string        # hash of sorted species-only set (for core clustering)
  format_id: string          # e.g., "gen9vgc_regg_bo3"
  source:
    dataset_name: string
    dataset_version: string
    raw_id: string           # replay id / match id from source
  reconstruction_quality:
    fields_known: int        # count of non-UNK fields across all 6 mons (expect ~42/42 with |showteam|)
    fields_total: int        # total possible fields (6 mons × 7 fields each = 42)
    source_method: string    # "showteam_direct" | "log_aggregation" | "vgcpastes_match" | "partial"
  pokemon:                   # exactly 6
    - Pokemon
    - Pokemon
    - Pokemon
    - Pokemon
    - Pokemon
    - Pokemon
```

#### Pokemon
```yaml
Pokemon:
  species: string            # canonical species name (always known from preview)
  item: string | "UNK"      # unknown if never revealed in logs
  ability: string | "UNK"
  tera_type: string | "UNK"
  moves:                     # length 4; may contain "UNK" for unrevealed moves
    - string | "UNK"
    - string | "UNK"
    - string | "UNK"
    - string | "UNK"
```

#### MatchExample (directed, from the perspective of "player A")
```yaml
MatchExample:
  example_id: string         # stable hash; unique row id
  match_group_id: string     # leakage-control grouping id (see dedup)
  split_keys:
    team_a_id: string
    team_b_id: string
    core_cluster_a: string
    core_cluster_b: string
    is_mirror: bool          # true if core_cluster_a == core_cluster_b
  metadata:
    source_dataset: string
    tournament_id: string | null
    set_id: string | null    # BO3 set identifier if available
    game_number: int | null  # 1/2/3 if available
    format_id: string
    timestamp_epoch: int | null
  team_a: TeamSheet          # "my team" in tool usage
  team_b: TeamSheet          # "opponent team"
  label:
    lead2_idx: [int, int]    # indices 0..5 into canonical-ordered team_a.pokemon
    back2_idx: [int, int]    # indices among remaining 4
    action90_id: int         # 0..89 mapped from (lead2, back2)
    label_quality:
      bring4_observed: bool  # true if all 4 brought mons appeared in switch logs
      notes: string | null
```

### 2.2 Parsing plan

#### Raw log format
Each entry in the JSON files is `{"battle-id": (epoch_seconds, raw_log_text)}`. The raw log is Pokémon Showdown pipe-delimited protocol. This is NOT structured OTS data — it must be parsed.

#### Stage: `download/ingest`
- Download JSON files from HuggingFace (`cameronangliss/vgc-battle-logs`).
- Record `manifest.json` with dataset name, version, retrieval date, file checksums.
- Primary files: `logs-gen9vgc2024reggbo3.json`, `logs-gen9vgc2025reggbo3.json` (BO3 tournament data). Supplement with ladder files for volume.

#### Stage: `parse`
For each raw battle log:
1. **Extract full OTS from `|showteam|`**: parse the `|showteam|` line for each player. Format: mons separated by `]`, fields within each mon separated by `|`. Extract species (field 0), item (field 2), ability (field 3), moves (field 4, comma-separated), tera type (last element of field 11, comma-separated). This gives complete OTS for all 6 mons per side.
2. **Extract leads**: first `|switch|` messages for each player at game start. Map switched-in species back to position in the `|showteam|`-parsed team.
3. **Extract bring-4 (partial)**: collect all unique mons that appear in any `|switch|` or `|drag|` message per side throughout the entire game. Record `num_revealed_a` and `num_revealed_b` (count of unique mons that appeared on each side). Mons that never appear in `|switch|` were *either* benched or brought-but-never-entered — these are indistinguishable. Set `bring4_observed = (num_revealed == 4)` per side.
4. **Handle edge cases**: mons with <4 moves (e.g., Ditto with only Transform) pad with `"UNK"` to length 4. Names in `|showteam|` use CamelCase (e.g., `FakeOut`, `AssaultVest`) — canonicalization happens in the next stage.
5. Create two directed examples by swapping perspective (player1→A, player2→B and vice versa).
6. Emit `MatchExample` JSONL rows with `source_method: "showteam_direct"` and `fields_known` count.

**Note on `|showteam|` vs partial extraction:** Earlier versions of this bible described a reconstruction pipeline for datasets where `|showteam|` is absent and OTS must be pieced together from `|move|`, `|-ability|`, `|-item|`, etc. That design is preserved in git history but is **not needed** for the current dataset, where `|showteam|` provides 100% OTS coverage. The model retains UNK embedding support for graceful degradation if applied to datasets without `|showteam|`.

#### Label Observability Policy (INTEGRITY-CRITICAL)

Bring-4/back-2 labels are **not always fully observable** from Showdown logs. If a game ends before all 4 brought mons enter the field, we cannot distinguish "brought but never switched in" from "left on bench." This is confirmed by the VGC-Bench codebase (`logs2trajs.py`), which randomly samples back choices and patches them post-hoc.

**Policy:**

1. **`bring4_observed` flag**: every `MatchExample` records `bring4_observed = (num_revealed_mons == 4)` for Team A's side. This is set during the parse stage and is immutable.

2. **Lead-2 is the primary supervised task.** Leads are always fully observable (first `|switch|` messages). All headline accuracy and scoring metrics are reported for lead-2.

3. **Bring-4/back-2 uses a tiered approach:**
   - **Tier 1 (fully supervised):** train and eval on the subset where `bring4_observed == True`. Report bring-4 accuracy, NLL, and all metrics on this clean subset. This is the defensible number for the paper.
   - **Tier 2 (partial-label / lower bound):** on examples where `bring4_observed == False`, the revealed mons are a *lower bound* on the true bring-4. These examples can still be used for training with a partial-label loss (e.g., only penalize if the model's predicted bring-4 contradicts revealed mons), but eval metrics on this subset are reported separately with a "partial observation" caveat.
   - Report the fraction of examples in each tier as a dataset statistic.

4. **The 90-way action space** (joint lead-2 + back-2) is only evaluated on Tier 1 examples. On Tier 2 examples, fall back to evaluating lead-2 only.

5. **In the coach tool UX:** always output both lead-2 and bring-4 recommendations (the model predicts the joint action regardless). But the paper clearly separates "fully supervised" metrics from "partially observed" metrics.

### 2.3 Canonicalization rules

**Name normalization**
- `|showteam|` uses CamelCase names (e.g., `FakeOut` → `Fake Out`, `AssaultVest` → `Assault Vest`, `GrassySurge` → `Grassy Surge`).
- Conversion rule: insert space before uppercase runs. Keep hyphenated forms unchanged: `Urshifu-Rapid-Strike` stays as-is. Maintain a small exception dict for edge cases.
- Maintain a bidirectional mapping table for reversibility.
- Species: canonical display names (no nicknames, no forme abbreviations).
- Moves, items, abilities, tera_type: canonical display names.

**Move order normalization**
- Treat the 4 moves as an unordered set:
  - `moves_sorted = sorted(moves)` with `"UNK"` sorted last.
- Model input must not depend on move order.

**Team order invariance**
- Define each mon's canonical key:
  - `pk = f"{species}|{item}|{ability}|{tera_type}|{','.join(moves_sorted)}"`
- Sort the 6 mons by `pk` ascending.
- `team_id = hash(join(sorted pk list with separator))`.

### 2.4 Dedup rules + leakage prevention

**Goal:** prevent information leakage across splits while preserving multi-modal signal.

**Key definitions:**
- `team_key(team)` = canonical string used to hash `team_id`.
- `matchup_key_undirected` = `sort([team_id_A, team_id_B])`.
- `match_group_id`:
  - If the raw source has a stable BO3 set ID, prefer `set_id` grouping.
  - Else use `matchup_key_undirected` + tournament tag + date bucket.

**Exact-duplicate detection**
An example is a duplicate (remove it) if ALL of these match another example:
- `team_id_A` identical,
- `team_id_B` identical,
- same `action90_id` (lead2_idx + back2_idx),
- same `format_id`.

**Critical: same matchup with different actions is NOT a duplicate.**
The same team pair facing each other multiple times with different expert decisions is valuable signal — it captures the multi-modality of team preview (multiple valid lines). Do NOT dedup by matchup pair alone.

**Cross-split leakage guard:**
Remove any `(team_id_A, team_id_B, action90_id)` triple that appears in both train and test. This prevents exact memorization while keeping the multi-modal signal within each split.

**Match group constraint:**
All rows with the same `match_group_id` must be in exactly one split.

### 2.5 Core clustering algorithm (species-only, ≥4 overlap CCs)

**Definition**
- Each team represented by `species_set(team) = {species_0..species_5}`.
- Edge between teams `i` and `j` if `|species_set_i ∩ species_set_j| ≥ 4`.
- Cluster = connected component of this graph.

**Efficient union-find via 4-subset index**
```python
def core_cluster_teams(team_species_sets):
    # team_species_sets: dict[team_id] -> sorted list[str] of 6 species
    uf = UnionFind(team_species_sets.keys())
    index = defaultdict(list)  # key: tuple of 4 species -> list[team_id]

    for team_id, species6 in team_species_sets.items():
        for comb4 in combinations(species6, 4):
            key = tuple(comb4)
            for other_id in index[key]:
                uf.union(team_id, other_id)
            index[key].append(team_id)

    root_to_cluster = {}
    clusters = {}
    for team_id in team_species_sets.keys():
        root = uf.find(team_id)
        if root not in root_to_cluster:
            root_to_cluster[root] = f"cluster_{len(root_to_cluster)}"
        clusters[team_id] = root_to_cluster[root]
    return clusters
```

### 2.6 Train/val/test creation

Produce two split regimes from the same canonical dataset.

#### Regime A: "Pilot my team vs the field" (MAIN)

**Objective:** test on held-out Team A variants within a core cluster. Opponents are unrestricted — they can appear in any split. This matches the deployment scenario: you have your team, you face the open field.

**Procedure**
1. Compute `core_cluster_a = core_cluster(team_a_id)` for all examples.
2. For each core cluster independently:
   - Collect all unique `team_a_id` values belonging to this cluster.
   - Shuffle with fixed seed.
   - Assign `team_a_id` groups to train/val/test (80/10/10).
   - If cluster is small, ensure ≥1 team in test when possible.
3. Assign each example to the split of its `team_a_id`:
   - If `team_a_id` is in train group → example goes to train.
   - If `team_a_id` is in val group → example goes to val.
   - If `team_a_id` is in test group → example goes to test.
4. **Opponents (Team B) are NOT constrained** — the same `team_b_id` can appear as opponent in train, val, and test. This is correct because at deployment time, you don't control who you face.
5. Remove any `(team_a_id, team_b_id, action90_id)` exact triples that appear across splits.
6. Enforce `match_group_id` does not cross splits.

**Directed-example conflict resolution (IMPLEMENTATION-CRITICAL):**
Each raw game produces two directed examples (swap sides). These share a `match_group_id`, but may have different `team_a_id` values pointing to different splits. Resolution policy:
- Assign the `match_group_id` to the split implied by the **higher-priority direction** (`test > val > train`). That is: if either direction's `team_a_id` maps to test, the entire match group goes to test.
- The other direction is **kept** if its `team_a_id` also maps to the same split, or **dropped** if it would violate Regime A (i.e., its `team_a_id` is a train team but the match group was assigned to test).
- Track `directed_example_drop_rate` as a dataset statistic. Expected to be small (<5%) but must be reported.
- Same logic applies to BO3 sets: all games in a set share a `match_group_id` and follow the same rule.

**Key property:** A test `team_a_id` never appears as Team A in train. It CAN appear as Team B (opponent) in train — this is an explicit design choice, not an oversight. The deployment story is: **"the coach is trained on the full historical metagame; it may have seen your team in opponent logs, which is realistic."** This does not leak *preview decisions* (labels), only OTS features — which are public knowledge via OTS anyway.

#### Regime B: Out-of-core holdout (OOD)

**Objective:** hold out entire core clusters; measure abstention increase.

**Procedure**
1. Rank clusters by number of examples or teams.
2. Select clusters for OOD test (e.g., 20% by weight) using fixed seed.
3. Assign `train_clusters`, `val_clusters`, `test_clusters_ood`.
4. Create:
   - `train`: examples where `core_cluster_a` ∈ `train_clusters`
   - `val`: examples where `core_cluster_a` ∈ `val_clusters`
   - `test_ood`: examples where `core_cluster_a` ∈ `test_clusters_ood`

**Expected behavior:** calibration degrades and abstention rises on `test_ood`, especially for unseen cores.

#### Reporting stratification (applies to both regimes)

For all test-set metrics, report separately:
- **Overall**: all test examples.
- **Mirror**: examples where `core_cluster_a == core_cluster_b`.
- **Non-mirror**: examples where `core_cluster_a != core_cluster_b`.

Mirrors are where you'll most often see multiple valid lines — perfect for showcasing calibrated uncertainty and top-3 recommendations.

## 3) MODELING SPEC (ranked options)

### Mandatory baselines

#### Baseline 1: Popularity / frequency
- Global: `P(action90)`.
- Core-conditional: `P(action90 | core_cluster_a, core_cluster_b)`.
- Team-conditional: `P(action90 | team_id_a, team_id_b)` with Laplace smoothing.
- Output: full 90-way probability vector.
- Confidence = max probability; abstain if below threshold.

#### Baseline 2: Multinomial logistic regression with engineered OTS features
- Sparse one-hot / feature hashing:
  - Per side, per mon (order-invariant): species, item, ability, tera, move presence.
  - Pairwise cross-features: `speciesA=x AND speciesB=y`, `moveA=m AND speciesB=y`.
- Model: multinomial logistic → 90 logits.
- Loss: cross-entropy.

#### Baseline 3 (optional): Gradient boosting (CatBoost)
- Use ordered target statistics to avoid leakage in target encoding.
- Keep optional if leakage risk is too high.

### Main model family

#### Option A (recommended): Permutation-equivariant Transformer over 12 mons

**Input token embedding** for each Pokémon:
- `E = E_species + E_item + E_ability + E_tera + sum(E_move_i) + E_side`
- Unknown fields map to learned `UNK` embeddings.

**Architecture:**
- 12 Pokémon tokens → L layers of self-attention (no positional encodings) → pooled representation via learned `[CLS]` token or mean-pool.
- Feed pooled vector into MLP head → 90 logits.

**Output:** 90-way softmax.

**Loss:** cross-entropy on action90.

**Regularization:** dropout in attention + MLP, label smoothing (0.02–0.05), weight decay.

**Training recipe:** AdamW, batch size 256–1024, early stopping on validation NLL, mixed precision (AMP).

#### Option B: DeepSets + cross-interaction pooling (lighter)
- Per-mon MLP → mon embeddings.
- Team embedding = sum/mean over mon embeddings.
- Cross interaction = pooled pairwise (dot products or MLP over concatenated pairs).
- Concatenate `[teamA_embed, teamB_embed, cross_embed]` → MLP → 90 logits.

#### Option C: Factorized selection head (lead + back)
- 15 lead-pair scores via softmax.
- Conditional back-pair distribution given lead.
- Loss: `L = CE(lead_pair) + CE(back_pair | lead_pair)`.
- Produces clean marginals: `P(mon in leads)`, `P(mon in back)`.

### Symmetry augmentation
For every parsed game, create two directed examples by swapping sides. This doubles data and reduces position bias.

## 4) UNCERTAINTY + CALIBRATION SPEC

### 4.1 Deep ensembles (mandatory)

- Train `M=5` independently initialized models (stretch: `M=7–10`).
- Predictive distribution: `p̄(y|x) = (1/M) Σ_m p_m(y|x)`.

**Uncertainty scores:**
- Confidence (for abstention): `conf = max_y p̄(y|x)`.
- Predictive entropy: `H(p̄)`.
- Ensemble mutual information (epistemic proxy): `MI = H(p̄) - (1/M) Σ_m H(p_m)`.

### 4.2 Calibration (mandatory)

**Temperature scaling (default):**
- Fit scalar `T > 0` on validation split to minimize NLL.
- For ensembles: average logits across members, then temperature scale.
- Save `T` and validation calibration metrics as artifacts.
- **Calibrate and choose all thresholds on validation only. Never touch test.**

### 4.3 Metrics and plots (mandatory)

Compute for: overall test set, within-core, out-of-core, mirror, non-mirror — all stratifications.

**Proper scoring:**
- **NLL (log loss):** mean `-log p(y_true)`.
- **Brier score (multiclass):** mean `||p - onehot(y_true)||²`.

**Classification:**
- Top-1 accuracy.
- Top-3 accuracy (aligned with tool UX).
- Top-5 accuracy (optional).

**Calibration:**
- **ECE:** bin by confidence, compute weighted `|acc - conf|` per bin.
- **Reliability diagram:** accuracy vs confidence per bin + confidence histogram.

### 4.4 Selective prediction / abstention (mandatory)

**Confidence score:** `conf = max_y p_cal(y|x)` (calibrated, post temperature scaling).

**Policy:**
- If `conf ≥ τ`: output top-3 plans with probabilities.
- Else: abstain → scouting report mode.

**Risk–coverage curves (mandatory):**
Sweep τ and report:
- Coverage = fraction not abstained.
- **Risk definition 1 (standard):** `1 - top1_accuracy` on non-abstained subset.
- **Risk definition 2 (product-aligned):** `P(expert action NOT in top-3)` on non-abstained subset. This directly matches the tool's UX: "did our top-3 recommendations include what the expert chose?"
- Report AURC and operating points at 95%, 80%, 60% coverage.

### 4.5 Optional conformal layer (stretch)

- Nonconformity score: multiclass APS-style.
- Output: prediction set with coverage guarantee ≥ `1−α`.
- Report: coverage on calibration split, average set size.

## 5) INTERPRETABILITY + EXPLANATIONS SPEC

### 5.1 Retrieval-based evidence (mandatory)

**Index:** build embedding for each `MatchExample` using pooled representation before final head.

**Query:** cosine similarity, retrieve `k=20–50` neighbors. Filters: min species overlap, same core cluster (optional), exclude exact same team pair.

**Output:** "Similar matchups (k=10 shown)" with species overlap, expert leads/brings, frequency table of top actions. Evidence summary: "In similar games, experts led X+Y 42% and brought Z in 61%."

**Leakage check:** retrieval corpus must not include test examples.

### 5.2 Domain-legible cue extraction (mandatory)

OTS role lexicon mapping moves/items/abilities to player-relevant tags:
- Speed control: Tailwind, Trick Room, Icy Wind, Electroweb
- Redirection: Follow Me, Rage Powder
- Disruption: Fake Out, Taunt, Encore, Will-O-Wisp
- Spread damage: Heat Wave, Muddy Water, Rock Slide
- Priority: Extreme Speed, Sucker Punch
- Weather/terrain: abilities + moves
- Defensive tech: Protect, Wide Guard, Helping Hand
- Item roles: Sitrus, Choice items, Focus Sash, Safety Goggles

This is a deterministic annotation layer, not simulation.

### 5.3 Threats explanation (mandatory)

- `P(opponent leads pair)` from symmetric model or marginals.
- Threat score = `P(opp leads mon)` × `coverage_score(mon → our team)`.
- Output: "Likely lead: X (52%) — has Fake Out + Intimidate; pressures your Y/Z leads."

### 5.4 "Why these leads" explanation (mandatory)

- Report marginals: `P(mon i is led)` and `P(mon i is brought)` from 90-way distribution.
- Highlight top lead reasons tied to OTS cues via lexicon.

### 5.5 Counterfactual / sensitivity (mandatory)

**Counterfactual 1:** Conditional response to likely opponent leads.
- Predict opponent top-2 lead pairs.
- For each: "If they lead X+Y, your best response is lead A+B; bring C/D."

**Counterfactual 2:** Feature masking sensitivity.
- Mask one cue at a time, measure delta in plan probabilities.
- Surface biggest deltas: "Your plan depends heavily on opponent having Trick Room."

## 6) ROBUSTNESS / SHIFT EVAL SPEC

### 6.1 Within-core results (mandatory)

Report for within-core test split (all stratifications: overall, mirror, non-mirror):
- Top-1 / Top-3 accuracy
- NLL and Brier
- ECE + reliability diagram
- Risk–coverage curves (both risk definitions from 4.4)
- Breakdown by cluster size deciles (optional)

### 6.2 Out-of-core OOD (mandatory)

For `test_ood`:
- Same metrics as above, plus:
  - Abstention rate at fixed τ chosen on within-core validation
  - Mean predictive entropy and MI shift
- Key claim: calibrated confidence drops and abstention rises OOD.

### 6.3 Moves-hidden stress test (mandatory)

**Note:** Because `|showteam|` provides full OTS, the model trains with complete information. The stress test measures robustness to **test-time information loss** — simulating scenarios where the user doesn't have the opponent's full team sheet, or where the model is applied to datasets without `|showteam|`.

Test-time ablations:
- Hide k=2 moves per mon (partial)
- Hide k=4 moves (full moves hidden)
- Optional: hide tera, hide items

Report: accuracy/top-3, NLL, ECE, abstention vs % moves hidden.

**Train-time mitigation (recommended):**
- Random masking with probability `p_mask=0.2–0.4`.
- Report whether masking improves robustness without harming within-core performance.

### 6.4 Confidence intervals (mandatory)

**Cluster-aware bootstrap:**
- Sample `TeamSheetKey_A` groups (or core clusters) with replacement.
- Recompute metrics each time.
- B=1000 bootstraps (200 for iteration). CI = percentile (2.5, 97.5).
- Compute 95% CIs for: NLL, Brier, ECE, Top-1, Top-3, abstention rate, AURC.

## 7) IMPLEMENTATION CONTRACTS FOR CLAUDE CODE

### 7.1 Required pipeline stages

**download/ingest**
- Inputs: HuggingFace dataset identifiers + output dir
- Outputs: raw JSON files + `manifest.json` with hashes

**parse**
- Inputs: raw JSON dir
- Outputs: parsed JSONL of `MatchExample` with full OTS from `|showteam|`, `bring4_observed` flags, parse stats report

**canonicalize**
- Inputs: parsed examples
- Outputs: canonicalized dataset (name normalization, team order, move order normalized), team_id hashes, deduped

**cluster**
- Inputs: canonicalized examples
- Outputs: cluster assignments (`team_id → cluster_id`), cluster stats

**split**
- Inputs: canonicalized examples + cluster assignments + split config
- Outputs: train/val/test splits for Regime A and Regime B, with directed-example conflict resolution and cross-split triple dedup applied

**assemble**
- Inputs: canonicalized examples + clusters + splits
- Outputs: per-split JSONL files ready for training, dataset manifest

**stats**
- Inputs: assembled dataset
- Outputs: comprehensive stats report + all integrity validation assertions

**train**
- Inputs: split_dir + model config
- Outputs: model checkpoint(s) + training logs

**calibrate**
- Inputs: model_ckpt + val split
- Outputs: calibration artifact(s) + calibration metrics

**eval**
- Inputs: model_ckpt + calibration artifacts + test split(s)
- Outputs: JSON metrics + paper-ready plots/tables + bootstrap CIs

**demo**
- Inputs: model_ckpt + calibration artifacts + two team sheets
- Outputs: top-3 plans + abstain/scouting + explanations + retrieval evidence

### 7.2 Required CLI contracts

```
download --out_dir …
parse --raw_dir … --out_dir … [--limit N]
canonicalize --in_path … --out_dir …
cluster --in_path … --out_dir …
split --in_path … --clusters … --out_dir … --seed 42
assemble --canonical_dir … --clusters … --splits_dir … --out_dir …
stats --data_dir … --validate
train --split_dir … --config … --out_dir …
calibrate --model_ckpt … --val_split … --out_dir …
eval --model_ckpt … --calib … --test_split … --out_dir …
demo --model_ckpt … --calib … --team_a … --team_b …
```

**Reproducibility contract:** every command logs full config, random seeds, writes `run_metadata.json` with git hash.

### 7.3 Required artifacts

- Canonical dataset files with `{TeamSheetA, TeamSheetB, label}` format
- `splits.json`: train/val/test ids for Regime A + Regime B
- `cluster_assignments.json`: `team_id → cluster_id`
- `dataset_report.json`: OTS completeness, bring4_observed rates, split sizes, cluster stats, dedup stats
- Model checkpoint(s)
- Calibration artifacts: temperature `T`, calibration metrics
- Evaluation metrics as machine-readable JSON
- Paper-ready figures:
  - Reliability diagram(s) (overall + stratified)
  - Risk–coverage curves (both risk definitions)
  - Top-k accuracy + NLL/Brier summary table
  - Moves-hidden stress test plots
  - Mirror vs non-mirror comparison table
- Retrieval index artifacts (if implemented)

### 7.4 Week-by-week plan

**Week 1: data pipeline + clustering + splits + baselines**
- Download vgc-battle-logs from HuggingFace.
- Implement package scaffolding: schemas, action space bijection (90-way), CLI entry points.
- Implement full Showdown protocol parser extracting OTS from `|showteam|`, leads, bring-4 with `bring4_observed` flag.
- Implement canonicalization (CamelCase → display names, move/team order normalization, stable hashing) + dedup.
- Implement core clustering (union-find over 4-subset index).
- Implement both split regimes (Regime A: pilot-my-team with directed-example conflict resolution, Regime B: OOD).
- Assemble final per-split JSONL files.
- Run comprehensive stats report + integrity validation.
- Implement popularity and linear baselines.
- Produce first within-core baseline metrics (lead-2 primary, action90 on Tier 1 subset).

**Week 2: main model + training loop + eval harness**
- Implement Transformer set model (Option A).
- Build training harness with configs and deterministic behavior.
- Implement evaluation metrics + reliability diagram + top-k tables.
- Establish paper-ready plotting pipeline with mirror/non-mirror stratification.

**Week 3: UQ stack + calibration + selective prediction**
- Train ensembles (5 members).
- Implement temperature scaling on validation.
- Implement risk–coverage evaluation (both risk definitions) and abstention thresholds.
- Compare within-core vs OOD abstention behavior.
- Lock MVP stats story.

**Week 4: explanations + robustness + demo polish + paper figures**
- Build retrieval index and evidence output.
- Implement threats + why-leads + counterfactual modules.
- Implement moves-hidden stress tests.
- Generate final CI-backed tables and figures.
- Polish CLI demo and write paper-quality report.

### 7.5 MVP vs Stretch

**MVP**
- Showdown protocol parser extracting full OTS from `|showteam|` + bring4_observed tracking.
- Canonical pipeline: parse → canonicalize → cluster → split → assemble → validate.
- Baseline + one set model.
- Calibration + abstention.
- Within-core evaluation with reliability + risk–coverage plots.
- Mirror vs non-mirror stratified reporting.
- Demo tool: top 3 plans, abstain/scouting, retrieval evidence.

**Stretch**
- Conformal wrapper for prediction sets.
- Strong OOD evaluation with MI-based abstention.
- Richer counterfactual response model.
- Embedding-based ANN retrieval.
- More robustness ablations (tera/items hidden).
- Factorized head (Option C) for cleaner marginals.

### 7.6 Suggested logical modules

- `data`: Showdown parser (`|showteam|` extraction), canonicalize, schema, dedup, manifests, assembly
- `splits`: clustering + split generation + validators
- `models`: baselines + set model(s) + heads
- `uq`: ensembles, calibration, abstention policies
- `eval`: metrics, plots, bootstraps, stratified reporting
- `tool`: coach demo interface (CLI-first; optional web UI later)
- `docs`: assumptions, reproducibility, dataset verification, source links

## 8) ACCEPTANCE CRITERIA

- [ ] Showdown protocol parser correctly extracts full OTS from `|showteam|` lines: species, item, ability, tera, all 4 moves.
- [ ] Parser records `bring4_observed` flag per side; dataset reports Tier 1 vs Tier 2 example counts.
- [ ] Lead-2 metrics are the primary headline numbers; bring-4/action90 metrics reported on Tier 1 (fully observed) subset only.
- [ ] OTS completeness ≈ 42/42 fields known for nearly all examples (edge cases: Ditto etc. with <4 moves).
- [ ] Data pipeline produces correct canonical schema; unit tests pass for move-order invariance, team-order invariance, stable hashes.
- [ ] Splits satisfy Regime A constraints: no test `team_a_id` appears as Team A in train.
- [ ] Splits satisfy Regime B constraints: OOD clusters never appear in training.
- [ ] No `(team_a_id, team_b_id, action90_id)` triples cross splits.
- [ ] No `match_group_id` crosses splits.
- [ ] Trained model outputs valid 90-way probabilities.
- [ ] Calibration artifacts exist; reliability plot produced.
- [ ] Risk–coverage curves generated for both risk definitions (top-1 miss, top-3 miss).
- [ ] All test metrics reported with mirror/non-mirror stratification.
- [ ] Demo tool outputs top-3 plans, abstains when `conf < τ`, shows explanations + retrieval evidence.
- [ ] Moves-hidden stress test implemented and reported.
- [ ] Cluster-aware bootstrap CIs computed for all main metrics.
- [ ] Paper-ready tables/figures generated by CLI scripts and are reproducible.

## 9) INTERVIEW TALK TRACK

- **Product pitch:** "A turn-zero OTS coach that recommends lead+bring plans with calibrated confidence and evidence from similar historical games."
- **Data engineering story:** "I initially designed a full reconstruction pipeline for partial OTS — cross-game aggregation, VGCPastes matching, deterministic move selection with confidence tracking — because the Showdown protocol normally only reveals moves/items/abilities when they're used in-game. But empirical verification showed that the OTS-filtered dataset contains `|showteam|` lines with 100% complete team sheets. So the reconstruction wasn't needed, but the architecture still supports it: the model has UNK embeddings and train-time masking for graceful degradation on datasets without full OTS. I'd rather design for the general case and discover the simplification than the other way around."
- **Label observability:** "Even with full OTS from `|showteam|`, the bring-4 *labels* are only ~85% fully observable — in short games, back mons never enter the field so you can't tell 'brought but didn't need' from 'left on bench.' I flag each example with `bring4_observed`, report lead-2 (always observable) as the primary task, and evaluate bring-4 only on the fully-observed subset. This prevents inflated claims about supervision quality."
- **Why uncertainty matters:** Team preview is multi-modal — many lines are viable against the same opponent. We need proper scoring and abstention, not overconfident single picks.
- **Split design:** "I use a 'pilot my team vs the field' evaluation: hold out Team A variants within species-core clusters, let opponents float freely. This matches real deployment and avoids the data-wasting constraint of requiring both teams in the same split."
- **Leakage control:** "Dedup on `(matchup, action)` triples — not matchup pairs — because the same teams facing each other with different expert decisions is signal, not leakage."
- **Calibration + abstention:** Deep ensembles + temperature scaling on validation only. Risk–coverage where risk = "expert not in top-3" directly matches the tool's UX.
- **Mirror matches:** Kept naturally in the test set and reported as a separate stratum. Mirrors are where multi-modality is highest — perfect for showcasing calibrated uncertainty.
- **Interpretability:** Retrieval-based explanations ("in similar matchups, experts led X+Y 42%"), domain-legible threat analysis tied to OTS cues, counterfactual plans for opponent's likely leads.
- **Robustness:** Moves-hidden stress test simulates deployment without full OTS. Train-time masking improves stability. The model architecture handles UNK natively even though the current dataset has full OTS — this generalizes to future formats.
- **Engineering quality:** Reproducible CLI pipeline, manifests/hashes, artifact contracts, paper-grade plots.
- **What I learned:** How leakage shows up in game logs (same matchup ≠ same decision), why "both teams in same split" wastes data vs the deployment-aligned split, how core-based clusters change conclusions vs random splits, why bring-4 labels require careful observability tracking even when OTS features are complete (short games hide back mons), and the value of empirically verifying data assumptions before committing to architecture (the reconstruction pipeline I designed wasn't needed — but I'm glad I designed it before discovering that).
- **Future:** Conformal prediction sets, conditional response planning, and richer embedding-based retrieval.
