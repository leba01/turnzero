"""Stage 5: Train/val/test split generation (Regime A + Regime B).

Regime A ("Pilot my team vs the field"):
    Hold out Team A variants *within* each core cluster.
    Opponents (Team B) are unrestricted across splits.
    Key constraints:
      - team_a_id → split is deterministic (80/10/10 within cluster)
      - match_group_id conflict resolution: highest-priority split wins
        (test > val > train); drop conflicting directed examples
      - cross-split triple dedup: (team_a_id, team_b_id, action90_id)
        triples must not appear in multiple splits

Regime B (Out-of-core OOD):
    Hold out entire core clusters.
    Clusters assigned to train(70%)/val(10%)/test_ood(20%) by team weight.
    Same match_group and triple dedup constraints.

Reference: docs/PROJECT_BIBLE.md Section 2.6

CLI contract:
    split --in_path <canonical_examples.jsonl> --clusters <cluster_assignments.json>
          --out_dir <dir> --seed 42
Outputs:
    splits.json           — {regime_a: {train/val/test: [example_ids]},
                              regime_b: {train/val/test: [example_ids]}}
    split_manifest.json   — statistics
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from turnzero.data.io_utils import read_jsonl, write_manifest

log = logging.getLogger(__name__)

_SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


# ---------------------------------------------------------------------------
# Regime A: within-core team_a holdout
# ---------------------------------------------------------------------------

def _assign_teams_within_cluster(
    cluster_to_team_a_ids: dict[str, list[str]],
    seed: int,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> dict[str, str]:
    """Assign each team_a_id to train/val/test within its cluster.

    For each cluster:
      - Shuffle team_a_ids with fixed seed.
      - Assign 80/10/10 by count.
      - Small clusters (≤2 teams): ensure ≥1 in test when possible.

    Returns:
        Mapping of team_a_id → "train" | "val" | "test".
    """
    rng = random.Random(seed)
    team_to_split: dict[str, str] = {}

    for cluster_id in sorted(cluster_to_team_a_ids.keys()):
        team_ids = sorted(cluster_to_team_a_ids[cluster_id])
        rng.shuffle(team_ids)
        n = len(team_ids)

        if n == 1:
            # Single team → train (can't hold out the only team)
            team_to_split[team_ids[0]] = "train"
        elif n == 2:
            # 2 teams → one train, one test
            team_to_split[team_ids[0]] = "train"
            team_to_split[team_ids[1]] = "test"
        elif n <= 5:
            # Small cluster: 1 test, 1 val, rest train
            team_to_split[team_ids[0]] = "test"
            team_to_split[team_ids[1]] = "val"
            for tid in team_ids[2:]:
                team_to_split[tid] = "train"
        else:
            # Standard 80/10/10
            n_val = max(1, round(n * val_frac))
            n_test = max(1, round(n * (1 - train_frac - val_frac)))
            n_train = n - n_val - n_test

            for tid in team_ids[:n_train]:
                team_to_split[tid] = "train"
            for tid in team_ids[n_train:n_train + n_val]:
                team_to_split[tid] = "val"
            for tid in team_ids[n_train + n_val:]:
                team_to_split[tid] = "test"

    return team_to_split


def _resolve_match_groups(
    examples: list[dict[str, Any]],
    team_to_split: dict[str, str],
) -> list[dict[str, Any]]:
    """Resolve directed-example conflicts within match groups.

    For each match_group_id:
      - Find the highest-priority split among its examples' team_a splits.
      - Keep examples whose team_a split matches the group split.
      - Drop examples that would violate the split (team_a in lower split
        but group assigned to higher split).

    Returns:
        Filtered list of examples (dropped examples removed).
    """
    # Group examples by match_group_id
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        groups[ex["match_group_id"]].append(ex)

    kept: list[dict[str, Any]] = []
    for mg_id, group_exs in groups.items():
        # Find highest-priority split in this group
        max_priority = -1
        for ex in group_exs:
            team_a_id = ex["team_a"]["team_id"]
            split = team_to_split.get(team_a_id, "train")
            max_priority = max(max_priority, _SPLIT_PRIORITY[split])

        group_split = {v: k for k, v in _SPLIT_PRIORITY.items()}[max_priority]

        # Keep only examples whose team_a split matches group split
        for ex in group_exs:
            team_a_id = ex["team_a"]["team_id"]
            if team_to_split.get(team_a_id, "train") == group_split:
                kept.append(ex)

    return kept


def _cross_split_triple_dedup(
    examples: list[dict[str, Any]],
    team_to_split: dict[str, str],
) -> list[dict[str, Any]]:
    """Remove (team_a_id, team_b_id, action90_id) triples appearing in multiple splits.

    When a triple appears in multiple splits, keep it in the highest-priority
    split and remove from lower-priority ones.
    """
    # Build triple → set of splits
    triple_splits: dict[tuple[str, str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i, ex in enumerate(examples):
        triple = (
            ex["team_a"]["team_id"],
            ex["team_b"]["team_id"],
            ex["label"]["action90_id"],
        )
        split = team_to_split.get(ex["team_a"]["team_id"], "train")
        triple_splits[triple][split].append(i)

    # Find indices to drop
    drop_indices: set[int] = set()
    for triple, splits_map in triple_splits.items():
        if len(splits_map) <= 1:
            continue
        # Keep highest-priority split, drop from others
        max_split = max(splits_map.keys(), key=lambda s: _SPLIT_PRIORITY[s])
        for split, indices in splits_map.items():
            if split != max_split:
                drop_indices.update(indices)

    return [ex for i, ex in enumerate(examples) if i not in drop_indices]


def split_regime_a(
    examples: list[dict[str, Any]],
    clusters: dict[str, str],
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Generate Regime A splits.

    Returns:
        (split_assignments, stats) where split_assignments maps
        "train"/"val"/"test" → list of example_ids.
    """
    # 1. Collect unique team_a_ids per cluster
    cluster_to_team_a_ids: dict[str, list[str]] = defaultdict(list)
    seen_team_a: set[str] = set()
    for ex in examples:
        tid = ex["team_a"]["team_id"]
        if tid not in seen_team_a:
            seen_team_a.add(tid)
            cid = clusters.get(tid, f"unknown_{tid}")
            cluster_to_team_a_ids[cid].append(tid)

    # 2. Assign team_a_ids to splits within each cluster
    team_to_split = _assign_teams_within_cluster(cluster_to_team_a_ids, seed)

    # 3. Resolve match_group conflicts
    resolved = _resolve_match_groups(examples, team_to_split)
    dropped_match_group = len(examples) - len(resolved)

    # 4. Cross-split triple dedup
    deduped = _cross_split_triple_dedup(resolved, team_to_split)
    dropped_triple_dedup = len(resolved) - len(deduped)

    # 5. Build final split assignments
    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for ex in deduped:
        tid = ex["team_a"]["team_id"]
        split = team_to_split.get(tid, "train")
        split_ids[split].append(ex["example_id"])

    stats = {
        "total_examples_in": len(examples),
        "dropped_match_group_conflict": dropped_match_group,
        "dropped_triple_dedup": dropped_triple_dedup,
        "total_examples_out": len(deduped),
        "train": len(split_ids["train"]),
        "val": len(split_ids["val"]),
        "test": len(split_ids["test"]),
        "unique_team_a_ids": len(seen_team_a),
        "clusters_with_team_a": len(cluster_to_team_a_ids),
        "directed_example_drop_rate": round(
            dropped_match_group / len(examples), 6
        ) if examples else 0,
    }

    return split_ids, stats


# ---------------------------------------------------------------------------
# Regime B: out-of-core holdout
# ---------------------------------------------------------------------------

def _assign_clusters_regime_b(
    cluster_to_team_a_ids: dict[str, list[str]],
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> dict[str, str]:
    """Assign entire clusters to train/val/test_ood by team weight.

    Two-pass approach:
      1. Sort clusters largest-first. The largest cluster (mega-cluster)
         goes to train — it typically contains >70% of teams.
      2. Shuffle remaining clusters, then greedily fill val (target ~1/3
         of remainder) and test_ood (remaining ~2/3).

    This ensures: (a) mega-cluster anchors train, (b) OOD test gets a
    diverse set of held-out clusters, (c) val exists for threshold tuning.

    Returns:
        Mapping of cluster_id → "train" | "val" | "test".
    """
    rng = random.Random(seed)

    # Sort clusters by size descending
    sorted_clusters = sorted(
        cluster_to_team_a_ids.keys(),
        key=lambda c: len(cluster_to_team_a_ids[c]),
        reverse=True,
    )

    total_teams = sum(len(tids) for tids in cluster_to_team_a_ids.values())
    target_train = total_teams * train_frac

    cluster_to_split: dict[str, str] = {}
    train_count = 0
    remaining: list[str] = []

    # Pass 1: fill train with largest clusters
    for cid in sorted_clusters:
        n = len(cluster_to_team_a_ids[cid])
        if train_count < target_train:
            cluster_to_split[cid] = "train"
            train_count += n
        else:
            remaining.append(cid)

    # Pass 2: divide remaining between val and test_ood
    rng.shuffle(remaining)
    remaining_teams = sum(len(cluster_to_team_a_ids[c]) for c in remaining)
    # val gets val_frac/(val_frac + test_frac) of remainder
    test_frac = 1.0 - train_frac - val_frac
    val_share = val_frac / (val_frac + test_frac) if (val_frac + test_frac) > 0 else 0.5
    target_val_remaining = remaining_teams * val_share

    val_count = 0
    for cid in remaining:
        n = len(cluster_to_team_a_ids[cid])
        if val_count < target_val_remaining:
            cluster_to_split[cid] = "val"
            val_count += n
        else:
            cluster_to_split[cid] = "test"

    return cluster_to_split


def split_regime_b(
    examples: list[dict[str, Any]],
    clusters: dict[str, str],
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Generate Regime B splits (out-of-core OOD).

    Returns:
        (split_assignments, stats) where split_assignments maps
        "train"/"val"/"test" → list of example_ids.
    """
    # 1. Collect unique team_a_ids per cluster
    cluster_to_team_a_ids: dict[str, list[str]] = defaultdict(list)
    seen_team_a: set[str] = set()
    for ex in examples:
        tid = ex["team_a"]["team_id"]
        if tid not in seen_team_a:
            seen_team_a.add(tid)
            cid = clusters.get(tid, f"unknown_{tid}")
            cluster_to_team_a_ids[cid].append(tid)

    # 2. Assign clusters to splits
    cluster_to_split = _assign_clusters_regime_b(cluster_to_team_a_ids, seed)

    # Build team_a_id → split via cluster
    team_to_split: dict[str, str] = {}
    for cid, tids in cluster_to_team_a_ids.items():
        split = cluster_to_split[cid]
        for tid in tids:
            team_to_split[tid] = split

    # 3. Resolve match_group conflicts
    resolved = _resolve_match_groups(examples, team_to_split)
    dropped_match_group = len(examples) - len(resolved)

    # 4. Cross-split triple dedup
    deduped = _cross_split_triple_dedup(resolved, team_to_split)
    dropped_triple_dedup = len(resolved) - len(deduped)

    # 5. Build final split assignments
    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for ex in deduped:
        tid = ex["team_a"]["team_id"]
        split = team_to_split.get(tid, "train")
        split_ids[split].append(ex["example_id"])

    # Cluster-level stats
    split_cluster_counts: dict[str, int] = defaultdict(int)
    for cid, split in cluster_to_split.items():
        split_cluster_counts[split] += 1

    stats = {
        "total_examples_in": len(examples),
        "dropped_match_group_conflict": dropped_match_group,
        "dropped_triple_dedup": dropped_triple_dedup,
        "total_examples_out": len(deduped),
        "train": len(split_ids["train"]),
        "val": len(split_ids["val"]),
        "test": len(split_ids["test"]),
        "train_clusters": split_cluster_counts["train"],
        "val_clusters": split_cluster_counts["val"],
        "test_clusters": split_cluster_counts["test"],
        "directed_example_drop_rate": round(
            dropped_match_group / len(examples), 6
        ) if examples else 0,
    }

    return split_ids, stats


# ---------------------------------------------------------------------------
# Integrity validators
# ---------------------------------------------------------------------------

def validate_regime_a(
    examples: list[dict[str, Any]],
    split_ids: dict[str, list[str]],
    clusters: dict[str, str],
) -> list[str]:
    """Validate Regime A split integrity. Returns list of violations."""
    violations: list[str] = []

    # Build example lookup
    ex_by_id: dict[str, dict] = {ex["example_id"]: ex for ex in examples}
    id_to_split: dict[str, str] = {}
    for split, ids in split_ids.items():
        for eid in ids:
            id_to_split[eid] = split

    # 1. No test team_a_id appears as team_a in train
    train_team_a: set[str] = set()
    test_team_a: set[str] = set()
    val_team_a: set[str] = set()
    for eid, split in id_to_split.items():
        if eid not in ex_by_id:
            continue
        tid = ex_by_id[eid]["team_a"]["team_id"]
        if split == "train":
            train_team_a.add(tid)
        elif split == "val":
            val_team_a.add(tid)
        elif split == "test":
            test_team_a.add(tid)

    leak = test_team_a & train_team_a
    if leak:
        violations.append(
            f"team_a_id leak: {len(leak)} test team_a_ids appear in train"
        )
    leak_val = val_team_a & train_team_a
    if leak_val:
        violations.append(
            f"team_a_id leak: {len(leak_val)} val team_a_ids appear in train"
        )

    # 2. No (team_a_id, team_b_id, action90_id) triples cross splits
    triple_splits: dict[tuple, set[str]] = defaultdict(set)
    for eid, split in id_to_split.items():
        if eid not in ex_by_id:
            continue
        ex = ex_by_id[eid]
        triple = (
            ex["team_a"]["team_id"],
            ex["team_b"]["team_id"],
            ex["label"]["action90_id"],
        )
        triple_splits[triple].add(split)

    cross_triples = sum(1 for s in triple_splits.values() if len(s) > 1)
    if cross_triples:
        violations.append(
            f"{cross_triples} (team_a, team_b, action90) triples cross splits"
        )

    # 3. No match_group_id crosses splits
    mg_splits: dict[str, set[str]] = defaultdict(set)
    for eid, split in id_to_split.items():
        if eid not in ex_by_id:
            continue
        mg = ex_by_id[eid]["match_group_id"]
        mg_splits[mg].add(split)

    cross_mg = sum(1 for s in mg_splits.values() if len(s) > 1)
    if cross_mg:
        violations.append(
            f"{cross_mg} match_group_ids cross splits"
        )

    return violations


def validate_regime_b(
    examples: list[dict[str, Any]],
    split_ids: dict[str, list[str]],
    clusters: dict[str, str],
) -> list[str]:
    """Validate Regime B split integrity. Returns list of violations."""
    violations: list[str] = []

    ex_by_id: dict[str, dict] = {ex["example_id"]: ex for ex in examples}
    id_to_split: dict[str, str] = {}
    for split, ids in split_ids.items():
        for eid in ids:
            id_to_split[eid] = split

    # 1. No OOD cluster_a appears in train
    train_clusters: set[str] = set()
    test_clusters: set[str] = set()
    for eid, split in id_to_split.items():
        if eid not in ex_by_id:
            continue
        tid = ex_by_id[eid]["team_a"]["team_id"]
        cid = clusters.get(tid, "unknown")
        if split == "train":
            train_clusters.add(cid)
        elif split == "test":
            test_clusters.add(cid)

    leak = test_clusters & train_clusters
    if leak:
        violations.append(
            f"cluster leak: {len(leak)} test clusters appear in train"
        )

    # 2. No triples cross splits
    triple_splits: dict[tuple, set[str]] = defaultdict(set)
    for eid, split in id_to_split.items():
        if eid not in ex_by_id:
            continue
        ex = ex_by_id[eid]
        triple = (
            ex["team_a"]["team_id"],
            ex["team_b"]["team_id"],
            ex["label"]["action90_id"],
        )
        triple_splits[triple].add(split)

    cross_triples = sum(1 for s in triple_splits.values() if len(s) > 1)
    if cross_triples:
        violations.append(
            f"{cross_triples} (team_a, team_b, action90) triples cross splits"
        )

    # 3. No match_group_id crosses splits
    mg_splits: dict[str, set[str]] = defaultdict(set)
    for eid, split in id_to_split.items():
        if eid not in ex_by_id:
            continue
        mg = ex_by_id[eid]["match_group_id"]
        mg_splits[mg].add(split)

    cross_mg = sum(1 for s in mg_splits.values() if len(s) > 1)
    if cross_mg:
        violations.append(
            f"{cross_mg} match_group_ids cross splits"
        )

    return violations


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_split(
    in_path: str,
    clusters_path: str,
    out_dir: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Run split generation for both regimes.

    Args:
        in_path: Path to canonical match_examples.jsonl.
        clusters_path: Path to cluster_assignments.json.
        out_dir: Output directory for splits.json + manifest.
        seed: Random seed for reproducibility.

    Returns:
        Manifest dict with split statistics.
    """
    in_path_p = Path(in_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print(f"Reading canonical examples from {in_path_p} ...")
    t0 = time.time()

    # Load cluster assignments
    with open(clusters_path) as f:
        clusters: dict[str, str] = json.load(f)

    # Load all examples into memory (needed for match_group resolution)
    examples: list[dict[str, Any]] = list(read_jsonl(in_path_p))
    print(f"Loaded {len(examples)} examples.")

    # --- Regime A ---
    print("Generating Regime A splits (within-core team_a holdout) ...")
    t_a = time.time()
    regime_a_ids, regime_a_stats = split_regime_a(examples, clusters, seed)
    time_a = time.time() - t_a

    # Validate
    violations_a = validate_regime_a(examples, regime_a_ids, clusters)
    if violations_a:
        for v in violations_a:
            print(f"  [VIOLATION] Regime A: {v}")
    else:
        print("  Regime A: all integrity checks passed.")

    print(f"  Regime A: train={regime_a_stats['train']}, "
          f"val={regime_a_stats['val']}, test={regime_a_stats['test']} "
          f"(dropped mg={regime_a_stats['dropped_match_group_conflict']}, "
          f"triple={regime_a_stats['dropped_triple_dedup']}) "
          f"in {time_a:.2f}s")

    # --- Regime B ---
    print("Generating Regime B splits (out-of-core OOD) ...")
    t_b = time.time()
    regime_b_ids, regime_b_stats = split_regime_b(examples, clusters, seed)
    time_b = time.time() - t_b

    violations_b = validate_regime_b(examples, regime_b_ids, clusters)
    if violations_b:
        for v in violations_b:
            print(f"  [VIOLATION] Regime B: {v}")
    else:
        print("  Regime B: all integrity checks passed.")

    print(f"  Regime B: train={regime_b_stats['train']}, "
          f"val={regime_b_stats['val']}, test={regime_b_stats['test']} "
          f"(dropped mg={regime_b_stats['dropped_match_group_conflict']}, "
          f"triple={regime_b_stats['dropped_triple_dedup']}) "
          f"in {time_b:.2f}s")

    elapsed = time.time() - t0

    # Write splits.json
    splits_out = {
        "regime_a": regime_a_ids,
        "regime_b": regime_b_ids,
    }
    splits_path = out_dir_p / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits_out, f, indent=2)

    # Write manifest
    manifest: dict[str, Any] = {
        "in_path": str(in_path_p),
        "clusters_path": clusters_path,
        "seed": seed,
        "total_examples": len(examples),
        "regime_a": {
            **regime_a_stats,
            "violations": violations_a,
            "time_seconds": round(time_a, 2),
        },
        "regime_b": {
            **regime_b_stats,
            "violations": violations_b,
            "time_seconds": round(time_b, 2),
        },
        "total_time_seconds": round(elapsed, 2),
    }
    write_manifest(out_dir_p / "split_manifest.json", manifest)

    return manifest
