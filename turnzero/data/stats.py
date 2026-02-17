"""Stage 7: Comprehensive dataset stats + end-to-end integrity validation.

Reads assembled per-split JSONL files and produces a dataset_report.json
covering OTS completeness, label observability, split sizes, mirror rates,
action distribution, species frequency, and all integrity assertions.

Reference: docs/PROJECT_BIBLE.md Section 7.1, 7.3, 8

CLI contract:
    stats --data_dir <assembled_dir> --validate
Outputs:
    dataset_report.json
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from turnzero.data.io_utils import read_jsonl, write_manifest


# ---------------------------------------------------------------------------
# Stats collection
# ---------------------------------------------------------------------------

def _collect_split_stats(examples: list[dict]) -> dict[str, Any]:
    """Compute statistics for a single split."""
    n = len(examples)
    if n == 0:
        return {"count": 0}

    bring4_true = sum(1 for ex in examples if ex["label_quality"]["bring4_observed"])
    bring4_false = n - bring4_true

    # Mirror rate
    mirror = sum(1 for ex in examples
                 if ex.get("split_keys", {}).get("is_mirror", False))

    # OTS completeness
    fields_known_values = []
    for ex in examples:
        for side in ("team_a", "team_b"):
            rq = ex[side].get("reconstruction_quality")
            if rq:
                fields_known_values.append(rq["fields_known"])

    # Action90 distribution
    action_counts = Counter(ex["label"]["action90_id"] for ex in examples)
    top_10_actions = action_counts.most_common(10)

    # Unique teams
    unique_team_a = len({ex["team_a"]["team_id"] for ex in examples})
    unique_team_b = len({ex["team_b"]["team_id"] for ex in examples})
    unique_teams_all = len(
        {ex["team_a"]["team_id"] for ex in examples}
        | {ex["team_b"]["team_id"] for ex in examples}
    )

    # Species frequency (team_a side)
    species_counter: Counter[str] = Counter()
    for ex in examples:
        for mon in ex["team_a"]["pokemon"]:
            species_counter[mon["species"]] += 1

    return {
        "count": n,
        "bring4_observed_true": bring4_true,
        "bring4_observed_false": bring4_false,
        "bring4_observed_rate": round(bring4_true / n, 4),
        "mirror_count": mirror,
        "mirror_rate": round(mirror / n, 4),
        "non_mirror_count": n - mirror,
        "unique_team_a_ids": unique_team_a,
        "unique_team_b_ids": unique_team_b,
        "unique_teams_total": unique_teams_all,
        "ots_completeness": {
            "mean_fields_known": round(
                sum(fields_known_values) / len(fields_known_values), 2
            ) if fields_known_values else None,
            "min_fields_known": min(fields_known_values) if fields_known_values else None,
            "max_fields_known": max(fields_known_values) if fields_known_values else None,
            "pct_fully_known": round(
                sum(1 for v in fields_known_values if v == 42) / len(fields_known_values), 4
            ) if fields_known_values else None,
        },
        "action90_unique_actions": len(action_counts),
        "action90_top_10": [
            {"action_id": a, "count": c, "pct": round(c / n, 4)}
            for a, c in top_10_actions
        ],
        "species_top_20": [
            {"species": s, "count": c}
            for s, c in species_counter.most_common(20)
        ],
    }


# ---------------------------------------------------------------------------
# Integrity validators
# ---------------------------------------------------------------------------

def _validate_regime_a(
    splits: dict[str, list[dict]],
) -> list[str]:
    """Run Regime A integrity checks. Returns list of violations."""
    violations = []

    # 1. No test team_a_id appears as team_a in train
    train_team_a = {ex["team_a"]["team_id"] for ex in splits.get("train", [])}
    test_team_a = {ex["team_a"]["team_id"] for ex in splits.get("test", [])}
    val_team_a = {ex["team_a"]["team_id"] for ex in splits.get("val", [])}

    leak_test_train = test_team_a & train_team_a
    if leak_test_train:
        violations.append(
            f"Regime A: {len(leak_test_train)} test team_a_ids appear as team_a in train"
        )

    leak_val_train = val_team_a & train_team_a
    if leak_val_train:
        violations.append(
            f"Regime A: {len(leak_val_train)} val team_a_ids appear as team_a in train"
        )

    # 2. No (team_a_id, team_b_id, action90_id) triples cross splits
    triple_splits: dict[tuple, set[str]] = defaultdict(set)
    for split_name, exs in splits.items():
        for ex in exs:
            triple = (
                ex["team_a"]["team_id"],
                ex["team_b"]["team_id"],
                ex["label"]["action90_id"],
            )
            triple_splits[triple].add(split_name)

    cross_triples = sum(1 for s in triple_splits.values() if len(s) > 1)
    if cross_triples:
        violations.append(
            f"Regime A: {cross_triples} (team_a, team_b, action90) triples cross splits"
        )

    # 3. No match_group_id crosses splits
    mg_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, exs in splits.items():
        for ex in exs:
            mg_splits[ex["match_group_id"]].add(split_name)

    cross_mg = sum(1 for s in mg_splits.values() if len(s) > 1)
    if cross_mg:
        violations.append(
            f"Regime A: {cross_mg} match_group_ids cross splits"
        )

    return violations


def _validate_regime_b(
    splits: dict[str, list[dict]],
) -> list[str]:
    """Run Regime B integrity checks. Returns list of violations."""
    violations = []

    # 1. No test cluster_a appears in train
    train_clusters = {
        ex.get("split_keys", {}).get("core_cluster_a", "unknown")
        for ex in splits.get("train", [])
    }
    test_clusters = {
        ex.get("split_keys", {}).get("core_cluster_a", "unknown")
        for ex in splits.get("test", [])
    }

    leak = test_clusters & train_clusters
    if leak:
        violations.append(
            f"Regime B: {len(leak)} test core_cluster_a values appear in train"
        )

    # 2. No triples cross splits
    triple_splits: dict[tuple, set[str]] = defaultdict(set)
    for split_name, exs in splits.items():
        for ex in exs:
            triple = (
                ex["team_a"]["team_id"],
                ex["team_b"]["team_id"],
                ex["label"]["action90_id"],
            )
            triple_splits[triple].add(split_name)

    cross_triples = sum(1 for s in triple_splits.values() if len(s) > 1)
    if cross_triples:
        violations.append(
            f"Regime B: {cross_triples} (team_a, team_b, action90) triples cross splits"
        )

    # 3. No match_group_id crosses splits
    mg_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, exs in splits.items():
        for ex in exs:
            mg_splits[ex["match_group_id"]].add(split_name)

    cross_mg = sum(1 for s in mg_splits.values() if len(s) > 1)
    if cross_mg:
        violations.append(
            f"Regime B: {cross_mg} match_group_ids cross splits"
        )

    return violations


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_stats(
    data_dir: str,
    validate: bool = True,
) -> dict[str, Any]:
    """Compute comprehensive stats and run integrity validation.

    Args:
        data_dir: Path to assembled data directory (with regime_a/, regime_b/).
        validate: If True, run integrity validation assertions.

    Returns:
        Dataset report dict.
    """
    data_dir_p = Path(data_dir)
    t0 = time.time()

    report: dict[str, Any] = {"data_dir": str(data_dir_p)}
    all_violations: list[str] = []

    for regime in ("regime_a", "regime_b"):
        regime_dir = data_dir_p / regime
        if not regime_dir.exists():
            print(f"  Skipping {regime} (directory not found)")
            continue

        print(f"Computing stats for {regime} ...")
        regime_report: dict[str, Any] = {}
        regime_splits: dict[str, list[dict]] = {}

        for split_name in ("train", "val", "test"):
            split_path = regime_dir / f"{split_name}.jsonl"
            if not split_path.exists():
                regime_report[split_name] = {"count": 0}
                regime_splits[split_name] = []
                continue

            examples = list(read_jsonl(split_path))
            regime_splits[split_name] = examples
            regime_report[split_name] = _collect_split_stats(examples)
            print(f"  {split_name}: {len(examples)} examples")

        # Overall stats across all splits
        all_examples = []
        for exs in regime_splits.values():
            all_examples.extend(exs)
        regime_report["overall"] = _collect_split_stats(all_examples)

        # Validation
        if validate:
            print(f"  Running integrity validation for {regime} ...")
            if regime == "regime_a":
                violations = _validate_regime_a(regime_splits)
            else:
                violations = _validate_regime_b(regime_splits)

            regime_report["integrity_violations"] = violations
            all_violations.extend(violations)

            if violations:
                for v in violations:
                    print(f"    [FAIL] {v}")
            else:
                print(f"    All integrity checks PASSED.")

        report[regime] = regime_report

    elapsed = time.time() - t0
    report["validation_passed"] = len(all_violations) == 0
    report["all_violations"] = all_violations
    report["stats_time_seconds"] = round(elapsed, 2)

    # Print summary
    print(f"\n{'='*60}")
    print("DATASET REPORT SUMMARY")
    print(f"{'='*60}")
    for regime in ("regime_a", "regime_b"):
        if regime not in report:
            continue
        r = report[regime]
        overall = r.get("overall", {})
        print(f"\n{regime}:")
        print(f"  Total examples:       {overall.get('count', 0)}")
        print(f"  bring4_observed rate:  {overall.get('bring4_observed_rate', 'N/A')}")
        print(f"  Mirror rate:           {overall.get('mirror_rate', 'N/A')}")
        ots = overall.get("ots_completeness", {})
        print(f"  OTS completeness:      mean={ots.get('mean_fields_known', 'N/A')}/42, "
              f"pct_full={ots.get('pct_fully_known', 'N/A')}")
        print(f"  Unique actions used:   {overall.get('action90_unique_actions', 'N/A')}/90")
        for s in ("train", "val", "test"):
            split_r = r.get(s, {})
            count = split_r.get("count", 0)
            mirror = split_r.get("mirror_rate", "N/A")
            b4 = split_r.get("bring4_observed_rate", "N/A")
            print(f"  {s:6s}: {count:>6d} examples  "
                  f"(mirror={mirror}, bring4={b4})")

    print(f"\nIntegrity: {'PASSED' if report['validation_passed'] else 'FAILED'}")
    if all_violations:
        for v in all_violations:
            print(f"  [FAIL] {v}")
    print(f"Stats computed in {elapsed:.2f}s")

    # Write report
    write_manifest(data_dir_p / "dataset_report.json", report)

    return report
