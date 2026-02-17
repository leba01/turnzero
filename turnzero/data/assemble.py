"""Stage 6: Assemble per-split JSONL files for training.

Reads canonical examples, cluster assignments, and split assignments,
then writes per-split JSONL files with split_keys attached
(core_cluster_a, core_cluster_b, is_mirror).

Reference: docs/PROJECT_BIBLE.md Section 7.1

CLI contract:
    assemble --canonical_path <match_examples.jsonl>
             --clusters <cluster_assignments.json>
             --splits <splits.json>
             --out_dir <dir>
Outputs:
    regime_a/train.jsonl, regime_a/val.jsonl, regime_a/test.jsonl
    regime_b/train.jsonl, regime_b/val.jsonl, regime_b/test.jsonl
    assemble_manifest.json
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from turnzero.data.io_utils import read_jsonl, write_manifest


def _attach_split_keys(
    example: dict[str, Any],
    clusters: dict[str, str],
) -> dict[str, Any]:
    """Attach split_keys to an example dict.

    Adds:
        split_keys.team_a_id, split_keys.team_b_id,
        split_keys.core_cluster_a, split_keys.core_cluster_b,
        split_keys.is_mirror
    """
    team_a_id = example["team_a"]["team_id"]
    team_b_id = example["team_b"]["team_id"]
    cluster_a = clusters.get(team_a_id, "unknown")
    cluster_b = clusters.get(team_b_id, "unknown")

    example["split_keys"] = {
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
        "core_cluster_a": cluster_a,
        "core_cluster_b": cluster_b,
        "is_mirror": cluster_a == cluster_b,
    }
    return example


def run_assemble(
    canonical_path: str,
    clusters_path: str,
    splits_path: str,
    out_dir: str,
) -> dict[str, Any]:
    """Assemble per-split JSONL files for both regimes.

    Args:
        canonical_path: Path to canonical match_examples.jsonl.
        clusters_path: Path to cluster_assignments.json.
        splits_path: Path to splits.json.
        out_dir: Output directory for per-split JSONL + manifest.

    Returns:
        Manifest dict with assembly statistics.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print(f"Loading cluster assignments from {clusters_path} ...")
    with open(clusters_path) as f:
        clusters: dict[str, str] = json.load(f)

    print(f"Loading split assignments from {splits_path} ...")
    with open(splits_path) as f:
        splits: dict[str, dict[str, list[str]]] = json.load(f)

    # Build example_id → split lookups for each regime
    regime_lookups: dict[str, dict[str, str]] = {}
    for regime in ("regime_a", "regime_b"):
        lookup: dict[str, str] = {}
        for split_name, ids in splits[regime].items():
            for eid in ids:
                lookup[eid] = split_name
        regime_lookups[regime] = lookup

    # Open output files
    print(f"Reading canonical examples from {canonical_path} ...")
    t0 = time.time()

    writers: dict[str, Any] = {}
    for regime in ("regime_a", "regime_b"):
        regime_dir = out_dir_p / regime
        regime_dir.mkdir(parents=True, exist_ok=True)
        writers[regime] = {}
        for split_name in ("train", "val", "test"):
            writers[regime][split_name] = open(regime_dir / f"{split_name}.jsonl", "w")

    # Track stats
    counts: dict[str, dict[str, int]] = {
        regime: defaultdict(int) for regime in ("regime_a", "regime_b")
    }
    examples_read = 0
    mirror_counts: dict[str, dict[str, int]] = {
        regime: defaultdict(int) for regime in ("regime_a", "regime_b")
    }

    try:
        for d in read_jsonl(canonical_path):
            examples_read += 1
            eid = d["example_id"]

            # Attach split_keys (cluster info + is_mirror)
            enriched = _attach_split_keys(d, clusters)
            line = json.dumps(enriched, separators=(",", ":")) + "\n"

            # Write to appropriate split file for each regime
            for regime in ("regime_a", "regime_b"):
                split_name = regime_lookups[regime].get(eid)
                if split_name is not None:
                    writers[regime][split_name].write(line)
                    counts[regime][split_name] += 1
                    if enriched["split_keys"]["is_mirror"]:
                        mirror_counts[regime][split_name] += 1

            if examples_read % 10000 == 0:
                print(f"  {examples_read} examples processed ...")
    finally:
        for regime in writers:
            for f in writers[regime].values():
                f.close()

    elapsed = time.time() - t0

    # Print summary
    for regime in ("regime_a", "regime_b"):
        total = sum(counts[regime].values())
        print(f"  {regime}: train={counts[regime]['train']}, "
              f"val={counts[regime]['val']}, test={counts[regime]['test']} "
              f"(total={total})")

    print(f"Assembly complete in {elapsed:.2f}s")

    # Build manifest
    manifest: dict[str, Any] = {
        "canonical_path": canonical_path,
        "clusters_path": clusters_path,
        "splits_path": splits_path,
        "examples_read": examples_read,
        "regime_a": {
            "train": counts["regime_a"]["train"],
            "val": counts["regime_a"]["val"],
            "test": counts["regime_a"]["test"],
            "total": sum(counts["regime_a"].values()),
            "mirror_counts": {
                "train": mirror_counts["regime_a"]["train"],
                "val": mirror_counts["regime_a"]["val"],
                "test": mirror_counts["regime_a"]["test"],
            },
        },
        "regime_b": {
            "train": counts["regime_b"]["train"],
            "val": counts["regime_b"]["val"],
            "test": counts["regime_b"]["test"],
            "total": sum(counts["regime_b"].values()),
            "mirror_counts": {
                "train": mirror_counts["regime_b"]["train"],
                "val": mirror_counts["regime_b"]["val"],
                "test": mirror_counts["regime_b"]["test"],
            },
        },
        "assembly_time_seconds": round(elapsed, 2),
    }
    write_manifest(out_dir_p / "assemble_manifest.json", manifest)

    return manifest
