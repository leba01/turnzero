"""Stage 4: Core-cluster teams by species overlap (≥4/6 → connected components).

Uses union-find with a 4-subset index for efficient edge detection.
Two teams are in the same core cluster if they share ≥4 species,
transitively closed via connected components.

Reference: docs/PROJECT_BIBLE.md Section 2.5

CLI contract:
    cluster --in_path <canonical_examples.jsonl> --out_dir <dir>
Outputs:
    cluster_assignments.json  — {team_id: cluster_id}
    cluster_manifest.json     — stats
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from turnzero.data.io_utils import read_jsonl, write_manifest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set / union-find with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by rank
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def __contains__(self, x: str) -> bool:
        return x in self._parent


# ---------------------------------------------------------------------------
# Core clustering algorithm
# ---------------------------------------------------------------------------

def core_cluster_teams(
    team_species: dict[str, list[str]],
) -> dict[str, str]:
    """Assign each team to a core cluster via ≥4/6 species overlap.

    Args:
        team_species: mapping of team_id → sorted list of 6 species names.

    Returns:
        Mapping of team_id → cluster_id (deterministic string labels).

    Algorithm:
        For each team, enumerate all C(6,4)=15 subsets of 4 species.
        Build an inverted index: 4-species-tuple → list of team_ids.
        For each collision in the index, union the teams.
        Connected components become clusters.
    """
    uf = UnionFind()
    index: dict[tuple[str, ...], list[str]] = defaultdict(list)

    for team_id, species6 in team_species.items():
        uf.add(team_id)
        for comb4 in combinations(sorted(species6), 4):
            key = comb4
            for other_id in index[key]:
                uf.union(team_id, other_id)
            index[key].append(team_id)

    # Assign deterministic cluster labels (sorted by first team_id seen)
    root_to_cluster: dict[str, str] = {}
    clusters: dict[str, str] = {}
    # Sort team_ids for deterministic cluster numbering
    for team_id in sorted(team_species.keys()):
        root = uf.find(team_id)
        if root not in root_to_cluster:
            root_to_cluster[root] = f"cluster_{len(root_to_cluster)}"
        clusters[team_id] = root_to_cluster[root]

    return clusters


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_cluster(in_path: str, out_dir: str) -> dict[str, Any]:
    """Read canonical examples, cluster teams, write assignments + stats.

    Args:
        in_path: Path to canonical match_examples.jsonl.
        out_dir: Output directory for cluster_assignments.json + manifest.

    Returns:
        Manifest dict with clustering statistics.
    """
    in_path_p = Path(in_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print(f"Reading canonical examples from {in_path_p} ...")
    t0 = time.time()

    # Collect unique team_id → species set from both sides
    team_species: dict[str, list[str]] = {}
    examples_read = 0

    for d in read_jsonl(in_path_p):
        examples_read += 1
        for side in ("team_a", "team_b"):
            team = d[side]
            tid = team["team_id"]
            if tid not in team_species:
                team_species[tid] = sorted(p["species"] for p in team["pokemon"])

        if examples_read % 10000 == 0:
            print(f"  {examples_read} examples scanned ...")

    print(f"Scanned {examples_read} examples, found {len(team_species)} unique teams.")

    # Run clustering
    print("Running core clustering (≥4/6 species overlap, union-find) ...")
    t_cluster = time.time()
    clusters = core_cluster_teams(team_species)
    cluster_time = time.time() - t_cluster

    # Compute cluster stats
    cluster_sizes: dict[str, int] = defaultdict(int)
    for cid in clusters.values():
        cluster_sizes[cid] += 1

    sizes = sorted(cluster_sizes.values(), reverse=True)
    num_clusters = len(cluster_sizes)
    singleton_count = sum(1 for s in sizes if s == 1)

    print(f"Clustered {len(team_species)} teams into {num_clusters} clusters "
          f"({singleton_count} singletons) in {cluster_time:.2f}s")

    # Write cluster assignments
    assignments_path = out_dir_p / "cluster_assignments.json"
    with open(assignments_path, "w") as f:
        json.dump(clusters, f, indent=2, sort_keys=True)

    elapsed = time.time() - t0

    # Build manifest
    manifest: dict[str, Any] = {
        "in_path": str(in_path_p),
        "examples_read": examples_read,
        "unique_teams": len(team_species),
        "num_clusters": num_clusters,
        "singleton_clusters": singleton_count,
        "largest_cluster_size": sizes[0] if sizes else 0,
        "median_cluster_size": sizes[len(sizes) // 2] if sizes else 0,
        "cluster_size_distribution": {
            "1": singleton_count,
            "2-5": sum(1 for s in sizes if 2 <= s <= 5),
            "6-20": sum(1 for s in sizes if 6 <= s <= 20),
            "21-100": sum(1 for s in sizes if 21 <= s <= 100),
            "101+": sum(1 for s in sizes if s > 100),
        },
        "top_10_clusters": [
            {"cluster_id": cid, "size": cluster_sizes[cid]}
            for cid in sorted(cluster_sizes, key=lambda c: cluster_sizes[c], reverse=True)[:10]
        ],
        "cluster_time_seconds": round(cluster_time, 2),
        "total_time_seconds": round(elapsed, 2),
    }
    write_manifest(out_dir_p / "cluster_manifest.json", manifest)

    return manifest
