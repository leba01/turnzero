"""Tests for Stage 4: core clustering (union-find + 4-subset index)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnzero.splits.cluster import UnionFind, core_cluster_teams, run_cluster


# ---------------------------------------------------------------------------
# UnionFind
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_find_self(self):
        uf = UnionFind()
        uf.add("a")
        assert uf.find("a") == "a"

    def test_union_and_find(self):
        uf = UnionFind()
        uf.add("a")
        uf.add("b")
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_transitive_union(self):
        uf = UnionFind()
        for x in "abcde":
            uf.add(x)
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("d", "e")
        assert uf.find("a") == uf.find("c")
        assert uf.find("d") == uf.find("e")
        assert uf.find("a") != uf.find("d")

    def test_union_idempotent(self):
        uf = UnionFind()
        uf.add("a")
        uf.add("b")
        uf.union("a", "b")
        uf.union("a", "b")
        uf.union("b", "a")
        assert uf.find("a") == uf.find("b")

    def test_contains(self):
        uf = UnionFind()
        assert "x" not in uf
        uf.add("x")
        assert "x" in uf


# ---------------------------------------------------------------------------
# core_cluster_teams
# ---------------------------------------------------------------------------

class TestCoreClusterTeams:
    def test_identical_teams_same_cluster(self):
        """Teams with identical species → same cluster."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],
            "t2": ["A", "B", "C", "D", "E", "F"],
        }
        clusters = core_cluster_teams(teams)
        assert clusters["t1"] == clusters["t2"]

    def test_four_overlap_same_cluster(self):
        """Teams sharing exactly 4 species → same cluster."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],
            "t2": ["A", "B", "C", "D", "X", "Y"],
        }
        clusters = core_cluster_teams(teams)
        assert clusters["t1"] == clusters["t2"]

    def test_five_overlap_same_cluster(self):
        """Teams sharing 5 species → same cluster."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],
            "t2": ["A", "B", "C", "D", "E", "X"],
        }
        clusters = core_cluster_teams(teams)
        assert clusters["t1"] == clusters["t2"]

    def test_three_overlap_different_clusters(self):
        """Teams sharing only 3 species → different clusters."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],
            "t2": ["A", "B", "C", "X", "Y", "Z"],
        }
        clusters = core_cluster_teams(teams)
        assert clusters["t1"] != clusters["t2"]

    def test_no_overlap_different_clusters(self):
        """Teams with disjoint species → different clusters."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],
            "t2": ["G", "H", "I", "J", "K", "L"],
        }
        clusters = core_cluster_teams(teams)
        assert clusters["t1"] != clusters["t2"]

    def test_transitive_clustering(self):
        """A-B share 4, B-C share 4, but A-C share only 2 → all same cluster."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],       # shares A,B,C,D with t2
            "t2": ["A", "B", "C", "D", "X", "Y"],       # shares A,B,X,Y with t3
            "t3": ["A", "B", "X", "Y", "P", "Q"],       # shares only A,B with t1 (< 4)
        }
        clusters = core_cluster_teams(teams)
        # t1-t2 connected, t2-t3 connected → all same cluster
        assert clusters["t1"] == clusters["t2"]
        assert clusters["t2"] == clusters["t3"]

    def test_singleton_cluster(self):
        """A team with no overlap ≥4 to anyone → its own cluster."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],
            "t2": ["G", "H", "I", "J", "K", "L"],
            "t3": ["M", "N", "O", "P", "Q", "R"],
        }
        clusters = core_cluster_teams(teams)
        unique_clusters = set(clusters.values())
        assert len(unique_clusters) == 3

    def test_single_team(self):
        """One team → one cluster."""
        teams = {"t1": ["A", "B", "C", "D", "E", "F"]}
        clusters = core_cluster_teams(teams)
        assert len(clusters) == 1
        assert "t1" in clusters

    def test_deterministic_cluster_ids(self):
        """Cluster IDs should be deterministic across runs."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],
            "t2": ["A", "B", "C", "D", "X", "Y"],
            "t3": ["G", "H", "I", "J", "K", "L"],
        }
        c1 = core_cluster_teams(teams)
        c2 = core_cluster_teams(teams)
        assert c1 == c2

    def test_cluster_id_format(self):
        """Cluster IDs should follow cluster_N format."""
        teams = {
            "t1": ["A", "B", "C", "D", "E", "F"],
            "t2": ["G", "H", "I", "J", "K", "L"],
        }
        clusters = core_cluster_teams(teams)
        for cid in clusters.values():
            assert cid.startswith("cluster_")

    def test_all_teams_assigned(self):
        """Every input team must get a cluster assignment."""
        teams = {f"t{i}": [f"S{i}_{j}" for j in range(6)] for i in range(10)}
        clusters = core_cluster_teams(teams)
        assert set(clusters.keys()) == set(teams.keys())

    def test_large_cluster_chain(self):
        """Chain of teams, each overlapping 4 with the next → one big cluster."""
        # t0: [S0, S1, S2, S3, S4, S5]
        # t1: [S0, S1, S2, S3, S6, S7]  (shares 4 with t0)
        # t2: [S0, S1, S6, S7, S8, S9]  (shares 4 with t1: S0,S1,S6,S7)
        teams = {}
        base = 0
        for i in range(5):
            species = [f"S{base}", f"S{base+1}"]
            # Add 4 new species, but share 4 with previous via overlap
            if i == 0:
                species = [f"S{j}" for j in range(6)]
            else:
                # Share S_{2*(i-1)}, S_{2*(i-1)+1} from prev's unique pair
                # and S_{shared0}, S_{shared1} from prev
                prev_shared = [f"S{2*(i-1)}", f"S{2*(i-1)+1}"]
                new = [f"S{2*i+4}", f"S{2*i+5}"]
                # Need 4 overlap with previous team
                # Previous team has species at indices based on construction
                pass

        # Simpler chain: each team shares exactly 4 species with the next
        teams = {
            "t0": ["A", "B", "C", "D", "E", "F"],
            "t1": ["A", "B", "C", "D", "G", "H"],    # 4 overlap with t0
            "t2": ["A", "B", "G", "H", "I", "J"],    # 4 overlap with t1
            "t3": ["A", "B", "I", "J", "K", "L"],    # 4 overlap with t2
        }
        clusters = core_cluster_teams(teams)
        # All should be in the same cluster via transitive closure
        assert len(set(clusters.values())) == 1


# ---------------------------------------------------------------------------
# run_cluster (integration)
# ---------------------------------------------------------------------------

def _make_canonical_jsonl(path: Path, examples: list[dict]) -> None:
    """Write dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _make_team_dict(team_id: str, species: list[str], format_id: str = "gen9test") -> dict:
    """Build a minimal team dict for testing."""
    return {
        "team_id": team_id,
        "species_key": f"sk_{team_id}",
        "format_id": format_id,
        "pokemon": [
            {"species": s, "item": "Item", "ability": "Ability",
             "tera_type": "Fire", "moves": ["M1", "M2", "M3", "M4"]}
            for s in species
        ],
    }


def _make_example_dict(
    example_id: str,
    team_a_id: str,
    team_a_species: list[str],
    team_b_id: str,
    team_b_species: list[str],
) -> dict:
    """Build a minimal canonical example dict."""
    return {
        "example_id": example_id,
        "match_group_id": f"mg_{example_id}",
        "battle_id": f"b_{example_id}",
        "team_a": _make_team_dict(team_a_id, team_a_species),
        "team_b": _make_team_dict(team_b_id, team_b_species),
        "label": {"lead2_idx": [0, 1], "back2_idx": [2, 3], "action90_id": 0},
        "label_quality": {"bring4_observed": True, "notes": None},
        "format_id": "gen9test",
        "metadata": {},
    }


class TestRunCluster:
    def test_basic_integration(self, tmp_path):
        """Cluster a small set of examples and verify outputs."""
        examples = [
            _make_example_dict("e1", "ta1", ["A", "B", "C", "D", "E", "F"],
                               "tb1", ["G", "H", "I", "J", "K", "L"]),
            _make_example_dict("e2", "ta2", ["A", "B", "C", "D", "X", "Y"],
                               "tb1", ["G", "H", "I", "J", "K", "L"]),
        ]
        in_path = tmp_path / "canonical" / "match_examples.jsonl"
        _make_canonical_jsonl(in_path, examples)

        out_dir = tmp_path / "clusters"
        manifest = run_cluster(str(in_path), str(out_dir))

        # Check outputs exist
        assert (out_dir / "cluster_assignments.json").exists()
        assert (out_dir / "cluster_manifest.json").exists()

        # Load and verify assignments
        with open(out_dir / "cluster_assignments.json") as f:
            assignments = json.load(f)

        # ta1 and ta2 share 4 species → same cluster
        assert assignments["ta1"] == assignments["ta2"]
        # tb1 is disjoint → different cluster
        assert assignments["tb1"] != assignments["ta1"]
        # 3 unique teams
        assert len(assignments) == 3

    def test_manifest_stats(self, tmp_path):
        """Manifest should contain correct statistics."""
        examples = [
            _make_example_dict("e1", "ta1", ["A", "B", "C", "D", "E", "F"],
                               "tb1", ["G", "H", "I", "J", "K", "L"]),
        ]
        in_path = tmp_path / "canonical" / "match_examples.jsonl"
        _make_canonical_jsonl(in_path, examples)

        out_dir = tmp_path / "clusters"
        manifest = run_cluster(str(in_path), str(out_dir))

        assert manifest["examples_read"] == 1
        assert manifest["unique_teams"] == 2
        assert manifest["num_clusters"] == 2
        assert manifest["singleton_clusters"] == 2

    def test_both_sides_clustered(self, tmp_path):
        """Teams from both team_a and team_b should appear in assignments."""
        examples = [
            _make_example_dict("e1", "ta1", ["A", "B", "C", "D", "E", "F"],
                               "tb1", ["A", "B", "C", "D", "X", "Y"]),
        ]
        in_path = tmp_path / "canonical" / "match_examples.jsonl"
        _make_canonical_jsonl(in_path, examples)

        out_dir = tmp_path / "clusters"
        run_cluster(str(in_path), str(out_dir))

        with open(out_dir / "cluster_assignments.json") as f:
            assignments = json.load(f)

        assert "ta1" in assignments
        assert "tb1" in assignments
        # They share 4 species → same cluster
        assert assignments["ta1"] == assignments["tb1"]

    def test_dedup_teams_across_examples(self, tmp_path):
        """Same team_id in multiple examples should only be counted once."""
        examples = [
            _make_example_dict("e1", "ta1", ["A", "B", "C", "D", "E", "F"],
                               "tb1", ["G", "H", "I", "J", "K", "L"]),
            _make_example_dict("e2", "ta1", ["A", "B", "C", "D", "E", "F"],
                               "tb2", ["M", "N", "O", "P", "Q", "R"]),
        ]
        in_path = tmp_path / "canonical" / "match_examples.jsonl"
        _make_canonical_jsonl(in_path, examples)

        out_dir = tmp_path / "clusters"
        manifest = run_cluster(str(in_path), str(out_dir))

        assert manifest["examples_read"] == 2
        assert manifest["unique_teams"] == 3  # ta1, tb1, tb2
