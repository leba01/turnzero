"""Tests for Stage 6: assemble per-split JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnzero.data.assemble import _attach_split_keys, run_assemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ex(
    example_id: str,
    team_a_id: str,
    team_b_id: str,
    action90_id: int = 0,
    match_group_id: str | None = None,
) -> dict:
    if match_group_id is None:
        match_group_id = f"mg_{example_id}"
    return {
        "example_id": example_id,
        "match_group_id": match_group_id,
        "battle_id": f"b_{example_id}",
        "team_a": {
            "team_id": team_a_id,
            "species_key": f"sk_{team_a_id}",
            "format_id": "gen9test",
            "pokemon": [
                {"species": f"S{i}", "item": "I", "ability": "A",
                 "tera_type": "Fire", "moves": ["M1", "M2", "M3", "M4"]}
                for i in range(6)
            ],
        },
        "team_b": {
            "team_id": team_b_id,
            "species_key": f"sk_{team_b_id}",
            "format_id": "gen9test",
            "pokemon": [
                {"species": f"S{i}", "item": "I", "ability": "A",
                 "tera_type": "Fire", "moves": ["M1", "M2", "M3", "M4"]}
                for i in range(6)
            ],
        },
        "label": {"lead2_idx": [0, 1], "back2_idx": [2, 3],
                  "action90_id": action90_id},
        "label_quality": {"bring4_observed": True, "notes": None},
        "format_id": "gen9test",
        "metadata": {},
    }


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# _attach_split_keys
# ---------------------------------------------------------------------------

class TestAttachSplitKeys:
    def test_adds_split_keys(self):
        clusters = {"ta": "c0", "tb": "c1"}
        ex = _ex("e1", "ta", "tb")
        result = _attach_split_keys(ex, clusters)
        assert result["split_keys"]["core_cluster_a"] == "c0"
        assert result["split_keys"]["core_cluster_b"] == "c1"
        assert result["split_keys"]["team_a_id"] == "ta"
        assert result["split_keys"]["team_b_id"] == "tb"
        assert result["split_keys"]["is_mirror"] is False

    def test_mirror_detected(self):
        clusters = {"ta": "c0", "tb": "c0"}
        ex = _ex("e1", "ta", "tb")
        result = _attach_split_keys(ex, clusters)
        assert result["split_keys"]["is_mirror"] is True

    def test_unknown_cluster_handled(self):
        clusters = {"ta": "c0"}  # tb not in clusters
        ex = _ex("e1", "ta", "tb")
        result = _attach_split_keys(ex, clusters)
        assert result["split_keys"]["core_cluster_a"] == "c0"
        assert result["split_keys"]["core_cluster_b"] == "unknown"


# ---------------------------------------------------------------------------
# run_assemble integration
# ---------------------------------------------------------------------------

class TestRunAssemble:
    def _setup(self, tmp_path):
        """Create canonical JSONL, clusters, and splits for testing."""
        examples = [
            _ex("e1", "ta1", "tb1"),
            _ex("e2", "ta2", "tb1"),
            _ex("e3", "ta3", "tb2"),
            _ex("e4", "ta4", "tb2"),
        ]
        clusters = {
            "ta1": "c0", "ta2": "c0", "ta3": "c1", "ta4": "c1",
            "tb1": "c0", "tb2": "c1",
        }
        splits = {
            "regime_a": {
                "train": ["e1", "e2"],
                "val": ["e3"],
                "test": ["e4"],
            },
            "regime_b": {
                "train": ["e1", "e3"],
                "val": ["e2"],
                "test": ["e4"],
            },
        }

        canonical_path = tmp_path / "canonical" / "match_examples.jsonl"
        _write_jsonl(canonical_path, examples)

        clusters_path = tmp_path / "clusters.json"
        with open(clusters_path, "w") as f:
            json.dump(clusters, f)

        splits_path = tmp_path / "splits.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f)

        return str(canonical_path), str(clusters_path), str(splits_path)

    def test_output_files_created(self, tmp_path):
        canon, clusters, splits = self._setup(tmp_path)
        out_dir = tmp_path / "assembled"
        run_assemble(canon, clusters, splits, str(out_dir))

        for regime in ("regime_a", "regime_b"):
            for split_name in ("train", "val", "test"):
                assert (out_dir / regime / f"{split_name}.jsonl").exists()
        assert (out_dir / "assemble_manifest.json").exists()

    def test_correct_example_counts(self, tmp_path):
        canon, clusters, splits = self._setup(tmp_path)
        out_dir = tmp_path / "assembled"
        manifest = run_assemble(canon, clusters, splits, str(out_dir))

        assert manifest["regime_a"]["train"] == 2
        assert manifest["regime_a"]["val"] == 1
        assert manifest["regime_a"]["test"] == 1
        assert manifest["regime_a"]["total"] == 4

        assert manifest["regime_b"]["train"] == 2
        assert manifest["regime_b"]["val"] == 1
        assert manifest["regime_b"]["test"] == 1
        assert manifest["regime_b"]["total"] == 4

    def test_split_keys_attached(self, tmp_path):
        canon, clusters, splits = self._setup(tmp_path)
        out_dir = tmp_path / "assembled"
        run_assemble(canon, clusters, splits, str(out_dir))

        examples = _read_jsonl(out_dir / "regime_a" / "train.jsonl")
        for ex in examples:
            assert "split_keys" in ex
            assert "core_cluster_a" in ex["split_keys"]
            assert "core_cluster_b" in ex["split_keys"]
            assert "is_mirror" in ex["split_keys"]

    def test_mirror_flag_correct(self, tmp_path):
        canon, clusters, splits = self._setup(tmp_path)
        out_dir = tmp_path / "assembled"
        run_assemble(canon, clusters, splits, str(out_dir))

        # e1: ta1(c0) vs tb1(c0) → mirror
        train = _read_jsonl(out_dir / "regime_a" / "train.jsonl")
        e1 = [ex for ex in train if ex["example_id"] == "e1"][0]
        assert e1["split_keys"]["is_mirror"] is True

        # e4: ta4(c1) vs tb2(c1) → mirror
        test = _read_jsonl(out_dir / "regime_a" / "test.jsonl")
        e4 = test[0]
        assert e4["split_keys"]["is_mirror"] is True

    def test_no_examples_lost(self, tmp_path):
        canon, clusters, splits = self._setup(tmp_path)
        out_dir = tmp_path / "assembled"
        manifest = run_assemble(canon, clusters, splits, str(out_dir))

        # All 4 examples should appear in each regime
        for regime in ("regime_a", "regime_b"):
            all_ids = set()
            for split_name in ("train", "val", "test"):
                examples = _read_jsonl(out_dir / regime / f"{split_name}.jsonl")
                for ex in examples:
                    all_ids.add(ex["example_id"])
            assert all_ids == {"e1", "e2", "e3", "e4"}

    def test_examples_not_in_split_excluded(self, tmp_path):
        """Examples not assigned to any split in a regime should not appear."""
        examples = [_ex("e1", "ta1", "tb1"), _ex("e2", "ta2", "tb1")]
        clusters = {"ta1": "c0", "ta2": "c0", "tb1": "c0"}
        splits = {
            "regime_a": {"train": ["e1"], "val": [], "test": []},
            "regime_b": {"train": ["e1", "e2"], "val": [], "test": []},
        }

        canonical_path = tmp_path / "data.jsonl"
        _write_jsonl(canonical_path, examples)
        clusters_path = tmp_path / "clusters.json"
        with open(clusters_path, "w") as f:
            json.dump(clusters, f)
        splits_path = tmp_path / "splits.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f)

        out_dir = tmp_path / "out"
        manifest = run_assemble(
            str(canonical_path), str(clusters_path), str(splits_path), str(out_dir)
        )

        # Regime A: only e1 assigned
        assert manifest["regime_a"]["total"] == 1
        # Regime B: both assigned
        assert manifest["regime_b"]["total"] == 2

    def test_mirror_counts_in_manifest(self, tmp_path):
        canon, clusters, splits = self._setup(tmp_path)
        out_dir = tmp_path / "assembled"
        manifest = run_assemble(canon, clusters, splits, str(out_dir))

        assert "mirror_counts" in manifest["regime_a"]
        assert "mirror_counts" in manifest["regime_b"]
