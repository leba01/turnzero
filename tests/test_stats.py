"""Tests for Stage 7: comprehensive dataset stats + integrity validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnzero.data.stats import (
    _collect_split_stats,
    _validate_regime_a,
    _validate_regime_b,
    run_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ex(
    example_id: str,
    team_a_id: str,
    team_b_id: str,
    cluster_a: str = "c0",
    cluster_b: str = "c1",
    bring4: bool = True,
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
            "reconstruction_quality": {
                "fields_known": 42, "fields_total": 42,
                "source_method": "showteam_direct",
            },
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
            "reconstruction_quality": {
                "fields_known": 42, "fields_total": 42,
                "source_method": "showteam_direct",
            },
        },
        "label": {"lead2_idx": [0, 1], "back2_idx": [2, 3],
                  "action90_id": action90_id},
        "label_quality": {"bring4_observed": bring4, "notes": None},
        "format_id": "gen9test",
        "metadata": {},
        "split_keys": {
            "team_a_id": team_a_id,
            "team_b_id": team_b_id,
            "core_cluster_a": cluster_a,
            "core_cluster_b": cluster_b,
            "is_mirror": cluster_a == cluster_b,
        },
    }


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


# ---------------------------------------------------------------------------
# _collect_split_stats
# ---------------------------------------------------------------------------

class TestCollectSplitStats:
    def test_empty(self):
        result = _collect_split_stats([])
        assert result["count"] == 0

    def test_basic_counts(self):
        examples = [
            _ex("e1", "ta", "tb", bring4=True),
            _ex("e2", "ta", "tb", bring4=False),
            _ex("e3", "ta", "tb", bring4=True),
        ]
        result = _collect_split_stats(examples)
        assert result["count"] == 3
        assert result["bring4_observed_true"] == 2
        assert result["bring4_observed_false"] == 1
        assert result["bring4_observed_rate"] == round(2 / 3, 4)

    def test_mirror_counting(self):
        examples = [
            _ex("e1", "ta", "tb", cluster_a="c0", cluster_b="c0"),  # mirror
            _ex("e2", "ta", "tb", cluster_a="c0", cluster_b="c1"),  # not mirror
        ]
        result = _collect_split_stats(examples)
        assert result["mirror_count"] == 1
        assert result["non_mirror_count"] == 1

    def test_ots_completeness(self):
        examples = [_ex("e1", "ta", "tb")]
        result = _collect_split_stats(examples)
        assert result["ots_completeness"]["mean_fields_known"] == 42.0
        assert result["ots_completeness"]["pct_fully_known"] == 1.0

    def test_unique_teams(self):
        examples = [
            _ex("e1", "ta1", "tb1"),
            _ex("e2", "ta2", "tb1"),
            _ex("e3", "ta1", "tb2"),
        ]
        result = _collect_split_stats(examples)
        assert result["unique_team_a_ids"] == 2
        assert result["unique_team_b_ids"] == 2
        assert result["unique_teams_total"] == 4


# ---------------------------------------------------------------------------
# Integrity validators
# ---------------------------------------------------------------------------

class TestValidateRegimeA:
    def test_clean_passes(self):
        splits = {
            "train": [_ex("e1", "ta1", "tb1")],
            "val": [_ex("e2", "ta2", "tb1")],
            "test": [_ex("e3", "ta3", "tb1")],
        }
        assert _validate_regime_a(splits) == []

    def test_detects_team_a_leak(self):
        splits = {
            "train": [_ex("e1", "ta1", "tb1")],
            "val": [],
            "test": [_ex("e2", "ta1", "tb2")],  # same team_a_id in test
        }
        violations = _validate_regime_a(splits)
        assert any("team_a_id" in v for v in violations)

    def test_detects_triple_cross(self):
        splits = {
            "train": [_ex("e1", "ta1", "tb1", action90_id=5)],
            "val": [],
            "test": [_ex("e2", "ta1", "tb1", action90_id=5)],
        }
        violations = _validate_regime_a(splits)
        assert any("triple" in v for v in violations)

    def test_detects_match_group_cross(self):
        splits = {
            "train": [_ex("e1", "ta1", "tb1", match_group_id="mg1")],
            "val": [],
            "test": [_ex("e2", "ta2", "tb2", match_group_id="mg1")],
        }
        violations = _validate_regime_a(splits)
        assert any("match_group" in v for v in violations)


class TestValidateRegimeB:
    def test_clean_passes(self):
        splits = {
            "train": [_ex("e1", "ta1", "tb1", cluster_a="c0")],
            "val": [_ex("e2", "ta2", "tb1", cluster_a="c1")],
            "test": [_ex("e3", "ta3", "tb1", cluster_a="c2")],
        }
        assert _validate_regime_b(splits) == []

    def test_detects_cluster_leak(self):
        splits = {
            "train": [_ex("e1", "ta1", "tb1", cluster_a="c0")],
            "val": [],
            "test": [_ex("e2", "ta2", "tb1", cluster_a="c0")],  # same cluster
        }
        violations = _validate_regime_b(splits)
        assert any("core_cluster_a" in v for v in violations)


# ---------------------------------------------------------------------------
# run_stats integration
# ---------------------------------------------------------------------------

class TestRunStats:
    def _setup(self, tmp_path) -> Path:
        """Create a minimal assembled directory."""
        data_dir = tmp_path / "assembled"
        for regime in ("regime_a", "regime_b"):
            train = [
                _ex("e1", "ta1", "tb1", cluster_a="c0", cluster_b="c0"),
                _ex("e2", "ta2", "tb1", cluster_a="c0", cluster_b="c0"),
            ]
            val = [_ex("e3", "ta3", "tb1", cluster_a="c1", cluster_b="c0")]
            test = [_ex("e4", "ta4", "tb2", cluster_a="c2", cluster_b="c1")]

            _write_jsonl(data_dir / regime / "train.jsonl", train)
            _write_jsonl(data_dir / regime / "val.jsonl", val)
            _write_jsonl(data_dir / regime / "test.jsonl", test)

        return data_dir

    def test_produces_report(self, tmp_path):
        data_dir = self._setup(tmp_path)
        report = run_stats(str(data_dir), validate=True)
        assert (data_dir / "dataset_report.json").exists()

    def test_report_structure(self, tmp_path):
        data_dir = self._setup(tmp_path)
        report = run_stats(str(data_dir), validate=True)
        assert "regime_a" in report
        assert "regime_b" in report
        assert "validation_passed" in report
        for regime in ("regime_a", "regime_b"):
            for split_name in ("train", "val", "test", "overall"):
                assert split_name in report[regime]

    def test_validation_passes_on_clean_data(self, tmp_path):
        data_dir = self._setup(tmp_path)
        report = run_stats(str(data_dir), validate=True)
        assert report["validation_passed"] is True
        assert report["all_violations"] == []

    def test_skip_validation(self, tmp_path):
        data_dir = self._setup(tmp_path)
        report = run_stats(str(data_dir), validate=False)
        assert "integrity_violations" not in report.get("regime_a", {})

    def test_counts_correct(self, tmp_path):
        data_dir = self._setup(tmp_path)
        report = run_stats(str(data_dir), validate=False)
        assert report["regime_a"]["train"]["count"] == 2
        assert report["regime_a"]["val"]["count"] == 1
        assert report["regime_a"]["test"]["count"] == 1
        assert report["regime_a"]["overall"]["count"] == 4
