"""Tests for turnzero.data.sequential_dataset."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from turnzero.data.bo3_context import SENTINEL_LEAD
from turnzero.data.dataset import Vocab
from turnzero.data.sequential_dataset import SequentialVGCDataset


def _make_mon(species: str = "Pikachu") -> dict:
    return {
        "species": species,
        "item": "Light Ball",
        "ability": "Static",
        "tera_type": "Electric",
        "moves": ["Thunderbolt", "Volt Tackle", "Iron Tail", "Quick Attack"],
    }


def _make_example(
    battle_id: str = "battle-1",
    team_a_species: list[str] | None = None,
    team_b_species: list[str] | None = None,
) -> dict:
    if team_a_species is None:
        team_a_species = ["Mon0", "Mon1", "Mon2", "Mon3", "Mon4", "Mon5"]
    if team_b_species is None:
        team_b_species = ["Opp0", "Opp1", "Opp2", "Opp3", "Opp4", "Opp5"]

    return {
        "example_id": f"ex-{battle_id}",
        "match_group_id": battle_id,
        "battle_id": battle_id,
        "team_a": {
            "team_id": "ta",
            "species_key": "sk",
            "format_id": "gen9vgc2025reggbo3",
            "pokemon": [_make_mon(sp) for sp in team_a_species],
        },
        "team_b": {
            "team_id": "tb",
            "species_key": "sk",
            "format_id": "gen9vgc2025reggbo3",
            "pokemon": [_make_mon(sp) for sp in team_b_species],
        },
        "label": {"lead2_idx": [0, 1], "back2_idx": [2, 3], "action90_id": 0},
        "label_quality": {"bring4_observed": True, "notes": ""},
        "format_id": "gen9vgc2025reggbo3",
        "metadata": {"source_dataset": "test", "timestamp_epoch": 0},
        "split_keys": {"is_mirror": False},
    }


@pytest.fixture
def vocab_and_files(tmp_path):
    """Create a minimal JSONL + context JSON for testing."""
    team_a_species = ["Mon0", "Mon1", "Mon2", "Mon3", "Mon4", "Mon5"]
    team_b_species = ["Opp0", "Opp1", "Opp2", "Opp3", "Opp4", "Opp5"]

    examples = [
        _make_example("battle-game1", team_a_species, team_b_species),
        _make_example("battle-game2", team_a_species, team_b_species),
        _make_example("battle-no-ctx", team_a_species, team_b_species),
    ]

    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Context: battle-game2 is game 2 with prior leads Mon0+Mon1
    context = {
        "battle-game2": {
            "game_num": 2,
            "sides": {
                "p1": {"prior_leads": ["Mon0", "Mon1"], "prior_result": 2},
                "p2": {"prior_leads": ["Opp2", "Opp3"], "prior_result": 1},
            },
            "species6": {
                "p1": sorted(team_a_species),
                "p2": sorted(team_b_species),
            },
            "curr_species6": {
                "p1": sorted(team_a_species),
                "p2": sorted(team_b_species),
            },
        }
    }
    ctx_path = tmp_path / "bo3_context.json"
    with open(ctx_path, "w") as f:
        json.dump(context, f)

    # Build vocab from examples
    vocab = Vocab.from_examples(examples)

    return vocab, jsonl_path, ctx_path


class TestSequentialDatasetNoContext:
    def test_sentinel_values(self, vocab_and_files):
        vocab, jsonl_path, _ = vocab_and_files
        ds = SequentialVGCDataset(jsonl_path, vocab, context_path=None)
        item = ds[0]
        assert item["prior_lead2_idx"] == SENTINEL_LEAD
        assert item["prior_result"] == 0
        assert item["game_num"] == 0

    def test_backward_compat_fields(self, vocab_and_files):
        """Base VGCDataset fields are still present."""
        vocab, jsonl_path, _ = vocab_and_files
        ds = SequentialVGCDataset(jsonl_path, vocab, context_path=None)
        item = ds[0]
        assert "team_a" in item
        assert "team_b" in item
        assert "action90_label" in item
        assert "lead2_label" in item
        assert item["team_a"].shape == (6, 8)


class TestSequentialDatasetWithContext:
    def test_game2_gets_context(self, vocab_and_files):
        vocab, jsonl_path, ctx_path = vocab_and_files
        ds = SequentialVGCDataset(jsonl_path, vocab, context_path=ctx_path)

        # battle-game2 is index 1 (p1 perspective → team_a = Mon0..Mon5)
        item = ds[1]
        assert item["prior_lead2_idx"] == 0  # Mon0+Mon1 = slots (0,1) = idx 0
        assert item["prior_result"] == 2  # p1 lost
        assert item["game_num"] == 1  # game 2 → mapped to 1

    def test_no_context_battle_gets_sentinel(self, vocab_and_files):
        vocab, jsonl_path, ctx_path = vocab_and_files
        ds = SequentialVGCDataset(jsonl_path, vocab, context_path=ctx_path)

        # battle-no-ctx is index 2
        item = ds[2]
        assert item["prior_lead2_idx"] == SENTINEL_LEAD
        assert item["prior_result"] == 0
        assert item["game_num"] == 0

    def test_game1_gets_sentinel(self, vocab_and_files):
        vocab, jsonl_path, ctx_path = vocab_and_files
        ds = SequentialVGCDataset(jsonl_path, vocab, context_path=ctx_path)

        # battle-game1 has no context entry
        item = ds[0]
        assert item["prior_lead2_idx"] == SENTINEL_LEAD
        assert item["prior_result"] == 0
        assert item["game_num"] == 0

    def test_dataset_length_unchanged(self, vocab_and_files):
        vocab, jsonl_path, ctx_path = vocab_and_files
        base_ds = SequentialVGCDataset(jsonl_path, vocab, context_path=None)
        seq_ds = SequentialVGCDataset(jsonl_path, vocab, context_path=ctx_path)
        assert len(seq_ds) == len(base_ds)
