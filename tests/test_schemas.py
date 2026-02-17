"""Tests for canonical data schemas."""

from __future__ import annotations

import json

import pytest
from turnzero.schemas import (
    Label,
    LabelQuality,
    MatchExample,
    Pokemon,
    ReconstructionQuality,
    TeamSheet,
)


# ---------------------------------------------------------------------------
# Pokemon
# ---------------------------------------------------------------------------

def test_pokemon_valid():
    p = Pokemon(species="Flutter Mane", item="Booster Energy", ability="Protosynthesis",
                tera_type="Fairy", moves=["Dazzling Gleam", "Moonblast", "Protect", "Shadow Ball"])
    assert p.species == "Flutter Mane"
    assert len(p.moves) == 4


def test_pokemon_defaults_to_unk():
    p = Pokemon(species="Ditto")
    assert p.item == "UNK"
    assert p.ability == "UNK"
    assert p.tera_type == "UNK"
    assert p.moves == ["UNK"] * 4


def test_pokemon_rejects_wrong_move_count():
    with pytest.raises(ValueError, match="moves must have exactly 4"):
        Pokemon(species="Ditto", moves=["Transform"])


def test_pokemon_round_trip():
    p = Pokemon(species="Incineroar", item="Safety Goggles", ability="Intimidate",
                tera_type="Ghost", moves=["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"])
    assert Pokemon.from_dict(p.to_dict()) == p


# ---------------------------------------------------------------------------
# TeamSheet
# ---------------------------------------------------------------------------

def _make_team(species_list: list[str] | None = None) -> TeamSheet:
    species_list = species_list or [
        "Flutter Mane", "Incineroar", "Rillaboom",
        "Urshifu-Rapid-Strike", "Tornadus", "Chien-Pao",
    ]
    pokemon = [Pokemon(species=s) for s in species_list]
    return TeamSheet(
        team_id="abc123",
        species_key="def456",
        format_id="gen9vgc_regg",
        pokemon=pokemon,
    )


def test_teamsheet_valid():
    ts = _make_team()
    assert len(ts.pokemon) == 6


def test_teamsheet_rejects_wrong_count():
    with pytest.raises(ValueError, match="exactly 6 pokemon"):
        TeamSheet(team_id="x", species_key="y", format_id="z",
                  pokemon=[Pokemon(species="A")])


def test_teamsheet_round_trip():
    ts = _make_team()
    d = ts.to_dict()
    ts2 = TeamSheet.from_dict(d)
    assert ts2.team_id == ts.team_id
    assert len(ts2.pokemon) == 6
    assert ts2.pokemon[0].species == "Flutter Mane"


def test_teamsheet_round_trip_with_rq():
    ts = _make_team()
    ts.reconstruction_quality = ReconstructionQuality(fields_known=42)
    d = ts.to_dict()
    ts2 = TeamSheet.from_dict(d)
    assert ts2.reconstruction_quality is not None
    assert ts2.reconstruction_quality.fields_known == 42
    assert ts2.reconstruction_quality.source_method == "showteam_direct"


def test_teamsheet_json_serializable():
    ts = _make_team()
    s = json.dumps(ts.to_dict())
    assert isinstance(s, str)


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

def test_label_valid():
    lbl = Label(lead2_idx=(0, 3), back2_idx=(1, 5), action90_id=0)
    assert lbl.lead2_idx == (0, 3)


def test_label_rejects_unsorted_lead():
    with pytest.raises(ValueError, match="sorted ascending"):
        Label(lead2_idx=(3, 0), back2_idx=(1, 5), action90_id=0)


def test_label_rejects_unsorted_back():
    with pytest.raises(ValueError, match="sorted ascending"):
        Label(lead2_idx=(0, 3), back2_idx=(5, 1), action90_id=0)


def test_label_rejects_overlap():
    with pytest.raises(ValueError, match="must not overlap"):
        Label(lead2_idx=(0, 1), back2_idx=(1, 2), action90_id=0)


def test_label_rejects_out_of_range():
    with pytest.raises(ValueError, match="must be in 0..5"):
        Label(lead2_idx=(0, 6), back2_idx=(1, 2), action90_id=0)


def test_label_rejects_bad_action_id():
    with pytest.raises(ValueError, match="action90_id"):
        Label(lead2_idx=(0, 1), back2_idx=(2, 3), action90_id=90)


def test_label_round_trip():
    lbl = Label(lead2_idx=(0, 3), back2_idx=(1, 5), action90_id=7)
    assert Label.from_dict(lbl.to_dict()) == lbl


# ---------------------------------------------------------------------------
# LabelQuality
# ---------------------------------------------------------------------------

def test_label_quality_round_trip():
    lq = LabelQuality(bring4_observed=True, notes="test")
    assert LabelQuality.from_dict(lq.to_dict()) == lq


def test_label_quality_defaults():
    lq = LabelQuality(bring4_observed=False)
    assert lq.notes is None


# ---------------------------------------------------------------------------
# MatchExample
# ---------------------------------------------------------------------------

def _make_example() -> MatchExample:
    team_a = _make_team()
    team_b = _make_team(["Calyrex-Ice", "Pelipper", "Amoonguss",
                          "Raging Bolt", "Cresselia", "Ogerpon-Cornerstone"])
    return MatchExample(
        example_id="ex001",
        match_group_id="mg001",
        battle_id="gen9vgc2024regg-123",
        team_a=team_a,
        team_b=team_b,
        label=Label(lead2_idx=(0, 1), back2_idx=(2, 3), action90_id=0),
        label_quality=LabelQuality(bring4_observed=True),
        format_id="gen9vgc_regg",
        metadata={"timestamp_epoch": 1700000000},
    )


def test_match_example_round_trip():
    ex = _make_example()
    d = ex.to_dict()
    ex2 = MatchExample.from_dict(d)
    assert ex2.example_id == ex.example_id
    assert ex2.team_a.team_id == ex.team_a.team_id
    assert ex2.label.action90_id == 0
    assert ex2.label_quality.bring4_observed is True


def test_match_example_json_round_trip():
    ex = _make_example()
    s = json.dumps(ex.to_dict())
    d = json.loads(s)
    ex2 = MatchExample.from_dict(d)
    assert ex2.example_id == "ex001"


def test_match_example_split_keys_optional():
    ex = _make_example()
    assert ex.split_keys is None
    d = ex.to_dict()
    assert "split_keys" not in d
