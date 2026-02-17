"""Tests for the Showdown protocol parser."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from turnzero.data.parser import (
    parse_showteam_line,
    parse_battle,
    produce_directed_examples,
    run_parse,
    _match_to_showteam,
    _build_label,
    _team_id,
    _species_key,
)
from turnzero.schemas import Pokemon


# ---------------------------------------------------------------------------
# Real |showteam| line from logs-gen9vgc2024regg.json battle 2254013652
# ---------------------------------------------------------------------------
SHOWTEAM_P1 = (
    "|showteam|p1|"
    "Rillaboom||AssaultVest|GrassySurge|FakeOut,WoodHammer,GrassyGlide,HighHorsepower"
    "|||F|||50|,,,,,Ground"
    "]Urshifu-Rapid-Strike||MysticWater|UnseenFist|AquaJet,SurgingStrikes,CloseCombat,Detect"
    "|||M|||50|,,,,,Grass"
    "]Tornadus||CovertCloak|Prankster|Tailwind,BleakwindStorm,RainDance,Protect"
    "|||M|||50|,,,,,Dark"
    "]Zacian-Crowned||RustedSword|IntrepidSword|behemothblade,PlayRough,SacredSword,Protect"
    "|||||||50|,,,,,Fairy"
    "]Chien-Pao||FocusSash|SwordofRuin|IcicleCrash,SuckerPunch,SacredSword,Protect"
    "|||||||50|,,,,,Stellar"
    "]Umbreon||SafetyGoggles|InnerFocus|FoulPlay,Yawn,Taunt,HelpingHand"
    "|||F|||50|,,,,,Water"
)


# ---------------------------------------------------------------------------
# parse_showteam_line
# ---------------------------------------------------------------------------

class TestParseShowteamLine:
    def test_extracts_six_mons(self):
        side, pokemon = parse_showteam_line(SHOWTEAM_P1)
        assert side == "p1"
        assert len(pokemon) == 6

    def test_first_mon_fields(self):
        _, pokemon = parse_showteam_line(SHOWTEAM_P1)
        p = pokemon[0]
        assert p.species == "Rillaboom"
        assert p.item == "AssaultVest"
        assert p.ability == "GrassySurge"
        assert p.tera_type == "Ground"
        assert p.moves == ["FakeOut", "WoodHammer", "GrassyGlide", "HighHorsepower"]

    def test_form_species(self):
        _, pokemon = parse_showteam_line(SHOWTEAM_P1)
        assert pokemon[1].species == "Urshifu-Rapid-Strike"
        assert pokemon[3].species == "Zacian-Crowned"

    def test_all_tera_types(self):
        _, pokemon = parse_showteam_line(SHOWTEAM_P1)
        teras = [p.tera_type for p in pokemon]
        assert teras == ["Ground", "Grass", "Dark", "Fairy", "Stellar", "Water"]

    def test_move_count_always_four(self):
        _, pokemon = parse_showteam_line(SHOWTEAM_P1)
        for p in pokemon:
            assert len(p.moves) == 4

    def test_ditto_padded(self):
        """Ditto has only 1 move; should be padded to 4 with UNK."""
        line = (
            "|showteam|p2|Ditto||ChoiceScarf|Imposter|Transform"
            "|||||||50|,,,,,Normal"
            "]Filler||Item|Ability|A,B,C,D|||||||50|,,,,,Fire"
            "]Filler2||Item|Ability|A,B,C,D|||||||50|,,,,,Fire"
            "]Filler3||Item|Ability|A,B,C,D|||||||50|,,,,,Fire"
            "]Filler4||Item|Ability|A,B,C,D|||||||50|,,,,,Fire"
            "]Filler5||Item|Ability|A,B,C,D|||||||50|,,,,,Fire"
        )
        side, pokemon = parse_showteam_line(line)
        assert side == "p2"
        assert pokemon[0].species == "Ditto"
        assert pokemon[0].moves == ["Transform", "UNK", "UNK", "UNK"]

    def test_rejects_non_showteam(self):
        with pytest.raises(ValueError, match="Not a showteam"):
            parse_showteam_line("|poke|p1|Pikachu")


# ---------------------------------------------------------------------------
# _match_to_showteam
# ---------------------------------------------------------------------------

class TestMatchToShowteam:
    def test_exact_match(self):
        species = ["Rillaboom", "Urshifu-Rapid-Strike", "Tornadus"]
        assert _match_to_showteam("Tornadus", species) == 2

    def test_prefix_match_showteam_longer(self):
        species = ["Ogerpon-Cornerstone", "Rillaboom"]
        # Switch says base form, showteam has full form
        assert _match_to_showteam("Ogerpon", species) == 0

    def test_prefix_match_switch_longer(self):
        species = ["Ogerpon", "Rillaboom"]
        # Showteam says base, switch has tera form
        assert _match_to_showteam("Ogerpon-Cornerstone-Tera", species) == 0

    def test_no_match_returns_none(self):
        species = ["Rillaboom", "Tornadus"]
        assert _match_to_showteam("Pikachu", species) is None

    def test_skips_used_indices(self):
        species = ["Rillaboom", "Rillaboom"]  # pathological
        assert _match_to_showteam("Rillaboom", species, used={0}) == 1


# ---------------------------------------------------------------------------
# Synthetic battle log for integration tests
# ---------------------------------------------------------------------------

def _make_battle_log() -> str:
    """Build a minimal valid battle log with showteam + leads + 4 revealed."""
    p1_team = (
        "Mon0||Item0|Ability0|M0a,M0b,M0c,M0d|||||||50|,,,,,Fire"
        "]Mon1||Item1|Ability1|M1a,M1b,M1c,M1d|||||||50|,,,,,Water"
        "]Mon2||Item2|Ability2|M2a,M2b,M2c,M2d|||||||50|,,,,,Grass"
        "]Mon3||Item3|Ability3|M3a,M3b,M3c,M3d|||||||50|,,,,,Electric"
        "]Mon4||Item4|Ability4|M4a,M4b,M4c,M4d|||||||50|,,,,,Ice"
        "]Mon5||Item5|Ability5|M5a,M5b,M5c,M5d|||||||50|,,,,,Dark"
    )
    p2_team = (
        "Opp0||OItem0|OAbil0|OM0a,OM0b,OM0c,OM0d|||||||50|,,,,,Fire"
        "]Opp1||OItem1|OAbil1|OM1a,OM1b,OM1c,OM1d|||||||50|,,,,,Water"
        "]Opp2||OItem2|OAbil2|OM2a,OM2b,OM2c,OM2d|||||||50|,,,,,Grass"
        "]Opp3||OItem3|OAbil3|OM3a,OM3b,OM3c,OM3d|||||||50|,,,,,Electric"
        "]Opp4||OItem4|OAbil4|OM4a,OM4b,OM4c,OM4d|||||||50|,,,,,Ice"
        "]Opp5||OItem5|OAbil5|OM5a,OM5b,OM5c,OM5d|||||||50|,,,,,Dark"
    )
    return "\n".join([
        "|t:|1700000000",
        f"|showteam|p1|{p1_team}",
        f"|showteam|p2|{p2_team}",
        "|poke|p1|Mon0|",
        "|poke|p1|Mon1|",
        "|poke|p1|Mon2|",
        "|poke|p1|Mon3|",
        "|poke|p1|Mon4|",
        "|poke|p1|Mon5|",
        "|poke|p2|Opp0|",
        "|poke|p2|Opp1|",
        "|poke|p2|Opp2|",
        "|poke|p2|Opp3|",
        "|poke|p2|Opp4|",
        "|poke|p2|Opp5|",
        "|start",
        # Leads: p1 sends Mon0 + Mon1, p2 sends Opp2 + Opp3
        "|switch|p1a: Mon0|Mon0, L50|100/100",
        "|switch|p1b: Mon1|Mon1, L50|100/100",
        "|switch|p2a: Opp2|Opp2, L50|100/100",
        "|switch|p2b: Opp3|Opp3, L50|100/100",
        "|turn|1",
        # Mid-game switches reveal 2 more per side (total 4 each)
        "|switch|p1a: Mon3|Mon3, L50|100/100",
        "|switch|p2a: Opp0|Opp0, L50|100/100",
        "|switch|p1b: Mon4|Mon4, L50|100/100",
        "|switch|p2b: Opp5|Opp5, L50|100/100",
        "|win|p1",
    ])


def _make_partial_log() -> str:
    """Battle where only 3 mons revealed per side (bring4_observed=False)."""
    p1_team = (
        "Mon0||I0|A0|M0a,M0b,M0c,M0d|||||||50|,,,,,Fire"
        "]Mon1||I1|A1|M1a,M1b,M1c,M1d|||||||50|,,,,,Water"
        "]Mon2||I2|A2|M2a,M2b,M2c,M2d|||||||50|,,,,,Grass"
        "]Mon3||I3|A3|M3a,M3b,M3c,M3d|||||||50|,,,,,Electric"
        "]Mon4||I4|A4|M4a,M4b,M4c,M4d|||||||50|,,,,,Ice"
        "]Mon5||I5|A5|M5a,M5b,M5c,M5d|||||||50|,,,,,Dark"
    )
    p2_team = p1_team.replace("Mon", "Opp").replace("I", "OI").replace("A", "OA").replace("M", "OM")
    return "\n".join([
        f"|showteam|p1|{p1_team}",
        f"|showteam|p2|{p2_team}",
        "|poke|p1|Mon0|", "|poke|p1|Mon1|", "|poke|p1|Mon2|",
        "|poke|p1|Mon3|", "|poke|p1|Mon4|", "|poke|p1|Mon5|",
        "|poke|p2|Opp0|", "|poke|p2|Opp1|", "|poke|p2|Opp2|",
        "|poke|p2|Opp3|", "|poke|p2|Opp4|", "|poke|p2|Opp5|",
        "|start",
        "|switch|p1a: Mon0|Mon0, L50|100/100",
        "|switch|p1b: Mon1|Mon1, L50|100/100",
        "|switch|p2a: Opp0|Opp0, L50|100/100",
        "|switch|p2b: Opp1|Opp1, L50|100/100",
        "|turn|1",
        # Only 1 more switch per side (3 total, not 4)
        "|switch|p1a: Mon2|Mon2, L50|100/100",
        "|switch|p2a: Opp2|Opp2, L50|100/100",
        "|win|p1",
    ])


# ---------------------------------------------------------------------------
# parse_battle
# ---------------------------------------------------------------------------

class TestParseBattle:
    def test_full_observation(self):
        parsed = parse_battle(_make_battle_log())
        assert len(parsed["showteam"]["p1"]) == 6
        assert len(parsed["showteam"]["p2"]) == 6
        assert parsed["leads"]["p1"] == ["Mon0", "Mon1"]
        assert parsed["leads"]["p2"] == ["Opp2", "Opp3"]
        assert len(parsed["revealed"]["p1"]) == 4
        assert len(parsed["revealed"]["p2"]) == 4
        assert parsed["timestamp"] == 1700000000

    def test_partial_observation(self):
        parsed = parse_battle(_make_partial_log())
        assert len(parsed["revealed"]["p1"]) == 3
        assert len(parsed["revealed"]["p2"]) == 3

    def test_no_showteam_raises(self):
        with pytest.raises(ValueError, match="No .showteam."):
            parse_battle("|start\n|switch|p1a: X|X, L50|100/100\n|turn|1")

    def test_no_start_raises(self):
        log = (
            "|showteam|p1|A||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]B||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]C||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]D||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]E||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]F||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire\n"
            "|showteam|p2|G||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]H||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]I||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]J||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]K||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
            "]L||I|Ab|M1,M2,M3,M4|||||||50|,,,,,Fire"
        )
        with pytest.raises(ValueError, match="No .start."):
            parse_battle(log)


# ---------------------------------------------------------------------------
# produce_directed_examples
# ---------------------------------------------------------------------------

class TestProduceDirectedExamples:
    def test_produces_two_examples(self):
        parsed = parse_battle(_make_battle_log())
        examples = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        assert len(examples) == 2

    def test_swapped_perspectives(self):
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        # First example: p1 is team_a
        assert ex[0].team_a.pokemon[0].species == "Mon0"
        assert ex[0].team_b.pokemon[0].species == "Opp0"
        # Second example: p2 is team_a
        assert ex[1].team_a.pokemon[0].species == "Opp0"
        assert ex[1].team_b.pokemon[0].species == "Mon0"

    def test_unique_example_ids(self):
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        assert ex[0].example_id != ex[1].example_id

    def test_same_match_group_id(self):
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        assert ex[0].match_group_id == ex[1].match_group_id == "test-battle-1"

    def test_lead_indices_correct(self):
        """p1 leads Mon0 (idx 0) + Mon1 (idx 1)."""
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        assert ex[0].label.lead2_idx == (0, 1)

    def test_back_indices_when_bring4_observed(self):
        """p1 reveals Mon0,1,3,4 → leads (0,1), back = (3,4)."""
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        assert ex[0].label_quality.bring4_observed is True
        assert ex[0].label.back2_idx == (3, 4)

    def test_bring4_false_for_partial(self):
        parsed = parse_battle(_make_partial_log())
        ex = produce_directed_examples("partial-1", parsed, "gen9vgc2024regg")
        assert ex[0].label_quality.bring4_observed is False
        assert ex[0].label_quality.notes is not None
        assert "inferred" in ex[0].label_quality.notes

    def test_action90_id_valid(self):
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        for e in ex:
            assert 0 <= e.label.action90_id < 90

    def test_teams_have_six_mons(self):
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        for e in ex:
            assert len(e.team_a.pokemon) == 6
            assert len(e.team_b.pokemon) == 6

    def test_reconstruction_quality(self):
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        rq = ex[0].team_a.reconstruction_quality
        assert rq is not None
        assert rq.source_method == "showteam_direct"
        assert rq.fields_known == 42  # 6 mons x 7 non-species fields

    def test_json_round_trip(self):
        parsed = parse_battle(_make_battle_log())
        ex = produce_directed_examples("test-battle-1", parsed, "gen9vgc2024regg")
        for e in ex:
            d = e.to_dict()
            s = json.dumps(d)
            assert json.loads(s)  # valid JSON


# ---------------------------------------------------------------------------
# Hash stability
# ---------------------------------------------------------------------------

class TestHashing:
    def test_team_id_order_invariant(self):
        mons = [Pokemon(species=f"Mon{i}", item=f"I{i}", ability=f"A{i}",
                        tera_type="Fire", moves=[f"M{i}a", f"M{i}b", f"M{i}c", f"M{i}d"])
                for i in range(6)]
        h1 = _team_id(mons)
        h2 = _team_id(list(reversed(mons)))
        assert h1 == h2

    def test_species_key_order_invariant(self):
        mons = [Pokemon(species=f"Mon{i}") for i in range(6)]
        k1 = _species_key(mons)
        k2 = _species_key(list(reversed(mons)))
        assert k1 == k2

    def test_different_teams_different_ids(self):
        mons_a = [Pokemon(species=f"Mon{i}") for i in range(6)]
        mons_b = [Pokemon(species=f"Other{i}") for i in range(6)]
        assert _team_id(mons_a) != _team_id(mons_b)


# ---------------------------------------------------------------------------
# run_parse (integration with temp file)
# ---------------------------------------------------------------------------

class TestRunParse:
    def test_small_synthetic(self, tmp_path):
        """Write a synthetic raw file, run the parser, check output."""
        raw = {
            "battle-001": [1700000000, _make_battle_log()],
            "battle-002": [1700001000, _make_partial_log()],
        }
        raw_path = tmp_path / "raw.json"
        raw_path.write_text(json.dumps(raw))

        out_dir = tmp_path / "parsed"
        manifest = run_parse(str(raw_path), str(out_dir))

        # Manifest checks
        assert manifest["battles_parsed"] == 2
        assert manifest["parse_errors"] == 0
        assert manifest["examples_written"] == 4  # 2 per battle

        # JSONL checks
        jsonl = out_dir / "match_examples.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text().strip().split("\n")
        assert len(lines) == 4

        # Check IDs are unique
        ids = [json.loads(l)["example_id"] for l in lines]
        assert len(set(ids)) == 4

    def test_limit_param(self, tmp_path):
        raw = {f"b-{i}": [1700000000, _make_battle_log()] for i in range(10)}
        raw_path = tmp_path / "raw.json"
        raw_path.write_text(json.dumps(raw))

        out_dir = tmp_path / "parsed"
        manifest = run_parse(str(raw_path), str(out_dir), limit=3)
        assert manifest["battles_attempted"] == 3
        assert manifest["examples_written"] == 6
