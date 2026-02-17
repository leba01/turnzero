"""Tests for the canonicalization stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnzero.action_space import lead_back_to_action90
from turnzero.data.canonicalize import (
    _canonical_key,
    _dedup_key,
    _team_id,
    _species_key,
    camel_to_display,
    canonicalize_example,
    canonicalize_pokemon,
    run_canonicalize,
)
from turnzero.schemas import (
    Label,
    LabelQuality,
    MatchExample,
    Pokemon,
    ReconstructionQuality,
    TeamSheet,
)


# ---------------------------------------------------------------------------
# camel_to_display
# ---------------------------------------------------------------------------

class TestCamelToDisplay:
    def test_basic_camel_case(self):
        assert camel_to_display("FakeOut") == "Fake Out"
        assert camel_to_display("AssaultVest") == "Assault Vest"
        assert camel_to_display("GrassySurge") == "Grassy Surge"
        assert camel_to_display("HighHorsepower") == "High Horsepower"

    def test_single_word_unchanged(self):
        assert camel_to_display("Protect") == "Protect"
        assert camel_to_display("Tailwind") == "Tailwind"
        assert camel_to_display("Rillaboom") == "Rillaboom"
        assert camel_to_display("Thunderbolt") == "Thunderbolt"

    def test_unk_passthrough(self):
        assert camel_to_display("UNK") == "UNK"

    def test_hyphenated_passthrough(self):
        assert camel_to_display("Urshifu-Rapid-Strike") == "Urshifu-Rapid-Strike"
        assert camel_to_display("Calyrex-Ice") == "Calyrex-Ice"
        assert camel_to_display("Chien-Pao") == "Chien-Pao"
        assert camel_to_display("Indeedee-F") == "Indeedee-F"

    def test_exception_prepositions(self):
        assert camel_to_display("SwordofRuin") == "Sword of Ruin"
        assert camel_to_display("TabletsofRuin") == "Tablets of Ruin"
        assert camel_to_display("VesselofRuin") == "Vessel of Ruin"
        assert camel_to_display("BeadsofRuin") == "Beads of Ruin"

    def test_exception_hyphens(self):
        assert camel_to_display("Uturn") == "U-turn"
        assert camel_to_display("WillOWisp") == "Will-O-Wisp"
        assert camel_to_display("FreezeDry") == "Freeze-Dry"
        assert camel_to_display("XScissor") == "X-Scissor"

    def test_exception_all_lowercase(self):
        assert camel_to_display("behemothblade") == "Behemoth Blade"
        assert camel_to_display("behemothbash") == "Behemoth Bash"

    def test_idempotent(self):
        """Already-canonical names should pass through unchanged."""
        assert camel_to_display("Fake Out") == "Fake Out"
        assert camel_to_display("Assault Vest") == "Assault Vest"
        assert camel_to_display("Iron Valiant") == "Iron Valiant"

    def test_species_with_space(self):
        """Species like 'Iron Valiant' already have spaces."""
        assert camel_to_display("Iron Valiant") == "Iron Valiant"
        assert camel_to_display("Raging Bolt") == "Raging Bolt"


# ---------------------------------------------------------------------------
# canonicalize_pokemon
# ---------------------------------------------------------------------------

class TestCanonicalizePokemon:
    def test_normalizes_all_fields(self):
        p = Pokemon(
            species="Rillaboom",
            item="AssaultVest",
            ability="GrassySurge",
            tera_type="Ground",
            moves=["FakeOut", "WoodHammer", "GrassyGlide", "HighHorsepower"],
        )
        cp = canonicalize_pokemon(p)
        assert cp.species == "Rillaboom"
        assert cp.item == "Assault Vest"
        assert cp.ability == "Grassy Surge"
        assert cp.tera_type == "Ground"

    def test_sorts_moves_alphabetically(self):
        p = Pokemon(
            species="Mon",
            item="Item",
            ability="Ability",
            tera_type="Fire",
            moves=["ZMove", "AMove", "MMove", "BMove"],
        )
        cp = canonicalize_pokemon(p)
        # camel_to_display on single-words doesn't change them, but
        # "ZMove" → "ZMove" (single word). Let me use multi-word moves:
        assert cp.moves == sorted(cp.moves)

    def test_unk_sorted_last(self):
        p = Pokemon(
            species="Ditto",
            item="ChoiceScarf",
            ability="Imposter",
            tera_type="Normal",
            moves=["Transform", "UNK", "UNK", "UNK"],
        )
        cp = canonicalize_pokemon(p)
        assert cp.moves == ["Transform", "UNK", "UNK", "UNK"]

    def test_moves_normalized_then_sorted(self):
        p = Pokemon(
            species="Mon",
            item="Item",
            ability="Ability",
            tera_type="Fire",
            moves=["WoodHammer", "FakeOut", "HighHorsepower", "GrassyGlide"],
        )
        cp = canonicalize_pokemon(p)
        assert cp.moves == ["Fake Out", "Grassy Glide", "High Horsepower", "Wood Hammer"]


# ---------------------------------------------------------------------------
# canonicalize_example — label remapping
# ---------------------------------------------------------------------------

def _make_team(species_prefix: str, format_id: str = "gen9test") -> TeamSheet:
    """Build a 6-mon team with predictable canonical key ordering.

    Mon keys are constructed so that sorting by canonical key gives a
    known permutation from the original order.
    """
    # Create mons in a specific order where canonical sort reorders them.
    # Species starting with Z sorts after A, etc.
    mons = []
    # Order: C, A, F, B, E, D → canonical sort → A, B, C, D, E, F
    for letter in ["C", "A", "F", "B", "E", "D"]:
        mons.append(Pokemon(
            species=f"{species_prefix}{letter}",
            item="Item",
            ability="Ability",
            tera_type="Fire",
            moves=["MoveA", "MoveB", "MoveC", "MoveD"],
        ))
    return TeamSheet(
        team_id="placeholder",
        species_key="placeholder",
        format_id=format_id,
        pokemon=mons,
        reconstruction_quality=ReconstructionQuality(fields_known=42),
    )


class TestCanonicalizeExample:
    def test_pokemon_sorted_by_canonical_key(self):
        """After canonicalization, pokemon should be sorted by canonical key."""
        team_a = _make_team("A")
        team_b = _make_team("B")

        ex = MatchExample(
            example_id="test1",
            match_group_id="mg1",
            battle_id="b1",
            team_a=team_a,
            team_b=team_b,
            label=Label(lead2_idx=(0, 1), back2_idx=(2, 3),
                        action90_id=lead_back_to_action90((0, 1), (2, 3))),
            label_quality=LabelQuality(bring4_observed=True),
            format_id="gen9test",
        )
        canon = canonicalize_example(ex)

        # Check team_a pokemon are sorted by species
        species_a = [p.species for p in canon.team_a.pokemon]
        assert species_a == sorted(species_a)

        # Check team_b pokemon are sorted by species
        species_b = [p.species for p in canon.team_b.pokemon]
        assert species_b == sorted(species_b)

    def test_label_indices_remapped(self):
        """Label indices should track pokemon through reordering.

        Original team_a order: [C, A, F, B, E, D] (indices 0-5)
        Canonical order:       [A, B, C, D, E, F]

        So original index mapping is:
          old 0 (C) → new 2
          old 1 (A) → new 0
          old 2 (F) → new 5
          old 3 (B) → new 1
          old 4 (E) → new 4
          old 5 (D) → new 3

        If original lead2_idx = (0, 1) [C and A]:
          new lead2_idx = sorted(2, 0) = (0, 2)

        If original back2_idx = (3, 4) [B and E]:
          new back2_idx = sorted(1, 4) = (1, 4)
        """
        team_a = _make_team("A")
        team_b = _make_team("B")

        ex = MatchExample(
            example_id="test1",
            match_group_id="mg1",
            battle_id="b1",
            team_a=team_a,
            team_b=team_b,
            label=Label(lead2_idx=(0, 1), back2_idx=(3, 4),
                        action90_id=lead_back_to_action90((0, 1), (3, 4))),
            label_quality=LabelQuality(bring4_observed=True),
            format_id="gen9test",
        )
        canon = canonicalize_example(ex)

        # C (old 0) → new 2, A (old 1) → new 0
        assert canon.label.lead2_idx == (0, 2)
        # B (old 3) → new 1, E (old 4) → new 4
        assert canon.label.back2_idx == (1, 4)

    def test_action90_recomputed(self):
        """action90_id should match the remapped lead/back indices."""
        team_a = _make_team("A")
        team_b = _make_team("B")

        ex = MatchExample(
            example_id="test1",
            match_group_id="mg1",
            battle_id="b1",
            team_a=team_a,
            team_b=team_b,
            label=Label(lead2_idx=(0, 1), back2_idx=(3, 4),
                        action90_id=lead_back_to_action90((0, 1), (3, 4))),
            label_quality=LabelQuality(bring4_observed=True),
            format_id="gen9test",
        )
        canon = canonicalize_example(ex)

        expected = lead_back_to_action90(canon.label.lead2_idx, canon.label.back2_idx)
        assert canon.label.action90_id == expected

    def test_lead_back_no_overlap(self):
        """Remapped lead and back indices must not overlap."""
        team_a = _make_team("A")
        team_b = _make_team("B")

        ex = MatchExample(
            example_id="test1",
            match_group_id="mg1",
            battle_id="b1",
            team_a=team_a,
            team_b=team_b,
            label=Label(lead2_idx=(2, 5), back2_idx=(0, 4),
                        action90_id=lead_back_to_action90((2, 5), (0, 4))),
            label_quality=LabelQuality(bring4_observed=True),
            format_id="gen9test",
        )
        canon = canonicalize_example(ex)

        lead_set = set(canon.label.lead2_idx)
        back_set = set(canon.label.back2_idx)
        assert not lead_set & back_set

    def test_team_id_recomputed(self):
        """team_id should be different from placeholder after canonicalization."""
        team_a = _make_team("A")
        team_b = _make_team("B")

        ex = MatchExample(
            example_id="test1",
            match_group_id="mg1",
            battle_id="b1",
            team_a=team_a,
            team_b=team_b,
            label=Label(lead2_idx=(0, 1), back2_idx=(2, 3),
                        action90_id=lead_back_to_action90((0, 1), (2, 3))),
            label_quality=LabelQuality(bring4_observed=True),
            format_id="gen9test",
        )
        canon = canonicalize_example(ex)

        assert canon.team_a.team_id != "placeholder"
        assert canon.team_b.team_id != "placeholder"
        assert len(canon.team_a.team_id) == 16
        assert len(canon.team_b.team_id) == 16

    def test_example_id_preserved(self):
        """example_id identifies the game/perspective, not content — keep it."""
        team_a = _make_team("A")
        team_b = _make_team("B")

        ex = MatchExample(
            example_id="test1",
            match_group_id="mg1",
            battle_id="b1",
            team_a=team_a,
            team_b=team_b,
            label=Label(lead2_idx=(0, 1), back2_idx=(2, 3),
                        action90_id=lead_back_to_action90((0, 1), (2, 3))),
            label_quality=LabelQuality(bring4_observed=True),
            format_id="gen9test",
        )
        canon = canonicalize_example(ex)
        assert canon.example_id == "test1"
        assert canon.match_group_id == "mg1"

    def test_json_round_trip(self):
        """Canonicalized example should survive JSON serialization."""
        team_a = _make_team("A")
        team_b = _make_team("B")

        ex = MatchExample(
            example_id="test1",
            match_group_id="mg1",
            battle_id="b1",
            team_a=team_a,
            team_b=team_b,
            label=Label(lead2_idx=(0, 1), back2_idx=(2, 3),
                        action90_id=lead_back_to_action90((0, 1), (2, 3))),
            label_quality=LabelQuality(bring4_observed=True),
            format_id="gen9test",
        )
        canon = canonicalize_example(ex)
        d = canon.to_dict()
        s = json.dumps(d)
        restored = MatchExample.from_dict(json.loads(s))
        assert restored.team_a.team_id == canon.team_a.team_id
        assert restored.label.action90_id == canon.label.action90_id


# ---------------------------------------------------------------------------
# Hash stability
# ---------------------------------------------------------------------------

class TestHashStability:
    def test_move_order_invariance(self):
        """Same pokemon with different move orders → same team_id."""
        mons_a = [Pokemon(
            species=f"Mon{i}", item="Item", ability="Ability",
            tera_type="Fire", moves=["A", "B", "C", "D"],
        ) for i in range(6)]
        mons_b = [Pokemon(
            species=f"Mon{i}", item="Item", ability="Ability",
            tera_type="Fire", moves=["D", "C", "B", "A"],
        ) for i in range(6)]
        assert _team_id(mons_a) == _team_id(mons_b)

    def test_team_order_invariance(self):
        """Same team in different pokemon order → same team_id."""
        mons = [Pokemon(
            species=f"Mon{i}", item=f"I{i}", ability=f"A{i}",
            tera_type="Fire", moves=[f"M{i}a", f"M{i}b", f"M{i}c", f"M{i}d"],
        ) for i in range(6)]
        assert _team_id(mons) == _team_id(list(reversed(mons)))

    def test_different_teams_different_ids(self):
        mons_a = [Pokemon(species=f"Mon{i}") for i in range(6)]
        mons_b = [Pokemon(species=f"Other{i}") for i in range(6)]
        assert _team_id(mons_a) != _team_id(mons_b)

    def test_canonical_names_change_hash(self):
        """CamelCase names produce different hash from display names."""
        mons_camel = [Pokemon(
            species="Mon", item="AssaultVest", ability="GrassySurge",
            tera_type="Fire", moves=["FakeOut", "Protect", "Tailwind", "Detect"],
        )] + [Pokemon(species=f"Filler{i}") for i in range(5)]

        mons_display = [Pokemon(
            species="Mon", item="Assault Vest", ability="Grassy Surge",
            tera_type="Fire", moves=["Fake Out", "Protect", "Tailwind", "Detect"],
        )] + [Pokemon(species=f"Filler{i}") for i in range(5)]

        assert _team_id(mons_camel) != _team_id(mons_display)

    def test_species_key_order_invariant(self):
        mons = [Pokemon(species=f"Mon{i}") for i in range(6)]
        assert _species_key(mons) == _species_key(list(reversed(mons)))


# ---------------------------------------------------------------------------
# Deduplication logic
# ---------------------------------------------------------------------------

class TestDedup:
    def _make_example(self, team_a_id: str = "ta", team_b_id: str = "tb",
                      action90: int = 0, format_id: str = "gen9test",
                      example_id: str = "ex1") -> MatchExample:
        """Build a minimal MatchExample with specific dedup-relevant fields."""
        from turnzero.action_space import action90_to_lead_back
        lead2, back2 = action90_to_lead_back(action90)

        def _team(tid: str) -> TeamSheet:
            return TeamSheet(
                team_id=tid,
                species_key="sk",
                format_id=format_id,
                pokemon=[Pokemon(species=f"Mon{i}") for i in range(6)],
            )

        return MatchExample(
            example_id=example_id,
            match_group_id="mg",
            battle_id="b",
            team_a=_team(team_a_id),
            team_b=_team(team_b_id),
            label=Label(lead2_idx=lead2, back2_idx=back2, action90_id=action90),
            label_quality=LabelQuality(bring4_observed=True),
            format_id=format_id,
        )

    def test_exact_duplicate_detected(self):
        ex1 = self._make_example()
        ex2 = self._make_example(example_id="ex2")
        assert _dedup_key(ex1) == _dedup_key(ex2)

    def test_different_action_not_duplicate(self):
        """Same matchup with different actions is NOT a duplicate."""
        ex1 = self._make_example(action90=0)
        ex2 = self._make_example(action90=1)
        assert _dedup_key(ex1) != _dedup_key(ex2)

    def test_different_format_not_duplicate(self):
        ex1 = self._make_example(format_id="regg")
        ex2 = self._make_example(format_id="reggbo3")
        assert _dedup_key(ex1) != _dedup_key(ex2)

    def test_swapped_teams_not_duplicate(self):
        """(A vs B) is not the same as (B vs A)."""
        ex1 = self._make_example(team_a_id="ta", team_b_id="tb")
        ex2 = self._make_example(team_a_id="tb", team_b_id="ta")
        assert _dedup_key(ex1) != _dedup_key(ex2)


# ---------------------------------------------------------------------------
# run_canonicalize (integration)
# ---------------------------------------------------------------------------

def _make_synthetic_jsonl(path: Path, examples: list[MatchExample]) -> None:
    """Write MatchExamples as JSONL for integration tests."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), separators=(",", ":")) + "\n")


def _make_test_example(species_prefix: str = "Mon",
                       opp_prefix: str = "Opp",
                       lead2: tuple[int, int] = (0, 1),
                       back2: tuple[int, int] = (2, 3),
                       battle_id: str = "b1",
                       perspective: str = "p1") -> MatchExample:
    """Build a test MatchExample with CamelCase names."""
    import hashlib

    def _hash(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    mons_a = [Pokemon(
        species=f"{species_prefix}{i}",
        item="AssaultVest" if i == 0 else f"Item{i}",
        ability="GrassySurge" if i == 0 else f"Ability{i}",
        tera_type="Fire",
        moves=[f"Move{i}A", f"Move{i}B", f"Move{i}C", f"Move{i}D"],
    ) for i in range(6)]

    mons_b = [Pokemon(
        species=f"{opp_prefix}{i}",
        item=f"OItem{i}",
        ability=f"OAbility{i}",
        tera_type="Water",
        moves=[f"OMove{i}A", f"OMove{i}B", f"OMove{i}C", f"OMove{i}D"],
    ) for i in range(6)]

    action90 = lead_back_to_action90(lead2, back2)

    return MatchExample(
        example_id=_hash(f"{battle_id}|{perspective}"),
        match_group_id=battle_id,
        battle_id=battle_id,
        team_a=TeamSheet(
            team_id=_hash("team_a"),
            species_key=_hash("sk_a"),
            format_id="gen9test",
            pokemon=mons_a,
            reconstruction_quality=ReconstructionQuality(fields_known=42),
        ),
        team_b=TeamSheet(
            team_id=_hash("team_b"),
            species_key=_hash("sk_b"),
            format_id="gen9test",
            pokemon=mons_b,
            reconstruction_quality=ReconstructionQuality(fields_known=42),
        ),
        label=Label(lead2_idx=lead2, back2_idx=back2, action90_id=action90),
        label_quality=LabelQuality(bring4_observed=True),
        format_id="gen9test",
    )


class TestRunCanonicalize:
    def test_basic_integration(self, tmp_path):
        """Canonicalize a small synthetic file."""
        examples = [
            _make_test_example(battle_id="b1", perspective="p1"),
            _make_test_example(battle_id="b1", perspective="p2",
                               species_prefix="Opp", opp_prefix="Mon"),
        ]
        in_path = tmp_path / "parsed" / "match_examples.jsonl"
        in_path.parent.mkdir(parents=True)
        _make_synthetic_jsonl(in_path, examples)

        out_dir = tmp_path / "canonical"
        manifest = run_canonicalize(str(in_path), str(out_dir))

        assert manifest["examples_read"] == 2
        assert manifest["examples_written"] == 2
        assert manifest["duplicates_removed"] == 0
        assert manifest["errors"] == 0

        # Check output file exists and has correct line count
        out_jsonl = out_dir / "match_examples.jsonl"
        assert out_jsonl.exists()
        lines = out_jsonl.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_dedup_removes_exact_duplicates(self, tmp_path):
        """Identical examples (same quadruple) should be deduped."""
        ex = _make_test_example()
        # Write the same example twice (different example_id but same content)
        examples = [ex, ex]
        in_path = tmp_path / "input.jsonl"
        _make_synthetic_jsonl(in_path, examples)

        out_dir = tmp_path / "out"
        manifest = run_canonicalize(str(in_path), str(out_dir))

        assert manifest["examples_read"] == 2
        assert manifest["examples_written"] == 1
        assert manifest["duplicates_removed"] == 1

    def test_different_actions_kept(self, tmp_path):
        """Same matchup with different actions should both be kept."""
        ex1 = _make_test_example(lead2=(0, 1), back2=(2, 3))
        ex2 = _make_test_example(lead2=(0, 2), back2=(1, 3))
        in_path = tmp_path / "input.jsonl"
        _make_synthetic_jsonl(in_path, [ex1, ex2])

        out_dir = tmp_path / "out"
        manifest = run_canonicalize(str(in_path), str(out_dir))

        assert manifest["examples_written"] == 2
        assert manifest["duplicates_removed"] == 0

    def test_output_names_canonical(self, tmp_path):
        """CamelCase names should be converted to display format."""
        ex = _make_test_example()
        in_path = tmp_path / "input.jsonl"
        _make_synthetic_jsonl(in_path, [ex])

        out_dir = tmp_path / "out"
        run_canonicalize(str(in_path), str(out_dir))

        out_jsonl = out_dir / "match_examples.jsonl"
        d = json.loads(out_jsonl.read_text().strip())
        first_mon = d["team_a"]["pokemon"][0]
        # "AssaultVest" → "Assault Vest"
        assert first_mon["item"] == "Assault Vest"
        # "GrassySurge" → "Grassy Surge"
        assert first_mon["ability"] == "Grassy Surge"

    def test_output_pokemon_sorted(self, tmp_path):
        """Pokemon should be sorted by canonical key in output."""
        ex = _make_test_example()
        in_path = tmp_path / "input.jsonl"
        _make_synthetic_jsonl(in_path, [ex])

        out_dir = tmp_path / "out"
        run_canonicalize(str(in_path), str(out_dir))

        out_jsonl = out_dir / "match_examples.jsonl"
        d = json.loads(out_jsonl.read_text().strip())
        species_a = [p["species"] for p in d["team_a"]["pokemon"]]
        assert species_a == sorted(species_a)

    def test_manifest_written(self, tmp_path):
        """Manifest JSON should be written alongside output."""
        ex = _make_test_example()
        in_path = tmp_path / "input.jsonl"
        _make_synthetic_jsonl(in_path, [ex])

        out_dir = tmp_path / "out"
        run_canonicalize(str(in_path), str(out_dir))

        manifest_path = out_dir / "canonicalize_manifest.json"
        assert manifest_path.exists()
        m = json.loads(manifest_path.read_text())
        assert m["examples_read"] == 1
        assert m["examples_written"] == 1
