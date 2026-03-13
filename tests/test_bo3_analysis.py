"""Smoke tests for turnzero.analysis.bo3 module."""

from __future__ import annotations

from turnzero.analysis.bo3 import (
    GameInfo,
    Transition,
    _wilson_ci,
    extract_bo3_linkage,
    parse_game_info,
    _parse_ident,
    _species_from_details,
    _match_to_showteam,
)


class TestWilsonCI:
    def test_zero_n(self):
        assert _wilson_ci(0.5, 0) == (0.0, 0.0)

    def test_returns_interval(self):
        lo, hi = _wilson_ci(0.5, 100)
        assert 0 < lo < 0.5 < hi < 1.0

    def test_perfect_rate(self):
        lo, hi = _wilson_ci(1.0, 50)
        assert hi == 1.0
        assert lo > 0.9


class TestParseIdent:
    def test_basic(self):
        side, nick = _parse_ident("p1a: Pikachu")
        assert side == "p1"
        assert nick == "Pikachu"

    def test_p2(self):
        side, nick = _parse_ident("p2b: Charizard")
        assert side == "p2"
        assert nick == "Charizard"


class TestSpeciesFromDetails:
    def test_basic(self):
        assert _species_from_details("Pikachu, L50, M") == "Pikachu"

    def test_no_details(self):
        assert _species_from_details("Pikachu") == "Pikachu"


class TestMatchToShowteam:
    def test_exact_match(self):
        assert _match_to_showteam("Pikachu", ["Charizard", "Pikachu", "Mewtwo"]) == 1

    def test_form_match(self):
        assert _match_to_showteam("Pikachu-Gmax", ["Charizard", "Pikachu", "Mewtwo"]) == 1

    def test_no_match(self):
        assert _match_to_showteam("Mew", ["Charizard", "Pikachu", "Mewtwo"]) is None

    def test_skip_used(self):
        assert _match_to_showteam("Pikachu", ["Pikachu", "Pikachu"], used={0}) == 1


class TestExtractBo3Linkage:
    def test_empty(self):
        assert extract_bo3_linkage({}) == {}


class TestDataclasses:
    def test_game_info_instantiation(self):
        gi = GameInfo(
            battle_id="b1", set_id="s1", game_num=1,
            player_names={"p1": "A", "p2": "B"},
            winner_side="p1",
            leads={"p1": frozenset(["X"]), "p2": frozenset(["Y"])},
            revealed={"p1": frozenset(["X"]), "p2": frozenset(["Y"])},
            species6={"p1": frozenset(["X"]), "p2": frozenset(["Y"])},
            species_key={"p1": "X", "p2": "Y"},
        )
        assert gi.battle_id == "b1"

    def test_transition_instantiation(self):
        t = Transition(
            set_id="s1", side="p1", prior_result="won_prior",
            lead_changed=True, bring4_changed=None,
            is_mirror=False, team_entropy=None,
        )
        assert t.lead_changed is True
