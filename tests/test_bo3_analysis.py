"""Smoke tests for turnzero.analysis.bo3 module."""

from __future__ import annotations

from turnzero.analysis.bo3 import (
    _wilson_ci,
    extract_bo3_linkage,
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
    def test_extracts_side_and_nickname(self):
        assert _parse_ident("p1a: Pikachu") == ("p1", "Pikachu")
        assert _parse_ident("p2b: Charizard") == ("p2", "Charizard")


class TestSpeciesFromDetails:
    def test_extracts_species(self):
        assert _species_from_details("Pikachu, L50, M") == "Pikachu"
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
