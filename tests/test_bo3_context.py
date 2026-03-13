"""Tests for turnzero.data.bo3_context."""

import pytest

from turnzero.data.bo3_context import (
    SENTINEL_LEAD,
    _resolve_lead_pair_idx,
)


class TestResolveLeadPairIdx:
    """Test _resolve_lead_pair_idx with various match scenarios."""

    def test_exact_match(self):
        team = ["Rillaboom", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        leads = frozenset({"Rillaboom", "Incineroar"})
        # Slots 0 and 1 → pair (0,1) → idx 0
        assert _resolve_lead_pair_idx(leads, team) == 0

    def test_exact_match_non_adjacent(self):
        team = ["Rillaboom", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        leads = frozenset({"Flutter Mane", "Tornadus"})
        # Slots 2 and 4 → pair (2,4) → idx
        from turnzero.data.bo3_context import LEAD_PAIR_TO_IDX
        expected = LEAD_PAIR_TO_IDX[(2, 4)]
        assert _resolve_lead_pair_idx(leads, team) == expected

    def test_no_match_returns_sentinel(self):
        team = ["Rillaboom", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        leads = frozenset({"Pikachu", "Charizard"})
        assert _resolve_lead_pair_idx(leads, team) == SENTINEL_LEAD

    def test_partial_match_returns_sentinel(self):
        """One lead matches, other doesn't → sentinel."""
        team = ["Rillaboom", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        leads = frozenset({"Rillaboom", "Pikachu"})
        assert _resolve_lead_pair_idx(leads, team) == SENTINEL_LEAD

    def test_form_variant_fallback(self):
        """Form variants match via base form in both directions."""
        team = ["Ursaluna-Bloodmoon", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        # Lead has base form, team has variant
        assert _resolve_lead_pair_idx(frozenset({"Ursaluna", "Incineroar"}), team) == 0
        # Lead has variant, team has base form
        team2 = ["Ursaluna", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        assert _resolve_lead_pair_idx(frozenset({"Ursaluna-Bloodmoon", "Incineroar"}), team2) == 0
        # Both leads need fallback
        team3 = ["Ogerpon-Hearthflame", "Urshifu-Rapid-Strike", "Flutter Mane", "Tornadus", "Farigiraf", "Incineroar"]
        from turnzero.data.bo3_context import LEAD_PAIR_TO_IDX
        assert _resolve_lead_pair_idx(frozenset({"Ogerpon", "Urshifu"}), team3) == LEAD_PAIR_TO_IDX[(0, 1)]

    def test_duplicate_base_form(self):
        """Two mons share the same base form — should match first available."""
        team = ["Urshifu", "Urshifu-Rapid-Strike", "Flutter Mane", "Tornadus", "Farigiraf", "Incineroar"]
        leads = frozenset({"Urshifu", "Incineroar"})
        # Urshifu matches slot 0 exactly, Incineroar matches slot 5
        from turnzero.data.bo3_context import LEAD_PAIR_TO_IDX
        expected = LEAD_PAIR_TO_IDX[(0, 5)]
        assert _resolve_lead_pair_idx(leads, team) == expected

    def test_insufficient_leads_returns_sentinel(self):
        """Empty or single-element lead sets return sentinel."""
        team = ["Rillaboom", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        assert _resolve_lead_pair_idx(frozenset(), team) == SENTINEL_LEAD
        assert _resolve_lead_pair_idx(frozenset({"Rillaboom"}), team) == SENTINEL_LEAD
