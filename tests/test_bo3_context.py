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
        """Ursaluna vs Ursaluna-Bloodmoon should match via base form."""
        team = ["Ursaluna-Bloodmoon", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        leads = frozenset({"Ursaluna", "Incineroar"})
        # Ursaluna matches Ursaluna-Bloodmoon at slot 0, Incineroar at slot 1
        assert _resolve_lead_pair_idx(leads, team) == 0

    def test_form_variant_reverse(self):
        """Prior leads have form suffix, team doesn't."""
        team = ["Ursaluna", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        leads = frozenset({"Ursaluna-Bloodmoon", "Incineroar"})
        assert _resolve_lead_pair_idx(leads, team) == 0

    def test_both_form_variants(self):
        """Both leads need base-form fallback."""
        team = ["Ogerpon-Hearthflame", "Urshifu-Rapid-Strike", "Flutter Mane", "Tornadus", "Farigiraf", "Incineroar"]
        leads = frozenset({"Ogerpon", "Urshifu"})
        from turnzero.data.bo3_context import LEAD_PAIR_TO_IDX
        expected = LEAD_PAIR_TO_IDX[(0, 1)]
        assert _resolve_lead_pair_idx(leads, team) == expected

    def test_duplicate_base_form(self):
        """Two mons share the same base form — should match first available."""
        team = ["Urshifu", "Urshifu-Rapid-Strike", "Flutter Mane", "Tornadus", "Farigiraf", "Incineroar"]
        leads = frozenset({"Urshifu", "Incineroar"})
        # Urshifu matches slot 0 exactly, Incineroar matches slot 5
        from turnzero.data.bo3_context import LEAD_PAIR_TO_IDX
        expected = LEAD_PAIR_TO_IDX[(0, 5)]
        assert _resolve_lead_pair_idx(leads, team) == expected

    def test_empty_leads_returns_sentinel(self):
        team = ["Rillaboom", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        leads = frozenset()
        assert _resolve_lead_pair_idx(leads, team) == SENTINEL_LEAD

    def test_single_lead_returns_sentinel(self):
        team = ["Rillaboom", "Incineroar", "Flutter Mane", "Urshifu", "Tornadus", "Farigiraf"]
        leads = frozenset({"Rillaboom"})
        assert _resolve_lead_pair_idx(leads, team) == SENTINEL_LEAD
