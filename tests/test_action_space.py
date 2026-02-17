"""Tests for the 90-way action space bijection."""

from __future__ import annotations

import pytest
from turnzero.action_space import (
    ACTION_TABLE,
    LEAD_BACK_TO_ACTION,
    action90_to_lead_back,
    lead_back_to_action90,
)


def test_table_has_90_entries():
    assert len(ACTION_TABLE) == 90
    assert len(LEAD_BACK_TO_ACTION) == 90


def test_round_trip_all():
    """Every action id round-trips through both directions."""
    for action_id in range(90):
        lead, back = action90_to_lead_back(action_id)
        assert lead_back_to_action90(lead, back) == action_id


def test_lead_pairs_sorted():
    for lead, _ in ACTION_TABLE:
        assert lead[0] < lead[1], f"lead pair not sorted: {lead}"


def test_back_pairs_sorted():
    for _, back in ACTION_TABLE:
        assert back[0] < back[1], f"back pair not sorted: {back}"


def test_no_overlap_between_lead_and_back():
    for lead, back in ACTION_TABLE:
        assert not set(lead) & set(back), f"overlap: lead={lead}, back={back}"


def test_all_indices_in_range():
    for lead, back in ACTION_TABLE:
        for idx in lead + back:
            assert 0 <= idx <= 5


def test_each_action_uses_4_unique_indices():
    for lead, back in ACTION_TABLE:
        all_idx = set(lead) | set(back)
        assert len(all_idx) == 4


def test_lexicographic_order():
    """Actions are enumerated in lexicographic order of (lead, back)."""
    for i in range(len(ACTION_TABLE) - 1):
        assert ACTION_TABLE[i] < ACTION_TABLE[i + 1]


def test_invalid_action_id_raises():
    with pytest.raises(IndexError):
        action90_to_lead_back(-1)
    with pytest.raises(IndexError):
        action90_to_lead_back(90)


def test_invalid_lead_back_raises():
    with pytest.raises(KeyError):
        lead_back_to_action90((0, 1), (0, 2))  # overlap
    with pytest.raises(KeyError):
        lead_back_to_action90((1, 0), (2, 3))  # unsorted


def test_specific_first_and_last():
    """Spot-check first and last action."""
    lead0, back0 = action90_to_lead_back(0)
    assert lead0 == (0, 1)
    assert back0 == (2, 3)

    lead89, back89 = action90_to_lead_back(89)
    assert lead89 == (4, 5)
    assert back89 == (2, 3)
