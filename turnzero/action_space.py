"""90-way action space: (lead-2, back-2) bijection.

The action space is the Cartesian product of:
  - C(6,2) = 15 lead pairs (sorted)
  - C(4,2) = 6  back pairs from the remaining 4 (sorted)

Total: 15 x 6 = 90 actions.

Enumeration is lexicographic: lead pairs in ascending order, then for each
lead pair the back pairs (from remaining indices) in ascending order.

Reference: docs/PROJECT_BIBLE.md Section 2.1
"""

from __future__ import annotations

from itertools import combinations

# Pre-compute the full table at import time.
# Each entry: (lead2_idx, back2_idx) where both are sorted tuples.
ACTION_TABLE: list[tuple[tuple[int, int], tuple[int, int]]] = []
LEAD_BACK_TO_ACTION: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

_all_indices = range(6)
for _lead in combinations(_all_indices, 2):
    _remaining = sorted(set(_all_indices) - set(_lead))
    for _back in combinations(_remaining, 2):
        _action_id = len(ACTION_TABLE)
        ACTION_TABLE.append((_lead, _back))
        LEAD_BACK_TO_ACTION[(_lead, _back)] = _action_id

assert len(ACTION_TABLE) == 90


def lead_back_to_action90(lead2: tuple[int, int], back2: tuple[int, int]) -> int:
    """Map a (lead-2, back-2) pair to its action90 id.

    Both tuples must be sorted ascending. Raises KeyError on invalid input.
    """
    return LEAD_BACK_TO_ACTION[(lead2, back2)]


def action90_to_lead_back(action_id: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Map an action90 id back to (lead-2, back-2)."""
    if not 0 <= action_id < 90:
        raise IndexError(f"action_id must be in [0, 90), got {action_id}")
    return ACTION_TABLE[action_id]
