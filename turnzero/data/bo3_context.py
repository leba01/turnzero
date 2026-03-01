"""BO3 sequential context builder.

Builds a lookup table mapping (battle_id, perspective_side) to context
features for the SequentialOTSTransformer: prior_lead2_idx, prior_result,
game_num.

Reuses extract_bo3_linkage() and parse_game_info() from the BO3 adaptation
analysis script.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

# Lead-pair index lookup — same ordering as dataset.py / action_space.py
LEAD_PAIRS: list[tuple[int, int]] = list(combinations(range(6), 2))
LEAD_PAIR_TO_IDX: dict[tuple[int, int], int] = {
    pair: idx for idx, pair in enumerate(LEAD_PAIRS)
}
SENTINEL_LEAD = 15  # no prior info


def _base_form(species: str) -> str:
    """Strip form suffix to base species name."""
    return species.split("-")[0]


def _resolve_lead_pair_idx(
    prior_leads: frozenset[str], team_a_species: list[str],
) -> int:
    """Map prior game's lead species to a lead2_idx relative to team_a ordering.

    Tries exact match first, then base-form fallback. Returns SENTINEL_LEAD
    (15) if the two leads can't be matched to two distinct team_a slots.
    """
    matched_slots: list[int] = []
    used: set[int] = set()

    # Pass 1: exact match
    for lead_sp in sorted(prior_leads):  # sort for determinism
        for i, team_sp in enumerate(team_a_species):
            if i not in used and lead_sp == team_sp:
                matched_slots.append(i)
                used.add(i)
                break

    # Pass 2: base-form fallback for unmatched leads
    if len(matched_slots) < 2:
        remaining_leads = sorted(prior_leads - {team_a_species[i] for i in matched_slots})
        for lead_sp in remaining_leads:
            lead_base = _base_form(lead_sp)
            for i, team_sp in enumerate(team_a_species):
                if i not in used and _base_form(team_sp) == lead_base:
                    matched_slots.append(i)
                    used.add(i)
                    break

    if len(matched_slots) != 2:
        return SENTINEL_LEAD

    pair = tuple(sorted(matched_slots))
    return LEAD_PAIR_TO_IDX.get(pair, SENTINEL_LEAD)


def build_context_lookup(
    raw_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build BO3 context lookup from raw battle logs.

    Returns dict keyed by ``battle_id`` with per-side context::

        {
            battle_id: {
                "game_num": int,              # 1, 2, or 3
                "p1": {"prior_leads": [...], "prior_result": int},
                "p2": {"prior_leads": [...], "prior_result": int},
            },
            ...
        }

    Downstream, the dataset resolves which side is ``team_a`` by matching
    species.
    """
    from scripts.run_bo3_adaptation import extract_bo3_linkage, parse_game_info

    set_to_games = extract_bo3_linkage(raw_data)

    # Parse all needed games
    game_cache: dict[str, Any] = {}
    for sid, games in set_to_games.items():
        if len(games) < 2:
            continue
        for gn in sorted(games):
            bid = games[gn]
            if bid not in game_cache:
                _, log_text = raw_data[bid]
                gi = parse_game_info(bid, sid, gn, log_text)
                if gi is not None:
                    game_cache[bid] = gi

    # Build context for games 2 and 3
    context: dict[str, dict[str, Any]] = {}

    for sid, games in set_to_games.items():
        gnums = sorted(games.keys())
        if len(gnums) < 2:
            continue
        for i in range(1, len(gnums)):
            prior_bid = games[gnums[i - 1]]
            curr_bid = games[gnums[i]]
            if prior_bid not in game_cache or curr_bid not in game_cache:
                continue

            prior_gi = game_cache[prior_bid]
            curr_gi = game_cache[curr_bid]

            # Verify same players
            if prior_gi.player_names != curr_gi.player_names:
                continue

            sides_ctx: dict[str, dict[str, Any]] = {}
            for side in ("p1", "p2"):
                # Prior result: 1=won, 2=lost, 0=sentinel (no winner)
                if prior_gi.winner_side is None:
                    pr = 0
                elif prior_gi.winner_side == side:
                    pr = 1
                else:
                    pr = 2

                sides_ctx[side] = {
                    "prior_leads": sorted(prior_gi.leads[side]),
                    "prior_result": pr,
                }

            context[curr_bid] = {
                "game_num": gnums[i],
                "sides": sides_ctx,
                # Store species for perspective resolution
                "species6": {
                    s: sorted(prior_gi.species6[s]) for s in ("p1", "p2")
                },
                # Store current game's species for matching
                "curr_species6": {
                    s: sorted(curr_gi.species6[s]) for s in ("p1", "p2")
                },
            }

    return context


def save_context_lookup(
    context: dict[str, dict[str, Any]], path: str | Path,
) -> None:
    """Persist context lookup as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(context, f)
    print(f"Saved BO3 context: {len(context):,} entries → {path}")


def load_context_lookup(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load context lookup from JSON."""
    with open(path) as f:
        return json.load(f)
