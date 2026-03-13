"""Shared BO3 analysis utilities.

Provides data structures and functions for parsing and linking games within
best-of-3 sets from Showdown battle logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = ROOT / "data" / "raw" / "logs-gen9vgc2025reggbo3.json"
CLUSTER_JSON = ROOT / "outputs" / "eval" / "cluster_analysis.json"
OUT_EVAL = ROOT / "outputs" / "eval"
OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class GameInfo:
    battle_id: str
    set_id: str
    game_num: int
    player_names: dict[str, str]  # {"p1": "Name", "p2": "Name"}
    winner_side: str | None       # "p1" or "p2" or None (forfeit/tie)
    leads: dict[str, frozenset[str]]    # {"p1": frozenset(species), ...}
    revealed: dict[str, frozenset[str]]
    species6: dict[str, frozenset[str]]  # full 6 species per side
    species_key: dict[str, str]   # "|".join(sorted(species)) per side


@dataclass
class Transition:
    set_id: str
    side: str
    prior_result: str        # "won_prior" | "lost_prior"
    lead_changed: bool
    bring4_changed: bool | None  # None if either game not Tier 1
    is_mirror: bool
    team_entropy: float | None   # from cluster_analysis.json


# ── Extraction ────────────────────────────────────────────────────────


_RE_SET_ID = re.compile(r'/game-(bestof3-[^"]+)')
_RE_GAME_NUM = re.compile(r"Game (\d+)")


def extract_bo3_linkage(
    raw_data: dict[str, Any],
) -> dict[str, dict[int, str]]:
    """Scan logs for |uhtml|bestof| lines and group by set_id.

    Returns {set_id: {game_num: battle_id}}.
    """
    sets: dict[str, dict[int, str]] = {}
    for battle_id, (_, log_text) in raw_data.items():
        for line in log_text.split("\n"):
            if "|uhtml|bestof|" not in line:
                continue
            m_set = _RE_SET_ID.search(line)
            m_game = _RE_GAME_NUM.search(line)
            if m_set and m_game:
                sid = m_set.group(1)
                gnum = int(m_game.group(1))
                sets.setdefault(sid, {})[gnum] = battle_id
            break  # only one bestof line per log
    return sets


def _parse_ident(ident: str) -> tuple[str, str]:
    """Parse 'p1a: Nickname' -> ('p1', 'Nickname')."""
    ident = ident.strip()
    side = ident.split(":")[0][:2]
    nickname = ident.split(": ", 1)[1]
    return side, nickname


def _species_from_details(details: str) -> str:
    return details.split(",")[0].strip()


def _match_to_showteam(
    switch_species: str, showteam_species: list[str], used: set[int] | None = None
) -> int | None:
    skip = used or set()
    for i, st in enumerate(showteam_species):
        if i not in skip and switch_species == st:
            return i
    for i, st in enumerate(showteam_species):
        if i in skip:
            continue
        if switch_species.startswith(st + "-") or st.startswith(switch_species + "-"):
            return i
    sw_base = switch_species.split("-")[0]
    for i, st in enumerate(showteam_species):
        if i in skip:
            continue
        if st.split("-")[0] == sw_base:
            return i
    return None


def parse_game_info(
    battle_id: str, set_id: str, game_num: int, log_text: str
) -> GameInfo | None:
    """Parse a single game log for player names, winner, leads, revealed, species."""
    lines = log_text.split("\n")

    # ── Player names ──
    player_names: dict[str, str] = {}
    for line in lines:
        if "|player|" in line:
            parts = line.split("|")
            # |player|p1|Name|avatar|rating
            if len(parts) >= 4 and parts[2] in ("p1", "p2") and parts[3]:
                player_names[parts[2]] = parts[3]

    if "p1" not in player_names or "p2" not in player_names:
        return None

    # ── Winner ──
    winner_name = None
    for line in lines:
        if "|win|" in line:
            parts = line.split("|")
            if len(parts) >= 3:
                winner_name = parts[2]
            break

    winner_side = None
    if winner_name:
        for side, name in player_names.items():
            if name == winner_name:
                winner_side = side
                break

    # ── Showteam ──
    from turnzero.data.parser import parse_showteam_line

    showteam: dict[str, list] = {}
    for line in lines:
        if "|showteam|" in line:
            try:
                side, pokemon = parse_showteam_line(line)
                showteam[side] = pokemon
            except (ValueError, IndexError):
                pass

    if "p1" not in showteam or "p2" not in showteam:
        return None

    st_species = {s: [p.species for p in showteam[s]] for s in ("p1", "p2")}

    # ── Leads + revealed (same logic as parser.py) ──
    start_idx = None
    for i, line in enumerate(lines):
        if "|start" in line:
            start_idx = i
            break
    if start_idx is None:
        return None

    nick_to_idx: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    leads: dict[str, list[str]] = {"p1": [], "p2": []}
    revealed: dict[str, set[str]] = {"p1": set(), "p2": set()}
    leads_done = False

    for line in lines[start_idx + 1 :]:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1].strip()
        if tag == "turn":
            leads_done = True
        if tag in ("switch", "drag") and len(parts) >= 5:
            side, nick = _parse_ident(parts[2])
            sw_sp = _species_from_details(parts[3])
            if nick not in nick_to_idx[side]:
                idx = _match_to_showteam(sw_sp, st_species[side])
                if idx is not None:
                    nick_to_idx[side][nick] = idx
                else:
                    continue
            species = st_species[side][nick_to_idx[side][nick]]
            revealed[side].add(species)
            if not leads_done and len(leads[side]) < 2:
                leads[side].append(species)

    for side in ("p1", "p2"):
        if len(leads[side]) != 2:
            return None

    species6 = {s: frozenset(st_species[s]) for s in ("p1", "p2")}
    species_key = {
        s: "|".join(sorted(st_species[s])) for s in ("p1", "p2")
    }

    return GameInfo(
        battle_id=battle_id,
        set_id=set_id,
        game_num=game_num,
        player_names=player_names,
        winner_side=winner_side,
        leads={s: frozenset(leads[s]) for s in ("p1", "p2")},
        revealed={s: frozenset(revealed[s]) for s in ("p1", "p2")},
        species6=species6,
        species_key=species_key,
    )
