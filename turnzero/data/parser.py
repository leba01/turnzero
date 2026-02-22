"""Showdown protocol parser for TurnZero.

Extracts full OTS from |showteam| lines, leads from first |switch| events,
and bring-4 from all |switch|/|drag| events throughout the game.

Reference: docs/PROJECT_BIBLE.md Section 2.2
"""

from __future__ import annotations

import hashlib
import json
import logging

import time
from pathlib import Path
from typing import Any

from turnzero.action_space import lead_back_to_action90
from turnzero.data.io_utils import write_manifest
from turnzero.schemas import (
    Label,
    LabelQuality,
    MatchExample,
    Pokemon,
    ReconstructionQuality,
    TeamSheet,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# |showteam| extraction
# ---------------------------------------------------------------------------

def parse_showteam_line(raw_line: str) -> tuple[str, list[Pokemon]]:
    """Parse a |showteam|pX|... line into (side, list of 6 Pokemon).

    Field layout per mon (pipe-delimited within ]-separated mons):
      [0] species  [1] (nickname)  [2] item  [3] ability
      [4] moves(csv)  ... [7] gender ...  [-1] IVs+tera (csv, tera is last)

    Note: field count varies (12 or 13) depending on whether gender is
    present, so tera is always extracted from the LAST field.
    """
    prefix = "|showteam|"
    idx = raw_line.find(prefix)
    if idx < 0:
        raise ValueError(f"Not a showteam line: {raw_line[:80]}")

    rest = raw_line[idx + len(prefix) :]
    side, mons_str = rest.split("|", 1)

    pokemon: list[Pokemon] = []
    for mon_str in mons_str.split("]"):
        if not mon_str.strip():
            continue
        fields = mon_str.split("|")
        if len(fields) < 5:
            continue  # skip malformed fragments

        species = fields[0]
        item = fields[2] if len(fields) > 2 and fields[2] else "UNK"
        ability = fields[3] if len(fields) > 3 and fields[3] else "UNK"

        moves_raw = fields[4].split(",") if len(fields) > 4 and fields[4] else []
        moves = moves_raw[:4]
        while len(moves) < 4:
            moves.append("UNK")

        # Tera type: last comma-separated element of the LAST field.
        # Field count varies (12-13) so we use fields[-1] not fields[11].
        tera_type = "UNK"
        last_field = fields[-1]
        if last_field:
            tera_parts = last_field.split(",")
            if tera_parts and tera_parts[-1]:
                tera_type = tera_parts[-1]

        pokemon.append(Pokemon(
            species=species, item=item, ability=ability,
            tera_type=tera_type, moves=moves,
        ))

    if len(pokemon) != 6:
        raise ValueError(f"Expected 6 mons from showteam, got {len(pokemon)}")

    return side, pokemon


# ---------------------------------------------------------------------------
# Helpers (adapted from scripts/parse_sample_stats.py)
# ---------------------------------------------------------------------------

def _parse_ident(ident: str) -> tuple[str, str]:
    """Parse 'p1a: Nickname' -> ('p1', 'Nickname')."""
    ident = ident.strip()
    side = ident.split(":")[0][:2]
    nickname = ident.split(": ", 1)[1]
    return side, nickname


def _species_from_details(details: str) -> str:
    """Extract species from details like 'Urshifu-Rapid-Strike, L50, M'."""
    return details.split(",")[0].strip()


def _match_to_showteam(switch_species: str, showteam_species: list[str],
                       used: set[int] | None = None) -> int | None:
    """Match a switch species to its showteam index.

    Tries: exact match, prefix match (either direction), base-species match.
    """
    skip = used or set()

    # Exact
    for i, st in enumerate(showteam_species):
        if i not in skip and switch_species == st:
            return i

    # Prefix (handles form variants like Ogerpon-Cornerstone-Tera)
    for i, st in enumerate(showteam_species):
        if i in skip:
            continue
        if switch_species.startswith(st + "-") or st.startswith(switch_species + "-"):
            return i

    # Base species (first hyphen segment)
    sw_base = switch_species.split("-")[0]
    for i, st in enumerate(showteam_species):
        if i in skip:
            continue
        if st.split("-")[0] == sw_base:
            return i

    return None


def _compute_hash(text: str) -> str:
    """Truncated sha256 hex digest (16 chars = 64 bits)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _team_id(pokemon: list[Pokemon]) -> str:
    """Order-invariant team hash from full build."""
    keys = []
    for p in pokemon:
        ms = ",".join(sorted(p.moves))
        keys.append(f"{p.species}|{p.item}|{p.ability}|{p.tera_type}|{ms}")
    keys.sort()
    return _compute_hash("|".join(keys))


def _species_key(pokemon: list[Pokemon]) -> str:
    """Order-invariant hash of species set."""
    return _compute_hash("|".join(sorted(p.species for p in pokemon)))


def _fields_known(pokemon: list[Pokemon]) -> int:
    """Count non-UNK fields across all 6 mons. Max 42 (7 per mon).

    The 7 countable fields per mon are: item, ability, tera_type, + 4 moves.
    Species is excluded because it is always known from team preview.
    """
    n = 0
    for p in pokemon:
        n += (p.item != "UNK")
        n += (p.ability != "UNK")
        n += (p.tera_type != "UNK")
        n += sum(m != "UNK" for m in p.moves)
    return n


# ---------------------------------------------------------------------------
# Core battle parser
# ---------------------------------------------------------------------------

def parse_battle(log_text: str) -> dict[str, Any]:
    """Parse one battle log.

    Returns::

        {
            "showteam": {"p1": [Pokemon, ...], "p2": [Pokemon, ...]},
            "leads":    {"p1": [species, species], "p2": [species, species]},
            "revealed": {"p1": {species, ...}, "p2": {species, ...}},
            "timestamp": int | None,
        }

    Raises ValueError on structural parse failures.
    """
    lines = log_text.split("\n")

    # --- Phase 1: |showteam| ---
    showteam: dict[str, list[Pokemon]] = {}
    for line in lines:
        if "|showteam|" in line:
            side, pokemon = parse_showteam_line(line)
            showteam[side] = pokemon

    for side in ("p1", "p2"):
        if side not in showteam:
            raise ValueError(f"No |showteam| for {side}")

    # --- Phase 2: find |start| ---
    start_idx = None
    for i, line in enumerate(lines):
        if "|start" in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No |start| marker")

    # Species lists for matching
    st_species = {s: [p.species for p in showteam[s]] for s in ("p1", "p2")}

    # --- Phase 3: scan for leads + revealed mons ---
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
                    log.debug("Unmatched switch species %s in %s", sw_sp, side)
                    continue

            species = st_species[side][nick_to_idx[side][nick]]
            revealed[side].add(species)

            if not leads_done and len(leads[side]) < 2:
                leads[side].append(species)

    for side in ("p1", "p2"):
        if len(leads[side]) != 2:
            raise ValueError(f"{side} has {len(leads[side])} leads, expected 2")

    # --- Timestamp from |t:| ---
    timestamp = None
    for line in lines:
        if line.startswith("|t|") or line.startswith("|t:|"):
            try:
                timestamp = int(line.split("|")[-1])
            except (ValueError, IndexError):
                pass
            break

    return {
        "showteam": showteam,
        "leads": leads,
        "revealed": revealed,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def _build_label(pokemon: list[Pokemon], leads_species: list[str],
                 revealed_species: set[str]) -> tuple[Label, LabelQuality]:
    """Build Label + LabelQuality for one directed example (team-A side).

    Lead-2 is always fully determined. Back-2 is only reliable when
    bring4_observed is True (4 unique mons appeared on the field).
    For Tier 2 (partial), back-2 is filled deterministically from
    the lowest available indices.
    """
    species_list = [p.species for p in pokemon]

    # Lead indices in showteam order
    lead_indices = []
    for sp in leads_species:
        try:
            lead_indices.append(species_list.index(sp))
        except ValueError:
            raise ValueError(f"Lead species '{sp}' not in team: {species_list}")
    lead2 = tuple(sorted(lead_indices))

    # Revealed indices
    revealed_idx = set()
    for sp in revealed_species:
        if sp in species_list:
            revealed_idx.add(species_list.index(sp))

    bring4_observed = len(revealed_idx) == 4

    if bring4_observed:
        back_set = revealed_idx - set(lead2)
        back2 = tuple(sorted(back_set))
        notes = None
    else:
        # Deterministic fill: known back mons first, then lowest remaining.
        # Sort the final pair since known_back indices may be > remaining.
        known_back = sorted(revealed_idx - set(lead2))
        remaining = sorted(set(range(6)) - set(lead2) - revealed_idx)
        candidates = (known_back + remaining)[:2]
        back2 = tuple(sorted(candidates))
        notes = f"back2 inferred (revealed={len(revealed_idx)}/4)"

    action90_id = lead_back_to_action90(lead2, back2)
    return (
        Label(lead2_idx=lead2, back2_idx=back2, action90_id=action90_id),
        LabelQuality(bring4_observed=bring4_observed, notes=notes),
    )


# ---------------------------------------------------------------------------
# Directed example construction
# ---------------------------------------------------------------------------

def produce_directed_examples(
    battle_id: str,
    parsed: dict[str, Any],
    format_id: str,
) -> list[MatchExample]:
    """Produce two directed MatchExamples by swapping sides."""
    examples = []

    for a_side, b_side, persp in [("p1", "p2", "p1_as_a"), ("p2", "p1", "p2_as_a")]:
        a_mons = parsed["showteam"][a_side]
        b_mons = parsed["showteam"][b_side]

        team_a = TeamSheet(
            team_id=_team_id(a_mons),
            species_key=_species_key(a_mons),
            format_id=format_id,
            pokemon=a_mons,
            reconstruction_quality=ReconstructionQuality(
                fields_known=_fields_known(a_mons),
            ),
        )
        team_b = TeamSheet(
            team_id=_team_id(b_mons),
            species_key=_species_key(b_mons),
            format_id=format_id,
            pokemon=b_mons,
            reconstruction_quality=ReconstructionQuality(
                fields_known=_fields_known(b_mons),
            ),
        )

        label, quality = _build_label(
            a_mons, parsed["leads"][a_side], parsed["revealed"][a_side],
        )

        examples.append(MatchExample(
            example_id=_compute_hash(f"{battle_id}|{persp}"),
            match_group_id=battle_id,
            battle_id=battle_id,
            team_a=team_a,
            team_b=team_b,
            label=label,
            label_quality=quality,
            format_id=format_id,
            metadata={
                "source_dataset": "cameronangliss/vgc-battle-logs",
                "timestamp_epoch": parsed["timestamp"],
            },
        ))

    return examples


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def run_parse(raw_path: str, out_dir: str, limit: int | None = None) -> dict[str, Any]:
    """Parse all battles from a raw JSON file and write JSONL + manifest.

    Args:
        raw_path: Path to raw JSON file (e.g. logs-gen9vgc2024regg.json).
        out_dir: Directory for output files.
        limit: If set, only parse the first N battles.

    Returns:
        Manifest dict with parse statistics.
    """
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)

    # Derive format_id from filename: logs-gen9vgc2024regg.json -> gen9vgc2024regg
    stem = raw_path.stem
    format_id = stem.removeprefix("logs-")

    print(f"Loading {raw_path} ...")
    t0 = time.time()
    with open(raw_path) as f:
        data = json.load(f)
    load_time = time.time() - t0
    print(f"Loaded {len(data)} battles in {load_time:.1f}s")

    battle_ids = list(data.keys())
    if limit is not None:
        battle_ids = battle_ids[:limit]

    # Parse
    jsonl_path = out_dir / "match_examples.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples_written = 0
    n_success = 0
    n_errors = 0
    error_samples: list[dict[str, str]] = []
    bring4_count = 0
    total_sides = 0

    t0 = time.time()
    with open(jsonl_path, "w") as f:
        for i, bid in enumerate(battle_ids):
            _ts, log_text = data[bid]
            try:
                parsed = parse_battle(log_text)
                examples = produce_directed_examples(bid, parsed, format_id)
                for ex in examples:
                    f.write(json.dumps(ex.to_dict(), separators=(",", ":")) + "\n")
                    examples_written += 1
                    total_sides += 1
                    if ex.label_quality.bring4_observed:
                        bring4_count += 1
                n_success += 1
            except Exception as e:
                n_errors += 1
                if len(error_samples) < 20:
                    error_samples.append({"battle_id": bid, "error": str(e)})

            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(battle_ids)} battles processed ...")

    parse_time = time.time() - t0
    print(f"Parsed {n_success}/{len(battle_ids)} battles "
          f"({n_errors} errors) in {parse_time:.1f}s")
    print(f"Wrote {examples_written} examples to {jsonl_path}")

    bring4_rate = bring4_count / total_sides if total_sides else 0
    print(f"bring4_observed rate: {bring4_rate:.4f} ({bring4_count}/{total_sides})")

    manifest = {
        "raw_path": str(raw_path),
        "format_id": format_id,
        "total_battles_in_file": len(data),
        "battles_attempted": len(battle_ids),
        "battles_parsed": n_success,
        "parse_errors": n_errors,
        "error_rate": round(n_errors / len(battle_ids), 6) if battle_ids else 0,
        "examples_written": examples_written,
        "bring4_observed_count": bring4_count,
        "bring4_observed_rate": round(bring4_rate, 4),
        "total_directed_sides": total_sides,
        "parse_time_seconds": round(parse_time, 1),
        "error_samples": error_samples,
    }
    write_manifest(out_dir / "parse_manifest.json", manifest)
    return manifest
