"""Sample 200 random battles from logs-gen9vgc2024regg.json and report
parse-stage observability statistics (bring4_observed rate, revealed
moves/items/abilities/tera per mon, num_revealed_mons distribution).

Output: data/exploration/parse_sample_stats.json
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
SAMPLE_SIZE = 200
RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "logs-gen9vgc2024regg.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "exploration" / "parse_sample_stats.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ident(ident: str):
    """Parse 'p1a: Nickname' -> ('p1', 'Nickname')."""
    ident = ident.strip()
    slot = ident.split(":")[0]       # "p1a"
    side = slot[:2]                  # "p1"
    nickname = ident.split(": ", 1)[1]
    return side, nickname


def normalize_poke_species(species: str) -> str:
    """Strip the -* wildcard suffix from team preview species names."""
    if species.endswith("-*"):
        return species[:-2]
    return species


def match_switch_to_poke(switch_species: str, poke_species_list: list) -> str | None:
    """Prefix-match a switch-line species to a poke-line species.

    E.g. 'Urshifu-Rapid-Strike' matches 'Urshifu' (after -* strip).
    Returns the matched poke species or None.
    """
    for poke_sp in poke_species_list:
        if switch_species == poke_sp or switch_species.startswith(poke_sp + "-"):
            return poke_sp
    # Fallback: the switch species itself might be in the list verbatim
    if switch_species in poke_species_list:
        return switch_species
    return None


def parse_species_from_details(details: str) -> str:
    """Extract species from a details field like 'Urshifu-Rapid-Strike, L50, M'."""
    return details.split(",")[0].strip()


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_battle(log_text: str) -> dict:
    """Parse a single battle log and return per-side data."""
    lines = log_text.split("\n")

    # --- Phase 1: team preview (|poke| lines) ---
    team_species = {"p1": [], "p2": []}
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 4 and parts[1] == "poke":
            side = parts[2].strip()  # "p1" or "p2"
            species_raw = parse_species_from_details(parts[3])
            species = normalize_poke_species(species_raw)
            team_species[side].append(species)

    for side in ("p1", "p2"):
        if len(team_species[side]) != 6:
            raise ValueError(f"{side} has {len(team_species[side])} poke lines, expected 6")

    # --- Phase 2: find |start| marker ---
    start_idx = None
    for i, line in enumerate(lines):
        if "|start" in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No |start| marker found")

    # --- Phase 3 & 4: scan lines after |start| ---
    # Per-side tracking
    nick_to_species = {"p1": {}, "p2": {}}
    leads = {"p1": [], "p2": []}
    revealed_nicks = {"p1": set(), "p2": set()}
    moves = {"p1": defaultdict(set), "p2": defaultdict(set)}
    items = {"p1": defaultdict(set), "p2": defaultdict(set)}
    abilities = {"p1": defaultdict(set), "p2": defaultdict(set)}
    tera = {"p1": defaultdict(set), "p2": defaultdict(set)}

    leads_done = False  # True after we see |turn|

    for line in lines[start_idx + 1:]:
        parts = line.split("|")
        if len(parts) < 3:
            continue

        tag = parts[1].strip()

        # Detect end of lead phase
        if tag == "turn":
            leads_done = True

        # --- |switch| and |drag| ---
        if tag in ("switch", "drag") and len(parts) >= 5:
            side, nickname = parse_ident(parts[2])
            switch_species = parse_species_from_details(parts[3])

            # Map nickname to poke species on first encounter
            if nickname not in nick_to_species[side]:
                matched = match_switch_to_poke(switch_species, team_species[side])
                if matched is None:
                    matched = switch_species  # fallback
                nick_to_species[side][nickname] = matched

            revealed_nicks[side].add(nickname)

            # Record leads (switches before first |turn|)
            if not leads_done and len(leads[side]) < 2:
                leads[side].append(nick_to_species[side][nickname])

        # --- |move| ---
        elif tag == "move" and len(parts) >= 4:
            side, nickname = parse_ident(parts[2])
            move_name = parts[3].strip()
            if nickname in nick_to_species[side]:
                species = nick_to_species[side][nickname]
                moves[side][species].add(move_name)

        # --- |-ability| ---
        elif tag == "-ability" and len(parts) >= 4:
            side, nickname = parse_ident(parts[2])
            ability_name = parts[3].strip()
            if nickname in nick_to_species[side]:
                species = nick_to_species[side][nickname]
                abilities[side][species].add(ability_name)

        # --- |-item| ---
        elif tag == "-item" and len(parts) >= 4:
            side, nickname = parse_ident(parts[2])
            item_name = parts[3].strip()
            if nickname in nick_to_species[side]:
                species = nick_to_species[side][nickname]
                items[side][species].add(item_name)

        # --- |-enditem| ---
        elif tag == "-enditem" and len(parts) >= 4:
            side, nickname = parse_ident(parts[2])
            item_name = parts[3].strip()
            if nickname in nick_to_species[side]:
                species = nick_to_species[side][nickname]
                items[side][species].add(item_name)

        # --- |-terastallize| ---
        elif tag == "-terastallize" and len(parts) >= 4:
            side, nickname = parse_ident(parts[2])
            tera_type = parts[3].strip()
            if nickname in nick_to_species[side]:
                species = nick_to_species[side][nickname]
                tera[side][species].add(tera_type)

    # --- Build per-side results ---
    result = {}
    for side in ("p1", "p2"):
        revealed_species = [nick_to_species[side][n] for n in revealed_nicks[side]
                            if n in nick_to_species[side]]
        # Deduplicate (same poke species seen under different nicknames shouldn't happen,
        # but form changes map to the same poke species via nick_to_species)
        revealed_species_unique = list(dict.fromkeys(revealed_species))

        result[side] = {
            "team": team_species[side],
            "leads": leads[side],
            "num_revealed_mons": len(revealed_species_unique),
            "revealed_mons": revealed_species_unique,
            "moves_per_mon": {sp: sorted(moves[side][sp]) for sp in revealed_species_unique},
            "items_per_mon": {sp: sorted(items[side][sp]) for sp in revealed_species_unique},
            "abilities_per_mon": {sp: sorted(abilities[side][sp]) for sp in revealed_species_unique},
            "tera_per_mon": {sp: sorted(tera[side][sp]) for sp in revealed_species_unique},
        }

    return result


# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------

def compute_stats(parsed_battles: list, total_battles: int) -> dict:
    all_sides = []
    for battle in parsed_battles:
        for side in ("p1", "p2"):
            entry = battle["parsed"][side]
            entry["battle_id"] = battle["battle_id"]
            entry["side"] = side
            all_sides.append(entry)

    total_sides = len(all_sides)

    # bring4_observed
    bring4_count = sum(1 for s in all_sides if s["num_revealed_mons"] == 4)

    # num_revealed_mons distribution
    revealed_dist = Counter(s["num_revealed_mons"] for s in all_sides)

    # Per-mon averages (only over mons that actually appeared)
    total_mons = 0
    total_moves = 0
    total_items = 0
    total_abilities = 0
    total_tera = 0

    for s in all_sides:
        for species in s["revealed_mons"]:
            total_mons += 1
            total_moves += len(s["moves_per_mon"].get(species, []))
            total_items += len(s["items_per_mon"].get(species, []))
            total_abilities += len(s["abilities_per_mon"].get(species, []))
            total_tera += len(s["tera_per_mon"].get(species, []))

    return {
        "sample_size": SAMPLE_SIZE,
        "total_battles_in_file": total_battles,
        "seed": SEED,
        "total_sides_parsed": total_sides,
        "bring4_observed_count": bring4_count,
        "bring4_observed_rate": round(bring4_count / total_sides, 4) if total_sides else 0,
        "num_revealed_mons_distribution": {str(k): v for k, v in sorted(revealed_dist.items())},
        "total_mons_appeared": total_mons,
        "avg_revealed_moves_per_mon": round(total_moves / total_mons, 4) if total_mons else 0,
        "avg_revealed_items_per_mon": round(total_items / total_mons, 4) if total_mons else 0,
        "avg_revealed_abilities_per_mon": round(total_abilities / total_mons, 4) if total_mons else 0,
        "avg_revealed_tera_per_mon": round(total_tera / total_mons, 4) if total_mons else 0,
        "per_side_data": all_sides,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {RAW_PATH} ...")
    with open(RAW_PATH) as f:
        data = json.load(f)

    total_battles = len(data)
    keys = list(data.keys())
    print(f"Total battles in file: {total_battles}")

    # Sample
    rng = random.Random(SEED)
    sample_keys = rng.sample(keys, SAMPLE_SIZE)

    # Parse
    parsed = []
    errors = []
    for battle_id in sample_keys:
        _timestamp, log_text = data[battle_id]
        try:
            result = parse_battle(log_text)
            parsed.append({"battle_id": battle_id, "parsed": result})
        except Exception as e:
            errors.append({"battle_id": battle_id, "error": str(e)})

    print(f"Parsed {len(parsed)}/{SAMPLE_SIZE} battles ({len(errors)} errors)")

    # Stats
    stats = compute_stats(parsed, total_battles)
    stats["parse_errors"] = errors

    # Write
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")

    # Print summary (exclude bulky per_side_data)
    summary = {k: v for k, v in stats.items() if k not in ("per_side_data",)}
    print("\n" + "=" * 60)
    print("PARSE SAMPLE STATS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
