"""OTS role lexicon for domain-legible cue extraction.

Maps moves, items, and abilities to player-relevant tactical role tags.
Used by the coach demo to annotate opponent teams with readable cues like
"speed_control", "redirection", "disruption", etc.

Reference: docs/PROJECT_BIBLE.md Section 5.2
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Role lexicon: role_tag → set of moves / items / abilities
# ---------------------------------------------------------------------------

ROLE_LEXICON: dict[str, set[str]] = {
    # --- Move-based roles ---
    "speed_control": {
        "Tailwind", "Trick Room", "Icy Wind", "Electroweb",
        "Scary Face", "Bulldoze", "Drum Beating", "String Shot",
        "Rock Tomb", "Cotton Spore",
    },
    "redirection": {
        "Follow Me", "Rage Powder", "Spotlight",
    },
    "fake_out": {
        "Fake Out",
    },
    "priority": {
        "Extreme Speed", "Sucker Punch", "Aqua Jet", "Grassy Glide",
        "Mach Punch", "Bullet Punch", "Ice Shard", "Shadow Sneak",
        "Quick Attack", "First Impression", "Accelerock",
        "Jet Punch", "Water Shuriken",
    },
    "spread": {
        "Heat Wave", "Muddy Water", "Rock Slide", "Dazzling Gleam",
        "Earthquake", "Surf", "Discharge", "Blizzard", "Icy Wind",
        "Snarl", "Hyper Voice", "Electroweb", "Lava Plume",
        "Bulldoze", "Breaking Swipe", "Make It Rain",
        "Expanding Force", "Brutal Swing",
    },
    "protect": {
        "Protect", "Detect", "Wide Guard", "Quick Guard",
        "Spiky Shield", "Baneful Bunker", "King's Shield",
        "Max Guard", "Silk Trap", "Obstruct", "Burning Bulwark",
    },
    "disruption": {
        "Taunt", "Encore", "Will-O-Wisp", "Thunder Wave",
        "Spore", "Sleep Powder", "Yawn", "Haze", "Clear Smog",
        "Imprison", "Trick", "Switcheroo", "Disable",
        "Torment", "Helping Hand", "Ally Switch",
        "Parting Shot", "U-turn", "Volt Switch", "Flip Turn",
    },
    "weather_setter": {
        "Rain Dance", "Sunny Day", "Sandstorm", "Snowscape",
        "Hail",
    },
    "terrain_setter": {
        "Electric Terrain", "Grassy Terrain",
        "Psychic Terrain", "Misty Terrain",
    },
    "setup": {
        "Swords Dance", "Nasty Plot", "Calm Mind", "Dragon Dance",
        "Quiver Dance", "Iron Defense", "Bulk Up", "Coil",
        "Shell Smash", "Geomancy", "Belly Drum",
        "Curse", "Agility", "Autotomize", "Work Up",
        "Coaching", "Decorate",
    },
    "recovery": {
        "Recover", "Roost", "Moonlight", "Synthesis",
        "Slack Off", "Soft-Boiled", "Morning Sun",
        "Shore Up", "Strength Sap", "Leech Seed",
        "Drain Punch", "Giga Drain", "Horn Leech",
    },
    # --- Item-based roles ---
    "choice_item": {
        "Choice Scarf", "Choice Band", "Choice Specs",
    },
    "sash": {
        "Focus Sash",
    },
    "berry": {
        "Sitrus Berry", "Lum Berry", "Wiki Berry", "Aguav Berry",
        "Figy Berry", "Iapapa Berry", "Mago Berry",
        "Shuca Berry", "Coba Berry", "Kasib Berry",
        "Yache Berry", "Occa Berry", "Charti Berry",
        "Chople Berry", "Roseli Berry", "Haban Berry",
        "Wacan Berry",
    },
    # --- Ability-based roles ---
    "intimidate": {
        "Intimidate",
    },
    "weather_ability": {
        "Drizzle", "Drought", "Sand Stream", "Snow Warning",
        "Orichalcum Pulse", "Desolate Land", "Primordial Sea",
    },
    "terrain_ability": {
        "Electric Surge", "Grassy Surge",
        "Psychic Surge", "Misty Surge",
        "Hadron Engine",
    },
}

# ---------------------------------------------------------------------------
# Reverse lexicon: move/item/ability → set of role tags
# ---------------------------------------------------------------------------

REVERSE_LEXICON: dict[str, set[str]] = {}

for _role, _entries in ROLE_LEXICON.items():
    for _entry in _entries:
        REVERSE_LEXICON.setdefault(_entry, set()).add(_role)


# ---------------------------------------------------------------------------
# Team annotation
# ---------------------------------------------------------------------------

def annotate_team(team_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Annotate each mon with role tags derived from its OTS.

    Args:
        team_dict: a TeamSheet dict with a ``"pokemon"`` key containing
            6 mon dicts, each with ``species``, ``item``, ``ability``,
            ``tera_type``, and ``moves`` fields.

    Returns:
        List of 6 annotation dicts::

            [{"species": "Incineroar", "roles": ["fake_out", "intimidate", ...]}, ...]

        Roles are sorted alphabetically.
    """
    annotations = []
    for mon in team_dict["pokemon"]:
        roles: set[str] = set()

        # Check moves
        for move in mon.get("moves", []):
            if move and move != "UNK":
                roles.update(REVERSE_LEXICON.get(move, set()))

        # Check item
        item = mon.get("item", "UNK")
        if item and item != "UNK":
            roles.update(REVERSE_LEXICON.get(item, set()))

        # Check ability
        ability = mon.get("ability", "UNK")
        if ability and ability != "UNK":
            roles.update(REVERSE_LEXICON.get(ability, set()))

        annotations.append({
            "species": mon["species"],
            "roles": sorted(roles),
        })

    return annotations
