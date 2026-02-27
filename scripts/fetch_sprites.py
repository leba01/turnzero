"""Download Showdown-style community sprites for all known species.

Fetches from the Pokémon Showdown sprite repository (community fan-art, not official).
Handles form names (e.g., Ogerpon-Hearthflame → ogerpon-hearthflame).

Usage:
    uv run python scripts/fetch_sprites.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

# Showdown sprite base URL (community sprites, not official Nintendo assets)
SHOWDOWN_BASE = "https://play.pokemonshowdown.com/sprites/gen5/"

# Map species names to Showdown slug format
def species_to_slug(species: str) -> str:
    """Convert species name to Showdown sprite filename slug."""
    slug = species.lower()
    # Remove special characters but keep hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Handle specific form mappings
    return slug


def download_sprite(slug: str, out_dir: Path) -> bool:
    """Download a single sprite. Returns True if successful."""
    url = f"{SHOWDOWN_BASE}{slug}.png"
    out_path = out_dir / f"{slug}.png"

    if out_path.exists():
        return True

    try:
        urllib.request.urlretrieve(url, str(out_path))
        return True
    except Exception:
        return False


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "web" / "public" / "data"
    sprite_dir = root / "web" / "public" / "sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)

    # Load species list from pokemon_data.json
    pokemon_data_path = data_dir / "pokemon_data.json"
    if not pokemon_data_path.exists():
        print("Run export_web_data.py first!")
        return

    with open(pokemon_data_path) as f:
        pokemon_data = json.load(f)

    species_list = pokemon_data["species"]
    print(f"Fetching sprites for {len(species_list)} species...")

    success = 0
    failed = []
    manifest = {}

    for i, species in enumerate(species_list):
        slug = species_to_slug(species)
        if download_sprite(slug, sprite_dir):
            success += 1
            manifest[species] = f"{slug}.png"
        else:
            failed.append(species)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(species_list)} ({success} ok, {len(failed)} failed)")
            time.sleep(0.5)  # Rate limit

    # Save manifest
    manifest_path = data_dir / "sprite_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {success}/{len(species_list)} sprites downloaded")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:20])}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    # Save failed list for debugging
    if failed:
        with open(data_dir / "sprites_failed.json", "w") as f:
            json.dump(failed, f, indent=2)


if __name__ == "__main__":
    main()
