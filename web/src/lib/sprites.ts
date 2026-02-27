/** Sprite URL utility — uses Showdown CDN, no local hosting needed. */

const SHOWDOWN_BASE = "https://play.pokemonshowdown.com/sprites";

/** Convert species name to Showdown slug: lowercase, remove non-alphanumeric except hyphens. */
export function speciesSlug(species: string): string {
  return species
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, "");
}

/** Get sprite URL for a species using Showdown HOME sprites (Gen 9 quality). */
export function spriteUrl(species: string): string {
  if (!species || species === "UNK" || species === "<UNK>") {
    return `${SHOWDOWN_BASE}/gen5/0.png`; // Fallback missingno
  }
  return `${SHOWDOWN_BASE}/home/${speciesSlug(species)}.png`;
}

/** Get small icon sprite (40x30) for compact displays. */
export function iconUrl(species: string): string {
  if (!species || species === "UNK" || species === "<UNK>") {
    return `${SHOWDOWN_BASE}/gen5/0.png`;
  }
  return `${SHOWDOWN_BASE}/gen5/${speciesSlug(species)}.png`;
}
