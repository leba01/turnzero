/** Sprite URL utility — uses Showdown CDN, no local hosting needed. */

const SHOWDOWN_BASE = "https://play.pokemonshowdown.com/sprites";

/** Species whose actual name contains hyphens (not form separators). */
const HYPHENATED_BASE = new Set([
  "ho-oh",
  "porygon-z",
  "chien-pao",
  "chi-yu",
  "wo-chien",
  "ting-lu",
  "jangmo-o",
  "hakamo-o",
  "kommo-o",
]);

/**
 * Convert species name to Showdown sprite slug.
 *
 * Showdown sprite filenames use `{baseId}.png` for base species and
 * `{baseId}-{formId}.png` for forms, where both IDs are lowercase
 * alphanumeric only. Hyphens in the species' *actual* name (like
 * Chien-Pao) get stripped; hyphens separating base from form stay.
 */
export function speciesSlug(species: string): string {
  const lower = species.toLowerCase();
  const strip = (s: string) => s.replace(/[^a-z0-9]/g, "");

  // Known hyphenated base name (no form)
  if (HYPHENATED_BASE.has(lower)) return strip(lower);

  // Known hyphenated base + form suffix (e.g. "Kommo-o-Totem")
  for (const base of HYPHENATED_BASE) {
    if (lower.startsWith(base + "-")) {
      return `${strip(base)}-${strip(lower.slice(base.length + 1))}`;
    }
  }

  // No hyphen → strip non-alphanumeric
  const idx = lower.indexOf("-");
  if (idx === -1) return strip(lower);

  // Standard base-form split on first hyphen
  return `${strip(lower.slice(0, idx))}-${strip(lower.slice(idx + 1))}`;
}

/**
 * Format species name for display.
 *
 * "Urshifu-Rapid-Strike" → "Urshifu (Rapid Strike)"
 * "Chien-Pao"            → "Chien-Pao"  (hyphen is real name)
 * "Flutter Mane"          → "Flutter Mane"
 */
export function displayName(species: string): string {
  if (!species || species === "UNK" || species === "<UNK>") return "?";

  const lower = species.toLowerCase();

  // Hyphenated base species — name is correct as-is
  if (HYPHENATED_BASE.has(lower)) return species;

  // Hyphenated base + form
  for (const base of HYPHENATED_BASE) {
    if (lower.startsWith(base + "-")) {
      const baseName = species.slice(0, base.length);
      const formName = species.slice(base.length + 1).replace(/-/g, " ");
      return `${baseName} (${formName})`;
    }
  }

  // Standard form split
  const idx = species.indexOf("-");
  if (idx === -1) return species;

  const baseName = species.slice(0, idx);
  const formName = species.slice(idx + 1).replace(/-/g, " ");
  return `${baseName} (${formName})`;
}

/** Get sprite URL for a species using Showdown Gen 5 BW-style pixel sprites. */
export function spriteUrl(species: string): string {
  if (!species || species === "UNK" || species === "<UNK>") {
    return `${SHOWDOWN_BASE}/gen5/0.png`;
  }
  return `${SHOWDOWN_BASE}/gen5/${speciesSlug(species)}.png`;
}

/** Get small icon sprite (40x30) for compact displays. */
export function iconUrl(species: string): string {
  if (!species || species === "UNK" || species === "<UNK>") {
    return `${SHOWDOWN_BASE}/gen5/0.png`;
  }
  return `${SHOWDOWN_BASE}/gen5/${speciesSlug(species)}.png`;
}
