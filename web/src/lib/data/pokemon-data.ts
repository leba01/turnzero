/**
 * Static Pokemon data loader.
 *
 * Provides species, item, ability, tera type, and move lists
 * for autocomplete / validation in the team builder UI.
 */

export interface PokemonData {
  species: string[];
  items: string[];
  abilities: string[];
  tera_types: string[];
  moves: string[];
}

/** Load Pokemon data lists from the static JSON file. */
export async function loadPokemonData(): Promise<PokemonData> {
  const response = await fetch("/data/pokemon_data.json");
  if (!response.ok) {
    throw new Error(`Failed to load pokemon data: ${response.status}`);
  }
  return response.json() as Promise<PokemonData>;
}

let _speciesTypes: Record<string, string[]> | null = null;

/** Returns base type(s) for a species, e.g. ["Fighting", "Water"]. Cached after first fetch. */
export async function getSpeciesTypes(): Promise<Record<string, string[]>> {
  if (_speciesTypes) return _speciesTypes;
  const response = await fetch("/data/species_types.json");
  if (!response.ok) throw new Error(`Failed to load species types: ${response.status}`);
  _speciesTypes = await response.json() as Record<string, string[]>;
  return _speciesTypes;
}
