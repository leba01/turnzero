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
