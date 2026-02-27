import type { TeamSheet } from '@/types/pokemon';

/**
 * Pre-loaded example: standard VGC goodstuffs vs Calyrex-Shadow.
 * Sourced from the standard_vgc test vector — a recognizable, high-signal matchup.
 */
export const EXAMPLE_TEAM_A: TeamSheet = {
  pokemon: [
    {
      species: 'Incineroar',
      item: 'Safety Goggles',
      ability: 'Intimidate',
      tera_type: 'Ghost',
      moves: ['Fake Out', 'Flare Blitz', 'Knock Off', 'Parting Shot'],
    },
    {
      species: 'Flutter Mane',
      item: 'Choice Specs',
      ability: 'Protosynthesis',
      tera_type: 'Fairy',
      moves: ['Moonblast', 'Shadow Ball', 'Dazzling Gleam', 'Mystical Fire'],
    },
    {
      species: 'Rillaboom',
      item: 'Miracle Seed',
      ability: 'Grassy Surge',
      tera_type: 'Grass',
      moves: ['Grassy Glide', 'Wood Hammer', 'Fake Out', 'U-turn'],
    },
    {
      species: 'Urshifu-Rapid-Strike',
      item: 'Focus Sash',
      ability: 'Unseen Fist',
      tera_type: 'Water',
      moves: ['Surging Strikes', 'Close Combat', 'Aqua Jet', 'Detect'],
    },
    {
      species: 'Tornadus',
      item: 'Focus Sash',
      ability: 'Prankster',
      tera_type: 'Ghost',
      moves: ['Tailwind', 'Rain Dance', 'Taunt', 'Bleakwind Storm'],
    },
    {
      species: 'Landorus',
      item: 'Life Orb',
      ability: 'Sheer Force',
      tera_type: 'Poison',
      moves: ['Earth Power', 'Sludge Bomb', 'Protect', 'Sandsear Storm'],
    },
  ],
};

export const EXAMPLE_TEAM_B: TeamSheet = {
  pokemon: [
    {
      species: 'Calyrex-Shadow',
      item: 'Focus Sash',
      ability: 'UNK',
      tera_type: 'Grass',
      moves: ['Astral Barrage', 'Psyshock', 'Nasty Plot', 'Protect'],
    },
    {
      species: 'Incineroar',
      item: 'Safety Goggles',
      ability: 'Intimidate',
      tera_type: 'Water',
      moves: ['Fake Out', 'Flare Blitz', 'Knock Off', 'Parting Shot'],
    },
    {
      species: 'Whimsicott',
      item: 'Covert Cloak',
      ability: 'Prankster',
      tera_type: 'Steel',
      moves: ['Tailwind', 'Moonblast', 'Encore', 'Protect'],
    },
    {
      species: 'Rillaboom',
      item: 'Assault Vest',
      ability: 'Grassy Surge',
      tera_type: 'Fire',
      moves: ['Grassy Glide', 'Wood Hammer', 'Fake Out', 'U-turn'],
    },
    {
      species: 'Chien-Pao',
      item: 'Life Orb',
      ability: 'Sword of Ruin',
      tera_type: 'Ice',
      moves: ['Ice Spinner', 'Crunch', 'Sacred Sword', 'Protect'],
    },
    {
      species: 'Landorus',
      item: 'Choice Scarf',
      ability: 'Sheer Force',
      tera_type: 'Flying',
      moves: ['Earth Power', 'Sludge Bomb', 'Psychic', 'U-turn'],
    },
  ],
};
