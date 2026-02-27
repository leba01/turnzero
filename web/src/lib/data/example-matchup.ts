import type { TeamSheet } from '@/types/pokemon';

/** An example matchup with a human-readable label and description. */
export interface ExampleMatchup {
  id: string;
  label: string;
  description: string;
  teamA: TeamSheet;
  teamB: TeamSheet;
}

/**
 * Pre-loaded example: standard VGC goodstuffs vs Calyrex-Shadow.
 * Moderate confidence — models mostly agree on bring-4 but split on lead pair.
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

// ── High-confidence example: Commander Dondozo ──────────────────────────
// 36% confidence, 5/5 agreement. Weezing-Galar leads to disable Commander
// with Neutralizing Gas, then deploy Dondozo+Tatsugiri safely in back.

const HIGH_TEAM_A: TeamSheet = {
  pokemon: [
    {
      species: 'Calyrex-Shadow',
      item: 'Life Orb',
      ability: 'As One (Spectrier)',
      tera_type: 'Normal',
      moves: ['Astral Barrage', 'Hyper Beam', 'Protect', 'Psychic'],
    },
    {
      species: 'Dondozo',
      item: 'Leftovers',
      ability: 'Unaware',
      tera_type: 'Grass',
      moves: ['Body Press', 'Earthquake', 'Order Up', 'Protect'],
    },
    {
      species: 'Ogerpon-Cornerstone',
      item: 'Cornerstone Mask',
      ability: 'Sturdy',
      tera_type: 'Rock',
      moves: ['Follow Me', 'Ivy Cudgel', 'Power Whip', 'Spiky Shield'],
    },
    {
      species: 'Roaring Moon',
      item: 'Booster Energy',
      ability: 'Protosynthesis',
      tera_type: 'Flying',
      moves: ['Acrobatics', 'Knock Off', 'Protect', 'Tailwind'],
    },
    {
      species: 'Tatsugiri',
      item: 'Focus Sash',
      ability: 'Commander',
      tera_type: 'Stellar',
      moves: ['Draco Meteor', 'Helping Hand', 'Muddy Water', 'Protect'],
    },
    {
      species: 'Weezing-Galar',
      item: 'Covert Cloak',
      ability: 'Neutralizing Gas',
      tera_type: 'Water',
      moves: ['Poison Gas', 'Protect', 'Strange Steam', 'Toxic Spikes'],
    },
  ],
};

const HIGH_TEAM_B: TeamSheet = {
  pokemon: [
    {
      species: 'Farigiraf',
      item: 'Electric Seed',
      ability: 'Armor Tail',
      tera_type: 'Water',
      moves: ['Foul Play', 'Helping Hand', 'Psychic Noise', 'Trick Room'],
    },
    {
      species: 'Iron Hands',
      item: 'Assault Vest',
      ability: 'Quark Drive',
      tera_type: 'Bug',
      moves: ['Drain Punch', 'Fake Out', 'Low Kick', 'Wild Charge'],
    },
    {
      species: 'Miraidon',
      item: 'Choice Specs',
      ability: 'Hadron Engine',
      tera_type: 'Fairy',
      moves: ['Dazzling Gleam', 'Draco Meteor', 'Electro Drift', 'Volt Switch'],
    },
    {
      species: 'Ogerpon-Hearthflame',
      item: 'Hearthflame Mask',
      ability: 'Mold Breaker',
      tera_type: 'Fire',
      moves: ['Follow Me', 'Ivy Cudgel', 'Spiky Shield', 'Wood Hammer'],
    },
    {
      species: 'Urshifu-Rapid-Strike',
      item: 'Focus Sash',
      ability: 'Unseen Fist',
      tera_type: 'Stellar',
      moves: ['Aqua Jet', 'Close Combat', 'Protect', 'Surging Strikes'],
    },
    {
      species: 'Whimsicott',
      item: 'Covert Cloak',
      ability: 'Prankster',
      tera_type: 'Dark',
      moves: ['Encore', 'Light Screen', 'Moonblast', 'Tailwind'],
    },
  ],
};

// ── Abstain example: off-meta Dugtrio/Empoleon ──────────────────────────
// 1.5% confidence, 0/5 agreement. Near-uniform distribution across all 90
// actions. The model honestly says "I don't know."

const ABSTAIN_TEAM_A: TeamSheet = {
  pokemon: [
    {
      species: 'Dugtrio',
      item: 'Focus Sash',
      ability: 'Arena Trap',
      tera_type: 'Ghost',
      moves: ['Bulldoze', 'Endeavor', 'Helping Hand', 'Memento'],
    },
    {
      species: 'Empoleon',
      item: 'Rocky Helmet',
      ability: 'Competitive',
      tera_type: 'Grass',
      moves: ['Flash Cannon', 'Protect', 'Water Pledge', 'Yawn'],
    },
    {
      species: 'Giratina-Origin',
      item: 'Griseous Core',
      ability: 'Levitate',
      tera_type: 'Steel',
      moves: ['Draco Meteor', 'Shadow Force', 'Shadow Sneak', 'Will-O-Wisp'],
    },
    {
      species: 'Metagross',
      item: 'Weakness Policy',
      ability: 'Clear Body',
      tera_type: 'Dark',
      moves: ['Earthquake', 'Heavy Slam', 'Protect', 'Psychic Fangs'],
    },
    {
      species: 'Primarina',
      item: 'Life Orb',
      ability: 'Liquid Voice',
      tera_type: 'Poison',
      moves: ['Haze', 'Hyper Voice', 'Moonblast', 'Protect'],
    },
    {
      species: 'Ting-Lu',
      item: 'Assault Vest',
      ability: 'Vessel of Ruin',
      tera_type: 'Poison',
      moves: ['Fissure', 'Ruination', 'Stomping Tantrum', 'Throat Chop'],
    },
  ],
};

const ABSTAIN_TEAM_B: TeamSheet = {
  pokemon: [
    {
      species: 'Chi-Yu',
      item: 'Choice Scarf',
      ability: 'Beads of Ruin',
      tera_type: 'Grass',
      moves: ['Dark Pulse', 'Heat Wave', 'Overheat', 'Tera Blast'],
    },
    {
      species: 'Farigiraf',
      item: 'Safety Goggles',
      ability: 'Armor Tail',
      tera_type: 'Dark',
      moves: ['Ally Switch', 'Protect', 'Psychic', 'Shadow Ball'],
    },
    {
      species: 'Kyogre',
      item: 'Choice Specs',
      ability: 'Drizzle',
      tera_type: 'Water',
      moves: ['Ice Beam', 'Origin Pulse', 'Thunder', 'Water Spout'],
    },
    {
      species: 'Landorus',
      item: 'Life Orb',
      ability: 'Sheer Force',
      tera_type: 'Water',
      moves: ['Earth Power', 'Focus Blast', 'Protect', 'Sludge Bomb'],
    },
    {
      species: 'Tornadus',
      item: 'Covert Cloak',
      ability: 'Prankster',
      tera_type: 'Ground',
      moves: ['Hurricane', 'Protect', 'Rain Dance', 'Tailwind'],
    },
    {
      species: 'Urshifu-Rapid-Strike',
      item: 'Assault Vest',
      ability: 'Unseen Fist',
      tera_type: 'Grass',
      moves: ['Close Combat', 'Ice Spinner', 'Surging Strikes', 'U-turn'],
    },
  ],
};

// ── Exported example catalog ────────────────────────────────────────────

export const EXAMPLES: ExampleMatchup[] = [
  {
    id: 'moderate',
    label: 'Standard Meta',
    description: 'Models split on lead pair',
    teamA: EXAMPLE_TEAM_A,
    teamB: EXAMPLE_TEAM_B,
  },
  {
    id: 'high',
    label: 'Commander',
    description: 'All 5 models agree',
    teamA: HIGH_TEAM_A,
    teamB: HIGH_TEAM_B,
  },
  {
    id: 'abstain',
    label: 'Off-Meta',
    description: 'Model abstains',
    teamA: ABSTAIN_TEAM_A,
    teamB: ABSTAIN_TEAM_B,
  },
];
