'use client';

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import type { Pokemon, TeamSheet } from '@/types/pokemon';

interface PasteInputProps {
  onParsed: (team: TeamSheet) => void;
}

const PLACEHOLDER = `Incineroar @ Safety Goggles
Ability: Intimidate
Tera Type: Ghost
- Fake Out
- Flare Blitz
- Knock Off
- Parting Shot

Flutter Mane @ Choice Specs
Ability: Protosynthesis
Tera Type: Fairy
- Moonblast
- Shadow Ball
- Dazzling Gleam
- Mystical Fire

(paste your full 6-mon team in Showdown format)`;

function parseShowdownPaste(text: string): Pokemon[] {
  const blocks = text
    .split(/\n\s*\n/)
    .map((b) => b.trim())
    .filter(Boolean);

  const mons: Pokemon[] = [];

  for (const block of blocks) {
    const lines = block.split('\n').map((l) => l.trim());
    if (lines.length === 0) continue;

    // First line: "Species @ Item" or "Species"
    const firstLine = lines[0];
    let species = 'UNK';
    let item = 'UNK';

    const atMatch = firstLine.match(/^(.+?)\s*@\s*(.+)$/);
    if (atMatch) {
      species = atMatch[1].trim();
      item = atMatch[2].trim();
    } else {
      species = firstLine.trim();
    }

    // Strip nickname: "Nickname (Species)" pattern
    const nickMatch = species.match(/^.+?\((.+)\)$/);
    if (nickMatch) {
      species = nickMatch[1].trim();
    }
    // Strip gender suffix
    species = species.replace(/\s*\(M\)\s*$/, '').replace(/\s*\(F\)\s*$/, '').trim();

    let ability = 'UNK';
    let tera_type = 'UNK';
    const moves: string[] = [];

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i];
      if (line.startsWith('Ability:')) {
        ability = line.replace('Ability:', '').trim();
      } else if (line.startsWith('Tera Type:')) {
        tera_type = line.replace('Tera Type:', '').trim();
      } else if (line.startsWith('-')) {
        moves.push(line.replace(/^-\s*/, '').trim());
      }
    }

    // Pad moves to 4
    while (moves.length < 4) moves.push('UNK');

    mons.push({
      species,
      item,
      ability,
      tera_type,
      moves: [moves[0], moves[1], moves[2], moves[3]],
    });
  }

  return mons;
}

export function PasteInput({ onParsed }: PasteInputProps) {
  const [text, setText] = React.useState('');
  const [error, setError] = React.useState<string | null>(null);

  const handleParse = () => {
    setError(null);
    const mons = parseShowdownPaste(text);

    if (mons.length === 0) {
      setError('Could not parse any Pokémon from the paste.');
      return;
    }
    if (mons.length !== 6) {
      setError(`Expected 6 Pokémon, found ${mons.length}. Pad or trim your team.`);
      return;
    }

    onParsed({ pokemon: mons as TeamSheet['pokemon'] });
  };

  return (
    <div className="flex flex-col gap-3">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={PLACEHOLDER}
        rows={14}
        className="w-full border-2 border-night bg-white p-3 font-[family-name:var(--font-body)] text-xs sm:text-sm text-night placeholder:text-rock focus:border-jam focus:outline-none"
      />

      {error && (
        <Badge variant="destructive" className="w-fit bg-red-100 text-red-700">
          {error}
        </Badge>
      )}

      <Button onClick={handleParse} disabled={!text.trim()} className="w-fit">
        Parse Team
      </Button>
    </div>
  );
}
