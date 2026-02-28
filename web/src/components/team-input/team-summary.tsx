'use client';

import Image from 'next/image';
import { spriteUrl, displayName } from '@/lib/sprites';
import { SpeciesName } from '@/components/ui/species-name';
import { Button } from '@/components/ui/button';
import type { TeamSheet } from '@/types/pokemon';

interface TeamSummaryProps {
  team: TeamSheet;
  onEdit: () => void;
}

export function TeamSummary({ team, onEdit }: TeamSummaryProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="grid flex-1 grid-cols-3 gap-y-1">
        {team.pokemon.map((mon, i) => (
          <div key={i} className="flex flex-col items-center">
            <Image
              src={spriteUrl(mon.species)}
              alt={mon.species}
              width={48}
              height={48}
              className="object-contain"
              unoptimized
            />
            <span className="text-center font-[family-name:var(--font-label)] text-[9px] sm:text-[11px] text-night">
              <SpeciesName species={mon.species} />
            </span>
          </div>
        ))}
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={onEdit}
        className="shrink-0 border-night text-night hover:bg-mist"
      >
        EDIT
      </Button>
    </div>
  );
}
