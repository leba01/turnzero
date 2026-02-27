'use client';

import Image from 'next/image';
import { spriteUrl } from '@/lib/sprites';
import { Button } from '@/components/ui/button';
import type { TeamSheet } from '@/types/pokemon';

interface TeamSummaryProps {
  team: TeamSheet;
  onEdit: () => void;
}

export function TeamSummary({ team, onEdit }: TeamSummaryProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex flex-1 flex-wrap items-center gap-3">
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
            <span className="max-w-14 truncate text-center font-[family-name:var(--font-label)] text-[9px] text-night">
              {mon.species}
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
