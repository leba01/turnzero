'use client';

import Image from 'next/image';
import { spriteUrl } from '@/lib/sprites';
import type { TeamSheet } from '@/types/pokemon';
import { cn } from '@/lib/utils';

interface TeamDisplayProps {
  team: TeamSheet;
  className?: string;
}

export function TeamDisplay({ team, className }: TeamDisplayProps) {
  return (
    <div className={cn('flex gap-2', className)}>
      {team.pokemon.map((mon, i) => (
        <div key={i} className="flex flex-col items-center gap-0.5">
          <Image
            src={spriteUrl(mon.species)}
            alt={mon.species}
            width={48}
            height={48}
            className="object-contain"
            unoptimized
          />
          <span className="max-w-12 truncate text-center font-[family-name:var(--font-label)] text-[8px] text-night">
            {mon.species !== 'UNK' ? mon.species : '?'}
          </span>
        </div>
      ))}
    </div>
  );
}
