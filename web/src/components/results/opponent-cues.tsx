'use client';

import Image from 'next/image';
import { spriteUrl, displayName } from '@/lib/sprites';
import { SpeciesName } from '@/components/ui/species-name';
import { Badge } from '@/components/ui/badge';
import type { OpponentCue } from '@/types/pokemon';

interface OpponentCuesProps {
  cues: OpponentCue[];
}

const ROLE_COLORS: Record<string, string> = {
  // Offensive
  spread: 'bg-jam/20 text-jam border-jam',
  priority: 'bg-jam/20 text-jam border-jam',
  setup: 'bg-jam/20 text-jam border-jam',
  // Defensive
  protect: 'bg-sky/30 text-night border-sky',
  recovery: 'bg-sky/30 text-night border-sky',
  redirection: 'bg-sky/30 text-night border-sky',
  // Control
  speed_control: 'bg-moss/20 text-moss border-moss',
  disruption: 'bg-moss/20 text-moss border-moss',
  fake_out: 'bg-moss/20 text-moss border-moss',
  // Weather/Terrain
  weather_setter: 'bg-grass/20 text-night border-grass',
  terrain_setter: 'bg-grass/20 text-night border-grass',
  weather_ability: 'bg-grass/20 text-night border-grass',
  terrain_ability: 'bg-grass/20 text-night border-grass',
  // Items
  choice_item: 'bg-mist/30 text-night border-mist',
  sash: 'bg-mist/30 text-night border-mist',
  berry: 'bg-mist/30 text-night border-mist',
  // Abilities
  intimidate: 'bg-secondary/50 text-night border-secondary',
};

function getRoleStyle(role: string): string {
  return ROLE_COLORS[role] || 'bg-muted text-night border-rock';
}

export function OpponentCues({ cues }: OpponentCuesProps) {
  return (
    <div className="flex flex-col gap-2">
      {cues.map((cue, i) => (
        <div key={i} className="flex items-center gap-3">
          <Image
            src={spriteUrl(cue.species)}
            alt={cue.species}
            width={40}
            height={40}
            className="shrink-0 object-contain"
            unoptimized
          />
          <span className="min-w-0 break-words font-[family-name:var(--font-label)] text-xs sm:text-sm text-night">
            <SpeciesName species={cue.species} />
          </span>
          <div className="flex flex-wrap gap-1">
            {cue.roles.length > 0 ? (
              cue.roles.map((role) => (
                <Badge
                  key={role}
                  variant="outline"
                  className={`text-[9px] sm:text-[11px] ${getRoleStyle(role)}`}
                >
                  {role.replace(/_/g, ' ')}
                </Badge>
              ))
            ) : (
              <span className="font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-rock">
                (no cues)
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
