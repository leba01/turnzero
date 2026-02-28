'use client';

import Image from 'next/image';
import { spriteUrl, displayName } from '@/lib/sprites';
import { SpeciesName } from '@/components/ui/species-name';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from '@/components/ui/tooltip';
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

const LEGEND = [
  {
    label: 'Offensive',
    style: 'bg-jam/20 text-jam border-jam',
    blurb: 'Spread damage (Earthquake, Heat Wave), priority moves (Sucker Punch, Extreme Speed), or setup (Swords Dance, Nasty Plot).',
  },
  {
    label: 'Defensive',
    style: 'bg-sky/30 text-night border-sky',
    blurb: 'Protect variants (Wide Guard, Quick Guard), recovery (Recover, Drain Punch), or redirection (Follow Me, Rage Powder).',
  },
  {
    label: 'Control',
    style: 'bg-moss/20 text-moss border-moss',
    blurb: 'Speed control (Tailwind, Trick Room, Icy Wind), disruption (Taunt, Thunder Wave, sleep moves), or Fake Out.',
  },
  {
    label: 'Field',
    style: 'bg-grass/20 text-night border-grass',
    blurb: 'Sets weather or terrain via move (Rain Dance, Electric Terrain) or ability (Drizzle, Drought, Sand Stream).',
  },
  {
    label: 'Item',
    style: 'bg-mist/30 text-night border-mist',
    blurb: 'Carries a Choice item (locking moves), Focus Sash (survives a KO at full HP), or a Berry (Sitrus, Lum, etc.).',
  },
  {
    label: 'Ability',
    style: 'bg-secondary/50 text-night border-secondary',
    blurb: 'Intimidate — drops adjacent foes\' Attack on switch-in, a major factor in lead and positioning decisions.',
  },
];

export function OpponentCues({ cues }: OpponentCuesProps) {
  return (
    <div className="flex flex-col gap-3">
      <TooltipProvider>
        <div className="flex flex-wrap gap-1.5">
          {LEGEND.map(({ label, style, blurb }) => (
            <Tooltip key={label}>
              <TooltipTrigger asChild>
                <Badge variant="outline" className={`cursor-default text-[9px] sm:text-[11px] ${style}`}>
                  {label}
                </Badge>
              </TooltipTrigger>
              <TooltipContent className="max-w-[220px]">
                {blurb}
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      </TooltipProvider>
      <div className="flex flex-col gap-2">
      {cues.map((cue, i) => (
        <div key={i} className="flex items-center gap-3">
          <Image
            src={spriteUrl(cue.species)}
            alt={cue.species}
            width={48}
            height={48}
            className="shrink-0 object-contain"
            unoptimized
          />
          <span className="min-w-0 break-words font-[family-name:var(--font-label)] text-xs sm:text-sm text-night">
            <SpeciesName species={cue.species} />
          </span>
          <div className="flex flex-wrap gap-1">
            {cue.roles.length > 0 ? (
              <TooltipProvider>
                {cue.roles.map((role) => (
                  <Tooltip key={role}>
                    <TooltipTrigger asChild>
                      <Badge
                        variant="outline"
                        className={`cursor-default text-[9px] sm:text-[11px] ${getRoleStyle(role)}`}
                      >
                        {role.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                      </Badge>
                    </TooltipTrigger>
                    {cue.triggers[role]?.length > 0 && (
                      <TooltipContent>
                        via: {cue.triggers[role].join(', ')}
                      </TooltipContent>
                    )}
                  </Tooltip>
                ))}
              </TooltipProvider>
            ) : (
              <span className="font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-rock">
                (no cues)
              </span>
            )}
          </div>
        </div>
      ))}
      </div>
    </div>
  );
}
