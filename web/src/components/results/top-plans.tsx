'use client';

import Image from 'next/image';
import { spriteUrl } from '@/lib/sprites';
import type { ActionPlan } from '@/types/pokemon';
import { cn } from '@/lib/utils';

interface TopPlansProps {
  plans: ActionPlan[];
  species: string[];
  abstain: boolean;
}

function MonGroup({
  label,
  indices,
  species,
  dimmed,
}: {
  label: string;
  indices: [number, number];
  species: string[];
  dimmed?: boolean;
}) {
  return (
    <div className={cn('flex flex-col items-center gap-1', dimmed && 'opacity-40')}>
      <span className="font-[family-name:var(--font-label)] text-[8px] uppercase text-rock">
        {label}
      </span>
      <div className="flex gap-1">
        {indices.map((idx) => (
          <div key={idx} className="flex flex-col items-center">
            <Image
              src={spriteUrl(species[idx])}
              alt={species[idx]}
              width={40}
              height={40}
              className="object-contain"
              unoptimized
            />
            <span className="max-w-12 truncate font-[family-name:var(--font-label)] text-[7px] text-night">
              {species[idx]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function TopPlans({ plans, species, abstain }: TopPlansProps) {
  if (abstain) {
    return (
      <div className="border-2 border-red-400 bg-red-50 p-4 shadow-[4px_4px_0px_#E63946]">
        <h3 className="mb-2 font-[family-name:var(--font-heading)] text-xs text-red-600">
          LOW CONFIDENCE
        </h3>
        <p className="font-[family-name:var(--font-label)] text-xs text-red-500">
          SCOUTING MODE — The model is uncertain about this matchup. Review the marginals
          and similar matchups below for guidance.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {plans.map((plan) => {
        const pct = plan.probability * 100;
        return (
          <div
            key={plan.action90_id}
            className="flex items-center gap-3 border-2 border-night bg-white p-2 shadow-[2px_2px_0px_#3D5A80]"
          >
            {/* Rank */}
            <div className="flex size-8 shrink-0 items-center justify-center border-2 border-night bg-jam font-[family-name:var(--font-heading)] text-xs text-white">
              {plan.rank}
            </div>

            {/* Mon groups */}
            <div className="flex flex-1 items-center gap-4">
              <MonGroup label="Lead" indices={plan.lead} species={species} />
              <div className="h-10 w-px bg-night" />
              <MonGroup label="Back" indices={plan.back} species={species} />
              <div className="h-10 w-px bg-rock opacity-30" />
              <MonGroup label="Bench" indices={plan.bench} species={species} dimmed />
            </div>

            {/* Probability */}
            <div className="flex w-20 shrink-0 flex-col items-end gap-1">
              <span className="font-[family-name:var(--font-label)] text-xs text-night">
                {pct.toFixed(1)}%
              </span>
              <div className="h-2 w-full border border-night bg-muted">
                <div className="h-full bg-jam" style={{ width: `${Math.min(pct * 5, 100)}%` }} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
