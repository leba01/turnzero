'use client';

import Image from 'next/image';
import { spriteUrl, displayName } from '@/lib/sprites';
import { Badge } from '@/components/ui/badge';
import type { ActionPlan } from '@/types/pokemon';
import { cn } from '@/lib/utils';

const ENSEMBLE_SIZE = 5;

interface TopPlansProps {
  plans: ActionPlan[];
  species: string[];
  abstain: boolean;
  confidence: number;
  ensembleAgreement: number;
}

function confidenceTier(agreement: number, abstain: boolean): {
  label: string;
  sublabel: string;
  color: string;
  bgColor: string;
  borderColor: string;
  dots: boolean[];
} {
  const dots = Array.from({ length: ENSEMBLE_SIZE }, (_, i) => i < agreement);
  if (abstain) return {
    label: 'Uncertain',
    sublabel: 'The model is guessing — treat as scouting info',
    color: 'text-red-500',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-400',
    dots,
  };
  if (agreement >= 4) return {
    label: 'Strong',
    sublabel: 'Models agree — this is a common pattern in tournament data',
    color: 'text-moss',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-moss',
    dots,
  };
  if (agreement >= 3) return {
    label: 'Moderate',
    sublabel: 'Bring-4 is likely clear, but lead order varies in the data',
    color: 'text-night',
    bgColor: 'bg-sky-50',
    borderColor: 'border-night',
    dots,
  };
  return {
    label: 'Low',
    sublabel: 'Multiple viable plans in the data — check marginals and evidence',
    color: 'text-rock',
    bgColor: 'bg-stone-50',
    borderColor: 'border-rock',
    dots,
  };
}

function MonSprite({
  name,
  size,
  dimmed,
}: {
  name: string;
  size: number;
  dimmed?: boolean;
}) {
  return (
    <div className={cn('flex flex-col items-center', dimmed && 'opacity-40')}>
      <Image
        src={spriteUrl(name)}
        alt={name}
        width={size}
        height={size}
        className="object-contain"
        unoptimized
      />
      <span className="whitespace-nowrap text-center font-[family-name:var(--font-label)] text-[9px] sm:text-[11px] text-night">
        {displayName(name)}
      </span>
    </div>
  );
}

function PlanRow({
  plan,
  species,
  isHero,
}: {
  plan: ActionPlan;
  species: string[];
  isHero?: boolean;
}) {
  const pct = plan.probability * 100;
  const leadSize = isHero ? 56 : 48;
  const backSize = isHero ? 44 : 36;

  return (
    <div
      className={cn(
        'col-span-5 grid items-center gap-x-1 sm:gap-x-3 border-2 border-night bg-white p-2',
        isHero
          ? 'shadow-[4px_4px_0px_#3D5A80] p-3'
          : 'shadow-[2px_2px_0px_#3D5A80]',
      )}
      style={{ gridTemplateColumns: 'subgrid' }}
    >
      <div
        className={cn(
          'flex shrink-0 items-center justify-center border-2 border-night font-[family-name:var(--font-heading)] text-white',
          isHero ? 'size-10 bg-jam text-sm sm:text-base' : 'size-7 bg-night text-[10px] sm:text-sm',
        )}
      >
        {plan.rank}
      </div>

      <div className="flex items-center justify-center gap-3">
        <MonSprite name={species[plan.lead[0]]} size={leadSize} />
        <MonSprite name={species[plan.lead[1]]} size={leadSize} />
      </div>

      <div className="h-full w-px bg-night" />

      <div className="flex items-center justify-center gap-3">
        <MonSprite name={species[plan.back[0]]} size={backSize} dimmed />
        <MonSprite name={species[plan.back[1]]} size={backSize} dimmed />
      </div>

      <div className="flex flex-col items-end gap-1">
        <span
          className={cn(
            'font-[family-name:var(--font-label)] text-night',
            isHero ? 'text-sm sm:text-base' : 'text-xs sm:text-sm',
          )}
        >
          {pct.toFixed(1)}%
        </span>
        <div
          className={cn(
            'border border-night bg-muted',
            isHero ? 'h-2.5 w-24' : 'h-2 w-16',
          )}
        >
          <div
            className="h-full bg-jam"
            style={{ width: `${Math.min(pct * 5, 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}

export function TopPlans({ plans, species, abstain, confidence, ensembleAgreement }: TopPlansProps) {
  const tier = confidenceTier(ensembleAgreement, abstain);

  // Hero treatment: #1 is emphasized when it's 30% better than avg of #2 and #3.
  const hasStrongFavorite =
    !abstain &&
    plans.length >= 3 &&
    plans[0].probability > ((plans[1].probability + plans[2].probability) / 2) * 1.3;

  const vsRandom = confidence * 90;

  return (
    <div className="flex flex-col gap-3">
      <div className={cn(
        'flex items-center gap-3 border-2 p-3',
        tier.borderColor,
        tier.bgColor,
      )}>
        <div className="flex shrink-0 flex-col items-center gap-1">
          <div className="flex gap-1">
            {tier.dots.map((filled, i) => (
              <div
                key={i}
                className={cn(
                  'size-2.5 border border-current',
                  filled ? tier.color : 'bg-transparent opacity-30',
                  filled && (abstain ? 'bg-red-500' : ensembleAgreement >= 4 ? 'bg-moss' : ensembleAgreement >= 3 ? 'bg-night' : 'bg-rock'),
                )}
              />
            ))}
          </div>
          <span className={cn('font-[family-name:var(--font-label)] text-[9px] sm:text-[11px]', tier.color)}>
            {ensembleAgreement}/{ENSEMBLE_SIZE}
          </span>
        </div>

        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <span className={cn('font-[family-name:var(--font-heading)] text-sm sm:text-base', tier.color)}>
              {tier.label}
            </span>
            {abstain && (
              <Badge className="animate-pulse bg-red-500 font-[family-name:var(--font-label)] text-[8px] sm:text-[10px] text-white">
                ABSTAIN
              </Badge>
            )}
          </div>
          <span className="font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-rock">
            {tier.sublabel}
          </span>
        </div>

        <div className="ml-auto flex shrink-0 flex-col items-end">
          <span className={cn('font-[family-name:var(--font-label)] text-xs sm:text-sm', tier.color)}>
            {vsRandom.toFixed(0)}x
          </span>
          <span className="font-[family-name:var(--font-body)] text-[8px] sm:text-[10px] text-rock">
            vs random
          </span>
        </div>
      </div>

      <div className="grid gap-y-3" style={{ gridTemplateColumns: 'auto auto auto auto 1fr' }}>
        {plans.map((plan, i) => (
          <PlanRow
            key={plan.action90_id}
            plan={plan}
            species={species}
            isHero={i === 0 && hasStrongFavorite}
          />
        ))}
      </div>

      <span className="font-[family-name:var(--font-body)] text-[9px] sm:text-[11px] text-rock">
        {(confidence * 100).toFixed(1)}% max probability (1/{90} = {(100 / 90).toFixed(1)}% random baseline) · H={plans.length > 0 ? (
          -Math.log2(confidence).toFixed(1)
        ) : '—'} bits
      </span>
    </div>
  );
}
