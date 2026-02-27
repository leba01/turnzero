'use client';

import Image from 'next/image';
import { spriteUrl } from '@/lib/sprites';
import { Badge } from '@/components/ui/badge';
import type { ActionPlan } from '@/types/pokemon';
import { cn } from '@/lib/utils';

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
  const dots = Array.from({ length: 5 }, (_, i) => i < agreement);
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
    sublabel: 'Clear game plan — models converge on the same strategy',
    color: 'text-moss',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-moss',
    dots,
  };
  if (agreement >= 3) return {
    label: 'Moderate',
    sublabel: 'Likely bring-4 is clear, but lead order is flexible',
    color: 'text-night',
    bgColor: 'bg-sky-50',
    borderColor: 'border-night',
    dots,
  };
  return {
    label: 'Low',
    sublabel: 'Multiple viable plans — check marginals for guidance',
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
      <span
        className={cn(
          'truncate text-center font-[family-name:var(--font-label)] text-night',
          size >= 48 ? 'max-w-16 text-[9px]' : 'max-w-12 text-[8px]',
        )}
      >
        {name}
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
  const spriteSize = isHero ? 52 : 36;
  const backSize = isHero ? 36 : 28;
  const mobileSpriteSize = isHero ? 40 : 28;
  const mobileBackSize = isHero ? 28 : 22;

  return (
    <div
      className={cn(
        'flex items-center gap-3 border-2 border-night bg-white p-2',
        isHero
          ? 'shadow-[4px_4px_0px_#3D5A80] p-3'
          : 'shadow-[2px_2px_0px_#3D5A80]',
      )}
    >
      {/* Rank */}
      <div
        className={cn(
          'flex shrink-0 items-center justify-center border-2 border-night font-[family-name:var(--font-heading)] text-white',
          isHero ? 'size-10 bg-jam text-sm' : 'size-7 bg-night text-[10px]',
        )}
      >
        {plan.rank}
      </div>

      {/* Lead pair */}
      <div className="flex items-end gap-1">
        <span className="hidden sm:contents">
          <MonSprite name={species[plan.lead[0]]} size={spriteSize} />
          <MonSprite name={species[plan.lead[1]]} size={spriteSize} />
        </span>
        <span className="contents sm:hidden">
          <MonSprite name={species[plan.lead[0]]} size={mobileSpriteSize} />
          <MonSprite name={species[plan.lead[1]]} size={mobileSpriteSize} />
        </span>
      </div>

      {/* Divider */}
      <div className={cn('bg-night', isHero ? 'h-12 w-px' : 'h-8 w-px')} />

      {/* Back pair */}
      <div className="flex items-end gap-1">
        <span className="hidden sm:contents">
          <MonSprite name={species[plan.back[0]]} size={backSize} dimmed />
          <MonSprite name={species[plan.back[1]]} size={backSize} dimmed />
        </span>
        <span className="contents sm:hidden">
          <MonSprite name={species[plan.back[0]]} size={mobileBackSize} dimmed />
          <MonSprite name={species[plan.back[1]]} size={mobileBackSize} dimmed />
        </span>
      </div>

      {/* Probability */}
      <div className="ml-auto flex shrink-0 flex-col items-end gap-1">
        <span
          className={cn(
            'font-[family-name:var(--font-label)] text-night',
            isHero ? 'text-sm' : 'text-xs',
          )}
        >
          {pct.toFixed(1)}%
        </span>
        <div
          className={cn(
            'border border-night bg-muted',
            isHero ? 'h-2.5 w-16 sm:w-24' : 'h-2 w-12 sm:w-16',
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

  // Hero treatment when #1 separates from the pack.
  // In a 90-way space with low baselines, even a modest lead is real signal.
  // Compare #1 against the average of #2 and #3 — if it pulls ahead of
  // the cluster, it deserves emphasis.
  const hasStrongFavorite =
    !abstain &&
    plans.length >= 3 &&
    plans[0].probability > ((plans[1].probability + plans[2].probability) / 2) * 1.3;

  // Relative strength: how many times better than the 1/90 uniform baseline.
  const vsRandom = confidence / (1 / 90);

  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <h3 className="font-[family-name:var(--font-label)] text-xs uppercase tracking-wider text-night">
        Recommended Plans
      </h3>

      {/* Confidence banner */}
      <div className={cn(
        'flex items-center gap-3 border-2 p-3',
        tier.borderColor,
        tier.bgColor,
      )}>
        {/* Agreement dots */}
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
          <span className={cn('font-[family-name:var(--font-label)] text-[9px]', tier.color)}>
            {ensembleAgreement}/5
          </span>
        </div>

        {/* Label and explanation */}
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <span className={cn('font-[family-name:var(--font-heading)] text-sm', tier.color)}>
              {tier.label}
            </span>
            {abstain && (
              <Badge className="animate-pulse bg-red-500 font-[family-name:var(--font-label)] text-[8px] text-white">
                ABSTAIN
              </Badge>
            )}
          </div>
          <span className="font-[family-name:var(--font-body)] text-[10px] text-rock">
            {tier.sublabel}
          </span>
        </div>

        {/* Relative strength pill */}
        <div className="ml-auto flex shrink-0 flex-col items-end">
          <span className={cn('font-[family-name:var(--font-label)] text-xs', tier.color)}>
            {vsRandom.toFixed(0)}x
          </span>
          <span className="font-[family-name:var(--font-body)] text-[8px] text-rock">
            vs random
          </span>
        </div>
      </div>

      {/* Plans — hero only when there's a clear favorite */}
      {plans.map((plan, i) => (
        <PlanRow
          key={plan.action90_id}
          plan={plan}
          species={species}
          isHero={i === 0 && hasStrongFavorite}
        />
      ))}

      {/* Fine print for the nerds */}
      <span className="font-[family-name:var(--font-body)] text-[9px] text-rock">
        {(confidence * 100).toFixed(1)}% max probability (1/{90} = {(100 / 90).toFixed(1)}% random baseline) · H={plans.length > 0 ? (
          -Math.log2(confidence).toFixed(1)
        ) : '—'} bits
      </span>
    </div>
  );
}
