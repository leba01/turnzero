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

function confidenceLabel(agreement: number, abstain: boolean): { text: string; color: string } {
  if (abstain) return { text: 'Uncertain — scouting mode', color: 'text-red-500' };
  if (agreement >= 4) return { text: `High confidence · ${agreement}/5 models agree`, color: 'text-moss' };
  if (agreement >= 3) return { text: `Moderate confidence · ${agreement}/5 models agree`, color: 'text-night' };
  return { text: `Low confidence · ${agreement}/5 models agree`, color: 'text-rock' };
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
          size >= 48 ? 'max-w-16 text-[9px]' : 'max-w-12 text-[7px]',
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
        <MonSprite name={species[plan.lead[0]]} size={spriteSize} />
        <MonSprite name={species[plan.lead[1]]} size={spriteSize} />
      </div>

      {/* Divider */}
      <div className={cn('bg-night', isHero ? 'h-12 w-px' : 'h-8 w-px')} />

      {/* Back pair */}
      <div className="flex items-end gap-1">
        <MonSprite name={species[plan.back[0]]} size={backSize} dimmed />
        <MonSprite name={species[plan.back[1]]} size={backSize} dimmed />
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
  const label = confidenceLabel(ensembleAgreement, abstain);

  // Hero treatment when #1 separates from the pack.
  // In a 90-way space with low baselines, even a modest lead is real signal.
  // Compare #1 against the average of #2 and #3 — if it pulls ahead of
  // the cluster, it deserves emphasis.
  const hasStrongFavorite =
    !abstain &&
    plans.length >= 3 &&
    plans[0].probability > ((plans[1].probability + plans[2].probability) / 2) * 1.3;

  return (
    <div className="flex flex-col gap-3">
      {/* Header: title + confidence inline */}
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-[family-name:var(--font-label)] text-xs uppercase tracking-wider text-night">
          Recommended Plans
        </h3>
        <div className="flex items-center gap-2">
          <span className={cn('font-[family-name:var(--font-body)] text-[11px]', label.color)}>
            {label.text}
          </span>
          {abstain && (
            <Badge className="animate-pulse bg-red-500 font-[family-name:var(--font-label)] text-[8px] text-white">
              ABSTAIN
            </Badge>
          )}
        </div>
      </div>

      {/* Abstain warning */}
      {abstain && (
        <div className="border-2 border-red-400 bg-red-50 p-3 shadow-[2px_2px_0px_#E63946]">
          <p className="font-[family-name:var(--font-body)] text-[11px] text-red-600">
            The model is uncertain about this matchup. Experts themselves disagree frequently
            on similar teams. Check the marginals and similar matchups below for guidance.
          </p>
        </div>
      )}

      {/* Plans — hero only when there's a clear favorite */}
      {plans.map((plan, i) => (
        <PlanRow
          key={plan.action90_id}
          plan={plan}
          species={species}
          isHero={i === 0 && hasStrongFavorite}
        />
      ))}

      {/* Confidence fine print */}
      <span className="font-[family-name:var(--font-body)] text-[9px] text-rock">
        {(confidence * 100).toFixed(1)}% max probability · H={plans.length > 0 ? (
          -Math.log2(confidence).toFixed(1)
        ) : '—'} bits
      </span>
    </div>
  );
}
