'use client';

import Image from 'next/image';
import { spriteUrl } from '@/lib/sprites';
import type { Marginals } from '@/types/pokemon';

interface MarginalsDisplayProps {
  marginals: Marginals;
  species: string[];
}

const LEAD_PAIRS: [number, number][] = [];
for (let i = 0; i < 6; i++) {
  for (let j = i + 1; j < 6; j++) {
    LEAD_PAIRS.push([i, j]);
  }
}

function ProbBar({
  sprite,
  label,
  value,
  maxValue,
  color,
}: {
  sprite?: string;
  label: string;
  value: number;
  maxValue: number;
  color: string;
}) {
  const pct = maxValue > 0 ? (value / maxValue) * 100 : 0;
  return (
    <div className="flex items-center gap-2">
      {sprite && (
        <Image
          src={spriteUrl(sprite)}
          alt={label}
          width={28}
          height={28}
          className="shrink-0 object-contain"
          unoptimized
        />
      )}
      <span className="w-20 shrink-0 truncate font-[family-name:var(--font-label)] text-[10px] text-night sm:w-28">
        {label}
      </span>
      <div className="h-3 flex-1 border border-night bg-muted">
        <div className="h-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="w-12 shrink-0 text-right font-[family-name:var(--font-body)] text-[10px] text-night">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export function MarginalsDisplay({ marginals, species }: MarginalsDisplayProps) {
  // Sort indices by probability descending
  const leadOrder = [...Array(6).keys()].sort(
    (a, b) => marginals.lead_probs[b] - marginals.lead_probs[a],
  );
  const bringOrder = [...Array(6).keys()].sort(
    (a, b) => marginals.bring_probs[b] - marginals.bring_probs[a],
  );
  const leadMax = Math.max(...marginals.lead_probs);
  const bringMax = Math.max(...marginals.bring_probs);

  // Top 5 lead pairs
  const pairOrder = [...Array(15).keys()]
    .sort((a, b) => marginals.lead_pair_probs[b] - marginals.lead_pair_probs[a])
    .slice(0, 5);
  const pairMax = Math.max(...marginals.lead_pair_probs);

  return (
    <div className="flex flex-col gap-4">
      {/* Lead probabilities */}
      <div>
        <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-wider text-rock">
          Lead Probabilities
        </h4>
        <div className="flex flex-col gap-1">
          {leadOrder.map((i) => (
            <ProbBar
              key={i}
              sprite={species[i]}
              label={species[i]}
              value={marginals.lead_probs[i]}
              maxValue={leadMax}
              color="#A8D8EA"
            />
          ))}
        </div>
      </div>

      {/* Bring probabilities */}
      <div>
        <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-wider text-rock">
          Bring Probabilities
        </h4>
        <div className="flex flex-col gap-1">
          {bringOrder.map((i) => (
            <ProbBar
              key={i}
              sprite={species[i]}
              label={species[i]}
              value={marginals.bring_probs[i]}
              maxValue={bringMax}
              color="#98C1D9"
            />
          ))}
        </div>
      </div>

      {/* Top lead pairs */}
      <div>
        <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-wider text-rock">
          Top Lead Pairs
        </h4>
        <div className="flex flex-col gap-1">
          {pairOrder.map((j) => {
            const [a, b] = LEAD_PAIRS[j];
            return (
              <ProbBar
                key={j}
                label={`${species[a]} + ${species[b]}`}
                value={marginals.lead_pair_probs[j]}
                maxValue={pairMax}
                color="#C5D86D"
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
