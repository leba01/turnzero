'use client';

import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import type { FeatureSensitivity } from '@/types/pokemon';

interface SensitivityDisplayProps {
  sensitivity: FeatureSensitivity | null;
}

const FIELD_LABELS: { key: keyof FeatureSensitivity; label: string }[] = [
  { key: 'species', label: 'Species' },
  { key: 'items', label: 'Items' },
  { key: 'ability', label: 'Ability' },
  { key: 'tera', label: 'Tera' },
  { key: 'moves', label: 'Moves' },
];

export function SensitivityDisplay({ sensitivity }: SensitivityDisplayProps) {
  if (!sensitivity) {
    return (
      <div className="flex flex-col gap-2">
        {FIELD_LABELS.map(({ label }) => (
          <div key={label} className="flex items-center gap-2">
            <span className="w-16 font-[family-name:var(--font-label)] text-[10px] text-rock">
              {label}
            </span>
            <Skeleton className="h-4 flex-1" />
          </div>
        ))}
        <span className="font-[family-name:var(--font-body)] text-[10px] text-rock">
          Computing sensitivity...
        </span>
      </div>
    );
  }

  const entries = FIELD_LABELS.map(({ key, label }) => ({
    label,
    kl: sensitivity[key],
  })).sort((a, b) => b.kl - a.kl);

  const maxKl = entries[0]?.kl || 1;

  return (
    <div className="flex flex-col gap-2">
      {entries.map(({ label, kl }, i) => {
        const pct = maxKl > 0 ? (kl / maxKl) * 100 : 0;
        const isTop = i === 0;
        return (
          <div key={label} className="flex items-center gap-2">
            <span className="w-16 shrink-0 font-[family-name:var(--font-label)] text-[10px] text-night">
              {label}
            </span>
            <div className="h-4 flex-1 border border-night bg-muted">
              <div
                className="h-full"
                style={{
                  width: `${pct}%`,
                  backgroundColor: isTop ? '#F4845F' : '#98C1D9',
                }}
              />
            </div>
            <span className="w-16 shrink-0 text-right font-[family-name:var(--font-body)] text-[10px] text-night">
              +{kl.toFixed(3)} KL
            </span>
            {isTop && (
              <Badge className="bg-jam font-[family-name:var(--font-label)] text-[8px] text-white">
                TOP
              </Badge>
            )}
          </div>
        );
      })}
    </div>
  );
}
