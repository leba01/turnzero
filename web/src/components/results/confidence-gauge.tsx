'use client';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface ConfidenceGaugeProps {
  confidence: number;
  entropy: number;
  mutual_information: number;
  abstain: boolean;
}

export function ConfidenceGauge({
  confidence,
  entropy,
  mutual_information,
  abstain,
}: ConfidenceGaugeProps) {
  const pct = Math.min(confidence * 100, 100);

  return (
    <div className="flex flex-col gap-2">
      {/* Bar */}
      <div className="h-5 w-full border-2 border-night bg-muted">
        <div
          className={cn(
            'h-full transition-all duration-300',
            abstain ? 'bg-red-400' : 'bg-jam',
          )}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Stats */}
      <div className="flex flex-wrap gap-2">
        <Badge
          variant="outline"
          className={cn(
            'font-[family-name:var(--font-label)] text-[10px]',
            abstain && 'border-red-400 text-red-600',
          )}
        >
          CONF: {(confidence * 100).toFixed(2)}%
        </Badge>
        <Badge variant="outline" className="font-[family-name:var(--font-label)] text-[10px]">
          ENT: {entropy.toFixed(3)}
        </Badge>
        <Badge variant="outline" className="font-[family-name:var(--font-label)] text-[10px]">
          MI: {mutual_information.toFixed(4)}
        </Badge>
        {abstain && (
          <Badge className="animate-pulse bg-red-500 font-[family-name:var(--font-label)] text-[10px] text-white">
            ABSTAIN
          </Badge>
        )}
      </div>
    </div>
  );
}
