'use client';

import { Skeleton } from '@/components/ui/skeleton';
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

const N = FIELD_LABELS.length; // 5 = pentagon
const PAD = 52; // padding for labels
const INNER = 160; // inner drawing area
const SIZE = INNER + PAD * 2; // full SVG viewBox
const CX = SIZE / 2;
const CY = SIZE / 2;
const R = 68; // max radius for the data polygon
const RINGS = 3; // concentric guide rings
const LABEL_R = R + 24; // radius for axis labels

/** Get (x, y) on pentagon for axis i at fractional radius r (0..1). */
function pentPoint(i: number, r: number): [number, number] {
  // Start from top (-π/2) and go clockwise
  const angle = (2 * Math.PI * i) / N - Math.PI / 2;
  return [CX + R * r * Math.cos(angle), CY + R * r * Math.sin(angle)];
}

/** Build an SVG polygon points string for a ring at fractional radius. */
function ringPoints(r: number): string {
  return Array.from({ length: N }, (_, i) => pentPoint(i, r).join(',')).join(' ');
}

/** Pentagon radar chart — pure SVG, no dependencies. */
function RadarPentagon({
  entries,
  maxKl,
}: {
  entries: { label: string; kl: number }[];
  maxKl: number;
}) {
  // Normalize values to 0..1
  const values = entries.map((e) => (maxKl > 0 ? e.kl / maxKl : 0));

  // Data polygon points
  const dataPoints = values.map((v, i) => pentPoint(i, v).join(',')).join(' ');

  // Find the top (highest) axis index
  const topIdx = values.indexOf(Math.max(...values));

  return (
    <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="mx-auto w-full max-w-[240px]">
      {/* Concentric guide rings */}
      {Array.from({ length: RINGS }, (_, ring) => {
        const r = (ring + 1) / RINGS;
        return (
          <polygon
            key={ring}
            points={ringPoints(r)}
            fill="none"
            stroke="#3D5A80"
            strokeWidth={ring === RINGS - 1 ? 1 : 0.5}
            opacity={0.15}
          />
        );
      })}

      {/* Axis lines from center to each vertex */}
      {Array.from({ length: N }, (_, i) => {
        const [x, y] = pentPoint(i, 1);
        return (
          <line
            key={i}
            x1={CX}
            y1={CY}
            x2={x}
            y2={y}
            stroke="#3D5A80"
            strokeWidth={0.5}
            opacity={0.15}
          />
        );
      })}

      {/* Filled data polygon */}
      <polygon
        points={dataPoints}
        fill="#98C1D9"
        fillOpacity={0.3}
        stroke="#98C1D9"
        strokeWidth={1.5}
      />

      {/* Data points as dots */}
      {values.map((v, i) => {
        const [x, y] = pentPoint(i, v);
        const isTop = i === topIdx;
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r={isTop ? 4 : 3}
            fill={isTop ? '#F4845F' : '#98C1D9'}
            stroke="white"
            strokeWidth={1.5}
          />
        );
      })}

      {/* Axis labels */}
      {entries.map((e, i) => {
        const angle = (2 * Math.PI * i) / N - Math.PI / 2;
        const lx = CX + LABEL_R * Math.cos(angle);
        const ly = CY + LABEL_R * Math.sin(angle);
        const isTop = i === topIdx;

        // Text anchor: left-ish for right side, right-ish for left side, middle for top/bottom
        let anchor: 'start' | 'middle' | 'end' = 'middle';
        if (Math.cos(angle) > 0.3) anchor = 'start';
        else if (Math.cos(angle) < -0.3) anchor = 'end';

        return (
          <text
            key={i}
            x={lx}
            y={ly}
            textAnchor={anchor}
            dominantBaseline="central"
            className="font-[family-name:var(--font-label)]"
            fontSize={10}
            fill={isTop ? '#F4845F' : '#3D5A80'}
            fontWeight={isTop ? 700 : 400}
          >
            {e.label}
          </text>
        );
      })}
    </svg>
  );
}

export function SensitivityDisplay({ sensitivity }: SensitivityDisplayProps) {
  if (!sensitivity) {
    return (
      <div className="flex flex-col gap-2">
        {FIELD_LABELS.map(({ label }) => (
          <div key={label} className="flex items-center gap-2">
            <span className="w-16 font-[family-name:var(--font-label)] text-[10px] sm:text-xs text-rock">
              {label}
            </span>
            <Skeleton className="h-4 flex-1" />
          </div>
        ))}
        <span className="font-[family-name:var(--font-body)] text-[10px] sm:text-xs text-rock">
          Computing sensitivity...
        </span>
      </div>
    );
  }

  // Keep original order for pentagon axes (don't sort — fixed positions)
  const entries = FIELD_LABELS.map(({ key, label }) => ({
    label,
    kl: sensitivity[key],
  }));

  const maxKl = Math.max(...entries.map((e) => e.kl)) || 1;
  const topEntry = entries.reduce((a, b) => (b.kl > a.kl ? b : a));

  return (
    <div className="flex flex-col items-center gap-3">
      <RadarPentagon entries={entries} maxKl={maxKl} />

      {/* Legend: compact row of values below the chart */}
      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1">
        {entries.map((e) => {
          const isTop = e === topEntry;
          return (
            <span
              key={e.label}
              className="font-[family-name:var(--font-body)] text-[10px] sm:text-xs"
              style={{ color: isTop ? '#F4845F' : '#8D8D8D' }}
            >
              <span className="font-[family-name:var(--font-label)] font-medium" style={{ color: isTop ? '#F4845F' : '#3D5A80' }}>
                {e.label}
              </span>
              {' '}
              +{e.kl.toFixed(3)}
            </span>
          );
        })}
      </div>
    </div>
  );
}
