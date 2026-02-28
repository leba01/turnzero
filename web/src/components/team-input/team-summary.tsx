'use client';

import { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import { Popover as PopoverPrimitive } from 'radix-ui';
import { spriteUrl } from '@/lib/sprites';
import { SpeciesName } from '@/components/ui/species-name';
import { Button } from '@/components/ui/button';
import { getSpeciesTypes } from '@/lib/data/pokemon-data';
import { TYPE_COLORS } from '@/lib/data/type-colors';
import type { TeamSheet, Pokemon } from '@/types/pokemon';

function TypeBadge({ type }: { type: string }) {
  const cls = TYPE_COLORS[type] ?? 'bg-night/20 text-night';
  return (
    <span className={`inline-block rounded px-1.5 py-px text-[9px] font-bold leading-none tracking-wide ${cls}`}>
      {type.toUpperCase()}
    </span>
  );
}

interface MonPopupProps {
  mon: Pokemon;
  baseTypes: string[] | null;
}

function MonPopup({ mon, baseTypes }: MonPopupProps) {
  const val = (v: string) =>
    !v || v === 'UNK' || v === '<UNK>' ? '—' : v;

  const teraVal = val(mon.tera_type);

  const rows: [string, string][] = [
    ['ITEM', val(mon.item)],
    ['ABIL', val(mon.ability)],
  ];

  return (
    <div className="w-44 rounded border-2 border-night bg-white px-2.5 py-2
                    font-[family-name:var(--font-body)] text-[10px] sm:text-[11px]
                    text-night shadow-[2px_2px_0px_#3D5A80]">
      {/* Species name */}
      <div className="font-[family-name:var(--font-label)] text-[11px] sm:text-[12px] font-bold leading-tight mb-1">
        <SpeciesName species={mon.species} />
      </div>

      {/* Base types */}
      {baseTypes && baseTypes.length > 0 && (
        <div className="flex gap-1 mb-1">
          {baseTypes.map(t => <TypeBadge key={t} type={t} />)}
        </div>
      )}

      {/* Tera type */}
      {teraVal !== '—' && (
        <div className="mb-1.5">
          <span className={`inline-block rounded px-1.5 py-px text-[9px] font-bold leading-none tracking-wide
                            ${TYPE_COLORS[teraVal] ?? 'bg-night/20 text-night'}`}>
            TERA&nbsp;{teraVal.toUpperCase()}
          </span>
        </div>
      )}

      <div className="mb-1 border-t border-night/20" />

      {rows.map(([label, value]) => (
        <div key={label} className="flex gap-1.5 leading-snug">
          <span className="shrink-0 font-bold opacity-60 w-8">{label}</span>
          <span className="truncate">{value}</span>
        </div>
      ))}

      <div className="my-1 border-t border-night/20" />

      {mon.moves.map((move, j) => (
        <div key={j} className="flex items-center gap-1 leading-snug">
          <span className="text-steel">◆</span>
          <span className="truncate">{val(move)}</span>
        </div>
      ))}
    </div>
  );
}

interface TeamSummaryProps {
  team: TeamSheet;
  onEdit: () => void;
}

export function TeamSummary({ team, onEdit }: TeamSummaryProps) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [speciesTypes, setSpeciesTypes] = useState<Record<string, string[]> | null>(null);

  useEffect(() => {
    getSpeciesTypes().then(setSpeciesTypes).catch(() => {});
  }, []);

  const openCard = (i: number) => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    setOpenIdx(i);
  };
  const scheduleClose = () => {
    closeTimer.current = setTimeout(() => setOpenIdx(null), 150);
  };

  return (
    <div className="flex items-center gap-2">
      <div className="grid flex-1 grid-cols-3 gap-y-1">
        {team.pokemon.map((mon, i) => (
          <PopoverPrimitive.Root
            key={i}
            open={openIdx === i}
            onOpenChange={(open) => { if (!open) setOpenIdx(null); }}
          >
            <PopoverPrimitive.Trigger asChild>
              <div
                className="flex flex-col items-center cursor-pointer select-none"
                onMouseEnter={() => openCard(i)}
                onMouseLeave={scheduleClose}
              >
                <Image
                  src={spriteUrl(mon.species)}
                  alt={mon.species}
                  width={48}
                  height={48}
                  className="object-contain"
                  unoptimized
                />
                <span className="text-center font-[family-name:var(--font-label)] text-[9px] sm:text-[11px] text-night">
                  <SpeciesName species={mon.species} />
                </span>
              </div>
            </PopoverPrimitive.Trigger>
            <PopoverPrimitive.Portal>
              <PopoverPrimitive.Content
                side="top"
                sideOffset={6}
                className="z-50"
                onMouseEnter={() => openCard(i)}
                onMouseLeave={scheduleClose}
              >
                <MonPopup
                  mon={mon}
                  baseTypes={speciesTypes?.[mon.species] ?? null}
                />
              </PopoverPrimitive.Content>
            </PopoverPrimitive.Portal>
          </PopoverPrimitive.Root>
        ))}
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={onEdit}
        className="shrink-0 border-night text-night hover:bg-mist"
      >
        EDIT
      </Button>
    </div>
  );
}
