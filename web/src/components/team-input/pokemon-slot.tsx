'use client';

import * as React from 'react';
import Image from 'next/image';
import { spriteUrl, displayName } from '@/lib/sprites';
import { ComboboxSearch } from './combobox-search';
import type { Pokemon } from '@/types/pokemon';

const UNK = 'UNK';

function emptyPokemon(): Pokemon {
  return { species: UNK, item: UNK, ability: UNK, tera_type: UNK, moves: [UNK, UNK, UNK, UNK] };
}

interface PokemonSlotProps {
  index: number;
  pokemon: Pokemon;
  onChange: (pokemon: Pokemon) => void;
  pokemonData: {
    species: string[];
    items: string[];
    abilities: string[];
    tera_types: string[];
    moves: string[];
  } | null;
}

export function PokemonSlot({ index, pokemon, onChange, pokemonData }: PokemonSlotProps) {
  const hasSpecies = pokemon.species && pokemon.species !== UNK;

  const handleSpeciesChange = (species: string) => {
    if (species !== pokemon.species) {
      onChange({ ...emptyPokemon(), species });
    }
  };

  const updateField = <K extends keyof Pokemon>(field: K, value: Pokemon[K]) => {
    onChange({ ...pokemon, [field]: value });
  };

  const updateMove = (moveIndex: number, value: string) => {
    const moves = [...pokemon.moves] as [string, string, string, string];
    moves[moveIndex] = value;
    onChange({ ...pokemon, moves });
  };

  return (
    <div className="flex gap-3 border-2 border-night bg-white p-3 shadow-[2px_2px_0px_#3D5A80]">
      {/* Sprite */}
      <div className="flex shrink-0 flex-col items-center gap-1">
        <div className="flex size-16 items-center justify-center border-2 border-night bg-muted sm:size-24">
          {hasSpecies ? (
            <Image
              src={spriteUrl(pokemon.species)}
              alt={pokemon.species}
              width={96}
              height={96}
              className="image-rendering-pixelated object-contain"
              unoptimized
            />
          ) : (
            <span className="font-[family-name:var(--font-label)] text-[10px] sm:text-sm text-rock">
              #{index + 1}
            </span>
          )}
        </div>
        {hasSpecies && (
          <span className="max-w-20 truncate text-center font-[family-name:var(--font-label)] text-[10px] sm:text-sm text-night sm:max-w-full">
            {displayName(pokemon.species)}
          </span>
        )}
      </div>

      {/* Fields */}
      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        {/* Species */}
        <FieldRow label="MON">
          <ComboboxSearch
            value={pokemon.species}
            onChange={handleSpeciesChange}
            options={pokemonData?.species ?? []}
            placeholder="Species"
          />
        </FieldRow>

        {/* Item + Ability row */}
        <div className="flex gap-2">
          <FieldRow label="ITEM" className="flex-1">
            <ComboboxSearch
              value={pokemon.item}
              onChange={(v) => updateField('item', v)}
              options={pokemonData?.items ?? []}
              placeholder="Item"
            />
          </FieldRow>
          <FieldRow label="ABIL" className="flex-1">
            <ComboboxSearch
              value={pokemon.ability}
              onChange={(v) => updateField('ability', v)}
              options={pokemonData?.abilities ?? []}
              placeholder="Ability"
            />
          </FieldRow>
        </div>

        {/* Tera */}
        <FieldRow label="TERA">
          <ComboboxSearch
            value={pokemon.tera_type}
            onChange={(v) => updateField('tera_type', v)}
            options={pokemonData?.tera_types ?? []}
            placeholder="Tera Type"
          />
        </FieldRow>

        {/* Moves */}
        <div className="grid grid-cols-2 gap-1.5">
          {pokemon.moves.map((move, i) => (
            <FieldRow key={i} label={`M${i + 1}`}>
              <ComboboxSearch
                value={move}
                onChange={(v) => updateMove(i, v)}
                options={pokemonData?.moves ?? []}
                placeholder={`Move ${i + 1}`}
              />
            </FieldRow>
          ))}
        </div>
      </div>
    </div>
  );
}

function FieldRow({
  label,
  children,
  className,
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={className}>
      <label className="mb-0.5 block font-[family-name:var(--font-label)] text-[10px] sm:text-sm uppercase tracking-wider text-rock">
        {label}
      </label>
      {children}
    </div>
  );
}
