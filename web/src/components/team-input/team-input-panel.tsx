'use client';

import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PokemonSlot } from './pokemon-slot';
import { PasteInput } from './paste-input';
import type { Pokemon, TeamSheet } from '@/types/pokemon';

interface TeamInputPanelProps {
  label: string;
  team: TeamSheet;
  onChange: (team: TeamSheet) => void;
  pokemonData: {
    species: string[];
    items: string[];
    abilities: string[];
    tera_types: string[];
    moves: string[];
  } | null;
}

export function TeamInputPanel({ label, team, onChange, pokemonData }: TeamInputPanelProps) {
  const handlePokemonChange = (index: number, pokemon: Pokemon) => {
    const newPokemon = [...team.pokemon] as TeamSheet['pokemon'];
    newPokemon[index] = pokemon;
    onChange({ pokemon: newPokemon });
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="font-[family-name:var(--font-label)] text-sm uppercase tracking-wider text-night">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="builder">
          <TabsList className="mb-4 border-2 border-night bg-muted">
            <TabsTrigger value="builder">Builder</TabsTrigger>
            <TabsTrigger value="paste">Paste</TabsTrigger>
          </TabsList>

          <TabsContent value="builder">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {team.pokemon.map((mon, i) => (
                <PokemonSlot
                  key={i}
                  index={i}
                  pokemon={mon}
                  onChange={(p) => handlePokemonChange(i, p)}
                  pokemonData={pokemonData}
                />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="paste">
            <PasteInput onParsed={onChange} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
