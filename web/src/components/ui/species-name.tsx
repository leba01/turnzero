import { displayName } from '@/lib/sprites';

/**
 * Renders a species name with smart wrapping: the form suffix "(Rapid Strike)"
 * stays on one line, but a break is allowed between the base name and the suffix.
 */
export function SpeciesName({ species }: { species: string }) {
  const name = displayName(species);
  const idx = name.indexOf(' (');
  if (idx === -1) return <>{name}</>;
  return (
    <>
      {name.slice(0, idx)}{' '}
      <span className="whitespace-nowrap">{name.slice(idx + 1)}</span>
    </>
  );
}
