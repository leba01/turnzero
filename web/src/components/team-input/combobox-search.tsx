'use client';

import * as React from 'react';
import { Popover } from 'radix-ui';
import { ChevronsUpDownIcon, CheckIcon } from 'lucide-react';
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from '@/components/ui/command';
import { cn } from '@/lib/utils';

const MAX_DISPLAY = 50;

interface ComboboxSearchProps {
  value: string;
  onChange: (value: string) => void;
  options: string[];
  placeholder: string;
  className?: string;
}

export function ComboboxSearch({
  value,
  onChange,
  options,
  placeholder,
  className,
}: ComboboxSearchProps) {
  const [open, setOpen] = React.useState(false);
  const [search, setSearch] = React.useState('');

  const displayValue = value && value !== 'UNK' ? value : '';

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          role="combobox"
          aria-expanded={open}
          className={cn(
            'flex h-8 w-full items-center justify-between',
            'border-2 border-night bg-white px-2 py-1',
            'font-[family-name:var(--font-body)] text-xs text-night',
            'hover:border-jam focus:border-jam focus:outline-none',
            'transition-colors disabled:opacity-50',
            className,
          )}
        >
          <span className={cn('truncate', !displayValue && 'text-rock')}>
            {displayValue || placeholder}
          </span>
          <ChevronsUpDownIcon className="ml-1 size-3 shrink-0 text-rock" />
        </button>
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          className={cn(
            'z-50 w-[var(--radix-popover-trigger-width)] min-w-[200px] max-w-[calc(100vw-2rem)]',
            'border-2 border-night bg-white shadow-[4px_4px_0px_#3D5A80]',
            'data-[state=open]:animate-in data-[state=open]:fade-in-0',
            'data-[state=closed]:animate-out data-[state=closed]:fade-out-0',
          )}
          sideOffset={4}
          align="start"
        >
          <Command shouldFilter={true}>
            <CommandInput
              placeholder={`Search ${placeholder.toLowerCase()}...`}
              value={search}
              onValueChange={setSearch}
              className="text-xs"
            />
            <CommandList className="max-h-[200px]">
              <CommandEmpty className="py-3 text-center text-xs text-rock">
                No match found.
              </CommandEmpty>
              <CommandGroup>
                {options.slice(0, MAX_DISPLAY).map((option) => (
                  <CommandItem
                    key={option}
                    value={option}
                    onSelect={(selected) => {
                      onChange(selected === value ? 'UNK' : selected);
                      setOpen(false);
                      setSearch('');
                    }}
                    className="text-xs"
                  >
                    <CheckIcon
                      className={cn(
                        'mr-1 size-3',
                        value === option ? 'opacity-100' : 'opacity-0',
                      )}
                    />
                    {option}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
