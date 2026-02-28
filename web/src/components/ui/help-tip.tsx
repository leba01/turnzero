'use client';

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface HelpTipProps {
  text: string;
}

export function HelpTip({ text }: HelpTipProps) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className="ml-1.5 inline-flex size-4 items-center justify-center rounded-full border border-rock/40 font-[family-name:var(--font-label)] text-[8px] sm:text-[10px] text-rock transition-colors hover:border-jam hover:bg-jam/10 hover:text-jam"
          >
            ?
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          sideOffset={4}
          className="max-w-56 border-2 border-night bg-white px-3 py-2 font-[family-name:var(--font-body)] text-[11px] sm:text-[13px] leading-relaxed text-night shadow-[2px_2px_0px_#3D5A80]"
        >
          {text}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
