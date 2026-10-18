import React from 'react';
import { ShieldAlert } from 'lucide-react';

export const FooterDisclaimer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950 py-4 px-6 text-slate-500 text-[11px] leading-relaxed">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-center sm:text-left">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-500/80 flex-shrink-0" />
          <span>
            FinPilot is an autonomous multi-agent research demonstration tool. All materials generated are for informational and academic review purposes only and do not constitute financial, legal, or investment advice.
          </span>
        </div>
        <div className="text-slate-500 font-mono text-[10px] whitespace-nowrap">
          Deterministic Financial Sourcing Layer &copy; 2026
        </div>
      </div>
    </footer>
  );
};
