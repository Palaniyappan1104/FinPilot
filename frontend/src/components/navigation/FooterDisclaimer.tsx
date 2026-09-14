import React from 'react';
import { ShieldAlert } from 'lucide-react';
import { STANDARD_REGULATORY_DISCLAIMER } from '../../services/mockData';

export const FooterDisclaimer: React.FC = () => {
  return (
    <footer
      data-testid="non-dismissible-disclaimer"
      className="bg-slate-900 text-slate-400 border-t border-slate-800 py-3 px-6 text-xs"
    >
      <div className="max-w-7xl mx-auto flex items-start space-x-3">
        <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-[11px] leading-relaxed text-slate-300">
            <span className="font-semibold text-slate-200 uppercase tracking-wider mr-1">
              Regulatory Disclosure:
            </span>
            {STANDARD_REGULATORY_DISCLAIMER}
          </p>
          <div className="mt-1 flex items-center justify-between text-[10px] text-slate-400">
            <span>FinPilot Multi-Agent Architecture © 2026. All rights reserved.</span>
            <span>Deterministic Research Decision Support • Not an automated trading system</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
