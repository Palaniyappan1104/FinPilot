import React from 'react';
import { AlertTriangle, ShieldAlert } from 'lucide-react';

interface CriticalRisksSectionProps {
  risks: string[];
  className?: string;
}

export const CriticalRisksSection: React.FC<CriticalRisksSectionProps> = ({
  risks,
  className = '',
}) => {
  if (!risks || risks.length === 0) return null;

  return (
    <div
      data-testid="critical-risks-section"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-4 ${className}`}
    >
      <div className="flex items-center space-x-2.5 pb-3 border-b border-slate-100">
        <div className="p-2 rounded-lg bg-rose-50 text-rose-600 border border-rose-100">
          <ShieldAlert className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-base font-bold text-slate-900">
            Critical Risk Factors & Adverse Scenarios
          </h3>
          <p className="text-xs text-slate-500">
            Top investment risks that could invalidate the positive thesis
          </p>
        </div>
      </div>

      <div className="space-y-2.5">
        {risks.map((risk, index) => (
          <div
            key={index}
            className="p-3.5 rounded-lg border border-rose-100 bg-rose-50/30 flex items-start space-x-3 text-xs"
          >
            <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-bold text-slate-800 mr-2">
                Risk Factor {index + 1}:
              </span>
              <span className="text-slate-700 leading-relaxed">{risk}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
