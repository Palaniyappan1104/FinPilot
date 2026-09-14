import React, { useEffect, useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { apiService } from '../../services/api';
import { CompanyInfo } from '../../types';

interface WatchlistQuickStartProps {
  onSelectCompany: (company: CompanyInfo) => void;
  className?: string;
}

export const WatchlistQuickStart: React.FC<WatchlistQuickStartProps> = ({
  onSelectCompany,
  className = '',
}) => {
  const [companies, setCompanies] = useState<CompanyInfo[]>([]);

  useEffect(() => {
    let active = true;
    apiService.searchCompanies('').then((data) => {
      if (active) {
        setCompanies(data.slice(0, 5));
      }
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div
      data-testid="watchlist-quick-start"
      className={`space-y-3 ${className}`}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Tracked Institutional Equities (Quick Start)
        </h3>
        <span className="text-[11px] text-slate-400">
          One-click research initiation
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {companies.map((comp) => {
          return (
            <div
              key={comp.ticker}
              data-testid={`watchlist-card-${comp.ticker}`}
              className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm hover:border-slate-300 transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-slate-900 px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200">
                    {comp.ticker}
                  </span>
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                    Tracked
                  </span>
                </div>

                <div className="mt-2">
                  <div className="text-xs font-bold text-slate-800 truncate">
                    {comp.name}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1 truncate">
                    {comp.sector}
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={() => onSelectCompany(comp)}
                className="mt-3 w-full py-1.5 px-2 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 text-[11px] font-semibold border border-slate-200 flex items-center justify-center transition-colors"
              >
                <span>Analyze {comp.ticker}</span>
                <ArrowRight className="w-3 h-3 ml-1 text-slate-400" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
