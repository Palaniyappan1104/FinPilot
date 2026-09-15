import React from 'react';
import { useLocation } from 'react-router-dom';
import { ShieldCheck, Info, BarChart2, Menu } from 'lucide-react';
import { useApp } from '../../context/AppContext';

interface HeaderProps {
  onToggleMobileMenu?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onToggleMobileMenu }) => {
  const location = useLocation();
  const { selectedCompany, activeAnalysis } = useApp();

  const getPageTitle = (path: string) => {
    if (path === '/') return 'System Dashboard';
    if (path.startsWith('/analysis/new')) return 'New Investment Research';
    if (path.includes('/progress')) return 'Multi-Agent Pipeline Monitor';
    if (path.startsWith('/multi-agent')) return 'Multi-Agent Architecture';
    if (path.startsWith('/reports')) return 'Investment Research Reports';
    if (path.startsWith('/documents')) return 'Research Vault (SEC & Filings)';
    if (path.startsWith('/profile')) return 'Investor Profile & Constraints';
    return 'Research Platform';
  };

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-4 sm:px-6 flex items-center justify-between sticky top-0 z-20">
      <div className="flex items-center space-x-3">
        {onToggleMobileMenu && (
          <button
            type="button"
            onClick={onToggleMobileMenu}
            aria-label="Open navigation menu"
            className="md:hidden text-slate-600 hover:text-slate-900 p-1.5 rounded-lg hover:bg-slate-100 transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        <h1 className="text-sm sm:text-base font-bold text-slate-900 truncate max-w-[200px] sm:max-w-none">
          {getPageTitle(location.pathname)}
        </h1>

        {activeAnalysis && activeAnalysis.status === 'running' && (
          <div className="hidden sm:flex items-center text-xs font-semibold px-2.5 py-1 rounded-full bg-sky-50 text-sky-700 border border-sky-200 animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-sky-500 mr-2" />
            Analyzing {activeAnalysis.ticker}
          </div>
        )}
      </div>

      <div className="flex items-center space-x-2 sm:space-x-4">
        {selectedCompany && (
          <div className="hidden lg:flex items-center space-x-2 text-xs text-slate-600 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg">
            <BarChart2 className="w-3.5 h-3.5 text-slate-400" />
            <span className="font-semibold text-slate-800">
              {selectedCompany.ticker}
            </span>
            <span className="text-slate-400">|</span>
            {selectedCompany.price > 0 ? (
              <>
                <span>
                  {selectedCompany.currency === 'INR' ? '₹' : '$'}
                  {selectedCompany.price.toLocaleString()}
                </span>
                <span
                  className={`font-semibold ${
                    selectedCompany.change >= 0
                      ? 'text-emerald-600'
                      : 'text-rose-600'
                  }`}
                >
                  {selectedCompany.change >= 0 ? '+' : ''}
                  {selectedCompany.changePercent}%
                </span>
              </>
            ) : (
              <span className="text-slate-400 text-[11px]">
                Market data unavailable
              </span>
            )}
          </div>
        )}

        <div className="flex items-center text-[10px] sm:text-[11px] font-medium text-slate-600 bg-slate-100 border border-slate-200 px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-md">
          <Info className="w-3.5 h-3.5 text-slate-400 mr-1" />
          <span>FinPilot Platform</span>
        </div>

        <div className="hidden sm:flex items-center text-[11px] font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-md">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 mr-1.5" />
          <span>Guardrails Active</span>
        </div>
      </div>
    </header>
  );
};
