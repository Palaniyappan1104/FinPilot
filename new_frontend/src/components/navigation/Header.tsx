import React from 'react';
import { Menu, Activity, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';

interface HeaderProps {
  onToggleMobileMenu: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onToggleMobileMenu }) => {
  return (
    <header className="h-16 border-b border-slate-200 bg-white/95 backdrop-blur px-4 sm:px-6 flex items-center justify-between sticky top-0 z-30 shadow-xs">
      <div className="flex items-center space-x-3">
        <button
          type="button"
          onClick={onToggleMobileMenu}
          className="md:hidden p-2 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          aria-label="Open mobile menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-2 text-xs">
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="text-slate-500 hidden sm:inline">Engine:</span>
          <span className="text-slate-800 font-mono font-semibold">Groq + yfinance + 5 Agents</span>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs">
          <Activity className="w-3.5 h-3.5 text-emerald-600" />
          <span className="text-[11px] font-medium hidden sm:inline">FastAPI Demo Engine:</span>
          <span className="text-[11px] text-emerald-600 font-bold">Online</span>
        </div>

        <Link
          to="/analysis/new"
          className="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors shadow-xs shadow-emerald-600/20"
        >
          + New Analysis
        </Link>
      </div>
    </header>
  );
};
