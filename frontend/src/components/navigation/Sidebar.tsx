import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Sparkles,
  Layers,
  FolderLock,
  UserCheck,
  ShieldAlert,
  X,
} from 'lucide-react';
import { useApp } from '../../context/AppContext';

interface SidebarProps {
  onClose?: () => void;
  className?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ onClose, className = '' }) => {
  const { profile, activeAnalysis } = useApp();

  const navItems = [
    {
      label: 'Dashboard',
      to: '/',
      icon: <LayoutDashboard className="w-4 h-4 mr-3 flex-shrink-0" />,
    },
    {
      label: 'New Analysis',
      to: '/analysis/new',
      icon: <Sparkles className="w-4 h-4 mr-3 flex-shrink-0" />,
    },
    {
      label: 'Research Vault',
      to: '/documents',
      icon: <FolderLock className="w-4 h-4 mr-3 flex-shrink-0" />,
    },
    {
      label: 'Multi-Agent Analysis',
      to: '/multi-agent',
      icon: <Layers className="w-4 h-4 mr-3 flex-shrink-0" />,
      badge: activeAnalysis?.status === 'running' ? 'Active' : undefined,
    },
    {
      label: 'Investor Profile',
      to: '/profile',
      icon: <UserCheck className="w-4 h-4 mr-3 flex-shrink-0" />,
    },
  ];

  return (
    <aside
      className={`w-64 bg-slate-900 text-slate-300 flex flex-col border-r border-slate-800 flex-shrink-0 min-h-screen ${className}`}
    >
      {/* Brand Header */}
      <div className="h-16 flex items-center justify-between px-6 border-b border-slate-800 bg-slate-950/50">
        <div className="flex items-center">
          <div className="w-8 h-8 rounded-lg bg-emerald-500 flex items-center justify-center text-slate-950 font-black text-lg tracking-wider mr-3 shadow-sm shadow-emerald-500/20">
            FP
          </div>
          <div>
            <div className="text-base font-bold text-white tracking-tight flex items-center">
              FinPilot
              <span className="ml-2 text-[10px] font-semibold px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                v0.1
              </span>
            </div>
            <p className="text-[11px] text-slate-400">AI Financial Research</p>
          </div>
        </div>

        {/* Mobile close button */}
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close navigation"
            className="md:hidden text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation List */}
      <nav aria-label="Main Navigation" className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          Platform
        </div>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            onClick={onClose}
            className={({ isActive }) =>
              `flex items-center justify-between px-3 py-2 text-xs font-medium rounded-lg transition-colors ${
                isActive
                  ? 'bg-slate-800 text-white shadow-sm border border-slate-700/60 font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
              }`
            }
          >
            <div className="flex items-center">
              {item.icon}
              <span>{item.label}</span>
            </div>
            {item.badge && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full font-bold bg-sky-950 text-sky-400 border border-sky-800 animate-pulse">
                {item.badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Active Profile Snippet */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/40 m-3 rounded-xl">
        <div className="flex items-center justify-between text-[11px] font-semibold text-slate-400 mb-1.5">
          <span>INVESTOR CONSTRAINTS</span>
          <span className="capitalize text-emerald-400 text-[10px]">
            {profile.risk_tolerance || 'Moderate'}
          </span>
        </div>
        <div className="text-xs text-slate-300 font-medium truncate">
          {profile.time_horizon || '3-5 years'}
        </div>
        <div className="text-[11px] text-slate-400 mt-0.5">
          Capital: ${Number(profile.capital_amount || 50000).toLocaleString()}
        </div>
      </div>

      {/* Regulatory Badge */}
      <div className="px-4 py-3 border-t border-slate-800/80 text-[10px] text-slate-400 flex items-center bg-slate-950/20">
        <ShieldAlert className="w-3.5 h-3.5 text-amber-500 mr-2 flex-shrink-0" />
        <span>Decision support only. Non-advisory.</span>
      </div>
    </aside>
  );
};
