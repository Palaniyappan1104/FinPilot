import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Sparkles,
  Layers,
  FileText,
  FolderLock,
  UserCheck,
  ShieldAlert,
  LogOut,
  X,
} from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { useAuth } from '../../context/AuthContext';

interface SidebarProps {
  onClose?: () => void;
  className?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ onClose, className = '' }) => {
  const { profile, storedReports } = useApp();
  const { user, signOut } = useAuth();

  const handleSignOut = async () => {
    try {
      await signOut();
    } catch (err) {
      console.error('Failed to sign out:', err);
    }
  };

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
      label: 'Multi-Agent Analysis',
      to: '/multi-agent',
      icon: <Layers className="w-4 h-4 mr-3 flex-shrink-0" />,
    },
    {
      label: 'Reports',
      to: '/reports',
      icon: <FileText className="w-4 h-4 mr-3 flex-shrink-0" />,
      badge: storedReports.length > 0 ? String(storedReports.length) : undefined,
    },
    {
      label: 'Research Vault',
      to: '/documents',
      icon: <FolderLock className="w-4 h-4 mr-3 flex-shrink-0" />,
    },
    {
      label: 'Investor Profile',
      to: '/profile',
      icon: <UserCheck className="w-4 h-4 mr-3 flex-shrink-0" />,
    },
  ];

  return (
    <aside
      className={`w-64 bg-white text-slate-700 flex flex-col border-r border-slate-200 flex-shrink-0 min-h-screen select-none ${className}`}
    >
      {/* Brand Header */}
      <div className="h-16 flex items-center justify-between px-6 border-b border-slate-200 bg-white">
        <div className="flex items-center">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-black text-lg tracking-wider mr-3 shadow-sm">
            FP
          </div>
          <div>
            <div className="text-base font-bold text-slate-900 tracking-tight flex items-center">
              FinPilot
              <span className="ml-2 text-[10px] font-semibold px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
                Groq AI
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
            className="md:hidden text-slate-400 hover:text-slate-700 p-1 rounded-lg hover:bg-slate-100 transition-colors"
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
              `flex items-center justify-between px-3 py-2.5 text-xs font-medium rounded-lg transition-colors ${
                isActive
                  ? 'bg-blue-50 text-blue-700 shadow-xs border border-blue-200 font-semibold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              }`
            }
          >
            <div className="flex items-center">
              {item.icon}
              <span>{item.label}</span>
            </div>
            {item.badge && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full font-bold bg-blue-100 text-blue-800 border border-blue-200">
                {item.badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Active Profile Snippet */}
      <div className="p-4 border border-slate-200 bg-slate-50 m-3 rounded-xl shadow-xs">
        <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1.5">
          <span>INVESTOR CONSTRAINTS</span>
          <span className="capitalize text-blue-700 font-bold text-[10px]">
            {profile.risk_tolerance || 'Moderate'}
          </span>
        </div>
        <div className="text-xs text-slate-800 font-semibold truncate">
          {profile.time_horizon || '3-5 years'}
        </div>
        <div className="text-[11px] text-slate-500 mt-0.5">
          Capital: ${Number(profile.capital_amount || 50000).toLocaleString()}
        </div>
      </div>

      {/* User Session & Sign Out */}
      {user && (
        <div className="px-3 pb-3 border-t border-slate-200">
          <div className="pt-3 px-1 mb-2">
            <p className="text-[11px] font-medium text-slate-500 truncate">
              {user.email}
            </p>
          </div>
          <button
            type="button"
            id="sidebar-sign-out-btn"
            onClick={handleSignOut}
            className="w-full flex items-center px-3 py-2 rounded-lg text-xs font-semibold text-red-600 hover:bg-red-50 transition-colors"
          >
            <LogOut className="w-4 h-4 mr-2 flex-shrink-0" />
            Sign Out
          </button>
        </div>
      )}

      {/* Regulatory Badge */}
      <div className="px-4 py-3 border-t border-slate-200 text-[10px] text-slate-400 flex items-center bg-slate-50">
        <ShieldAlert className="w-3.5 h-3.5 text-amber-500 mr-2 flex-shrink-0" />
        <span>Decision support only. Non-advisory.</span>
      </div>
    </aside>
  );
};
