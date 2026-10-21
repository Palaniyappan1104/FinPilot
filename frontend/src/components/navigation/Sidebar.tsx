import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  Sparkles,
  Layers,
  FolderLock,
  UserCheck,
  ShieldAlert,
  X,
  LogOut,
} from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { useAuth } from '../../context/AuthContext';

interface SidebarProps {
  onClose?: () => void;
  className?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ onClose, className = '' }) => {
  const { profile, activeAnalysis } = useApp();
  const { signOut, user } = useAuth();
  const navigate = useNavigate();

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

  const handleSignOut = async () => {
    await signOut();
    navigate('/login', { replace: true });
  };

  return (
    <aside
      className={`w-64 flex flex-col flex-shrink-0 min-h-screen ${className}`}
      style={{
        background: '#ffffff',
        borderRight: '1.5px solid #e2e8f0',
      }}
    >
      {/* Brand Header */}
      <div
        className="h-16 flex items-center justify-between px-5"
        style={{ borderBottom: '1.5px solid #e2e8f0' }}
      >
        <div className="flex items-center">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center font-black text-base text-white mr-3 flex-shrink-0"
            style={{ background: '#1e40af' }}
          >
            FP
          </div>
          <div>
            <div
              className="text-base font-bold tracking-tight flex items-center"
              style={{ color: '#0f172a' }}
            >
              FinPilot
              <span
                className="ml-2 text-[10px] font-semibold px-1.5 py-0.5 rounded"
                style={{
                  background: '#eff6ff',
                  color: '#1e40af',
                  border: '1px solid #bfdbfe',
                }}
              >
                v0.1
              </span>
            </div>
            <p className="text-[11px]" style={{ color: '#64748b' }}>
              AI Financial Research
            </p>
          </div>
        </div>

        {/* Mobile close button */}
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close navigation"
            className="md:hidden p-1 rounded-lg transition-colors"
            style={{ color: '#64748b' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#f1f5f9';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation List */}
      <nav aria-label="Main Navigation" className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div
          className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider"
          style={{ color: '#94a3b8' }}
        >
          Platform
        </div>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            onClick={onClose}
            className="flex items-center justify-between px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-150"
            style={({ isActive }) =>
              isActive
                ? {
                    background: '#eff6ff',
                    color: '#1e40af',
                    border: '1px solid #bfdbfe',
                    fontWeight: 600,
                  }
                : {
                    color: '#475569',
                    border: '1px solid transparent',
                  }
            }
            onMouseEnter={(e) => {
              const el = e.currentTarget as HTMLAnchorElement;
              if (!el.getAttribute('aria-current')) {
                el.style.background = '#f8fafc';
                el.style.color = '#1e3a8a';
              }
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget as HTMLAnchorElement;
              if (!el.getAttribute('aria-current')) {
                el.style.background = 'transparent';
                el.style.color = '#475569';
              }
            }}
          >
            <div className="flex items-center">
              {item.icon}
              <span>{item.label}</span>
            </div>
            {item.badge && (
              <span
                className="text-[10px] px-1.5 py-0.5 rounded-full font-bold animate-pulse"
                style={{
                  background: '#eff6ff',
                  color: '#1e40af',
                  border: '1px solid #bfdbfe',
                }}
              >
                {item.badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Active Profile Snippet */}
      <div
        className="mx-3 mb-3 p-3 rounded-xl"
        style={{
          background: '#f8fafc',
          border: '1.5px solid #e2e8f0',
        }}
      >
        <div
          className="flex items-center justify-between text-[11px] font-semibold mb-1.5"
          style={{ color: '#64748b' }}
        >
          <span>INVESTOR CONSTRAINTS</span>
          <span
            className="capitalize text-[10px] font-bold"
            style={{ color: '#1e40af' }}
          >
            {profile.risk_tolerance || 'Moderate'}
          </span>
        </div>
        <div className="text-xs font-medium truncate" style={{ color: '#1e293b' }}>
          {profile.time_horizon || '3-5 years'}
        </div>
        <div className="text-[11px] mt-0.5" style={{ color: '#64748b' }}>
          Capital: ${Number(profile.capital_amount || 50000).toLocaleString()}
        </div>
      </div>

      {/* Sign Out */}
      {user && (
        <div
          className="px-3 pb-3"
          style={{ borderTop: '1px solid #e2e8f0' }}
        >
          <div className="pt-3 px-1 mb-2">
            <p
              className="text-[11px] font-medium truncate"
              style={{ color: '#64748b' }}
            >
              {user.email}
            </p>
          </div>
          <button
            type="button"
            id="sidebar-sign-out-btn"
            onClick={handleSignOut}
            className="w-full flex items-center px-3 py-2 rounded-lg text-xs font-semibold transition-colors"
            style={{ color: '#dc2626' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#fef2f2';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <LogOut className="w-4 h-4 mr-2 flex-shrink-0" />
            Sign Out
          </button>
        </div>
      )}

      {/* Regulatory Badge */}
      <div
        className="px-4 py-3 text-[10px] flex items-center"
        style={{
          borderTop: '1px solid #e2e8f0',
          color: '#94a3b8',
        }}
      >
        <ShieldAlert className="w-3.5 h-3.5 mr-2 flex-shrink-0" style={{ color: '#f59e0b' }} />
        <span>Decision support only. Non-advisory.</span>
      </div>
    </aside>
  );
};
