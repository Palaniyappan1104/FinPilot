import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-teal-600 flex items-center justify-center text-white font-bold text-lg">
              F
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight text-slate-900">FinPilot</span>
              <span className="ml-2 text-xs font-semibold px-2 py-0.5 rounded bg-teal-100 text-teal-800">
                Phase 1 Foundation
              </span>
            </div>
          </div>
          <div className="text-xs text-slate-500 hidden sm:block">
            Multi-Agent AI Platform for Financial Research
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      <footer className="bg-white border-t border-slate-200 py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs text-slate-500">
          FinPilot &copy; 2026. Designed for research and decision-support. Not a trading bot.
        </div>
      </footer>
    </div>
  );
};
