import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  TrendingUp,
  Award,
  FolderLock,
  ArrowRight,
  ShieldCheck,
  Layers,
  FileText,
  Clock,
} from 'lucide-react';
import { useApp } from '../context/AppContext';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { storedReports } = useApp();

  const completedCount = storedReports.length;

  const quickWatchlist = [
    { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Consumer Electronics' },
    { ticker: 'NVDA', name: 'NVIDIA Corp.', sector: 'Semiconductors & AI' },
    { ticker: 'MSFT', name: 'Microsoft Corp.', sector: 'Cloud & Enterprise' },
    { ticker: 'TSLA', name: 'Tesla, Inc.', sector: 'Automotive & Energy' },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Banner */}
      <div className="bg-gradient-to-br from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 rounded-2xl p-8 sm:p-10 shadow-2xl relative overflow-hidden">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 text-xs font-semibold px-3 py-1 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/80">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Autonomous Multi-Agent Financial Research</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight leading-tight">
            Institutional Research & Investment Decision Support
          </h1>

          <p className="text-sm text-slate-300 leading-relaxed max-w-2xl">
            FinPilot coordinates specialist AI agents across Fundamental, Technical, News Sentiment, and Risk dimensions to synthesize objective, evidence-grounded dossiers tailored to your investor mandate.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => navigate('/analysis/new')}
              className="px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs sm:text-sm flex items-center space-x-2 transition-all shadow-lg shadow-emerald-950/50"
            >
              <Sparkles className="w-4 h-4" />
              <span>Start New Analysis</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              type="button"
              onClick={() => navigate('/multi-agent')}
              className="px-5 py-3 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-200 border border-slate-700 font-semibold text-xs sm:text-sm flex items-center space-x-2 transition-colors"
            >
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>Multi-Agent Specialist Flow</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Reports Generated</span>
            <FileText className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-black text-white">{completedCount}</div>
          <p className="text-[11px] text-slate-500">Autonomous research notes compiled</p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Active Specialists</span>
            <Layers className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-black text-white">5 Agents</div>
          <p className="text-[11px] text-slate-500">Fundamental, Technical, News, Risk & Synthesis</p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Market Data Engine</span>
            <TrendingUp className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-black text-white">yfinance Verified</div>
          <p className="text-[11px] text-slate-500">Prices, 252-day history, indicators & ratios</p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Research Vault</span>
            <FolderLock className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-black text-white">Ready</div>
          <p className="text-[11px] text-slate-500">Filing repository & SEC Edgar integration</p>
        </div>
      </div>

      {/* Quick Start Watchlist */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white">Quick Research Launchpad</h2>
            <p className="text-xs text-slate-400">Instant multi-agent analysis for leading market equities</p>
          </div>
          <span className="text-xs text-slate-500">Click to research</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {quickWatchlist.map((item) => (
            <button
              key={item.ticker}
              type="button"
              onClick={() => navigate(`/analysis/new?ticker=${item.ticker}`)}
              className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-emerald-500 hover:bg-slate-850 text-left transition-all group"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono font-bold text-emerald-400 text-sm">{item.ticker}</span>
                <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-emerald-400 transition-colors" />
              </div>
              <div className="text-xs font-semibold text-slate-200 mt-1">{item.name}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">{item.sector}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Recent Analyses Overview */}
      {storedReports.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-white">Recent Research Sessions</h2>
            <button
              type="button"
              onClick={() => navigate('/reports')}
              className="text-xs text-emerald-400 hover:underline font-semibold"
            >
              View all reports ({storedReports.length}) →
            </button>
          </div>

          <div className="divide-y divide-slate-800/80">
            {storedReports.slice(0, 3).map((rep) => (
              <div
                key={rep.id}
                className="py-3 flex items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-center space-x-3">
                  <span className="font-mono font-bold text-emerald-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                    {rep.ticker}
                  </span>
                  <span className="font-semibold text-white truncate">{rep.company}</span>
                </div>
                <div className="text-slate-400 truncate flex-1 hidden md:block">
                  {rep.executive_summary}
                </div>
                <button
                  type="button"
                  onClick={() => navigate(`/reports/${rep.id}`)}
                  className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs"
                >
                  View Dossier
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
