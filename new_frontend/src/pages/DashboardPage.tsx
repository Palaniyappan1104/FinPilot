import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  TrendingUp,
  FolderLock,
  ArrowRight,
  Layers,
  FileText,
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
      <div className="bg-gradient-to-br from-slate-900 via-slate-850 to-blue-950 border border-slate-800 rounded-2xl p-8 sm:p-10 shadow-lg relative overflow-hidden text-white">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 text-xs font-semibold px-3 py-1 rounded-full bg-blue-500/20 text-blue-300 border border-blue-400/30">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Autonomous Multi-Agent Financial Research</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight leading-tight">
            Institutional Research & Investment Decision Support
          </h1>

          <p className="text-sm text-slate-200 leading-relaxed max-w-2xl">
            FinPilot coordinates specialist AI agents across Fundamental, Technical, News Sentiment, and Risk dimensions to synthesize objective, evidence-grounded dossiers tailored to your investor mandate.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => navigate('/analysis/new')}
              className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs sm:text-sm flex items-center space-x-2 transition-all shadow-md shadow-blue-900/30"
            >
              <Sparkles className="w-4 h-4" />
              <span>Start New Analysis</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              type="button"
              onClick={() => navigate('/multi-agent')}
              className="px-5 py-3 rounded-xl bg-white/10 hover:bg-white/20 text-white border border-white/20 font-semibold text-xs sm:text-sm flex items-center space-x-2 transition-colors"
            >
              <Layers className="w-4 h-4 text-blue-300" />
              <span>Multi-Agent Specialist Flow</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Row (Clean Crisp White Cards) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Reports Generated</span>
            <FileText className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-2xl font-black text-slate-900">{completedCount}</div>
          <p className="text-[11px] text-slate-500">Autonomous research notes compiled</p>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Active Specialists</span>
            <Layers className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-2xl font-black text-slate-900">5 Agents</div>
          <p className="text-[11px] text-slate-500">Fundamental, Technical, News, Risk & Synthesis</p>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Market Data Engine</span>
            <TrendingUp className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-2xl font-black text-slate-900">yfinance Verified</div>
          <p className="text-[11px] text-slate-500">Prices, 252-day history, indicators & ratios</p>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Research Vault</span>
            <FolderLock className="w-4 h-4 text-purple-600" />
          </div>
          <div className="text-2xl font-black text-slate-900">Ready</div>
          <p className="text-[11px] text-slate-500">Filing repository & SEC Edgar integration</p>
        </div>
      </div>

      {/* Quick Start Watchlist */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-900">Quick Research Launchpad</h2>
            <p className="text-xs text-slate-500">Instant multi-agent analysis for leading market equities</p>
          </div>
          <span className="text-xs text-slate-400">Click to research</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {quickWatchlist.map((item) => (
            <button
              key={item.ticker}
              type="button"
              onClick={() => navigate(`/analysis/new?ticker=${item.ticker}`)}
              className="p-4 rounded-xl bg-slate-50 border border-slate-200 hover:border-blue-500 hover:bg-blue-50/50 text-left transition-all group"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono font-bold text-blue-700 text-sm">{item.ticker}</span>
                <ArrowRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-blue-600 transition-colors" />
              </div>
              <div className="text-xs font-bold text-slate-800 mt-1">{item.name}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">{item.sector}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Recent Analyses Overview */}
      {storedReports.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-slate-900">Recent Research Sessions</h2>
            <button
              type="button"
              onClick={() => navigate('/reports')}
              className="text-xs text-blue-600 hover:underline font-bold"
            >
              View all reports ({storedReports.length}) →
            </button>
          </div>

          <div className="divide-y divide-slate-100">
            {storedReports.slice(0, 3).map((rep) => (
              <div
                key={rep.id}
                className="py-3 flex items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-center space-x-3">
                  <span className="font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                    {rep.ticker}
                  </span>
                  <span className="font-bold text-slate-800 truncate">{rep.company}</span>
                </div>
                <div className="text-slate-500 truncate flex-1 hidden md:block">
                  {rep.executive_summary}
                </div>
                <button
                  type="button"
                  onClick={() => navigate(`/reports/${rep.id}`)}
                  className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold text-xs transition-colors"
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
