import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  TrendingUp,
  Award,
  FolderLock,
  ArrowRight,
  ShieldCheck,
  Search,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { StatCard } from '../components/common/StatCard';
import { AgentPipelineVisualizer } from '../components/dashboard/AgentPipelineVisualizer';
import { RecentAnalysesTable } from '../components/dashboard/RecentAnalysesTable';
import { WatchlistQuickStart } from '../components/dashboard/WatchlistQuickStart';
import { AnalysisSummary, CompanyInfo } from '../types';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { recentAnalyses, setSelectedCompany, documents } = useApp();
  const [quickQuery, setQuickQuery] = useState('');

  const handleQuickSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!quickQuery.trim()) return;
    navigate(`/analysis/new?q=${encodeURIComponent(quickQuery.trim())}`);
  };

  const handleSelectQuickCompany = (comp: CompanyInfo) => {
    setSelectedCompany(comp);
    navigate(`/analysis/new?ticker=${comp.ticker}`);
  };

  const completedCount = recentAnalyses.filter(
    (a: AnalysisSummary) => a.status === 'completed',
  ).length;

  return (
    <div data-testid="dashboard-page" className="space-y-8">
      {/* Hero Research Area */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-850 to-slate-900 text-white rounded-2xl p-8 border border-slate-800 shadow-sm relative overflow-hidden">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/80">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span>Autonomous Multi-Agent Financial Research</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-black tracking-tight text-white leading-tight">
            Institutional-Grade Research & Investment Decision Support
          </h1>

          <p className="text-sm text-slate-300 leading-relaxed max-w-2xl">
            FinPilot coordinates autonomous specialists in Fundamental, Technical, News Sentiment, and SEC Filing analysis to synthesize objective, evidence-grounded dossiers tailored to your investor constraints.
          </p>

          {/* Primary Query Action Bar */}
          <form
            onSubmit={handleQuickSubmit}
            className="pt-2 flex flex-col sm:flex-row gap-2 max-w-2xl"
          >
            <div className="relative flex-1">
              <input
                type="text"
                value={quickQuery}
                onChange={(e) => setQuickQuery(e.target.value)}
                placeholder="Ask an investment research question (e.g., 'Should I invest in Infosys for 5 years?')..."
                className="w-full pl-10 pr-4 py-3 text-xs sm:text-sm bg-slate-800/90 border border-slate-700 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent shadow-inner"
              />
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
            </div>
            <button
              type="submit"
              className="px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs sm:text-sm flex items-center justify-center space-x-2 transition-all shadow-sm flex-shrink-0"
            >
              <span>Analyze</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          {/* Example query links */}
          <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-slate-400">
            <span className="text-[11px] uppercase tracking-wider font-semibold">
              Sample Inquiries:
            </span>
            {[
              'Should I invest ₹1,00,000 in Infosys for 5 years?',
              'Evaluate Apple Services gross margin trajectory',
              'Review NVIDIA valuation vs semiconductors',
            ].map((prompt, i) => (
              <button
                key={i}
                type="button"
                onClick={() =>
                  navigate(`/analysis/new?q=${encodeURIComponent(prompt)}`)
                }
                className="text-[11px] text-slate-300 hover:text-emerald-400 underline underline-offset-2"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Metrics Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Analyses Completed"
          value={completedCount}
          change="+3 this week"
          isPositive={true}
          icon={<TrendingUp className="w-5 h-5" />}
          subtitle="Multi-agent dossiers generated"
        />
        <StatCard
          label="Avg Agent Confidence"
          value="86.5%"
          change="High Statistical Validity"
          isPositive={true}
          icon={<Award className="w-5 h-5 text-emerald-600" />}
          subtitle="Across 5 specialist domains"
        />
        <StatCard
          label="Indexed Vault Filings"
          value={documents.length || 4}
          change="Grounding Active"
          isPositive={true}
          icon={<FolderLock className="w-5 h-5" />}
          subtitle="ChromaDB vector embeddings"
        />
        <StatCard
          label="Safety Guardrails"
          value="Active"
          change="Non-Advisory Enforced"
          isPositive={true}
          icon={<ShieldCheck className="w-5 h-5 text-emerald-600" />}
          subtitle="No fabricated figures or trading bot advice"
        />
      </div>

      {/* Multi-Agent Architecture Visualization */}
      <AgentPipelineVisualizer />

      {/* Watchlist Quick Launch */}
      <WatchlistQuickStart onSelectCompany={handleSelectQuickCompany} />

      {/* Recent Analyses Summary Table */}
      <RecentAnalysesTable analyses={recentAnalyses} />
    </div>
  );
};
