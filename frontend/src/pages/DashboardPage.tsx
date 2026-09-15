import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  TrendingUp,
  Award,
  FolderLock,
  ArrowRight,
  ShieldCheck,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { StatCard } from '../components/common/StatCard';
import { RecentAnalysesTable } from '../components/dashboard/RecentAnalysesTable';
import { WatchlistQuickStart } from '../components/dashboard/WatchlistQuickStart';
import { AnalysisSummary, CompanyInfo } from '../types';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { recentAnalyses, setSelectedCompany, documents } = useApp();

  const handleSelectQuickCompany = (comp: CompanyInfo) => {
    setSelectedCompany(comp);
    navigate(`/analysis/new?ticker=${comp.ticker}`);
  };

  const completedAnalyses = recentAnalyses.filter(
    (a: AnalysisSummary) => a.status === 'completed',
  );
  const completedCount = completedAnalyses.length;

  const validConfidenceItems = completedAnalyses.filter((a) => a.confidence > 0);
  const avgConfidence =
    validConfidenceItems.length > 0
      ? `${(
          (validConfidenceItems.reduce((acc, curr) => acc + curr.confidence, 0) /
            validConfidenceItems.length) *
          100
        ).toFixed(1)}%`
      : 'Not available yet';

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

          {/* Primary Action Button */}
          <div className="pt-3 flex items-center space-x-4">
            <button
              type="button"
              data-testid="hero-start-analysis-btn"
              onClick={() => navigate('/analysis/new')}
              className="px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs sm:text-sm flex items-center space-x-2 transition-all shadow-md hover:shadow-lg shadow-emerald-950/40"
            >
              <Sparkles className="w-4 h-4 mr-1" />
              <span>Start New Analysis</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Analyses Completed"
          value={completedCount}
          change={completedCount > 0 ? `${completedCount} total` : undefined}
          isPositive={completedCount > 0}
          icon={<TrendingUp className="w-5 h-5" />}
          subtitle="Multi-agent dossiers generated"
        />
        <StatCard
          label="Avg Agent Confidence"
          value={avgConfidence}
          change={validConfidenceItems.length > 0 ? 'Empirical aggregate' : undefined}
          isPositive={validConfidenceItems.length > 0}
          icon={<Award className="w-5 h-5 text-emerald-600" />}
          subtitle="Across specialist domains"
        />
        <StatCard
          label="Indexed Vault Filings"
          value={documents.length}
          change={documents.length > 0 ? 'Grounding Active' : 'Vault Ready'}
          isPositive={documents.length > 0}
          icon={<FolderLock className="w-5 h-5" />}
          subtitle="ChromaDB vector embeddings"
        />
        <StatCard
          label="Safety Guardrails"
          value="Enforced"
          change="Non-Advisory"
          isPositive={true}
          icon={<ShieldCheck className="w-5 h-5 text-emerald-600" />}
          subtitle="No fabricated figures or trading bot advice"
        />
      </div>

      {/* Watchlist Quick Launch */}
      <WatchlistQuickStart onSelectCompany={handleSelectQuickCompany} />

      {/* Recent Analyses Summary Table */}
      <RecentAnalysesTable analyses={recentAnalyses} />
    </div>
  );
};
