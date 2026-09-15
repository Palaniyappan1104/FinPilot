import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layers,
  Sparkles,
  CheckCircle2,
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  ShieldAlert,
  FolderLock,
  ArrowRight,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { AgentPipelineVisualizer } from '../components/dashboard/AgentPipelineVisualizer';
import { SpecialistStatusGrid } from '../components/analysis/SpecialistStatusGrid';
import { AnalysisProgressBar } from '../components/analysis/AnalysisProgressBar';

export const MultiAgentPage: React.FC = () => {
  const navigate = useNavigate();
  const { activeAnalysis, activeReport } = useApp();

  const specialistSpecs = [
    {
      name: 'Technical Analyst',
      icon: <TrendingUp className="w-5 h-5 text-emerald-600" />,
      domain: 'Market Momentum & Price Action',
      description:
        'Analyzes 14-day RSI, SMA alignment (20/50/200), volume trends, and primary support/resistance levels.',
      source: 'Yahoo Finance Market Data Engine',
    },
    {
      name: 'Fundamental Analyst',
      icon: <FileSpreadsheet className="w-5 h-5 text-blue-600" />,
      domain: 'Valuation, Multiples & Cash Flow',
      description:
        'Calculates P/E ratios, EV/EBITDA multiples, operating cash flow health, and gross margin trajectory.',
      source: 'SEC Edgar Financial Statements & Yahoo Provider',
    },
    {
      name: 'News & Sentiment Analyst',
      icon: <Newspaper className="w-5 h-5 text-amber-600" />,
      domain: 'Market Sentiment & Headline Media',
      description:
        'Scores sentiment distribution across recent financial media, enterprise catalysts, and regulatory headlines.',
      source: 'Finnhub Curated News Feed',
    },
    {
      name: 'Research Vault Analyst',
      icon: <FolderLock className="w-5 h-5 text-purple-600" />,
      domain: 'RAG Retrieval Across Internal Documents',
      description:
        'Performs vector similarity search over user-uploaded filings and research PDFs in ChromaDB.',
      source: 'ChromaDB + Gemini Text Embeddings',
    },
    {
      name: 'Risk & Volatility Analyst',
      icon: <ShieldAlert className="w-5 h-5 text-rose-600" />,
      domain: 'Downside Drawdown & Beta Volatility',
      description:
        'Computes historical Value at Risk (95% VaR), drawdown stress scenarios, and market correlation risk.',
      source: 'Quantitative Risk Assessment Model',
    },
  ];

  return (
    <div data-testid="multi-agent-page" className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-emerald-600" />
            <h2 className="text-xl font-bold text-slate-900">
              Autonomous Multi-Agent Architecture
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Decoupled architectural telemetry and coordination view of the LangGraph decision support pipeline.
          </p>
        </div>

        <button
          type="button"
          onClick={() => navigate('/analysis/new')}
          className="inline-flex items-center px-4 py-2 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white transition-colors shadow-sm self-start sm:self-auto"
        >
          <Sparkles className="w-3.5 h-3.5 mr-1.5" />
          <span>Launch Analysis</span>
        </button>
      </div>

      {/* Pipeline Diagram */}
      <AgentPipelineVisualizer />

      {/* Active Session Telemetry */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Live Pipeline Session Telemetry
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Reflects the current analysis session in execution or most recently completed.
            </p>
          </div>

          {activeAnalysis ? (
            <span
              className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                activeAnalysis.status === 'completed'
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : activeAnalysis.status === 'running'
                    ? 'bg-sky-50 text-sky-700 border border-sky-200 animate-pulse'
                    : 'bg-rose-50 text-rose-700 border border-rose-200'
              }`}
            >
              Session: {activeAnalysis.status.toUpperCase()}
            </span>
          ) : (
            <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-slate-100 text-slate-600">
              Idle / Standby
            </span>
          )}
        </div>

        {activeAnalysis ? (
          <div className="space-y-6 pt-2">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
              <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                <span className="text-slate-400 block text-[11px]">Target Equity</span>
                <span className="font-bold text-slate-900 text-sm">
                  {activeAnalysis.companyName} ({activeAnalysis.ticker})
                </span>
              </div>
              <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                <span className="text-slate-400 block text-[11px]">Pipeline Session ID</span>
                <span className="font-mono text-xs text-slate-700 truncate block">
                  {activeAnalysis.analysisId}
                </span>
              </div>
              <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                <span className="text-slate-400 block text-[11px]">Progress Stage</span>
                <span className="font-semibold text-slate-800">
                  {activeAnalysis.progressStage}
                </span>
              </div>
            </div>

            <AnalysisProgressBar
              progressPercent={activeAnalysis.progressPercent}
              currentStage={activeAnalysis.progressStage}
            />

            <div>
              <h4 className="text-xs font-bold text-slate-600 uppercase tracking-wider mb-3">
                Specialist Agent Statuses
              </h4>
              <SpecialistStatusGrid
                specialistStatuses={activeAnalysis.specialistStatuses}
              />
            </div>

            {activeAnalysis.status === 'completed' && (
              <div className="flex items-center justify-between p-4 rounded-xl bg-emerald-50 border border-emerald-200">
                <div className="flex items-center space-x-2 text-xs text-emerald-900">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                  <span>Dossier synthesis finalized and linked to evidence sources.</span>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    if (activeReport?.report_id) {
                      navigate(`/reports/${activeReport.report_id}`);
                    } else {
                      navigate('/analysis/new');
                    }
                  }}
                  className="px-4 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-bold text-xs flex items-center space-x-1.5 transition-colors shadow-sm"
                >
                  <span>Examine Report</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="p-8 text-center text-slate-500 text-xs space-y-2">
            <p className="font-medium text-slate-700">
              No active multi-agent pipeline is currently running.
            </p>
            <p className="text-slate-400 max-w-md mx-auto">
              When you initiate an analysis from the Research & Analysis interface, real-time agent dispatch, CIO routing decisions, and specialist telemetry will display here.
            </p>
            <div className="pt-2">
              <button
                type="button"
                onClick={() => navigate('/analysis/new')}
                className="inline-flex items-center px-4 py-2 text-xs font-semibold rounded-lg bg-slate-900 hover:bg-slate-800 text-white transition-colors"
              >
                <span>Start New Analysis</span>
                <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Specialist Agent Specification Grid */}
      <div className="space-y-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">
            Autonomous Specialist Agents Reference
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Domain specialists coordinated by the Chief Investment Officer (CIO) Router.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {specialistSpecs.map((spec, i) => (
            <div
              key={i}
              className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3 flex flex-col justify-between"
            >
              <div className="space-y-2">
                <div className="flex items-center space-x-2.5">
                  <div className="p-2 rounded-lg bg-slate-50 border border-slate-100">
                    {spec.icon}
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900">{spec.name}</h4>
                    <span className="text-[10px] text-emerald-600 font-semibold block">
                      {spec.domain}
                    </span>
                  </div>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed">
                  {spec.description}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-400 flex items-center justify-between">
                <span>Source Provider:</span>
                <span className="font-medium text-slate-600">{spec.source}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
