import React, { useState } from 'react';
import {
  FileText,
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  ShieldAlert,
  Award,
  Download,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Printer,
} from 'lucide-react';
import { AnalyzeResponse } from '../../types';

interface ReportDossierProps {
  report: AnalyzeResponse;
  onReset?: () => void;
}

const AGENT_ICONS: Record<string, string> = {
  'Conversation Agent': '💬',
  'Technical Analyst': '📈',
  'Fundamental Analyst': '💼',
  'News & Sentiment Agent': '📰',
  'Risk Assessment Agent': '⚠️',
  'Synthesis Agent': '🧠',
};

export const ReportDossier: React.FC<ReportDossierProps> = ({ report, onReset }) => {
  const [activeTab, setActiveTab] = useState<number>(0);
  const [expandedBriefs, setExpandedBriefs] = useState<Record<string, boolean>>({});

  const toggleBrief = (name: string) => {
    setExpandedBriefs((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  const handleDownloadMarkdown = () => {
    const blob = new Blob([report.raw_markdown || ''], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `finpilot_${report.ticker || 'EQUITY'}_report.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const tabs = [
    { label: '📋 Summary', icon: <FileText className="w-4 h-4 mr-1.5" /> },
    { label: '📈 Technical', icon: <TrendingUp className="w-4 h-4 mr-1.5" /> },
    { label: '💼 Fundamental', icon: <FileSpreadsheet className="w-4 h-4 mr-1.5" /> },
    { label: '📰 News & Sentiment', icon: <Newspaper className="w-4 h-4 mr-1.5" /> },
    { label: '⚠️ Risk', icon: <ShieldAlert className="w-4 h-4 mr-1.5" /> },
    { label: '🎯 Verdict', icon: <Award className="w-4 h-4 mr-1.5" /> },
    { label: '📄 Full Report', icon: <Download className="w-4 h-4 mr-1.5" /> },
  ];

  const conf = report.confidence || 'Medium';
  const confBadgeColor =
    conf.toLowerCase() === 'high'
      ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
      : conf.toLowerCase() === 'low'
        ? 'bg-rose-950 text-rose-400 border-rose-800'
        : 'bg-amber-950 text-amber-400 border-amber-800';

  return (
    <div className="space-y-6">
      {/* Dossier Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xl shadow-slate-950/40">
        <div className="space-y-1">
          <div className="flex items-center space-x-3">
            <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
              {report.company}
            </h2>
            <span className="px-2.5 py-1 rounded-md bg-slate-800 text-emerald-400 font-mono text-sm font-bold border border-slate-700">
              {report.ticker}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Institutional Research Dossier synthesized by FinPilot Multi-Agent Architecture
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <div className={`px-3 py-1.5 rounded-lg border font-semibold text-xs flex items-center space-x-1.5 ${confBadgeColor}`}>
            <span>Confidence:</span>
            <span className="font-bold">{conf}</span>
          </div>

          <button
            type="button"
            onClick={handleDownloadMarkdown}
            className="inline-flex items-center px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-200 border border-slate-700 text-xs font-semibold transition-colors"
          >
            <Download className="w-3.5 h-3.5 mr-1 text-slate-400" />
            <span>Export .md</span>
          </button>

          {onReset && (
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition-colors shadow-sm"
            >
              <span>+ New Analysis</span>
            </button>
          )}
        </div>
      </div>

      {/* Agent Pipeline Summary Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
            🤖 Multi-Agent Execution Telemetry
          </h3>
          <span className="text-[11px] text-slate-500">5 Specialists + Chief Strategist</span>
        </div>

        <div className="divide-y divide-slate-800/50">
          {(report.agent_statuses || []).map((ag, idx) => {
            const icon = AGENT_ICONS[ag.name] || '🔄';
            const isCompleted = ag.status === 'completed';
            const isPartial = ag.status === 'partial';

            return (
              <div key={idx} className="py-2.5 flex items-center justify-between text-xs gap-3">
                <div className="flex items-center space-x-2.5 min-w-[200px]">
                  <span className="text-base">{icon}</span>
                  <span className="font-semibold text-slate-200">{ag.name}</span>
                </div>

                <div className="flex-1 text-slate-400 text-xs truncate">
                  {ag.summary}
                </div>

                <div>
                  {isCompleted ? (
                    <span className="px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 text-[10px] font-bold">
                      ✓ Done
                    </span>
                  ) : isPartial ? (
                    <span className="px-2 py-0.5 rounded-full bg-amber-950/80 text-amber-400 border border-amber-800/80 text-[10px] font-bold">
                      ◐ Partial
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded-full bg-rose-950/80 text-rose-400 border border-rose-800/80 text-[10px] font-bold">
                      ✕ Failed
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="border-b border-slate-800 flex items-center space-x-1 overflow-x-auto no-scrollbar">
        {tabs.map((tab, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => setActiveTab(idx)}
            className={`px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-colors flex items-center whitespace-nowrap ${
              activeTab === idx
                ? 'bg-slate-900 text-emerald-400 border-t-2 border-emerald-500 border-x border-slate-800'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content Panels */}
      <div className="bg-slate-900 border border-slate-800 rounded-b-xl rounded-tr-xl p-6 space-y-6">
        {/* Tab 0: Summary */}
        {activeTab === 0 && (
          <div className="space-y-6">
            {report.investor_profile && (
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400">
                  Investor Mandate
                </span>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {report.investor_profile}
                </p>
              </div>
            )}

            <div className="space-y-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Executive Summary
              </span>
              <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-800 text-xs text-slate-200 leading-relaxed whitespace-pre-line">
                {report.executive_summary}
              </div>
            </div>

            {report.key_metrics && Object.keys(report.key_metrics).length > 0 && (
              <div className="space-y-3">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Verified Key Metrics (yfinance)
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                  {Object.entries(report.key_metrics).map(([key, val]) => (
                    <div
                      key={key}
                      className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 flex flex-col justify-between"
                    >
                      <span className="text-[11px] text-slate-400">{key}</span>
                      <span className="text-sm font-bold text-slate-100 mt-1 font-mono">
                        {val}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {report.data_availability && (
              <div className="text-[11px] text-slate-400 bg-slate-950/40 p-3 rounded-lg border border-slate-800/80">
                <span className="font-semibold text-slate-300">Data Availability: </span>
                {report.data_availability}
              </div>
            )}
          </div>
        )}

        {/* Tab 1: Technical */}
        {activeTab === 1 && (
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-emerald-400">
              <TrendingUp className="w-5 h-5" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                Technical Analysis & Momentum Outlook
              </h3>
            </div>
            <div className="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-200 leading-relaxed whitespace-pre-line">
              {report.technical_outlook || 'No technical analysis available.'}
            </div>
          </div>
        )}

        {/* Tab 2: Fundamental */}
        {activeTab === 2 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sky-400">
                <FileSpreadsheet className="w-5 h-5" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                  Fundamental View & Financial Health
                </h3>
              </div>
              <div className="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-200 leading-relaxed whitespace-pre-line">
                {report.fundamental_view || 'No fundamental analysis available.'}
              </div>
            </div>

            {report.valuation_view && (
              <div className="space-y-2 pt-4 border-t border-slate-800">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Valuation Multiples & DCF Context
                </h4>
                <div className="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-200 leading-relaxed whitespace-pre-line">
                  {report.valuation_view}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: News & Sentiment */}
        {activeTab === 3 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-amber-400">
                <Newspaper className="w-5 h-5" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                  News Sentiment & Market Narrative
                </h3>
              </div>
              <div className="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-200 leading-relaxed whitespace-pre-line">
                {report.news_sentiment || 'No news sentiment available.'}
              </div>
            </div>

            {report.sources && report.sources.length > 0 && (
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Data Sources Actually Referenced
                </span>
                <ul className="space-y-1 text-xs text-slate-300 list-disc list-inside">
                  {report.sources.map((s, idx) => (
                    <li key={idx}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Risk */}
        {activeTab === 4 && (
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-rose-400">
              <ShieldAlert className="w-5 h-5" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                Institutional Risk Register & Sensitivities
              </h3>
            </div>
            <div className="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-200 leading-relaxed whitespace-pre-line">
              {report.risk_assessment || 'No risk assessment available.'}
            </div>
          </div>
        )}

        {/* Tab 5: Verdict */}
        {activeTab === 5 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-emerald-400">
                <Award className="w-5 h-5" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                  Final Synthesis & Verdict
                </h3>
              </div>
              <div className="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-100 leading-relaxed whitespace-pre-line font-medium">
                {report.final_verdict || 'No verdict available.'}
              </div>
            </div>

            {report.scenario_analysis && (
              <div className="space-y-2 pt-2 border-t border-slate-800">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Scenario Analysis (Base / Bull / Bear)
                </h4>
                <div className="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-300 leading-relaxed whitespace-pre-line">
                  {report.scenario_analysis}
                </div>
              </div>
            )}

            {report.catalysts_to_watch && (
              <div className="space-y-2 pt-2 border-t border-slate-800">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Catalysts to Watch & Invalidation Triggers
                </h4>
                <div className="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-300 leading-relaxed whitespace-pre-line">
                  {report.catalysts_to_watch}
                </div>
              </div>
            )}

            <div className="text-[11px] text-slate-500 italic pt-2 border-t border-slate-800">
              {report.disclaimer ||
                'This report is for informational purposes only and does not constitute financial advice.'}
            </div>
          </div>
        )}

        {/* Tab 6: Full Report & Specialist Briefs */}
        {activeTab === 6 && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white">Structured Research Note (Markdown)</h3>
                <p className="text-xs text-slate-400">Complete raw markdown output compiled by the Synthesis Agent</p>
              </div>

              <button
                type="button"
                onClick={handleDownloadMarkdown}
                className="inline-flex items-center px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors shadow-sm"
              >
                <Download className="w-4 h-4 mr-1.5" />
                <span>Download Report (.md)</span>
              </button>
            </div>

            <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 font-mono text-xs overflow-x-auto whitespace-pre-wrap max-h-[500px] leading-relaxed">
              {report.raw_markdown || 'No markdown output generated.'}
            </pre>

            {report.research_briefs && Object.keys(report.research_briefs).length > 0 && (
              <div className="space-y-3 pt-4 border-t border-slate-800">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  🔬 Raw Specialist Analyst Briefs
                </h4>
                <div className="space-y-2">
                  {Object.entries(report.research_briefs).map(([name, brief]) => {
                    const isOpen = Boolean(expandedBriefs[name]);
                    return (
                      <div
                        key={name}
                        className="rounded-xl border border-slate-800 bg-slate-950/60 overflow-hidden"
                      >
                        <button
                          type="button"
                          onClick={() => toggleBrief(name)}
                          className="w-full px-4 py-3 text-left flex items-center justify-between text-xs font-semibold text-slate-200 hover:bg-slate-800/40 transition-colors"
                        >
                          <span className="flex items-center space-x-2">
                            <span>{AGENT_ICONS[name] || '🔬'}</span>
                            <span>{name}</span>
                          </span>
                          {isOpen ? (
                            <ChevronDown className="w-4 h-4 text-slate-400" />
                          ) : (
                            <ChevronRight className="w-4 h-4 text-slate-400" />
                          )}
                        </button>
                        {isOpen && (
                          <div className="p-4 border-t border-slate-800 text-xs text-slate-300 leading-relaxed whitespace-pre-line bg-slate-950">
                            {brief}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
