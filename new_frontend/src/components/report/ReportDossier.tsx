import React, { useState } from 'react';
import {
  FileText,
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  ShieldAlert,
  Award,
  Download,
  ChevronDown,
  ChevronRight,
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
      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
      : conf.toLowerCase() === 'low'
        ? 'bg-rose-50 text-rose-700 border-rose-200'
        : 'bg-amber-50 text-amber-700 border-amber-200';

  return (
    <div className="space-y-6">
      {/* Dossier Header */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xs">
        <div className="space-y-1">
          <div className="flex items-center space-x-3">
            <h2 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">
              {report.company}
            </h2>
            <span className="px-2.5 py-1 rounded-md bg-blue-50 text-blue-700 font-mono text-sm font-bold border border-blue-200">
              {report.ticker}
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Institutional Research Dossier synthesized by FinPilot Multi-Agent Architecture
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <div className={`px-3 py-1.5 rounded-lg border font-bold text-xs flex items-center space-x-1.5 ${confBadgeColor}`}>
            <span>Confidence:</span>
            <span>{conf}</span>
          </div>

          <button
            type="button"
            onClick={handleDownloadMarkdown}
            className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 text-xs font-semibold transition-colors"
          >
            <Download className="w-3.5 h-3.5 mr-1 text-slate-500" />
            <span>Export .md</span>
          </button>

          {onReset && (
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-colors shadow-xs"
            >
              <span>+ New Analysis</span>
            </button>
          )}
        </div>
      </div>

      {/* Agent Pipeline Summary Card */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
        <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">
            🤖 Multi-Agent Execution Telemetry
          </h3>
          <span className="text-[11px] text-slate-400">5 Specialists + Chief Strategist</span>
        </div>

        <div className="divide-y divide-slate-100">
          {(report.agent_statuses || []).map((ag, idx) => {
            const icon = AGENT_ICONS[ag.name] || '🔄';
            const isCompleted = ag.status === 'completed';
            const isPartial = ag.status === 'partial';

            return (
              <div key={idx} className="py-2.5 flex items-center justify-between text-xs gap-3">
                <div className="flex items-center space-x-2.5 min-w-[200px]">
                  <span className="text-base">{icon}</span>
                  <span className="font-bold text-slate-800">{ag.name}</span>
                </div>

                <div className="flex-1 text-slate-600 text-xs truncate">
                  {ag.summary}
                </div>

                <div>
                  {isCompleted ? (
                    <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px] font-bold">
                      ✓ Done
                    </span>
                  ) : isPartial ? (
                    <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 text-[10px] font-bold">
                      ◐ Partial
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200 text-[10px] font-bold">
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
      <div className="border-b border-slate-200 flex items-center space-x-1 overflow-x-auto no-scrollbar">
        {tabs.map((tab, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => setActiveTab(idx)}
            className={`px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-colors flex items-center whitespace-nowrap ${
              activeTab === idx
                ? 'bg-white text-blue-700 border-t-2 border-blue-600 border-x border-slate-200 font-bold shadow-xs'
                : 'text-slate-500 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content Panels */}
      <div className="bg-white border border-slate-200 rounded-b-xl rounded-tr-xl p-6 shadow-xs space-y-6">
        {/* Tab 0: Summary */}
        {activeTab === 0 && (
          <div className="space-y-6">
            {report.investor_profile && (
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-blue-700">
                  Investor Mandate
                </span>
                <p className="text-xs text-slate-700 leading-relaxed">
                  {report.investor_profile}
                </p>
              </div>
            )}

            <div className="space-y-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                Executive Summary
              </span>
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-line">
                {report.executive_summary}
              </div>
            </div>

            {report.key_metrics && Object.keys(report.key_metrics).length > 0 && (
              <div className="space-y-3">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  Verified Key Metrics (yfinance)
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                  {Object.entries(report.key_metrics).map(([key, val]) => (
                    <div
                      key={key}
                      className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex flex-col justify-between"
                    >
                      <span className="text-[11px] text-slate-500">{key}</span>
                      <span className="text-sm font-bold text-slate-900 mt-1 font-mono">
                        {val}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {report.data_availability && (
              <div className="text-[11px] text-slate-500 bg-slate-50 p-3 rounded-lg border border-slate-200">
                <span className="font-semibold text-slate-700">Data Availability: </span>
                {report.data_availability}
              </div>
            )}
          </div>
        )}

        {/* Tab 1: Technical */}
        {activeTab === 1 && (
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-emerald-600">
              <TrendingUp className="w-5 h-5" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900">
                Technical Analysis & Momentum Outlook
              </h3>
            </div>
            <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-line">
              {report.technical_outlook || 'No technical analysis available.'}
            </div>
          </div>
        )}

        {/* Tab 2: Fundamental */}
        {activeTab === 2 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-blue-600">
                <FileSpreadsheet className="w-5 h-5" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900">
                  Fundamental View & Financial Health
                </h3>
              </div>
              <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-line">
                {report.fundamental_view || 'No fundamental analysis available.'}
              </div>
            </div>

            {report.valuation_view && (
              <div className="space-y-2 pt-4 border-t border-slate-100">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Valuation Multiples & DCF Context
                </h4>
                <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-line">
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
              <div className="flex items-center space-x-2 text-amber-600">
                <Newspaper className="w-5 h-5" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900">
                  News Sentiment & Market Narrative
                </h3>
              </div>
              <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-line">
                {report.news_sentiment || 'No news sentiment available.'}
              </div>
            </div>

            {report.sources && report.sources.length > 0 && (
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  Data Sources Actually Referenced
                </span>
                <ul className="space-y-1 text-xs text-slate-700 list-disc list-inside">
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
            <div className="flex items-center space-x-2 text-rose-600">
              <ShieldAlert className="w-5 h-5" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900">
                Institutional Risk Register & Sensitivities
              </h3>
            </div>
            <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-line">
              {report.risk_assessment || 'No risk assessment available.'}
            </div>
          </div>
        )}

        {/* Tab 5: Verdict */}
        {activeTab === 5 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-emerald-600">
                <Award className="w-5 h-5" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900">
                  Final Synthesis & Verdict
                </h3>
              </div>
              <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-line font-medium">
                {report.final_verdict || 'No verdict available.'}
              </div>
            </div>

            {report.scenario_analysis && (
              <div className="space-y-2 pt-2 border-t border-slate-100">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Scenario Analysis (Base / Bull / Bear)
                </h4>
                <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed whitespace-pre-line">
                  {report.scenario_analysis}
                </div>
              </div>
            )}

            {report.catalysts_to_watch && (
              <div className="space-y-2 pt-2 border-t border-slate-100">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Catalysts to Watch & Invalidation Triggers
                </h4>
                <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed whitespace-pre-line">
                  {report.catalysts_to_watch}
                </div>
              </div>
            )}

            <div className="text-[11px] text-slate-400 italic pt-2 border-t border-slate-100">
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
                <h3 className="text-sm font-bold text-slate-900">Structured Research Note (Markdown)</h3>
                <p className="text-xs text-slate-500">Complete raw markdown output compiled by the Synthesis Agent</p>
              </div>

              <button
                type="button"
                onClick={handleDownloadMarkdown}
                className="inline-flex items-center px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors shadow-xs"
              >
                <Download className="w-4 h-4 mr-1.5" />
                <span>Download Report (.md)</span>
              </button>
            </div>

            <pre className="p-4 rounded-xl bg-slate-900 text-slate-100 font-mono text-xs overflow-x-auto whitespace-pre-wrap max-h-[500px] leading-relaxed shadow-xs">
              {report.raw_markdown || 'No markdown output generated.'}
            </pre>

            {report.research_briefs && Object.keys(report.research_briefs).length > 0 && (
              <div className="space-y-3 pt-4 border-t border-slate-100">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  🔬 Raw Specialist Analyst Briefs
                </h4>
                <div className="space-y-2">
                  {Object.entries(report.research_briefs).map(([name, brief]) => {
                    const isOpen = Boolean(expandedBriefs[name]);
                    return (
                      <div
                        key={name}
                        className="rounded-xl border border-slate-200 bg-slate-50 overflow-hidden"
                      >
                        <button
                          type="button"
                          onClick={() => toggleBrief(name)}
                          className="w-full px-4 py-3 text-left flex items-center justify-between text-xs font-bold text-slate-800 hover:bg-slate-100 transition-colors"
                        >
                          <span className="flex items-center space-x-2">
                            <span>{AGENT_ICONS[name] || '🔬'}</span>
                            <span>{name}</span>
                          </span>
                          {isOpen ? (
                            <ChevronDown className="w-4 h-4 text-slate-500" />
                          ) : (
                            <ChevronRight className="w-4 h-4 text-slate-500" />
                          )}
                        </button>
                        {isOpen && (
                          <div className="p-4 border-t border-slate-200 text-xs text-slate-700 leading-relaxed whitespace-pre-line bg-white">
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
