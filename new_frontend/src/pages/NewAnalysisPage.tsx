import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Sparkles,
  Send,
  RefreshCw,
  AlertCircle,
  ArrowRight,
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  ShieldAlert,
  Bot,
  User,
  CheckCircle2,
  Clock,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useApp } from '../context/AppContext';
import { AnalyzeResponse, AgentStatus } from '../types';
import { ReportDossier } from '../components/report/ReportDossier';

type FlowStage = 'query' | 'clarify' | 'analyzing' | 'done';

const AGENT_SPECS = [
  { name: 'Conversation Agent', desc: 'Parsing query and framing the investor mandate' },
  { name: 'Technical Analyst', desc: 'Fetching price history and computing RSI / MACD / moving averages' },
  { name: 'Fundamental Analyst', desc: 'Pulling valuation ratios, margins and balance-sheet data' },
  { name: 'News & Sentiment Agent', desc: 'Searching reputable outlets for recent coverage' },
  { name: 'Risk Assessment Agent', desc: 'Scoring market, sector, company and macro risk' },
];

const AGENT_ICONS: Record<string, string> = {
  'Conversation Agent': '💬',
  'Technical Analyst': '📈',
  'Fundamental Analyst': '💼',
  'News & Sentiment Agent': '📰',
  'Risk Assessment Agent': '⚠️',
  'Synthesis Agent': '🧠',
};

const SUGGESTIONS = [
  'Should I invest in Apple (AAPL) for the long term?',
  'Analyse TSLA for a growth portfolio with 3-year horizon',
  'Evaluate NVIDIA valuation and downside risks before earnings',
  'Analyze Microsoft (MSFT) cloud moat and dividend sustainability',
];

export const NewAnalysisPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const { saveReport, activeReport, setActiveReport } = useApp();

  const [stage, setStage] = useState<FlowStage>('query');
  const [query, setQuery] = useState('');
  const [questions, setQuestions] = useState<string[]>([]);
  const [detectedTicker, setDetectedTicker] = useState<string | null>(null);
  const [detectedCompany, setDetectedCompany] = useState<string | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Pacing animation state
  const [activeAgentIndex, setActiveAgentIndex] = useState<number>(0);
  const [synthesisState, setSynthesisState] = useState<'queued' | 'running' | 'done'>('queued');
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Pre-fill query from URL params if provided
  useEffect(() => {
    const ticker = searchParams.get('ticker');
    if (ticker) {
      setQuery(`Comprehensive investment research on ${ticker.toUpperCase()}`);
    }
  }, [searchParams]);

  const handleReset = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setStage('query');
    setQuery('');
    setQuestions([]);
    setDetectedTicker(null);
    setDetectedCompany(null);
    setAnswers({});
    setErrorMsg(null);
    setActiveAgentIndex(0);
    setSynthesisState('queued');
    setElapsedSeconds(0);
  };

  // Stage 1: Clarify Trigger
  const handleStartClarify = async (overrideQuery?: string) => {
    const q = (overrideQuery || query).trim();
    if (!q) {
      setErrorMsg('Please enter a company, stock, or research query.');
      return;
    }
    setErrorMsg(null);
    setQuery(q);

    try {
      const clarifyRes = await apiService.clarify(q);
      setQuestions(clarifyRes.questions || []);
      setDetectedTicker(clarifyRes.detected_ticker || null);
      setDetectedCompany(clarifyRes.detected_company || null);
      setStage('clarify');
    } catch (err: any) {
      setErrorMsg(`Failed to clarify query: ${err.message || 'Make sure backend is running.'}`);
    }
  };

  // Stage 2: Proceed to Analysis
  const handleProceedToAnalysis = (skipAnswers = false) => {
    setErrorMsg(null);
    setStage('analyzing');
    setActiveAgentIndex(0);
    setSynthesisState('queued');
    setElapsedSeconds(0);

    const clarifyPayload = skipAnswers
      ? null
      : Object.fromEntries(Object.entries(answers).filter(([_, a]) => a.trim().length > 0));

    executePipeline(clarifyPayload);
  };

  // Stage 3: Real Pipeline execution + paced UI
  const executePipeline = async (clarifyAnswers: Record<string, string> | null) => {
    const controller = new AbortController();
    abortControllerRef.current = controller;

    let apiResult: AnalyzeResponse | null = null;
    let apiError: string | null = null;
    let apiFinished = false;

    // Background API call
    apiService
      .analyze(
        {
          query,
          clarification_answers: clarifyAnswers,
          ticker: detectedTicker,
          target_company: detectedCompany,
        },
        controller.signal,
      )
      .then((res) => {
        apiResult = res;
        apiFinished = true;
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          apiError = err.message || 'Pipeline execution failed.';
        }
        apiFinished = true;
      });

    // Paced visual progression loop (matching Streamlit 4s per agent pacing)
    const startTime = Date.now();
    let currentAgent = 0;

    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      setElapsedSeconds(elapsed);

      // Advance agent every 3 seconds while agents < 5
      const targetAgent = Math.min(Math.floor(elapsed / 3), AGENT_SPECS.length);
      setActiveAgentIndex(targetAgent);

      if (targetAgent >= AGENT_SPECS.length) {
        setSynthesisState('running');
      }

      // If backend finished
      if (apiFinished) {
        clearInterval(interval);

        if (apiError) {
          setErrorMsg(`Analysis failed: ${apiError}`);
          setStage('query');
        } else if (apiResult) {
          setSynthesisState('done');
          setActiveAgentIndex(AGENT_SPECS.length);
          setActiveReport(apiResult);
          saveReport(apiResult, query);
          setTimeout(() => {
            setStage('done');
          }, 800);
        }
      }
    }, 500);
  };

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-blue-900 via-indigo-950 to-slate-900 border border-indigo-800/40 rounded-2xl p-6 sm:p-8 text-white relative overflow-hidden shadow-sm">
        <div className="max-w-3xl space-y-2">
          <div className="inline-flex items-center space-x-2 text-xs font-semibold px-2.5 py-1 rounded-full bg-blue-500/20 text-blue-300 border border-blue-400/30">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Multi-Agent Research Pipeline</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-white">
            Institutional AI Financial Research
          </h1>
          <p className="text-xs sm:text-sm text-slate-200 leading-relaxed">
            Enter any stock, ETF, or investment thesis. FinPilot clarifies your investor mandate, fetches verified market data, runs 5 specialized agents concurrently, and synthesizes an institutional dossier.
          </p>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700 flex items-center justify-between shadow-xs">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMsg(null)}
            className="text-xs font-bold underline hover:text-red-900 ml-3"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Stage 1: Query Input */}
      {stage === 'query' && (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 space-y-4 shadow-xs">
            <h2 className="text-base sm:text-lg font-bold text-slate-900 flex items-center space-x-2">
              <span>🔍 What would you like to research?</span>
            </h2>

            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. 'Should I invest in Apple for the long term?' or 'Analyse TSLA for a growth portfolio'"
              className="w-full h-28 px-4 py-3 bg-slate-50 border border-slate-300 rounded-xl text-slate-900 placeholder-slate-400 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white focus:border-transparent transition-all"
            />

            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
              <div className="text-xs text-slate-500 font-medium">
                Supports all US and global tickers (e.g. AAPL, NVDA, MSFT, RELIANCE.NS)
              </div>

              <button
                type="button"
                onClick={() => handleStartClarify()}
                disabled={!query.trim()}
                className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-bold text-xs sm:text-sm flex items-center justify-center space-x-2 transition-all shadow-xs"
              >
                <span>🚀 Launch Analysis</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Suggestion Prompts */}
          <div className="space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Sample Research Queries
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {SUGGESTIONS.map((item, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => handleStartClarify(item)}
                  className="text-left p-3.5 rounded-xl bg-white border border-slate-200 hover:border-blue-400 hover:bg-blue-50/40 text-xs font-medium text-slate-700 transition-colors flex items-center justify-between group shadow-xs"
                >
                  <span>{item}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-blue-600 transition-colors flex-shrink-0 ml-2" />
                </button>
              ))}
            </div>
          </div>

          {/* Specialist Pillars Overview */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-1.5">
              <div className="flex items-center space-x-2 text-emerald-600 font-bold text-xs">
                <TrendingUp className="w-4 h-4" />
                <span>Technical Analysis</span>
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                RSI(14), MACD, SMA20/50/200, volatility & momentum signals.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-1.5">
              <div className="flex items-center space-x-2 text-blue-600 font-bold text-xs">
                <FileSpreadsheet className="w-4 h-4" />
                <span>Fundamental Health</span>
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Revenue growth, margins, P/E & cash flow valuation.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-1.5">
              <div className="flex items-center space-x-2 text-amber-600 font-bold text-xs">
                <Newspaper className="w-4 h-4" />
                <span>News & Sentiment</span>
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Provider headlines, corporate catalysts & analyst coverage.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-1.5">
              <div className="flex items-center space-x-2 text-rose-600 font-bold text-xs">
                <ShieldAlert className="w-4 h-4" />
                <span>Risk Assessment</span>
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Market beta, sector risk, balance sheet leverage & macro exposure.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stage 2: Clarification Stage */}
      {stage === 'clarify' && (
        <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 space-y-6 shadow-md max-w-2xl mx-auto animate-fadeIn">
          <div className="space-y-2 border-b border-slate-200 pb-4">
            {(detectedCompany || detectedTicker) && (
              <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-xs text-blue-700 font-semibold mb-2">
                <span>🎯 Target Identified:</span>
                <span className="font-bold text-slate-900">
                  {detectedCompany || ''} {detectedTicker ? `(${detectedTicker})` : ''}
                </span>
              </div>
            )}
            <h2 className="text-lg font-bold text-slate-900">
              🤔 Clarifying Questions
            </h2>
            <p className="text-xs text-slate-500">
              Help us calibrate the research depth to your exact requirements and investment horizon:
            </p>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 italic">
              "{query}"
            </div>
          </div>

          <div className="space-y-4">
            {questions.map((q, idx) => (
              <div key={idx} className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-700 block">
                  {q}
                </label>
                <input
                  type="text"
                  placeholder="Your answer (or leave blank to use defaults)..."
                  value={answers[q] || ''}
                  onChange={(e) =>
                    setAnswers((prev) => ({ ...prev, [q]: e.target.value }))
                  }
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-slate-900 placeholder-slate-400 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                />
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-slate-200">
            <button
              type="button"
              onClick={handleReset}
              className="text-xs font-medium text-slate-500 hover:text-slate-800 transition-colors"
            >
              ← Back to query
            </button>

            <div className="flex items-center space-x-3">
              <button
                type="button"
                onClick={() => handleProceedToAnalysis(true)}
                className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-colors"
              >
                Skip →
              </button>
              <button
                type="button"
                onClick={() => handleProceedToAnalysis(false)}
                className="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold transition-colors shadow-xs"
              >
                ▶ Run Multi-Agent Analysis
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Stage 3: Analyzing / Research Pipeline */}
      {stage === 'analyzing' && (
        <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 space-y-6 shadow-md max-w-2xl mx-auto animate-fadeIn">
          <div className="flex items-center justify-between border-b border-slate-200 pb-4">
            <div>
              <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
                <span>⚙️ Multi-Agent Research Pipeline Running...</span>
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Resolving data, computing indicators & orchestrating specialist briefs
              </p>
            </div>
            <div className="flex items-center space-x-1.5 text-xs text-blue-600 font-mono font-bold">
              <Clock className="w-4 h-4 animate-spin" />
              <span>{elapsedSeconds}s elapsed</span>
            </div>
          </div>

          {/* Agents execution list */}
          <div className="divide-y divide-slate-200 bg-slate-50 rounded-xl border border-slate-200 p-4">
            {AGENT_SPECS.map((ag, idx) => {
              const icon = AGENT_ICONS[ag.name] || '🔄';
              const isCompleted = idx < activeAgentIndex;
              const isRunning = idx === activeAgentIndex;

              return (
                <div key={idx} className="py-3 flex items-center justify-between gap-3">
                  <div className="flex items-center space-x-3">
                    <span className="text-base">{icon}</span>
                    <div>
                      <div className="text-xs font-bold text-slate-900">{ag.name}</div>
                      <div className="text-[11px] text-slate-500">{ag.desc}</div>
                    </div>
                  </div>

                  <div>
                    {isCompleted ? (
                      <span className="px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 text-[10px] font-bold">
                        ✓ Done
                      </span>
                    ) : isRunning ? (
                      <span className="px-2.5 py-1 rounded-full bg-amber-100 text-amber-800 border border-amber-300 text-[10px] font-bold animate-pulse">
                        ⟳ Running
                      </span>
                    ) : (
                      <span className="px-2.5 py-1 rounded-full bg-slate-200/70 text-slate-600 border border-slate-300 text-[10px] font-medium">
                        Queued
                      </span>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Synthesis Agent Row */}
            <div className="py-3 flex items-center justify-between gap-3">
              <div className="flex items-center space-x-3">
                <span className="text-base">🧠</span>
                <div>
                  <div className="text-xs font-bold text-slate-900">Synthesis Agent (Chief Strategist)</div>
                  <div className="text-[11px] text-slate-500">Reconciling evidence and assembling structured research note</div>
                </div>
              </div>

              <div>
                {synthesisState === 'done' ? (
                  <span className="px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 text-[10px] font-bold">
                    ✓ Done
                  </span>
                ) : synthesisState === 'running' ? (
                  <span className="px-2.5 py-1 rounded-full bg-blue-100 text-blue-800 border border-blue-300 text-[10px] font-bold animate-pulse">
                    ⟳ Synthesizing
                  </span>
                ) : (
                  <span className="px-2.5 py-1 rounded-full bg-slate-200/70 text-slate-600 border border-slate-300 text-[10px] font-medium">
                    Queued
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="text-center text-xs text-slate-500">
            {synthesisState === 'running'
              ? 'Finalizing research dossier on Groq...'
              : `Executing stage: ${AGENT_SPECS[Math.min(activeAgentIndex, AGENT_SPECS.length - 1)].name}...`}
          </div>
        </div>
      )}

      {/* Stage 4: Results Dossier */}
      {stage === 'done' && activeReport && (
        <ReportDossier report={activeReport} onReset={handleReset} />
      )}
    </div>
  );
};
