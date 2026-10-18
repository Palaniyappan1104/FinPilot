import React, { useState } from 'react';
import {
  Layers,
  Sparkles,
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  CheckCircle2,
  Clock,
  ArrowRight,
  AlertCircle,
  BarChart3,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useApp } from '../context/AppContext';
import { AnalyzeResponse } from '../types';

interface PresetStock {
  ticker: string;
  name: string;
}

const PRESET_STOCKS: PresetStock[] = [
  { ticker: 'AAPL', name: 'Apple Inc.' },
  { ticker: 'MSFT', name: 'Microsoft Corporation' },
  { ticker: 'NVDA', name: 'NVIDIA Corporation' },
  { ticker: 'TSLA', name: 'Tesla, Inc.' },
  { ticker: 'AMZN', name: 'Amazon.com, Inc.' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.' },
];

export const MultiAgentPage: React.FC = () => {
  const { saveReport } = useApp();
  const [selectedStock, setSelectedStock] = useState<PresetStock>(PRESET_STOCKS[0]);
  const [customTicker, setCustomTicker] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'all' | 'fundamental' | 'technical' | 'news'>('all');

  const activeTicker = (customTicker.trim() || selectedStock.ticker).toUpperCase();
  const activeName = customTicker.trim() ? customTicker.toUpperCase() : selectedStock.name;

  const templatedPrompt = `Perform an institutional multi-specialist research pass on ${activeName} (${activeTicker}), focusing specifically on:
1. Fundamental Valuation, Capital Structure & Margins
2. Technical Momentum, Indicator Readings (RSI/MACD/SMAs) & Price Trend
3. News Flow, Headline Catalysts & Market Sentiment`;

  const handleRunSpecialists = async () => {
    setIsRunning(true);
    setErrorMsg(null);
    setResult(null);

    try {
      const res = await apiService.analyze({
        query: templatedPrompt,
        ticker: activeTicker,
        target_company: activeName,
      });
      setResult(res);
      saveReport(res, `3-Specialist Analysis on ${activeTicker}`);
    } catch (err: any) {
      setErrorMsg(`Specialist pipeline failed: ${err.message || 'Check server connection'}`);
    } finally {
      setIsRunning(false);
    }
  };

  const getBrief = (agentName: string): string => {
    if (!result?.research_briefs) return '';
    return result.research_briefs[agentName] || '';
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xl">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-emerald-400">
            <Layers className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">Templated Specialist Flow</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            Multi-Agent Specialist Pipeline
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl">
            Select a stock to dispatch the 3 primary specialist agents (Fundamental, Technical, News & Sentiment) via structured prompts and inspect their combined synthesis.
          </p>
        </div>

        <button
          type="button"
          onClick={handleRunSpecialists}
          disabled={isRunning}
          className="px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-bold text-xs sm:text-sm flex items-center justify-center space-x-2 transition-all shadow-md shadow-emerald-950/40 self-start sm:self-auto"
        >
          {isRunning ? (
            <>
              <Clock className="w-4 h-4 animate-spin" />
              <span>Agents Running...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Run 3-Agent Pipeline</span>
            </>
          )}
        </button>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-950/80 border border-rose-800 text-xs text-rose-300 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Stock Selection & Prompt Templating Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Stock Picker */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">
            1. Select Target Stock
          </h2>

          <div className="grid grid-cols-2 gap-2">
            {PRESET_STOCKS.map((stock) => (
              <button
                key={stock.ticker}
                type="button"
                onClick={() => {
                  setSelectedStock(stock);
                  setCustomTicker('');
                }}
                className={`p-3 rounded-xl border text-left transition-all ${
                  selectedStock.ticker === stock.ticker && !customTicker
                    ? 'bg-emerald-950/60 border-emerald-500 text-white shadow-sm'
                    : 'bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="font-mono font-bold text-xs text-emerald-400">
                  {stock.ticker}
                </div>
                <div className="text-[11px] text-slate-400 truncate mt-0.5">
                  {stock.name}
                </div>
              </button>
            ))}
          </div>

          <div className="pt-2 border-t border-slate-800">
            <label className="text-[11px] font-semibold text-slate-400 block mb-1.5">
              Or specify custom ticker / exchange symbol:
            </label>
            <input
              type="text"
              placeholder="e.g. AMD, RELIANCE.NS, SAP.DE"
              value={customTicker}
              onChange={(e) => setCustomTicker(e.target.value)}
              className="w-full px-3.5 py-2 bg-slate-950 border border-slate-700/80 rounded-xl text-slate-100 placeholder-slate-500 text-xs uppercase font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>
        </div>

        {/* Right: Templated Prompt Card */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                2. Templated Specialist Dispatch Prompt
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                Automated
              </span>
            </div>
            <p className="text-xs text-slate-400">
              This prompt automatically binds target equity facts and orchestrates the 3 specialist agents concurrently:
            </p>

            <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 font-mono text-xs whitespace-pre-wrap leading-relaxed">
              {templatedPrompt}
            </pre>
          </div>

          <div className="pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
            <span>Specialists active:</span>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded bg-blue-950 text-blue-400 border border-blue-800 text-[10px] font-semibold">Fundamental</span>
              <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px] font-semibold">Technical</span>
              <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800 text-[10px] font-semibold">News & Sentiment</span>
            </div>
          </div>
        </div>
      </div>

      {/* Loading state indicator */}
      {isRunning && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center space-y-4 animate-pulse">
          <Clock className="w-8 h-8 text-emerald-400 animate-spin mx-auto" />
          <div>
            <h3 className="text-base font-bold text-white">Running 3 Specialist Agents for {activeTicker}...</h3>
            <p className="text-xs text-slate-400 mt-1">Fetching live market data via yfinance, executing GPT-OSS specialist passes, and synthesising combined takeaway.</p>
          </div>
        </div>
      )}

      {/* Results View */}
      {result && !isRunning && (
        <div className="space-y-6 animate-fadeIn">
          {/* Combined Synthesis Banner */}
          <div className="bg-gradient-to-br from-slate-900 to-slate-950 border border-emerald-800/60 rounded-2xl p-6 sm:p-8 space-y-4 shadow-xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400">
                  Combined Pipeline Synthesis
                </span>
                <h3 className="text-xl sm:text-2xl font-black text-white">
                  {result.company} ({result.ticker})
                </h3>
              </div>

              <div className="flex items-center space-x-2">
                <span className="px-3 py-1 rounded-lg bg-emerald-950 text-emerald-400 border border-emerald-800 text-xs font-bold">
                  Confidence: {result.confidence}
                </span>
              </div>
            </div>

            <div className="text-xs sm:text-sm text-slate-200 leading-relaxed whitespace-pre-line">
              {result.final_verdict}
            </div>

            {/* Quick Metrics Bar */}
            {result.key_metrics && Object.keys(result.key_metrics).length > 0 && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                {Object.entries(result.key_metrics).slice(0, 4).map(([k, v]) => (
                  <div key={k} className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">{k}</div>
                    <div className="text-xs font-bold font-mono text-emerald-400 mt-0.5">{v}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 3 Specialist Outputs Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* 1. Fundamental Analyst Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4 flex flex-col justify-between shadow-lg">
              <div className="space-y-3">
                <div className="flex items-center space-x-2.5 text-blue-400">
                  <div className="p-2 rounded-lg bg-blue-950 border border-blue-800">
                    <FileSpreadsheet className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider">Fundamental Analyst</h4>
                    <span className="text-[10px] text-slate-400">Valuation & Health</span>
                  </div>
                </div>

                <div className="text-xs text-slate-300 leading-relaxed whitespace-pre-line bg-slate-950/60 p-4 rounded-xl border border-slate-800 max-h-96 overflow-y-auto">
                  {result.fundamental_view || getBrief('Fundamental Analyst') || 'Analysis completed.'}
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
                <span>Model:</span>
                <span className="font-mono text-slate-300">GPT-OSS 20B</span>
              </div>
            </div>

            {/* 2. Technical Analyst Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4 flex flex-col justify-between shadow-lg">
              <div className="space-y-3">
                <div className="flex items-center space-x-2.5 text-emerald-400">
                  <div className="p-2 rounded-lg bg-emerald-950 border border-emerald-800">
                    <TrendingUp className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider">Technical Analyst</h4>
                    <span className="text-[10px] text-slate-400">Momentum & Indicators</span>
                  </div>
                </div>

                <div className="text-xs text-slate-300 leading-relaxed whitespace-pre-line bg-slate-950/60 p-4 rounded-xl border border-slate-800 max-h-96 overflow-y-auto">
                  {result.technical_outlook || getBrief('Technical Analyst') || 'Analysis completed.'}
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
                <span>Model:</span>
                <span className="font-mono text-slate-300">GPT-OSS 20B</span>
              </div>
            </div>

            {/* 3. News & Sentiment Agent Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4 flex flex-col justify-between shadow-lg">
              <div className="space-y-3">
                <div className="flex items-center space-x-2.5 text-amber-400">
                  <div className="p-2 rounded-lg bg-amber-950 border border-amber-800">
                    <Newspaper className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider">News & Sentiment</h4>
                    <span className="text-[10px] text-slate-400">Catalysts & Narratives</span>
                  </div>
                </div>

                <div className="text-xs text-slate-300 leading-relaxed whitespace-pre-line bg-slate-950/60 p-4 rounded-xl border border-slate-800 max-h-96 overflow-y-auto">
                  {result.news_sentiment || getBrief('News & Sentiment Agent') || 'Analysis completed.'}
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
                <span>Model:</span>
                <span className="font-mono text-slate-300">GPT-OSS 20B</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
