import React, { useState } from 'react';
import {
  Layers,
  Sparkles,
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  Clock,
  AlertCircle,
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
      {/* Header Banner */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-850 to-blue-950 border border-slate-800 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-sm text-white">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-blue-300">
            <Layers className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">Templated Specialist Flow</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            Multi-Agent Specialist Pipeline
          </h1>
          <p className="text-xs text-slate-200 max-w-2xl leading-relaxed">
            Select a stock to dispatch the 3 primary specialist agents (Fundamental, Technical, News & Sentiment) via structured prompts and inspect their combined synthesis.
          </p>
        </div>

        <button
          type="button"
          onClick={handleRunSpecialists}
          disabled={isRunning}
          className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-bold text-xs sm:text-sm flex items-center justify-center space-x-2 transition-all shadow-xs self-start sm:self-auto"
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
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700 flex items-center space-x-2 shadow-xs">
          <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Stock Selection & Prompt Templating Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Stock Picker */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-2xl p-6 space-y-4 shadow-xs">
          <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
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
                    ? 'bg-blue-50 border-blue-500 text-blue-900 shadow-xs'
                    : 'bg-slate-50 border-slate-200 text-slate-700 hover:border-slate-300'
                }`}
              >
                <div className="font-mono font-bold text-xs text-blue-700">
                  {stock.ticker}
                </div>
                <div className="text-[11px] text-slate-500 truncate mt-0.5 font-medium">
                  {stock.name}
                </div>
              </button>
            ))}
          </div>

          <div className="pt-2 border-t border-slate-100">
            <label className="text-[11px] font-semibold text-slate-500 block mb-1.5">
              Or specify custom ticker / exchange symbol:
            </label>
            <input
              type="text"
              placeholder="e.g. AMD, RELIANCE.NS, SAP.DE"
              value={customTicker}
              onChange={(e) => setCustomTicker(e.target.value)}
              className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-slate-900 placeholder-slate-400 text-xs uppercase font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
            />
          </div>
        </div>

        {/* Right: Templated Prompt Card */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-2xl p-6 space-y-4 flex flex-col justify-between shadow-xs">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                2. Templated Specialist Dispatch Prompt
              </h2>
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                Automated
              </span>
            </div>
            <p className="text-xs text-slate-500">
              This prompt automatically binds target equity facts and orchestrates the 3 specialist agents concurrently:
            </p>

            <pre className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-slate-800 font-mono text-xs whitespace-pre-wrap leading-relaxed">
              {templatedPrompt}
            </pre>
          </div>

          <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Specialists active:</span>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 text-[10px] font-bold">Fundamental</span>
              <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px] font-bold">Technical</span>
              <span className="px-2 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200 text-[10px] font-bold">News & Sentiment</span>
            </div>
          </div>
        </div>
      </div>

      {/* Loading state indicator */}
      {isRunning && (
        <div className="bg-white border border-slate-200 rounded-2xl p-8 text-center space-y-4 shadow-xs animate-pulse">
          <Clock className="w-8 h-8 text-blue-600 animate-spin mx-auto" />
          <div>
            <h3 className="text-base font-bold text-slate-900">Running 3 Specialist Agents for {activeTicker}...</h3>
            <p className="text-xs text-slate-500 mt-1">Fetching live market data via yfinance, executing Groq specialist passes, and synthesizing combined takeaway.</p>
          </div>
        </div>
      )}

      {/* Results View */}
      {result && !isRunning && (
        <div className="space-y-6 animate-fadeIn">
          {/* Combined Synthesis Banner */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 space-y-4 shadow-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-blue-600">
                  Combined Pipeline Synthesis
                </span>
                <h3 className="text-xl sm:text-2xl font-black text-slate-900">
                  {result.company} ({result.ticker})
                </h3>
              </div>

              <div className="flex items-center space-x-2">
                <span className="px-3 py-1 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-bold">
                  Confidence: {result.confidence}
                </span>
              </div>
            </div>

            <div className="text-xs sm:text-sm text-slate-700 leading-relaxed whitespace-pre-line">
              {result.final_verdict}
            </div>

            {/* Quick Metrics Bar */}
            {result.key_metrics && Object.keys(result.key_metrics).length > 0 && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                {Object.entries(result.key_metrics).slice(0, 4).map(([k, v]) => (
                  <div key={k} className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <div className="text-[10px] text-slate-500">{k}</div>
                    <div className="text-xs font-bold font-mono text-slate-900 mt-0.5">{v}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 3 Specialist Outputs Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* 1. Fundamental Analyst Card */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 flex flex-col justify-between shadow-xs">
              <div className="space-y-3">
                <div className="flex items-center space-x-2.5 text-blue-600">
                  <div className="p-2 rounded-lg bg-blue-50 border border-blue-200">
                    <FileSpreadsheet className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Fundamental Analyst</h4>
                    <span className="text-[10px] text-slate-500">Valuation & Health</span>
                  </div>
                </div>

                <div className="text-xs text-slate-700 leading-relaxed whitespace-pre-line bg-slate-50 p-4 rounded-xl border border-slate-200 max-h-96 overflow-y-auto">
                  {result.fundamental_view || getBrief('Fundamental Analyst') || 'Analysis completed.'}
                </div>
              </div>

              <div className="pt-3 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
                <span>Model:</span>
                <span className="font-mono font-medium text-slate-700">Groq Engine</span>
              </div>
            </div>

            {/* 2. Technical Analyst Card */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 flex flex-col justify-between shadow-xs">
              <div className="space-y-3">
                <div className="flex items-center space-x-2.5 text-emerald-600">
                  <div className="p-2 rounded-lg bg-emerald-50 border border-emerald-200">
                    <TrendingUp className="w-5 h-5 text-emerald-600" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Technical Analyst</h4>
                    <span className="text-[10px] text-slate-500">Momentum & Indicators</span>
                  </div>
                </div>

                <div className="text-xs text-slate-700 leading-relaxed whitespace-pre-line bg-slate-50 p-4 rounded-xl border border-slate-200 max-h-96 overflow-y-auto">
                  {result.technical_outlook || getBrief('Technical Analyst') || 'Analysis completed.'}
                </div>
              </div>

              <div className="pt-3 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
                <span>Model:</span>
                <span className="font-mono font-medium text-slate-700">Groq Engine</span>
              </div>
            </div>

            {/* 3. News & Sentiment Agent Card */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 flex flex-col justify-between shadow-xs">
              <div className="space-y-3">
                <div className="flex items-center space-x-2.5 text-amber-600">
                  <div className="p-2 rounded-lg bg-amber-50 border border-amber-200">
                    <Newspaper className="w-5 h-5 text-amber-600" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">News & Sentiment</h4>
                    <span className="text-[10px] text-slate-500">Catalysts & Narratives</span>
                  </div>
                </div>

                <div className="text-xs text-slate-700 leading-relaxed whitespace-pre-line bg-slate-50 p-4 rounded-xl border border-slate-200 max-h-96 overflow-y-auto">
                  {result.news_sentiment || getBrief('News & Sentiment Agent') || 'Analysis completed.'}
                </div>
              </div>

              <div className="pt-3 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
                <span>Model:</span>
                <span className="font-mono font-medium text-slate-700">Groq Engine</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
