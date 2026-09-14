import React, { useState } from 'react';
import { Search, BookOpen, Quote, Loader2 } from 'lucide-react';
import { ResearchQAResult } from '../../types';
import { mockApi } from '../../services/mockApi';

interface ResearchQAInterfaceProps {
  className?: string;
  defaultTicker?: string;
}

export const ResearchQAInterface: React.FC<ResearchQAInterfaceProps> = ({
  className = '',
  defaultTicker = 'AAPL',
}) => {
  const [query, setQuery] = useState('');
  const [ticker, setTicker] = useState(defaultTicker);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ResearchQAResult | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    try {
      const data = await mockApi.queryResearchDocuments(query, ticker);
      setResult(data);
    } finally {
      setIsLoading(false);
    }
  };

  const sampleQuestions = [
    'What was management discussion regarding AI capital expenditures?',
    'What is the debt maturity schedule and liquidity buffer?',
    'Are there material supply chain single-source dependencies?',
  ];

  return (
    <div
      data-testid="research-qa-interface"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-4 ${className}`}
    >
      <div className="pb-4 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <BookOpen className="w-5 h-5 text-purple-600" />
          <h3 className="text-base font-bold text-slate-900">
            Document Grounding & SEC Filing Q&A
          </h3>
        </div>
        <p className="text-xs text-slate-500 mt-0.5">
          Ask specific questions against indexed 10-K, 10-Q, and earnings transcripts with traceable citations.
        </p>
      </div>

      {/* Query Form */}
      <form onSubmit={handleSearch} className="space-y-3">
        <div className="flex flex-col sm:flex-row gap-2">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Ticker"
            className="w-full sm:w-28 px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg text-slate-900 font-bold uppercase focus:ring-2 focus:ring-purple-500 focus:outline-none"
          />
          <div className="relative flex-1">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. What does management state regarding Services gross margins?"
              className="w-full pl-9 pr-4 py-2 text-xs bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-purple-500 focus:outline-none"
            />
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          </div>
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="px-4 py-2 rounded-lg bg-purple-700 hover:bg-purple-800 disabled:opacity-40 text-white text-xs font-semibold flex items-center justify-center transition-colors shadow-sm"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                Querying...
              </>
            ) : (
              'Search Filings'
            )}
          </button>
        </div>

        {/* Suggested Queries */}
        <div className="flex items-center space-x-2 overflow-x-auto text-xs">
          <span className="text-[11px] text-slate-400 flex-shrink-0">
            Suggested:
          </span>
          {sampleQuestions.map((q, i) => (
            <button
              key={i}
              type="button"
              onClick={() => {
                setQuery(q);
              }}
              className="text-[11px] px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 whitespace-nowrap"
            >
              {q}
            </button>
          ))}
        </div>
      </form>

      {/* Answer & Citations Card */}
      {result && (
        <div
          data-testid="research-qa-result"
          className="mt-4 pt-4 border-t border-slate-100 space-y-4"
        >
          <div className="bg-purple-50/50 rounded-xl border border-purple-100 p-4">
            <div className="flex items-center justify-between text-xs font-semibold text-purple-900 mb-1.5">
              <span>Synthesized Finding</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-purple-200 text-purple-800">
                Confidence: {(result.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-slate-800 leading-relaxed">
              {result.answer}
            </p>
          </div>

          {/* Traceable Citations */}
          <div>
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2 flex items-center">
              <Quote className="w-3.5 h-3.5 text-purple-600 mr-1.5" />
              Direct Filing Citations
            </h4>
            <div className="space-y-2">
              {result.citations.map((cite, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-white rounded-lg border border-slate-200 text-xs space-y-1 hover:border-slate-300 transition-colors"
                >
                  <div className="flex items-center justify-between text-[11px] font-semibold text-slate-600">
                    <span className="text-purple-700 font-bold">
                      {cite.source_document}{' '}
                      {cite.page ? `(Page ${cite.page})` : ''}
                    </span>
                    <span className="text-slate-400">
                      Relevance: {(cite.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <blockquote className="text-slate-700 italic border-l-2 border-purple-300 pl-2.5 text-[11px]">
                    &ldquo;{cite.excerpt}&rdquo;
                  </blockquote>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
