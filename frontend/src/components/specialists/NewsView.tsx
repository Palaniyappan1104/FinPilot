import React from 'react';
import { Newspaper, Tag } from 'lucide-react';
import { NewsReportSection } from '../../types';

interface NewsViewProps {
  data?: NewsReportSection;
  className?: string;
}

export const NewsView: React.FC<NewsViewProps> = ({
  data,
  className = '',
}) => {
  if (!data) {
    return (
      <div className="p-6 text-center text-xs text-slate-500 bg-white rounded-xl border border-slate-200">
        No news sentiment data available.
      </div>
    );
  }

  return (
    <div
      data-testid="news-view"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6 ${className}`}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-2">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-lg bg-amber-50 text-amber-600 border border-amber-100">
            <Newspaper className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">
              News & Media Sentiment Engine
            </h3>
            <p className="text-xs text-slate-500">
              Scored sentiment distribution and high-relevance media coverage
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-500">
            Aggregate Sentiment:
          </span>
          <span className="text-xs font-bold px-2.5 py-1 rounded-md uppercase tracking-wider bg-emerald-50 text-emerald-700 border border-emerald-200">
            {data.overall_sentiment} ({data.sentiment_score > 0 ? '+' : ''}
            {data.sentiment_score.toFixed(2)})
          </span>
        </div>
      </div>

      {/* Sentiment Breakdown Bar */}
      <div>
        <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
          <span className="text-slate-600">Sentiment Distribution</span>
          <div className="flex space-x-3 text-[11px]">
            <span className="text-emerald-700">Positive: {data.positive_pct}%</span>
            <span className="text-slate-500">Neutral: {data.neutral_pct}%</span>
            <span className="text-rose-700">Negative: {data.negative_pct}%</span>
          </div>
        </div>
        <div className="w-full h-3 rounded-full overflow-hidden flex bg-slate-100">
          <div
            style={{ width: `${data.positive_pct}%` }}
            className="bg-emerald-500 h-full"
            title={`Positive: ${data.positive_pct}%`}
          />
          <div
            style={{ width: `${data.neutral_pct}%` }}
            className="bg-slate-300 h-full"
            title={`Neutral: ${data.neutral_pct}%`}
          />
          <div
            style={{ width: `${data.negative_pct}%` }}
            className="bg-rose-500 h-full"
            title={`Negative: ${data.negative_pct}%`}
          />
        </div>
      </div>

      {/* Key Market Themes */}
      {data.key_themes && data.key_themes.length > 0 && (
        <div>
          <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
            Identified Market Themes
          </h4>
          <div className="flex flex-wrap gap-2">
            {data.key_themes.map((theme, i) => (
              <span
                key={i}
                className="inline-flex items-center text-xs px-3 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200"
              >
                <Tag className="w-3 h-3 mr-1 text-slate-400" />
                {theme}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Top Curated Headlines */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2.5">
          Curated Financial Coverage
        </h4>
        <div className="space-y-2.5">
          {data.top_headlines.map((article) => (
            <div
              key={article.id}
              className="p-3.5 rounded-xl border border-slate-200 bg-slate-50/40 hover:bg-slate-50 transition-colors flex items-start justify-between"
            >
              <div className="space-y-1">
                <h5 className="text-xs font-bold text-slate-900 leading-snug">
                  {article.title}
                </h5>
                <div className="flex items-center space-x-2 text-[11px] text-slate-400">
                  <span className="font-semibold text-slate-600">
                    {article.source}
                  </span>
                  <span>•</span>
                  <span>
                    {new Date(article.published_at).toLocaleDateString()}
                  </span>
                  <span>•</span>
                  <span>
                    Relevance: {(article.relevance_score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ml-3 flex-shrink-0 ${
                  article.sentiment === 'positive'
                    ? 'bg-emerald-100 text-emerald-800'
                    : article.sentiment === 'negative'
                      ? 'bg-rose-100 text-rose-800'
                      : 'bg-slate-200 text-slate-700'
                }`}
              >
                {article.sentiment}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-1.5">
          Sentiment Narrative
        </h4>
        <p className="text-xs text-slate-600 leading-relaxed bg-slate-50/50 p-3.5 rounded-lg border border-slate-100">
          {data.summary}
        </p>
      </div>
    </div>
  );
};
