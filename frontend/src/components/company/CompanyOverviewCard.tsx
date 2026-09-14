import React from 'react';
import { TrendingUp, TrendingDown, ShieldCheck } from 'lucide-react';
import { CompanyInfo } from '../../types';

interface CompanyOverviewCardProps {
  company: CompanyInfo;
  className?: string;
  onAnalyzeClick?: () => void;
}

export const CompanyOverviewCard: React.FC<CompanyOverviewCardProps> = ({
  company,
  className = '',
  onAnalyzeClick,
}) => {
  const isPositive = company.change >= 0;

  return (
    <div
      data-testid="company-overview-card"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 overflow-hidden relative ${className}`}
    >
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-4">
        <div className="flex items-start space-x-3">
          <div className="w-12 h-12 rounded-xl bg-slate-900 text-white font-black text-base flex items-center justify-center flex-shrink-0">
            {company.ticker.slice(0, 3)}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-xl font-bold text-slate-900">{company.name}</h2>
              <span className="px-2 py-0.5 rounded text-xs font-bold bg-slate-100 text-slate-700 border border-slate-200">
                {company.ticker}
              </span>
            </div>
            <div className="text-xs text-slate-500 mt-0.5 flex items-center space-x-2">
              <span>{company.exchange}</span>
              <span>•</span>
              <span>{company.sector}</span>
              <span>•</span>
              <span>{company.industry}</span>
            </div>
          </div>
        </div>

        <div className="text-left sm:text-right">
          <div className="text-2xl font-black text-slate-900">
            {company.currency === 'INR' ? '₹' : '$'}
            {company.price.toLocaleString()}
          </div>
          <div
            className={`text-xs font-semibold flex items-center sm:justify-end ${
              isPositive ? 'text-emerald-600' : 'text-rose-600'
            }`}
          >
            {isPositive ? (
              <TrendingUp className="w-3.5 h-3.5 mr-1" />
            ) : (
              <TrendingDown className="w-3.5 h-3.5 mr-1" />
            )}
            <span>
              {isPositive ? '+' : ''}
              {company.change.toFixed(2)} ({isPositive ? '+' : ''}
              {company.changePercent}%)
            </span>
          </div>
        </div>
      </div>

      {/* Description */}
      <p className="text-xs leading-relaxed text-slate-600 my-4">
        {company.description}
      </p>

      {/* Key Financial Multiples Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-slate-100 text-xs">
        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
          <span className="text-slate-400 block text-[11px]">Market Cap</span>
          <span className="font-bold text-slate-800 text-sm mt-0.5 block">
            {company.marketCap}
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
          <span className="text-slate-400 block text-[11px]">P/E (TTM)</span>
          <span className="font-bold text-slate-800 text-sm mt-0.5 block">
            {company.peRatio}x
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
          <span className="text-slate-400 block text-[11px]">52-Week High</span>
          <span className="font-bold text-slate-800 text-sm mt-0.5 block">
            {company.currency === 'INR' ? '₹' : '$'}
            {company.fiftyTwoWeekHigh.toLocaleString()}
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
          <span className="text-slate-400 block text-[11px]">52-Week Low</span>
          <span className="font-bold text-slate-800 text-sm mt-0.5 block">
            {company.currency === 'INR' ? '₹' : '$'}
            {company.fiftyTwoWeekLow.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Action footer */}
      {onAnalyzeClick && (
        <div className="mt-4 pt-3 flex items-center justify-between border-t border-slate-100">
          <span className="text-[11px] text-slate-400 flex items-center">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-500 mr-1" />
            Specialist coverage active
          </span>
          <button
            type="button"
            onClick={onAnalyzeClick}
            className="inline-flex items-center px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-xs transition-colors shadow-sm"
          >
            Launch Multi-Agent Analysis
          </button>
        </div>
      )}
    </div>
  );
};
