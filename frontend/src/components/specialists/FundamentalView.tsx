import React from 'react';
import { FileSpreadsheet, CheckCircle2 } from 'lucide-react';
import { FundamentalReportSection } from '../../types';

interface FundamentalViewProps {
  data?: FundamentalReportSection;
  className?: string;
}

export const FundamentalView: React.FC<FundamentalViewProps> = ({
  data,
  className = '',
}) => {
  if (!data) {
    return (
      <div className="p-6 text-center text-xs text-slate-500 bg-white rounded-xl border border-slate-200">
        No fundamental analysis data available.
      </div>
    );
  }

  return (
    <div
      data-testid="fundamental-view"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center space-x-2.5 pb-4 border-b border-slate-100">
        <div className="p-2 rounded-lg bg-emerald-50 text-emerald-600 border border-emerald-100">
          <FileSpreadsheet className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-base font-bold text-slate-900">
            Fundamental Analysis Engine
          </h3>
          <p className="text-xs text-slate-500">
            SEC filing financial ratios, capital structure solvency, and margins
          </p>
        </div>
      </div>

      {/* Valuation Multiples Grid */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2.5">
          Valuation Multiples
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-400 uppercase">
              P/E Ratio (TTM)
            </span>
            <div className="text-lg font-black text-slate-900 mt-0.5">
              {data.pe_ratio.toFixed(1)}x
            </div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-400 uppercase">
              P/B Multiple
            </span>
            <div className="text-lg font-black text-slate-900 mt-0.5">
              {data.pb_ratio.toFixed(1)}x
            </div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-400 uppercase">
              EV / EBITDA
            </span>
            <div className="text-lg font-black text-slate-900 mt-0.5">
              {data.ev_ebitda.toFixed(1)}x
            </div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-400 uppercase">
              Free Cash Flow
            </span>
            <div className="text-lg font-black text-slate-900 mt-0.5">
              {data.free_cash_flow}
            </div>
          </div>
        </div>
      </div>

      {/* Profitability & Margins Grid */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2.5">
          Margins & Returns
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-400 uppercase">
              Gross Margin
            </span>
            <div className="text-lg font-black text-slate-900 mt-0.5">
              {data.gross_margin_pct.toFixed(1)}%
            </div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-400 uppercase">
              Operating Margin
            </span>
            <div className="text-lg font-black text-slate-900 mt-0.5">
              {data.operating_margin_pct.toFixed(1)}%
            </div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-400 uppercase">
              Net Profit Margin
            </span>
            <div className="text-lg font-black text-slate-900 mt-0.5">
              {data.net_margin_pct.toFixed(1)}%
            </div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-400 uppercase">
              Return on Equity (ROE)
            </span>
            <div className="text-lg font-black text-slate-900 mt-0.5">
              {data.roe_pct.toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Solvency & Growth */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">
            Debt-to-Equity
          </span>
          <div className="text-base font-bold text-slate-800 mt-0.5">
            {data.debt_to_equity.toFixed(2)}
          </div>
        </div>
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">
            Current Ratio
          </span>
          <div className="text-base font-bold text-slate-800 mt-0.5">
            {data.current_ratio.toFixed(2)}x
          </div>
        </div>
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">
            YoY Revenue Growth
          </span>
          <div className="text-base font-bold text-emerald-700 mt-0.5">
            +{data.revenue_growth_yoy_pct.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Summary Narrative */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
          Fundamental Summary
        </h4>
        <p className="text-xs text-slate-600 leading-relaxed bg-slate-50/50 p-3.5 rounded-lg border border-slate-100">
          {data.summary}
        </p>
      </div>

      {/* Key Findings List */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
          Solvency & Profitability Findings
        </h4>
        <ul className="space-y-1.5 text-xs text-slate-700">
          {data.key_findings.map((item, i) => (
            <li key={i} className="flex items-start space-x-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
