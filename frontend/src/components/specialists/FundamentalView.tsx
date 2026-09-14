import React from 'react';
import { FileSpreadsheet, CheckCircle2 } from 'lucide-react';
import { FundamentalReportSection } from '../../types';
import { MetricCard } from '../common/MetricCard';

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

  const valuationMetrics = [
    { label: 'P/E Ratio (TTM)', value: `${data.pe_ratio.toFixed(1)}x` },
    { label: 'P/B Multiple', value: `${data.pb_ratio.toFixed(1)}x` },
    { label: 'EV / EBITDA', value: `${data.ev_ebitda.toFixed(1)}x` },
    { label: 'Free Cash Flow', value: data.free_cash_flow },
  ];

  const marginMetrics = [
    { label: 'Gross Margin', value: `${data.gross_margin_pct.toFixed(1)}%` },
    { label: 'Operating Margin', value: `${data.operating_margin_pct.toFixed(1)}%` },
    { label: 'Net Profit Margin', value: `${data.net_margin_pct.toFixed(1)}%` },
    { label: 'Return on Equity (ROE)', value: `${data.roe_pct.toFixed(1)}%` },
  ];

  const solvencyMetrics = [
    {
      label: 'Debt-to-Equity',
      value: data.debt_to_equity.toFixed(2),
      valueClassName: 'text-base font-bold text-slate-800 mt-0.5',
    },
    {
      label: 'Current Ratio',
      value: `${data.current_ratio.toFixed(2)}x`,
      valueClassName: 'text-base font-bold text-slate-800 mt-0.5',
    },
    {
      label: 'YoY Revenue Growth',
      value: `+${data.revenue_growth_yoy_pct.toFixed(1)}%`,
      valueClassName: 'text-base font-bold text-emerald-700 mt-0.5',
    },
  ];

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
          {valuationMetrics.map((m) => (
            <MetricCard key={m.label} label={m.label} value={m.value} size="md" />
          ))}
        </div>
      </div>

      {/* Profitability & Margins Grid */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2.5">
          Margins & Returns
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {marginMetrics.map((m) => (
            <MetricCard key={m.label} label={m.label} value={m.value} size="md" />
          ))}
        </div>
      </div>

      {/* Solvency & Growth */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {solvencyMetrics.map((m) => (
          <MetricCard
            key={m.label}
            label={m.label}
            value={m.value}
            valueClassName={m.valueClassName}
            size="md"
          />
        ))}
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
