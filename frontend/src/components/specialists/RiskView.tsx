import React from 'react';
import { ShieldAlert, AlertTriangle, ShieldCheck } from 'lucide-react';
import { RiskReportSection } from '../../types';
import { MetricCard } from '../common/MetricCard';

interface RiskViewProps {
  data?: RiskReportSection;
  className?: string;
}

export const RiskView: React.FC<RiskViewProps> = ({
  data,
  className = '',
}) => {
  if (!data) {
    return (
      <div className="p-6 text-center text-xs text-slate-500 bg-white rounded-xl border border-slate-200">
        No risk analysis data available.
      </div>
    );
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'Severe':
        return 'text-rose-700 bg-rose-50 border-rose-200';
      case 'High':
        return 'text-amber-700 bg-amber-50 border-amber-200';
      case 'Moderate':
        return 'text-sky-700 bg-sky-50 border-sky-200';
      default:
        return 'text-emerald-700 bg-emerald-50 border-emerald-200';
    }
  };

  return (
    <div
      data-testid="risk-view"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6 ${className}`}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-2">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-lg bg-rose-50 text-rose-600 border border-rose-100">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Risk Analysis & Downside Assessment
            </h3>
            <p className="text-xs text-slate-500">
              Downside volatility modeling, historical stress, and risk factor classification
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-500">
            Composite Risk Score:
          </span>
          <span className="text-base font-black text-slate-900">
            {data.composite_risk_score} / 100
          </span>
          <span
            className={`text-xs font-bold px-2.5 py-1 rounded-md uppercase tracking-wider border ${getRiskColor(
              data.overall_risk_level,
            )}`}
          >
            {data.overall_risk_level} Risk
          </span>
        </div>
      </div>

      {/* Quantitative Risk Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <MetricCard
          label="Value at Risk (95% 1-Month VaR)"
          value={`${data.var_95_pct.toFixed(1)}%`}
          valueClassName="text-2xl font-black text-slate-900 mt-1"
          subtext="Maximum expected 1-month drawdown at 95% statistical confidence"
          size="lg"
        />

        <MetricCard
          label="Max Historical Drawdown (3-Year)"
          value={`-${data.max_drawdown_pct.toFixed(1)}%`}
          valueClassName="text-2xl font-black text-slate-900 mt-1"
          subtext="Peak-to-trough price decline experienced in trailing 36 months"
          size="lg"
        />
      </div>

      {/* Categorized Risk Factors */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2.5">
          Categorized Risk Breakdown
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/50">
            <span className="font-bold text-slate-800 flex items-center mb-1">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600 mr-1.5" />
              Market & Beta Risk
            </span>
            <p className="text-slate-600 text-[11px]">{data.market_risk}</p>
          </div>

          <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/50">
            <span className="font-bold text-slate-800 flex items-center mb-1">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600 mr-1.5" />
              Valuation Multiple Risk
            </span>
            <p className="text-slate-600 text-[11px]">{data.valuation_risk}</p>
          </div>

          <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/50">
            <span className="font-bold text-slate-800 flex items-center mb-1">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600 mr-1.5" />
              Operational & Supply Chain
            </span>
            <p className="text-slate-600 text-[11px]">{data.operational_risk}</p>
          </div>

          <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/50">
            <span className="font-bold text-slate-800 flex items-center mb-1">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600 mr-1.5" />
              Regulatory & Legal
            </span>
            <p className="text-slate-600 text-[11px]">{data.regulatory_risk}</p>
          </div>
        </div>
      </div>

      {/* Mitigating Factors */}
      {data.mitigating_factors && data.mitigating_factors.length > 0 && (
        <div>
          <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
            Identified Mitigating Factors
          </h4>
          <ul className="space-y-1.5 text-xs text-slate-700">
            {data.mitigating_factors.map((item, idx) => (
              <li key={idx} className="flex items-start space-x-2">
                <ShieldCheck className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Narrative */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-1.5">
          Risk Synthesis
        </h4>
        <p className="text-xs text-slate-600 leading-relaxed bg-slate-50/50 p-3.5 rounded-lg border border-slate-100">
          {data.summary}
        </p>
      </div>
    </div>
  );
};
