import React from 'react';
import { Activity, CheckCircle2 } from 'lucide-react';
import { TechnicalReportSection } from '../../types';
import { MetricCard } from '../common/MetricCard';

interface TechnicalViewProps {
  data?: TechnicalReportSection;
  className?: string;
}

export const TechnicalView: React.FC<TechnicalViewProps> = ({
  data,
  className = '',
}) => {
  if (!data) {
    return (
      <div className="p-6 text-center text-xs text-slate-500 bg-white rounded-xl border border-slate-200">
        No technical analysis data available for this session.
      </div>
    );
  }

  const isBullish = data.trend === 'bullish';

  return (
    <div
      data-testid="technical-view"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6 ${className}`}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-2">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-lg bg-sky-50 text-sky-600 border border-sky-100">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Technical Analysis Engine
            </h3>
            <p className="text-xs text-slate-500">
              Computed mathematical momentum indicators and trend boundaries
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-500">Trend:</span>
          <span
            className={`text-xs font-bold px-2.5 py-1 rounded-md uppercase tracking-wider ${
              isBullish
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : 'bg-amber-50 text-amber-700 border border-amber-200'
            }`}
          >
            {data.trend}
          </span>
        </div>
      </div>

      {/* Primary Indicator Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="RSI (14-Day)"
          value={data.rsi_14.toFixed(1)}
          size="lg"
          badge={
            <span
              className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase mt-1 inline-block ${
                data.rsi_status === 'overbought'
                  ? 'bg-rose-100 text-rose-800'
                  : data.rsi_status === 'oversold'
                    ? 'bg-emerald-100 text-emerald-800'
                    : 'bg-slate-200 text-slate-700'
              }`}
            >
              {data.rsi_status}
            </span>
          }
        />

        <MetricCard
          label="SMA Alignment"
          size="lg"
        >
          <div className="text-xs font-bold text-slate-800 mt-1 space-y-0.5">
            <div>20d: ${data.sma_20.toFixed(1)}</div>
            <div>50d: ${data.sma_50.toFixed(1)}</div>
            <div>200d: ${data.sma_200.toFixed(1)}</div>
          </div>
        </MetricCard>

        <MetricCard
          label="MACD Stance"
          value={data.macd_signal}
          valueClassName="text-xl font-black text-slate-900 mt-1 capitalize"
          subtext="Histogram slope positive"
          size="lg"
        />

        <MetricCard
          label="30d Volatility"
          value={`${data.volatility_30d_pct.toFixed(1)}%`}
          subtext="Annualized standard dev"
          size="lg"
        />
      </div>

      {/* Support & Resistance Bar */}
      <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 flex items-center justify-between text-xs">
        <div>
          <span className="text-[11px] font-semibold text-slate-400 uppercase block">
            Primary Support
          </span>
          <span className="text-sm font-bold text-emerald-700">
            ${data.support_level.toFixed(2)}
          </span>
        </div>

        <div className="h-4 w-px bg-slate-200" />

        <div className="text-center">
          <span className="text-[11px] font-semibold text-slate-400 uppercase block">
            Trading Range
          </span>
          <span className="text-xs font-medium text-slate-600">
            ${data.support_level.toFixed(0)} – ${data.resistance_level.toFixed(0)}
          </span>
        </div>

        <div className="h-4 w-px bg-slate-200" />

        <div className="text-right">
          <span className="text-[11px] font-semibold text-slate-400 uppercase block">
            Overhead Resistance
          </span>
          <span className="text-sm font-bold text-rose-700">
            ${data.resistance_level.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Summary Narrative */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
          Technical Summary
        </h4>
        <p className="text-xs text-slate-600 leading-relaxed bg-slate-50/50 p-3.5 rounded-lg border border-slate-100">
          {data.summary}
        </p>
      </div>

      {/* Key Findings List */}
      <div>
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
          Quantitative Observations
        </h4>
        <ul className="space-y-1.5 text-xs text-slate-700">
          {data.key_findings.map((item, i) => (
            <li key={i} className="flex items-start space-x-2">
              <CheckCircle2 className="w-4 h-4 text-sky-600 flex-shrink-0 mt-0.5" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
