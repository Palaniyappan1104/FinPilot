import React from 'react';
import {
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  BookOpen,
  ShieldAlert,
  CheckCircle2,
  Clock,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { SpecialistStatus, SpecialistType } from '../../types';
import { Badge } from '../common/Badge';

interface SpecialistStatusGridProps {
  specialistStatuses: Record<SpecialistType, SpecialistStatus>;
  className?: string;
  onSelectSpecialist?: (type: SpecialistType) => void;
}

export const SpecialistStatusGrid: React.FC<SpecialistStatusGridProps> = ({
  specialistStatuses,
  className = '',
  onSelectSpecialist,
}) => {
  const specialists: Array<{
    type: SpecialistType;
    title: string;
    description: string;
    icon: React.ReactNode;
    tools: string;
  }> = [
    {
      type: 'technical',
      title: 'Technical Analyst',
      description:
        'Computes mathematical indicators (RSI, SMA, MACD), trend support/resistance levels, and historical volatility.',
      icon: <TrendingUp className="w-5 h-5 text-sky-600" />,
      tools: 'Yahoo Finance Market Data Engine',
    },
    {
      type: 'fundamental',
      title: 'Fundamental Analyst',
      description:
        'Calculates valuation multiples (P/E, P/B, EV/EBITDA), profitability margins, ROE, and debt-to-equity ratios.',
      icon: <FileSpreadsheet className="w-5 h-5 text-emerald-600" />,
      tools: 'SEC Financial Data Provider',
    },
    {
      type: 'news',
      title: 'News & Sentiment Analyst',
      description:
        'Extracts recent media coverage, scores positive/negative sentiment, and identifies emerging market themes.',
      icon: <Newspaper className="w-5 h-5 text-amber-600" />,
      tools: 'Finnhub News & Market Sentiment API',
    },
    {
      type: 'research',
      title: 'Research Vault Analyst',
      description:
        'Embeds and queries local 10-K, 10-Q, and transcript filings to retrieve grounded citations and disclosures.',
      icon: <BookOpen className="w-5 h-5 text-purple-600" />,
      tools: 'ChromaDB Vector Retrieval & SEC Edgar',
    },
    {
      type: 'risk',
      title: 'Risk Analyst',
      description:
        'Calculates 95% Value at Risk (VaR), max drawdown, and categorized risk exposures across market and macro dimensions.',
      icon: <ShieldAlert className="w-5 h-5 text-rose-600" />,
      tools: 'Monte Carlo & Historical Risk Model',
    },
  ];

  const getStatusIcon = (status: SpecialistStatus) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-emerald-600" />;
      case 'running':
        return <Loader2 className="w-4 h-4 text-sky-600 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-rose-600" />;
      case 'insufficient_evidence':
        return <AlertCircle className="w-4 h-4 text-amber-600" />;
      default:
        return <Clock className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div
      data-testid="specialist-status-grid"
      className={`space-y-3 ${className}`}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Autonomous Specialist Execution (Phase 16.6.2)
        </h3>
        <span className="text-[11px] text-slate-400">
          Supervised by Chief Investment Officer (CIO) Agent
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {specialists.map((spec) => {
          const status = specialistStatuses[spec.type] || 'pending';

          return (
            <div
              key={spec.type}
              data-testid={`specialist-card-${spec.type}`}
              onClick={() => onSelectSpecialist?.(spec.type)}
              className={`bg-white rounded-xl border p-4 transition-all ${
                status === 'completed'
                  ? 'border-slate-200 shadow-sm hover:border-slate-300'
                  : status === 'running'
                    ? 'border-sky-300 bg-sky-50/20 shadow-sm ring-1 ring-sky-300'
                    : 'border-slate-200 opacity-80'
              } ${onSelectSpecialist ? 'cursor-pointer' : ''}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2.5">
                  <div className="p-2 rounded-lg bg-slate-50 border border-slate-100">
                    {spec.icon}
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900">
                      {spec.title}
                    </h4>
                    <span className="text-[10px] text-slate-400 block truncate max-w-[140px]">
                      {spec.tools}
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-1.5">
                  {getStatusIcon(status)}
                  <Badge status={status} size="sm" />
                </div>
              </div>

              <p className="mt-3 text-[11px] text-slate-600 leading-relaxed line-clamp-2">
                {spec.description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
