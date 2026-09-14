import React from 'react';
import { Target, CheckCircle2, Clock, ShieldCheck, AlertCircle } from 'lucide-react';
import { ReportRecommendation } from '../../types';
import { Badge } from '../common/Badge';

interface ExecutiveRecommendationProps {
  recommendation?: ReportRecommendation;
  overallSummary: string;
  specialistConsensus?: string;
  signalConflicts?: string[];
  className?: string;
}

export const ExecutiveRecommendation: React.FC<
  ExecutiveRecommendationProps
> = ({
  recommendation,
  overallSummary,
  specialistConsensus,
  signalConflicts = [],
  className = '',
}) => {
  return (
    <div
      data-testid="executive-recommendation"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-5 ${className}`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-3">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-lg bg-emerald-50 text-emerald-600 border border-emerald-100">
            <Target className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Executive Assessment & Decision Support
            </h3>
            <p className="text-xs text-slate-500">
              Synthesized by Report Generator Agent aligned with investor profile
            </p>
          </div>
        </div>

        {recommendation && (
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold text-slate-500">
              Analytical Stance:
            </span>
            <Badge stance={recommendation.stance} size="lg" />
          </div>
        )}
      </div>

      {/* Rationale & Summary */}
      <div className="space-y-3">
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
            Recommendation Rationale
          </span>
          <p className="text-xs leading-relaxed text-slate-800 font-medium">
            {recommendation?.rationale || overallSummary}
          </p>
        </div>

        {specialistConsensus && (
          <div className="text-xs text-slate-600 bg-white p-3 rounded-lg border border-slate-200">
            <span className="font-bold text-slate-800 block mb-0.5">
              Specialist Cross-Domain Consensus:
            </span>
            <p>{specialistConsensus}</p>
          </div>
        )}

        {signalConflicts.length > 0 && (
          <div className="p-3.5 rounded-lg bg-amber-50/60 border border-amber-200 text-xs">
            <span className="font-bold text-amber-900 flex items-center mb-1">
              <AlertCircle className="w-4 h-4 text-amber-600 mr-1.5" />
              Identified Signal Conflicts & Analytical Tensions
            </span>
            <ul className="list-disc list-inside text-amber-800 space-y-0.5 mt-1">
              {signalConflicts.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Suitability Matrix */}
      {recommendation && (
        <div className="pt-2 border-t border-slate-100">
          <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-3">
            Investor Profile Suitability Matrix
          </h4>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200/70">
              <div className="flex items-center text-slate-700 font-bold mb-1">
                <Clock className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
                Time Horizon Match
              </div>
              <p className="text-[11px] text-slate-600 leading-snug">
                {recommendation.time_horizon_suitability ||
                  'Not specified in report synthesis.'}
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200/70">
              <div className="flex items-center text-slate-700 font-bold mb-1">
                <ShieldCheck className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
                Risk Tolerance Match
              </div>
              <p className="text-[11px] text-slate-600 leading-snug">
                {recommendation.risk_tolerance_suitability ||
                  'Not specified in report synthesis.'}
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200/70">
              <div className="flex items-center text-slate-700 font-bold mb-1">
                <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
                Capital Sizing Notes
              </div>
              <p className="text-[11px] text-slate-600 leading-snug">
                {recommendation.capital_allocation_notes ||
                  'Not specified in report synthesis.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Key Monitoring Points */}
      {recommendation?.monitoring_points &&
        recommendation.monitoring_points.length > 0 && (
          <div className="pt-2 border-t border-slate-100">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
              Critical Milestones to Monitor
            </h4>
            <ul className="space-y-1.5 text-xs text-slate-700">
              {recommendation.monitoring_points.map((pt, i) => (
                <li key={i} className="flex items-start space-x-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0 mt-0.5" />
                  <span>{pt}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
    </div>
  );
};
