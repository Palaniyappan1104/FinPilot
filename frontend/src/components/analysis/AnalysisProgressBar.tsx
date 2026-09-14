import React from 'react';
import { CheckCircle2, CircleDot, Clock } from 'lucide-react';

interface AnalysisProgressBarProps {
  progressPercent: number;
  currentStage: string;
  className?: string;
}

export const AnalysisProgressBar: React.FC<AnalysisProgressBarProps> = ({
  progressPercent,
  currentStage,
  className = '',
}) => {
  const steps = [
    { label: 'Query & Profile', threshold: 20 },
    { label: 'CIO Allocation', threshold: 40 },
    { label: 'Specialist Execution', threshold: 60 },
    { label: 'Aggregation & Conflict Check', threshold: 80 },
    { label: 'Final Report Formatted', threshold: 100 },
  ];

  return (
    <div
      data-testid="analysis-progress-bar"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 ${className}`}
    >
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block">
            Multi-Agent Workflow Pipeline
          </span>
          <h3 className="text-sm font-semibold text-slate-900 mt-0.5">
            {currentStage}
          </h3>
        </div>
        <div className="text-right">
          <span className="text-xl font-black text-slate-900">
            {progressPercent}%
          </span>
          <span className="block text-[10px] text-slate-400 font-medium">
            COMPLETION
          </span>
        </div>
      </div>

      {/* Progress Track */}
      <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden mb-6">
        <div
          role="progressbar"
          aria-valuenow={progressPercent}
          aria-valuemin={0}
          aria-valuemax={100}
          className="bg-emerald-500 h-full rounded-full transition-all duration-500 ease-out"
          style={{ width: `${Math.min(100, Math.max(5, progressPercent))}%` }}
        />
      </div>

      {/* Stepper Pipeline */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-2 border-t border-slate-100">
        {steps.map((step, idx) => {
          const isDone = progressPercent >= step.threshold;
          const isCurrent =
            progressPercent < step.threshold &&
            (idx === 0 || progressPercent >= steps[idx - 1].threshold);

          return (
            <div
              key={step.label}
              className={`flex items-center space-x-2 text-xs p-2 rounded-lg ${
                isDone
                  ? 'text-emerald-800 bg-emerald-50/60 font-semibold'
                  : isCurrent
                    ? 'text-sky-800 bg-sky-50 font-bold border border-sky-200 animate-pulse'
                    : 'text-slate-400 bg-slate-50'
              }`}
            >
              {isDone ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              ) : isCurrent ? (
                <CircleDot className="w-4 h-4 text-sky-600 flex-shrink-0 animate-spin" />
              ) : (
                <Clock className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
              )}
              <span className="truncate">{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
