import React from 'react';
import { RecommendationStance, SpecialistStatus } from '../../types';

interface BadgeProps {
  children?: React.ReactNode;
  variant?:
    | 'default'
    | 'success'
    | 'warning'
    | 'danger'
    | 'info'
    | 'outline'
    | 'stance';
  stance?: RecommendationStance;
  status?: SpecialistStatus;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'default',
  stance,
  status,
  size = 'md',
  className = '',
}) => {
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs',
    lg: 'px-3 py-1.5 text-sm',
  }[size];

  // Auto-format stance
  if (stance) {
    const stanceStyles: Record<RecommendationStance, string> = {
      favorable:
        'bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold',
      cautious:
        'bg-amber-50 text-amber-700 border border-amber-200 font-semibold',
      neutral:
        'bg-slate-100 text-slate-700 border border-slate-200 font-semibold',
      unfavorable:
        'bg-rose-50 text-rose-700 border border-rose-200 font-semibold',
      insufficient_evidence:
        'bg-purple-50 text-purple-700 border border-purple-200 font-semibold',
    };

    return (
      <span
        className={`inline-flex items-center rounded-md uppercase tracking-wider ${stanceStyles[stance]} ${sizeClasses} ${className}`}
      >
        {children || stance.replace('_', ' ')}
      </span>
    );
  }

  // Auto-format specialist status
  if (status) {
    const statusStyles: Record<SpecialistStatus, string> = {
      pending: 'bg-slate-100 text-slate-600 border border-slate-200',
      running: 'bg-sky-50 text-sky-700 border border-sky-200 animate-pulse',
      completed: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
      failed: 'bg-rose-50 text-rose-700 border border-rose-200',
      insufficient_evidence: 'bg-amber-50 text-amber-700 border border-amber-200',
      omitted: 'bg-slate-100 text-slate-400 border border-slate-200',
    };

    return (
      <span
        className={`inline-flex items-center rounded-md font-medium capitalize ${statusStyles[status]} ${sizeClasses} ${className}`}
      >
        {children || status.replace('_', ' ')}
      </span>
    );
  }

  const variantStyles = {
    default: 'bg-slate-100 text-slate-800 border border-slate-200',
    success: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
    warning: 'bg-amber-50 text-amber-700 border border-amber-200',
    danger: 'bg-rose-50 text-rose-700 border border-rose-200',
    info: 'bg-sky-50 text-sky-700 border border-sky-200',
    outline: 'bg-transparent text-slate-700 border border-slate-300',
    stance: 'bg-slate-100 text-slate-800',
  }[variant];

  return (
    <span
      className={`inline-flex items-center rounded-md font-medium ${variantStyles} ${sizeClasses} ${className}`}
    >
      {children}
    </span>
  );
};
