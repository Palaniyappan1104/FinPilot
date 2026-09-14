import React from 'react';

export interface MetricCardProps {
  label: string;
  value?: React.ReactNode;
  unit?: string;
  subtext?: React.ReactNode;
  badge?: React.ReactNode;
  children?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  valueClassName?: string;
  testId?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  subtext,
  badge,
  children,
  size = 'md',
  className = '',
  valueClassName = '',
  testId,
}) => {
  const sizeClasses = {
    sm: 'p-2.5 rounded-lg',
    md: 'p-3.5 rounded-xl',
    lg: 'p-4 rounded-xl',
  }[size];

  const defaultValueClasses = {
    sm: 'text-sm font-bold text-slate-800 mt-0.5',
    md: 'text-lg font-black text-slate-900 mt-0.5',
    lg: 'text-xl font-black text-slate-900 mt-1',
  }[size];

  return (
    <div
      data-testid={testId}
      className={`bg-slate-50 border border-slate-100 ${sizeClasses} ${className}`}
    >
      <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
        {label}
      </div>

      {value !== undefined && (
        <div className={valueClassName || defaultValueClasses}>
          {value}
          {unit && <span className="ml-0.5 text-xs font-normal text-slate-500">{unit}</span>}
        </div>
      )}

      {children}

      {badge}

      {subtext && (
        <span className="text-[10px] sm:text-[11px] text-slate-500 mt-0.5 block">
          {subtext}
        </span>
      )}
    </div>
  );
};
