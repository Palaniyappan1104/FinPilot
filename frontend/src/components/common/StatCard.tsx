import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  change?: string;
  isPositive?: boolean;
  icon?: React.ReactNode;
  subtitle?: string;
  className?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  change,
  isPositive,
  icon,
  subtitle,
  className = '',
}) => {
  return (
    <div
      className={`bg-white rounded-xl border border-slate-200 p-5 shadow-sm transition-all hover:border-slate-300 ${className}`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-500">
          {label}
        </span>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>
      <div className="mt-2 flex items-baseline justify-between">
        <div className="text-2xl font-bold tracking-tight text-slate-900">
          {value}
        </div>
        {change && (
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full ${
              isPositive === true
                ? 'bg-emerald-50 text-emerald-700 font-semibold'
                : isPositive === false
                  ? 'bg-rose-50 text-rose-700 font-semibold'
                  : 'bg-slate-100 text-slate-600'
            }`}
          >
            {change}
          </span>
        )}
      </div>
      {subtitle && (
        <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
      )}
    </div>
  );
};
