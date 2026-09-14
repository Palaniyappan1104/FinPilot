import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  label = 'Loading research data...',
  size = 'md',
  className = '',
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  }[size];

  return (
    <div
      role="status"
      aria-label={label}
      className={`flex flex-col items-center justify-center p-6 text-slate-500 ${className}`}
    >
      <Loader2 className={`${sizeClasses} animate-spin text-slate-600 mb-2`} />
      {label && <span className="text-xs font-medium text-slate-600">{label}</span>}
    </div>
  );
};
