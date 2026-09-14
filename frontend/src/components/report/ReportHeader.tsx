import React from 'react';
import { Calendar, Hash, Award, Building } from 'lucide-react';
import { FinalReport } from '../../types';

interface ReportHeaderProps {
  report: FinalReport;
  className?: string;
}

export const ReportHeader: React.FC<ReportHeaderProps> = ({
  report,
  className = '',
}) => {
  return (
    <div
      data-testid="report-header"
      className={`bg-slate-900 text-white rounded-xl p-6 shadow-sm border border-slate-800 ${className}`}
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-emerald-400 text-xs font-semibold uppercase tracking-wider mb-1">
            <Building className="w-3.5 h-3.5" />
            <span>Multi-Agent Research Dossier</span>
          </div>
          <h2 className="text-2xl font-black tracking-tight text-white flex items-center">
            {report.company.name}
            <span className="ml-3 text-xs font-bold px-2 py-0.5 rounded bg-slate-800 text-emerald-300 border border-slate-700">
              {report.company.ticker}
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Sector: {report.company.sector} • Currency: {report.company.currency}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs">
          <div className="bg-slate-800/80 px-3.5 py-2 rounded-lg border border-slate-700/80">
            <div className="text-[10px] text-slate-400 font-semibold uppercase flex items-center">
              <Calendar className="w-3 h-3 mr-1" />
              Generated Date
            </div>
            <div className="font-medium text-slate-200 mt-0.5">
              {new Date(report.created_at).toLocaleDateString()}
            </div>
          </div>

          <div className="bg-slate-800/80 px-3.5 py-2 rounded-lg border border-slate-700/80">
            <div className="text-[10px] text-slate-400 font-semibold uppercase flex items-center">
              <Award className="w-3 h-3 mr-1 text-emerald-400" />
              Agent Confidence
            </div>
            <div className="font-bold text-emerald-400 mt-0.5">
              {(report.confidence * 100).toFixed(0)}% Statistical
            </div>
          </div>

          <div className="bg-slate-800/80 px-3.5 py-2 rounded-lg border border-slate-700/80">
            <div className="text-[10px] text-slate-400 font-semibold uppercase flex items-center">
              <Hash className="w-3 h-3 mr-1" />
              Report ID
            </div>
            <div className="font-mono text-slate-300 mt-0.5">
              {report.report_id}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
