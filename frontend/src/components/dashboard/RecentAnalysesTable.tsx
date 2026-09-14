import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Layers } from 'lucide-react';
import { AnalysisSummary } from '../../types';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';

interface RecentAnalysesTableProps {
  analyses: AnalysisSummary[];
  className?: string;
}

export const RecentAnalysesTable: React.FC<RecentAnalysesTableProps> = ({
  analyses,
  className = '',
}) => {
  const navigate = useNavigate();

  if (!analyses || analyses.length === 0) {
    return (
      <EmptyState
        title="No Research Analyses Yet"
        description="No research analyses completed yet. Launch a new analysis from the query box."
        icon={<FileText className="w-6 h-6 text-slate-500" />}
        actionLabel="Start New Analysis"
        onAction={() => navigate('/analysis/new')}
        className={className}
      />
    );
  }

  return (
    <div
      data-testid="recent-analyses-table"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden ${className}`}
    >
      <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-slate-900">
            Recent Multi-Agent Research Analyses
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Dossiers generated across specialist domains with statistical confidence
          </p>
        </div>
        <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700">
          {analyses.length} Reports
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-600 divide-y divide-slate-200">
          <thead className="bg-slate-50 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            <tr>
              <th className="px-6 py-3">Equity / Ticker</th>
              <th className="px-4 py-3">Analysis Date</th>
              <th className="px-4 py-3">Recommendation Stance</th>
              <th className="px-4 py-3">Confidence</th>
              <th className="px-4 py-3">Horizon / Risk</th>
              <th className="px-6 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {analyses.map((item) => (
              <tr
                key={item.analysis_id}
                className="hover:bg-slate-50/60 transition-colors"
              >
                <td className="px-6 py-3.5 font-semibold text-slate-900">
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-slate-900">
                      {item.company_name}
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 text-[11px] font-mono border border-slate-200">
                      {item.ticker}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3.5 text-slate-500">
                  {new Date(item.created_at).toLocaleDateString()}
                </td>
                <td className="px-4 py-3.5">
                  {item.recommendation_stance ? (
                    <Badge stance={item.recommendation_stance} size="sm" />
                  ) : (
                    <span className="text-slate-400 capitalize text-xs">
                      {item.status}
                    </span>
                  )}
                </td>
                <td className="px-4 py-3.5">
                  <div className="flex items-center space-x-2">
                    <div className="w-12 bg-slate-100 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-500 h-full"
                        style={{ width: `${item.confidence * 100}%` }}
                      />
                    </div>
                    <span className="font-bold text-slate-800 text-[11px]">
                      {(item.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3.5 text-[11px] text-slate-500">
                  <span className="block font-medium text-slate-700">
                    {item.horizon || '3-5 years'}
                  </span>
                  <span className="capitalize text-slate-400">
                    {item.risk_tolerance || 'Moderate'}
                  </span>
                </td>
                <td className="px-6 py-3.5 text-right space-x-2 whitespace-nowrap">
                  {item.report_id && (
                    <button
                      type="button"
                      onClick={() => navigate(`/reports/${item.report_id}`)}
                      className="inline-flex items-center px-2.5 py-1 text-xs font-semibold rounded-lg bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 transition-colors"
                    >
                      <FileText className="w-3.5 h-3.5 mr-1" />
                      Report
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() =>
                      navigate(`/analysis/${item.analysis_id}/specialists`)
                    }
                    className="inline-flex items-center px-2.5 py-1 text-xs font-semibold rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition-colors"
                  >
                    <Layers className="w-3.5 h-3.5 mr-1 text-slate-400" />
                    Specialists
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
