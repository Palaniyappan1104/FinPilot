import React from 'react';
import { Database, Link } from 'lucide-react';
import { AggregatedEvidenceItem } from '../../types';

interface EvidenceProvenanceTableProps {
  evidenceList: AggregatedEvidenceItem[];
  className?: string;
}

export const EvidenceProvenanceTable: React.FC<EvidenceProvenanceTableProps> = ({
  evidenceList,
  className = '',
}) => {
  if (!evidenceList || evidenceList.length === 0) {
    return (
      <div className="p-6 text-center text-xs text-slate-500 bg-white rounded-xl border border-slate-200">
        No evidence items recorded.
      </div>
    );
  }

  return (
    <div
      data-testid="evidence-provenance-table"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden ${className}`}
    >
      <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Database className="w-5 h-5 text-emerald-600" />
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Evidence & Grounded Sources Traceability (Phase 16.9.2)
            </h3>
            <p className="text-xs text-slate-500">
              Direct linkage between claims and specialist calculation tools / regulatory filings
            </p>
          </div>
        </div>
        <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700">
          {evidenceList.length} Verified Claims
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-600 divide-y divide-slate-200">
          <thead className="bg-slate-50 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            <tr>
              <th className="px-6 py-3">Claim / Analytical Finding</th>
              <th className="px-4 py-3">Specialist Domain</th>
              <th className="px-4 py-3">Source Tool / Filing</th>
              <th className="px-4 py-3">Confidence</th>
              <th className="px-4 py-3">Verified At</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {evidenceList.map((item) => (
              <tr key={item.id} className="hover:bg-slate-50/60 transition-colors">
                <td className="px-6 py-3.5 font-medium text-slate-900 leading-snug">
                  {item.claim}
                </td>
                <td className="px-4 py-3.5 whitespace-nowrap">
                  <span className="capitalize font-semibold text-slate-700 px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-[11px]">
                    {item.specialist_type}
                  </span>
                </td>
                <td className="px-4 py-3.5 text-slate-600 font-medium flex items-center space-x-1.5 whitespace-nowrap">
                  <Link className="w-3 h-3 text-slate-400" />
                  <span>{item.source_tool}</span>
                </td>
                <td className="px-4 py-3.5 whitespace-nowrap">
                  <span className="text-emerald-700 font-bold">
                    {(item.confidence * 100).toFixed(0)}%
                  </span>
                </td>
                <td className="px-4 py-3.5 text-slate-400 whitespace-nowrap">
                  {new Date(item.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
