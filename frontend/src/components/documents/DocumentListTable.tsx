import React from 'react';
import { FileText, CheckCircle2 } from 'lucide-react';
import { DocumentItem } from '../../types';
import { EmptyState } from '../common/EmptyState';

interface DocumentListTableProps {
  documents: DocumentItem[];
  className?: string;
}

export const DocumentListTable: React.FC<DocumentListTableProps> = ({
  documents,
  className = '',
}) => {
  if (documents.length === 0) {
    return (
      <EmptyState
        title="Filing Vault Is Empty"
        description="No research documents indexed in the vault yet. Upload SEC filings above to begin."
        icon={<FileText className="w-6 h-6 text-slate-500" />}
        className={className}
      />
    );
  }

  return (
    <div
      data-testid="document-list-table"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden ${className}`}
    >
      <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-slate-900">
            Indexed Filing Vault
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Documents available for grounded citation across specialist analyses
          </p>
        </div>
        <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700">
          {documents.length} Filings Active
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-600 divide-y divide-slate-200">
          <thead className="bg-slate-50 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            <tr>
              <th className="px-6 py-3">Document Title</th>
              <th className="px-4 py-3">Ticker</th>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3">Chunks</th>
              <th className="px-4 py-3">Size</th>
              <th className="px-4 py-3">Upload Date</th>
              <th className="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {documents.map((doc) => (
              <tr key={doc.id} className="hover:bg-slate-50/60 transition-colors">
                <td className="px-6 py-3 font-semibold text-slate-900 flex items-center space-x-2">
                  <FileText className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                  <span className="truncate max-w-xs">{doc.filename}</span>
                </td>
                <td className="px-4 py-3 font-bold text-slate-800">
                  {doc.ticker}
                </td>
                <td className="px-4 py-3">{doc.doc_type}</td>
                <td className="px-4 py-3 font-mono">{doc.chunk_count}</td>
                <td className="px-4 py-3 font-mono">
                  {(doc.size_bytes / 1024 / 1024).toFixed(2)} MB
                </td>
                <td className="px-4 py-3 text-slate-400">
                  {new Date(doc.upload_date).toLocaleDateString()}
                </td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                    <CheckCircle2 className="w-3 h-3 mr-1" />
                    Indexed
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
