import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  FileText,
  Clock,
  Trash2,
  ChevronRight,
  ArrowLeft,
  Sparkles,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { ReportDossier } from '../components/report/ReportDossier';

export const ReportsPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { storedReports, deleteReport } = useApp();

  const [selectedReportId, setSelectedReportId] = useState<string | null>(id || null);

  useEffect(() => {
    if (id) {
      setSelectedReportId(id);
    }
  }, [id]);

  const activeStoredReport = storedReports.find((r) => r.id === selectedReportId);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-3">
        <div>
          <div className="flex items-center space-x-2 text-blue-600">
            <FileText className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">Archive & Repository</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Generated Research Reports
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Institutional investment reports automatically archived from multi-agent analysis sessions.
          </p>
        </div>

        {selectedReportId && (
          <button
            type="button"
            onClick={() => setSelectedReportId(null)}
            className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 text-xs font-semibold transition-colors self-start sm:self-auto"
          >
            <ArrowLeft className="w-3.5 h-3.5 mr-1.5" />
            <span>Back to All Reports</span>
          </button>
        )}
      </div>

      {/* Selected Report Dossier View */}
      {selectedReportId && activeStoredReport ? (
        <div className="space-y-4">
          <ReportDossier
            report={activeStoredReport.data}
            onReset={() => navigate('/analysis/new')}
          />
        </div>
      ) : (
        /* Report List View */
        <div className="space-y-4">
          {storedReports.length === 0 ? (
            <div className="bg-white border border-slate-200 rounded-2xl p-12 text-center space-y-4 max-w-lg mx-auto shadow-xs">
              <div className="w-12 h-12 rounded-2xl bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600 mx-auto">
                <FileText className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-slate-900">No Reports Generated Yet</h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Whenever you run an investment analysis via New Analysis or Multi-Agent Analysis, its full institutional dossier is automatically preserved here.
                </p>
              </div>
              <button
                type="button"
                onClick={() => navigate('/analysis/new')}
                className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs inline-flex items-center space-x-1.5 transition-colors shadow-xs"
              >
                <Sparkles className="w-4 h-4" />
                <span>Start Your First Analysis</span>
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3">
              {storedReports.map((rep) => {
                const dateStr = new Date(rep.timestamp).toLocaleString(undefined, {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                });

                const conf = rep.confidence || 'Medium';
                const confColor =
                  conf.toLowerCase() === 'high'
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    : conf.toLowerCase() === 'low'
                      ? 'bg-rose-50 text-rose-700 border-rose-200'
                      : 'bg-amber-50 text-amber-700 border-amber-200';

                return (
                  <div
                    key={rep.id}
                    className="bg-white border border-slate-200 hover:border-slate-300 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-all shadow-xs group"
                  >
                    <div className="space-y-1.5 flex-1 min-w-0">
                      <div className="flex items-center space-x-3">
                        <span className="font-mono font-bold text-sm text-blue-700 bg-blue-50 px-2.5 py-1 rounded-md border border-blue-200">
                          {rep.ticker}
                        </span>
                        <h3 className="text-sm font-bold text-slate-900 truncate">
                          {rep.company}
                        </h3>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${confColor}`}>
                          {conf} Conf
                        </span>
                      </div>

                      <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                        {rep.executive_summary || rep.verdict || 'Institutional research analysis completed.'}
                      </p>

                      <div className="flex items-center space-x-2 text-[11px] text-slate-400 font-mono pt-1">
                        <Clock className="w-3 h-3" />
                        <span>{dateStr}</span>
                        <span>•</span>
                        <span className="truncate max-w-xs">{rep.query}</span>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2 sm:self-center self-end">
                      <button
                        type="button"
                        onClick={() => setSelectedReportId(rep.id)}
                        className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center space-x-1.5 transition-all shadow-xs"
                      >
                        <span>Examine Dossier</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>

                      <button
                        type="button"
                        onClick={() => deleteReport(rep.id)}
                        aria-label="Delete report"
                        className="p-2 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
