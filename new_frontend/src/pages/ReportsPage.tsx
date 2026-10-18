import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  FileText,
  Clock,
  Trash2,
  ExternalLink,
  ChevronRight,
  ArrowLeft,
  Sparkles,
  Download,
  Award,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { StoredReport } from '../types';
import { ReportDossier } from '../components/report/ReportDossier';

export const ReportsPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { storedReports, deleteReport, activeReport, setActiveReport } = useApp();

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
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center space-x-2 text-emerald-400">
            <FileText className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">Archive & Repository</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Generated Research Reports
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Institutional investment reports automatically archived from multi-agent analysis sessions.
          </p>
        </div>

        {selectedReportId && (
          <button
            type="button"
            onClick={() => setSelectedReportId(null)}
            className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-200 border border-slate-700 text-xs font-semibold transition-colors self-start sm:self-auto"
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
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center space-y-4 max-w-lg mx-auto shadow-lg">
              <div className="w-12 h-12 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-400 mx-auto">
                <FileText className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-white">No Reports Generated Yet</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Whenever you run an investment analysis via New Analysis or Multi-Agent Analysis, its full institutional dossier is automatically preserved here.
                </p>
              </div>
              <button
                type="button"
                onClick={() => navigate('/analysis/new')}
                className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs inline-flex items-center space-x-1.5 transition-colors shadow-sm"
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
                    ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                    : conf.toLowerCase() === 'low'
                      ? 'bg-rose-950 text-rose-400 border-rose-800'
                      : 'bg-amber-950 text-amber-400 border-amber-800';

                return (
                  <div
                    key={rep.id}
                    className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-all group"
                  >
                    <div className="space-y-1.5 flex-1 min-w-0">
                      <div className="flex items-center space-x-3">
                        <span className="font-mono font-bold text-sm text-emerald-400 bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
                          {rep.ticker}
                        </span>
                        <h3 className="text-sm font-bold text-white truncate">
                          {rep.company}
                        </h3>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${confColor}`}>
                          {conf} Conf
                        </span>
                      </div>

                      <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                        {rep.executive_summary || rep.verdict || 'Institutional research analysis completed.'}
                      </p>

                      <div className="flex items-center space-x-2 text-[11px] text-slate-500 font-mono pt-1">
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
                        className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-emerald-600 text-slate-200 hover:text-white border border-slate-700 text-xs font-semibold flex items-center space-x-1.5 transition-all shadow-sm"
                      >
                        <span>Examine Dossier</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>

                      <button
                        type="button"
                        onClick={() => deleteReport(rep.id)}
                        aria-label="Delete report"
                        className="p-2 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-950/40 transition-colors"
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
