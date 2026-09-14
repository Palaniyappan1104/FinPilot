import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { FileText } from 'lucide-react';
import { SpecialistTabContainer } from '../components/specialists/SpecialistTabContainer';
import { useApp } from '../context/AppContext';
import { mockApi } from '../services/mockApi';
import { FinalReport, SpecialistType } from '../types';

export const SpecialistsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { activeReport, setActiveReport, activeAnalysis } = useApp();

  const [report, setReport] = useState<FinalReport | null>(activeReport);
  const [loading, setLoading] = useState(!activeReport);

  const initialTab = (searchParams.get('tab') as SpecialistType) || 'technical';

  useEffect(() => {
    let active = true;
    if (!report) {
      setLoading(true);
      const targetReportId = activeAnalysis?.reportId || 'rep-aapl-001';
      mockApi.getReport(targetReportId).then((data: FinalReport) => {
        if (active) {
          setReport(data);
          setActiveReport(data);
          setLoading(false);
        }
      });
    }
    return () => {
      active = false;
    };
  }, [report, activeAnalysis, setActiveReport]);

  if (loading || !report) {
    return (
      <div className="p-12 text-center text-xs text-slate-500">
        Loading specialist data...
      </div>
    );
  }

  return (
    <div data-testid="specialists-page" className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold text-slate-900">
              Specialist Insights Dossier
            </h2>
            <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-slate-900 text-white">
              {report.company.ticker}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Deep-dive analytical models across quantitative technicals, fundamental filings, sentiment, and risk.
          </p>
        </div>

        <button
          type="button"
          onClick={() => navigate(`/reports/${report.report_id}`)}
          className="inline-flex items-center px-4 py-2 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white transition-colors shadow-sm self-start sm:self-auto"
        >
          <FileText className="w-3.5 h-3.5 mr-1.5" />
          View Synthesized Final Report
        </button>
      </div>

      <SpecialistTabContainer
        technical={report.technical}
        fundamental={report.fundamental}
        news={report.news}
        risk={report.risk}
        initialTab={initialTab}
      />
    </div>
  );
};
