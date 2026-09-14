import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { CheckCircle2, ArrowRight, Layers, FileText } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { AnalysisProgressBar } from '../components/analysis/AnalysisProgressBar';
import { SpecialistStatusGrid } from '../components/analysis/SpecialistStatusGrid';
import { apiService, ApiError, AnalysisStatusPayload } from '../services/api';
import { SpecialistType } from '../types';
import { EmptyState } from '../components/common/EmptyState';

export const AnalysisProgressPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { activeAnalysis, setActiveAnalysis } = useApp();

  const analysisId = id || activeAnalysis?.analysisId;
  const ticker = activeAnalysis?.ticker || '';

  const [statusPayload, setStatusPayload] = useState<AnalysisStatusPayload>({
    analysisId: analysisId || '',
    ticker: ticker || 'Target Equity',
    status: activeAnalysis?.status || 'running',
    progressPercent: activeAnalysis?.progressPercent || 35,
    progressStage: activeAnalysis?.progressStage || 'Executing multi-agent research pipeline',
    specialistStatuses: activeAnalysis?.specialistStatuses || {
      technical: 'running',
      fundamental: 'running',
      news: 'running',
      research: 'running',
      risk: 'running',
    },
    reportId: activeAnalysis?.reportId,
  });

  const [pollError, setPollError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    let pollCount = 0;
    const maxPolls = 60; // 90 seconds max

    if (!analysisId) {
      return;
    }

    const poll = async () => {
      try {
        const res = await apiService.getAnalysisStatus(analysisId);
        if (!isMounted) return;

        setStatusPayload(res);

        if (res.status === 'completed') {
          setActiveAnalysis((prev) =>
            prev
              ? {
                  ...prev,
                  status: 'completed',
                  progressPercent: 100,
                  progressStage: res.progressStage,
                  specialistStatuses: res.specialistStatuses,
                  reportId: res.reportId,
                }
              : null,
          );
          return; // Stop polling
        }

        if (res.status === 'failed') {
          setPollError(res.error || 'Analysis encountered a failure.');
          return; // Stop polling
        }

        if (res.status === 'clarification_needed') {
          return; // Stop polling
        }

        // Continue polling if still running
        pollCount++;
        if (pollCount < maxPolls && isMounted) {
          setTimeout(poll, 1500);
        } else if (pollCount >= maxPolls && isMounted) {
          setPollError('Analysis is taking longer than expected. Please check back shortly.');
        }
      } catch (err) {
        if (!isMounted) return;
        const msg = err instanceof ApiError ? err.message : 'Error checking analysis status.';
        setPollError(msg);
      }
    };

    poll();

    return () => {
      isMounted = false;
    };
  }, [analysisId, setActiveAnalysis]);

  const handleSelectSpecialist = (type: SpecialistType) => {
    navigate(`/analysis/${analysisId}/specialists?tab=${type}`);
  };

  if (!analysisId) {
    return (
      <div data-testid="analysis-progress-page" className="max-w-md mx-auto">
        <EmptyState
          title="No Active Analysis Pipeline"
          description="There is no ongoing multi-agent analysis session. Please start an analysis from the New Analysis page."
          icon={<Layers className="w-6 h-6 text-slate-500" />}
          actionLabel="Start New Analysis"
          onAction={() => navigate('/analysis/new')}
        />
      </div>
    );
  }

  const isComplete = statusPayload.status === 'completed' || statusPayload.progressPercent >= 100;
  const reportId = statusPayload.reportId || activeAnalysis?.reportId;

  return (
    <div data-testid="analysis-progress-page" className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold text-slate-900">
              Active Pipeline Monitor
            </h2>
            <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-slate-900 text-white">
              {ticker}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5 font-mono">
            Session ID: {analysisId}
          </p>
        </div>

        {isComplete && (
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={() => navigate(`/analysis/${analysisId}/specialists`)}
              className="inline-flex items-center px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 transition-colors"
            >
              <Layers className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
              Specialist Insights
            </button>
            {reportId && (
              <button
                type="button"
                onClick={() => navigate(`/reports/${reportId}`)}
                className="inline-flex items-center px-4 py-1.5 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white transition-colors shadow-sm"
              >
                <FileText className="w-3.5 h-3.5 mr-1.5" />
                View Investment Report
              </button>
            )}
          </div>
        )}
      </div>

      {pollError && (
        <div data-testid="poll-error-banner" className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-center justify-between">
          <span>{pollError}</span>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="text-xs font-bold underline hover:text-rose-950 ml-3"
          >
            Retry
          </button>
        </div>
      )}

      {/* Progress Bar & Stepper */}
      <AnalysisProgressBar
        progressPercent={statusPayload.progressPercent}
        currentStage={statusPayload.progressStage}
      />

      {/* Specialist Status Grid */}
      <SpecialistStatusGrid
        specialistStatuses={statusPayload.specialistStatuses}
        onSelectSpecialist={handleSelectSpecialist}
      />

      {/* Completed CTA Card */}
      {isComplete && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <CheckCircle2 className="w-8 h-8 text-emerald-600 flex-shrink-0" />
            <div>
              <h4 className="text-sm font-bold text-emerald-950">
                Multi-Agent Synthesis Complete
              </h4>
              <p className="text-xs text-emerald-800 mt-0.5">
                All 5 specialists have concluded evaluations. Cross-domain findings and evidence citations are compiled.
              </p>
            </div>
          </div>

          {reportId ? (
            <button
              type="button"
              onClick={() => navigate(`/reports/${reportId}`)}
              className="w-full sm:w-auto px-5 py-2.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-bold text-xs flex items-center justify-center space-x-2 transition-colors shadow-sm"
            >
              <span>Examine Final Report</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          ) : (
            <span className="text-xs text-slate-500 italic">Report identifier resolving...</span>
          )}
        </div>
      )}
    </div>
  );
};
