import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { CheckCircle2, ArrowRight, Layers, FileText } from 'lucide-react';
import { useApp, ActiveAnalysisState } from '../context/AppContext';
import { AnalysisProgressBar } from '../components/analysis/AnalysisProgressBar';
import { SpecialistStatusGrid } from '../components/analysis/SpecialistStatusGrid';
import { mockApi, AnalysisStatusPayload } from '../services/mockApi';
import { SpecialistStatus, SpecialistType } from '../types';

export const AnalysisProgressPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { activeAnalysis, setActiveAnalysis } = useApp();

  const [stepIndex, setStepIndex] = useState(2);
  const [isSimulating, setIsSimulating] = useState(true);

  const analysisId = id || activeAnalysis?.analysisId || 'an-active';
  const ticker = activeAnalysis?.ticker || 'AAPL';
  const reportId = activeAnalysis?.reportId || 'rep-aapl-001';

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    if (isSimulating && stepIndex < 4) {
      timer = setTimeout(() => {
        setStepIndex((prev: number) => prev + 1);
      }, 700);
    } else if (stepIndex >= 4) {
      setIsSimulating(false);
      setActiveAnalysis((prev: ActiveAnalysisState | null) =>
        prev
          ? {
              ...prev,
              status: 'completed',
              progressPercent: 100,
              progressStage: 'Final Investment Report synthesized & formatted',
              specialistStatuses: {
                technical: 'completed',
                fundamental: 'completed',
                news: 'completed',
                research: 'completed',
                risk: 'completed',
              },
            }
          : null,
      );
    }
    return () => clearTimeout(timer);
  }, [stepIndex, isSimulating, setActiveAnalysis]);

  const [statusPayload, setStatusPayload] = useState<{
    progressPercent: number;
    progressStage: string;
    specialistStatuses: Record<SpecialistType, SpecialistStatus>;
  }>({
    progressPercent: 75,
    progressStage: 'Aggregating cross-domain findings & checking conflicts',
    specialistStatuses: {
      technical: 'completed',
      fundamental: 'completed',
      news: 'completed',
      research: 'running',
      risk: 'running',
    },
  });

  useEffect(() => {
    let active = true;
    mockApi.getAnalysisStatus(analysisId, stepIndex).then((res: AnalysisStatusPayload) => {
      if (active) {
        setStatusPayload({
          progressPercent: res.progressPercent,
          progressStage: res.progressStage,
          specialistStatuses: res.specialistStatuses,
        });
      }
    });
    return () => {
      active = false;
    };
  }, [analysisId, stepIndex]);

  const handleSelectSpecialist = (type: SpecialistType) => {
    navigate(`/analysis/${analysisId}/specialists?tab=${type}`);
  };

  const isComplete = statusPayload.progressPercent >= 100;

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
            <button
              type="button"
              onClick={() => navigate(`/reports/${reportId}`)}
              className="inline-flex items-center px-4 py-1.5 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white transition-colors shadow-sm"
            >
              <FileText className="w-3.5 h-3.5 mr-1.5" />
              View Investment Report
            </button>
          </div>
        )}
      </div>

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

          <button
            type="button"
            onClick={() => navigate(`/reports/${reportId}`)}
            className="w-full sm:w-auto px-5 py-2.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-bold text-xs flex items-center justify-center space-x-2 transition-colors shadow-sm"
          >
            <span>Examine Final Report</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};
