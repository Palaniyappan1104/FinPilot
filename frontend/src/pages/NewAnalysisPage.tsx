import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { SlidersHorizontal, RefreshCw, AlertCircle } from 'lucide-react';
import { useApp, SESSION_ANALYSIS_STORAGE_KEY } from '../context/AppContext';
import { CompanySearch } from '../components/company/CompanySearch';
import { CompanyOverviewCard } from '../components/company/CompanyOverviewCard';
import { InvestorProfileForm } from '../components/profile/InvestorProfileForm';
import { ChatWindow } from '../components/chat/ChatWindow';
import { AnalysisProgressBar } from '../components/analysis/AnalysisProgressBar';
import { SpecialistStatusGrid } from '../components/analysis/SpecialistStatusGrid';
import { ReportView } from '../components/report/ReportView';
import { apiService, ApiError } from '../services/api';
import { CompanyInfo } from '../types';

/** Shape of the minimal token persisted to sessionStorage for mid-refresh resume. */
interface AnalysisResumeToken {
  analysisId: string;
  ticker: string;
  companyName: string;
}

function writeResumeToken(token: AnalysisResumeToken): void {
  try {
    sessionStorage.setItem(SESSION_ANALYSIS_STORAGE_KEY, JSON.stringify(token));
  } catch {
    // sessionStorage unavailable — resume on refresh won't work but won't crash
  }
}

function clearResumeToken(): void {
  try {
    sessionStorage.removeItem(SESSION_ANALYSIS_STORAGE_KEY);
  } catch {
    // Safe fail
  }
}

function readResumeToken(): AnalysisResumeToken | null {
  try {
    const raw = sessionStorage.getItem(SESSION_ANALYSIS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<AnalysisResumeToken>;
    // Validate shape — discard if any required field is missing or wrong type
    if (
      typeof parsed.analysisId === 'string' && parsed.analysisId.trim() &&
      typeof parsed.ticker === 'string' && parsed.ticker.trim() &&
      typeof parsed.companyName === 'string'
    ) {
      return parsed as AnalysisResumeToken;
    }
    clearResumeToken();
    return null;
  } catch {
    clearResumeToken();
    return null;
  }
}


export const NewAnalysisPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const {
    selectedCompany,
    setSelectedCompany,
    profile,
    updateProfile,
    startNewAnalysis,
    activeAnalysis,
    setActiveAnalysis,
    activeReport,
    setActiveReport,
  } = useApp();

  const [showProfileConfig, setShowProfileConfig] = useState(false);
  const [pollError, setPollError] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);

  // ── On mount: resume in-flight analysis if a token exists in sessionStorage ──
  useEffect(() => {
    // Only attempt resume if there is no already-live activeAnalysis in context
    // (e.g. user navigated away and back in the same session without refreshing)
    if (activeAnalysis) return;

    const token = readResumeToken();
    if (!token) return;

    let cancelled = false;
    setResuming(true);

    (async () => {
      try {
        const res = await apiService.getAnalysisStatus(token.analysisId);
        if (cancelled) return;

        if (res.status === 'completed') {
          // Restore completed state + fetch the report
          setActiveAnalysis({
            analysisId: token.analysisId,
            ticker: token.ticker,
            companyName: token.companyName,
            status: 'completed',
            progressPercent: 100,
            progressStage: res.progressStage || 'Complete',
            specialistStatuses: res.specialistStatuses,
            reportId: res.reportId,
          });
          clearResumeToken();
          if (res.reportId) {
            try {
              const rep = await apiService.getReport(res.reportId);
              if (!cancelled) setActiveReport(rep);
            } catch {
              // Report fetch failed — show completed state without report body
            }
          }
        } else if (res.status === 'running') {
          // Restore running state — the polling effect will pick it up automatically
          setActiveAnalysis({
            analysisId: token.analysisId,
            ticker: token.ticker,
            companyName: token.companyName,
            status: 'running',
            progressPercent: res.progressPercent,
            progressStage: res.progressStage || 'Resuming…',
            specialistStatuses: res.specialistStatuses,
          });
          // Token stays — will be cleared when polling finishes
        } else if (res.status === 'failed') {
          setPollError(res.error || 'Analysis failed before the page was reloaded.');
          setActiveAnalysis({
            analysisId: token.analysisId,
            ticker: token.ticker,
            companyName: token.companyName,
            status: 'failed',
            progressPercent: 0,
            progressStage: 'Failed',
            specialistStatuses: res.specialistStatuses,
            error: res.error,
          });
          clearResumeToken();
        } else {
          // Unknown / clarification_needed — treat as stale, clear token
          clearResumeToken();
        }
      } catch {
        if (!cancelled) {
          // Backend unreachable or ID not found — discard stale token silently
          clearResumeToken();
        }
      } finally {
        if (!cancelled) setResuming(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []); // Intentionally empty — runs once on mount only

  // Sync with URL query or ticker
  useEffect(() => {
    let active = true;
    const tickerParam = searchParams.get('ticker');
    if (tickerParam) {
      apiService.getCompany(tickerParam).then((comp) => {
        if (active && comp) {
          setSelectedCompany(comp);
          updateProfile({ ticker: comp.ticker, target_company: comp.name });
        }
      });
    }
    return () => {
      active = false;
    };
  }, [searchParams, setSelectedCompany, updateProfile]);

  // Polling effect for active analysis pipeline
  useEffect(() => {
    let isMounted = true;
    let pollCount = 0;
    const maxPolls = 60; // 90 seconds max

    if (!activeAnalysis?.analysisId || activeAnalysis.status !== 'running') {
      return;
    }

    // Persist/refresh token so a mid-analysis page refresh can resume
    writeResumeToken({
      analysisId: activeAnalysis.analysisId,
      ticker: activeAnalysis.ticker,
      companyName: activeAnalysis.companyName,
    });

    const poll = async () => {
      try {
        const res = await apiService.getAnalysisStatus(activeAnalysis.analysisId);
        if (!isMounted) return;

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
          clearResumeToken(); // analysis finished — token no longer needed

          if (res.reportId) {
            try {
              const rep = await apiService.getReport(res.reportId);
              if (isMounted) {
                setActiveReport(rep);
              }
            } catch {
              // Safe fallback
            }
          }
          return;
        }

        if (res.status === 'failed') {
          setPollError(res.error || 'Analysis execution failed.');
          setActiveAnalysis((prev) =>
            prev ? { ...prev, status: 'failed', error: res.error } : null,
          );
          clearResumeToken(); // analysis ended — clear token
          return;
        }

        // Still running - update progress and continue polling
        setActiveAnalysis((prev) =>
          prev
            ? {
                ...prev,
                progressPercent: res.progressPercent,
                progressStage: res.progressStage,
                specialistStatuses: res.specialistStatuses,
              }
            : null,
        );

        pollCount++;
        if (pollCount < maxPolls && isMounted) {
          setTimeout(poll, 1500);
        } else if (pollCount >= maxPolls && isMounted) {
          setPollError('Analysis is taking longer than expected. Please check status shortly.');
        }
      } catch (err) {
        if (!isMounted) return;
        const msg = err instanceof ApiError ? err.message : 'Error checking pipeline status.';
        setPollError(msg);
      }
    };

    poll();

    return () => {
      isMounted = false;
    };
  }, [activeAnalysis?.analysisId, activeAnalysis?.status, setActiveAnalysis, setActiveReport]);

  const handleSelectCompany = (comp: CompanyInfo) => {
    setSelectedCompany(comp);
    updateProfile({ ticker: comp.ticker, target_company: comp.name });
  };

  const handleLaunchAnalysis = async () => {
    if (!selectedCompany) return;
    setPollError(null);
    const analysisId = await startNewAnalysis(selectedCompany.ticker, profile);
    // Persist token so a page refresh mid-analysis can resume without re-triggering
    writeResumeToken({
      analysisId,
      ticker: selectedCompany.ticker.toUpperCase(),
      companyName: selectedCompany.name,
    });
  };

  const handleResetAnalysis = () => {
    clearResumeToken(); // discard any persisted resume token
    setActiveAnalysis(null);
    setActiveReport(null);
    setPollError(null);
  };


  return (
    <div data-testid="new-analysis-page" className="space-y-6">
      {/* Page Heading */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
        <div>
          <h2 className="text-xl font-bold text-slate-900">
            Initiate Financial Research
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Select a target company, calibrate investor profile boundaries, and engage the Conversation Agent.
          </p>
        </div>

        <div className="flex items-center space-x-2 self-start sm:self-auto">
          {activeAnalysis && (
            <button
              type="button"
              onClick={handleResetAnalysis}
              className="inline-flex items-center px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
              <span>Reset / New Session</span>
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowProfileConfig(!showProfileConfig)}
            className="inline-flex items-center px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
            <span>{showProfileConfig ? 'Hide Constraints' : 'Investor Constraints'}</span>
          </button>
        </div>
      </div>

      {/* Optional Profile Drawer / Card */}
      {showProfileConfig && (
        <div className="mb-4">
          <InvestorProfileForm
            onSaved={() => setShowProfileConfig(false)}
          />
        </div>
      )}

      {/* Resuming banner shown during mid-refresh status check */}
      {resuming && (
        <div className="p-4 rounded-xl bg-sky-50 border border-sky-200 text-xs text-sky-800 flex items-center space-x-2 animate-pulse">
          <RefreshCw className="w-4 h-4 text-sky-600 flex-shrink-0 animate-spin" />
          <span>Checking for a prior analysis session… please wait.</span>
        </div>
      )}

      {/* Error alert banner if polling failed */}
      {pollError && (

        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{pollError}</span>
          </div>
          <button
            type="button"
            onClick={() => setPollError(null)}
            className="text-xs font-bold underline hover:text-rose-950 ml-3"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Grid: Left Side Company Info, Right Side Chat Agent */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Company Search & Overview */}
        <div className="lg:col-span-5 space-y-4">
          <CompanySearch
            selectedTicker={selectedCompany?.ticker}
            onSelectCompany={handleSelectCompany}
          />

          {selectedCompany && (
            <CompanyOverviewCard
              company={selectedCompany}
              onAnalyzeClick={handleLaunchAnalysis}
            />
          )}

          {/* Quick Profile Summary Badge */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 text-xs space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-900">
                Active Investor Profile:
              </span>
              <span className="capitalize font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 text-[10px]">
                {profile.risk_tolerance || 'Moderate'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-slate-600 text-[11px] pt-1 border-t border-slate-100">
              <div>
                <span className="text-slate-400 block">Horizon:</span>
                <span className="font-medium text-slate-800">
                  {profile.time_horizon || '3-5 years'}
                </span>
              </div>
              <div>
                <span className="text-slate-400 block">Capital:</span>
                <span className="font-medium text-slate-800">
                  ${Number(profile.capital_amount || 50000).toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Conversational Query Agent & Clarification UI */}
        <div className="lg:col-span-7">
          <ChatWindow />
        </div>
      </div>

      {/* Inline Multi-Agent Execution Progress (Active while running) */}
      {activeAnalysis && activeAnalysis.status === 'running' && (
        <div className="space-y-4 pt-4 border-t border-slate-200 animate-fadeIn">
          <AnalysisProgressBar
            progressPercent={activeAnalysis.progressPercent}
            currentStage={activeAnalysis.progressStage}
          />
          <SpecialistStatusGrid
            specialistStatuses={activeAnalysis.specialistStatuses}
          />
        </div>
      )}

      {/* In-Place Final Report Dossier (Rendered when completed) */}
      {activeReport && (
        <div className="pt-6 border-t border-slate-200 space-y-4 animate-fadeIn">
          <div className="flex items-center justify-between bg-emerald-50 border border-emerald-200 rounded-xl p-4 text-xs text-emerald-900">
            <span className="font-bold">
              Institutional Research Dossier Generated for {activeReport.company.name} ({activeReport.company.ticker})
            </span>
            <span className="text-emerald-700 font-mono text-[11px]">
              Session ID: {activeAnalysis?.analysisId || activeReport.report_id}
            </span>
          </div>

          <ReportView report={activeReport} />
        </div>
      )}
    </div>
  );
};
