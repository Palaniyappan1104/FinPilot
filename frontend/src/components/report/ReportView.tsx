import React, { useState } from 'react';
import { CheckCircle2, ShieldAlert, ArrowLeft, Layers, ChevronDown, ChevronUp } from 'lucide-react';
import { FinalReport } from '../../types';
import { ReportHeader } from './ReportHeader';
import { ExecutiveRecommendation } from './ExecutiveRecommendation';
import { CriticalRisksSection } from './CriticalRisksSection';
import { EvidenceProvenanceTable } from './EvidenceProvenanceTable';
import { SpecialistTabContainer } from '../specialists/SpecialistTabContainer';

interface ReportViewProps {
  report: FinalReport;
  onBack?: () => void;
  className?: string;
}

export const ReportView: React.FC<ReportViewProps> = ({
  report,
  onBack,
  className = '',
}) => {
  const [showSpecialistBreakdown, setShowSpecialistBreakdown] = useState(true);
  return (
    <div data-testid="report-view" className={`space-y-6 print:space-y-4 ${className}`}>
      {/* Back button if passed */}
      {onBack && (
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center text-xs font-semibold text-slate-600 hover:text-slate-900 transition-colors print:hidden"
        >
          <ArrowLeft className="w-3.5 h-3.5 mr-1" />
          Back to Reports Overview
        </button>
      )}

      {/* 16.9.1 Report Header */}
      <ReportHeader report={report} />

      {/* 16.9.1 Executive Assessment & Recommendation */}
      <ExecutiveRecommendation
        recommendation={report.recommendation}
        overallSummary={report.overall_assessment.summary}
        specialistConsensus={report.overall_assessment.specialist_consensus}
        signalConflicts={report.overall_assessment.signal_conflicts}
      />

      {/* Key Supporting Reasons */}
      {report.key_reasons && report.key_reasons.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-3">
          <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
            <CheckCircle2 className="w-5 h-5 text-emerald-600" />
            <h3 className="text-base font-bold text-slate-900">
              Core Analytical Reasons
            </h3>
          </div>
          <ul className="space-y-2 text-xs text-slate-700">
            {report.key_reasons.map((reason, idx) => (
              <li key={idx} className="flex items-start space-x-2.5">
                <span className="w-5 h-5 rounded-full bg-emerald-50 text-emerald-700 font-bold text-[11px] flex items-center justify-center flex-shrink-0 mt-0.5 border border-emerald-200">
                  {idx + 1}
                </span>
                <span className="leading-relaxed font-medium">{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Specialist Deep Dive Tabs (Expandable Evidence Breakdown) */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <button
          type="button"
          onClick={() => setShowSpecialistBreakdown(!showSpecialistBreakdown)}
          className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-50/80 transition-colors text-left"
          aria-expanded={showSpecialistBreakdown}
        >
          <div className="flex items-center space-x-3">
            <Layers className="w-4 h-4 text-emerald-600" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Specialist Analytical Evidence & Breakdown
              </h3>
              <p className="text-[11px] text-slate-500">
                Inspect underlying Technical, Fundamental, News Sentiment, and Risk domain outputs
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400 font-medium hidden sm:inline">
              {showSpecialistBreakdown ? 'Hide Details' : 'Show Details'}
            </span>
            {showSpecialistBreakdown ? (
              <ChevronUp className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            )}
          </div>
        </button>

        {showSpecialistBreakdown && (
          <div className="p-6 pt-2 border-t border-slate-100">
            <SpecialistTabContainer
              technical={report.technical}
              fundamental={report.fundamental}
              news={report.news}
              risk={report.risk}
            />
          </div>
        )}
      </div>

      {/* 16.9.3 Critical Risk Factors */}
      <CriticalRisksSection risks={report.important_risks} />

      {/* 16.9.2 Grounded Evidence Provenance Table */}
      <EvidenceProvenanceTable evidenceList={report.evidence_sources} />

      {/* 16.9.4 Mandatory Regulatory Disclaimer Box */}
      <div
        data-testid="report-disclaimer-card"
        className="p-5 rounded-xl bg-amber-50/70 border border-amber-200 text-xs text-amber-950 flex items-start space-x-3"
      >
        <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="font-bold uppercase tracking-wider text-[11px] text-amber-900">
            Mandatory Regulatory Disclosure & Decision Support Limits
          </h4>
          <p className="leading-relaxed text-[11px] text-amber-800">
            {report.disclaimer}
          </p>
        </div>
      </div>
    </div>
  );
};
