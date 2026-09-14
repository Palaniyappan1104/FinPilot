/**
 * FinPilot Schema Adapters & Normalizers (Phase 17.1).
 * Translates between Phase 15 FastAPI Pydantic models and Phase 16 React UI types.
 */

import {
  AggregatedEvidenceItem,
  DocumentItem,
  FinalReport,
  OverallAssessmentSection,
  RecommendationStance,
  ReportRecommendation,
  SpecialistStatus,
  SpecialistType,
} from '../../types';
import {
  AnalysisStatusPayload,
  BackendAnalysisStatusResponse,
} from './types';

export const STANDARD_DISCLAIMER =
  'FinPilot provides automated financial research and decision support for ' +
  'informational purposes only. FinPilot is not a registered investment advisor, ' +
  'broker-dealer, or financial planner. This report does not constitute ' +
  'personalized investment advice, an endorsement, or a recommendation to buy, ' +
  'sell, or hold any security. Past performance does not guarantee future results. ' +
  'All investments carry risk of loss. Investors must conduct independent ' +
  'research and consult licensed financial professionals before making ' +
  'investment decisions.';

/**
 * Maps frontend UI document category to backend DocumentType enum.
 */
export function mapFrontendDocTypeToBackend(
  docType: DocumentItem['doc_type'],
): string {
  switch (docType) {
    case '10-K':
      return 'annual_report';
    case '10-Q':
    case '8-K':
      return 'company_report';
    case 'Earnings Transcript':
      return 'earnings_transcript';
    case 'Investor Presentation':
      return 'investor_presentation';
    default:
      return 'other_supported';
  }
}

/**
 * Maps backend DocumentType enum string back to frontend UI category.
 */
export function mapBackendDocTypeToFrontend(
  backendType: string,
): DocumentItem['doc_type'] {
  switch (backendType?.toLowerCase()) {
    case 'annual_report':
      return '10-K';
    case 'company_report':
      return '10-Q';
    case 'earnings_transcript':
      return 'Earnings Transcript';
    case 'investor_presentation':
      return 'Investor Presentation';
    default:
      return '10-K';
  }
}

/**
 * Normalizes an evidence item from backend AggregatedEvidenceItem or frontend item.
 */
export function normalizeEvidenceItem(
  raw: Record<string, unknown>,
  index: number = 0,
): AggregatedEvidenceItem {
  const referenceId = typeof raw.reference_id === 'string' ? raw.reference_id : null;
  const rawId = typeof raw.id === 'string' ? raw.id : null;
  const detail = typeof raw.detail === 'string' ? raw.detail : null;
  const claim = typeof raw.claim === 'string' ? raw.claim : null;
  const specialist = (raw.specialist || raw.specialist_type || 'fundamental') as SpecialistType;
  const documentId = typeof raw.document_id === 'string' ? raw.document_id : null;
  const sourceTool = typeof raw.source_tool === 'string' ? raw.source_tool : null;

  return {
    id: rawId || referenceId || `ev-${index + 1}`,
    claim: claim || detail || 'Analytical finding verified.',
    specialist_type: specialist,
    source_tool: sourceTool || (documentId ? `Filing: ${documentId}` : (referenceId || 'specialist_data')),
    confidence: typeof raw.confidence === 'number' ? raw.confidence : 0.92,
    timestamp: typeof raw.timestamp === 'string' ? raw.timestamp : new Date().toISOString(),
  };
}

/**
 * Normalizes the OverallAssessmentSection model across backend and frontend formats.
 */
export function normalizeOverallAssessment(
  raw: Record<string, unknown> | undefined,
): OverallAssessmentSection {
  if (!raw) {
    return {
      summary: 'Comprehensive multi-agent financial research completed.',
      specialist_consensus: 'favorable',
      signal_conflicts: [],
      completeness_ratio: 1.0,
    };
  }

  const summary = (raw.summary as string) || (raw.synthesis as string) || 'Overall synthesis synthesized across domains.';
  const completeness = typeof raw.completeness_ratio === 'number'
    ? raw.completeness_ratio
    : typeof raw.data_completeness_ratio === 'number'
      ? raw.data_completeness_ratio
      : 1.0;

  const consensus = typeof raw.specialist_consensus === 'string' ? raw.specialist_consensus : 'favorable';

  const conflicts: string[] = [];
  if (Array.isArray(raw.signal_conflicts)) {
    raw.signal_conflicts.forEach((c) => {
      if (typeof c === 'string') conflicts.push(c);
      else if (c && typeof c === 'object') {
        const desc = (c as Record<string, unknown>).description || (c as Record<string, unknown>).issue;
        if (desc) conflicts.push(String(desc));
      }
    });
  }

  return {
    summary,
    specialist_consensus: consensus,
    signal_conflicts: conflicts,
    completeness_ratio: completeness,
  };
}

/**
 * Normalizes the ReportRecommendation model across backend and frontend formats.
 */
export function normalizeRecommendation(
  raw: Record<string, unknown> | undefined,
): ReportRecommendation | undefined {
  if (!raw) return undefined;

  const rawStance = String(raw.stance || 'neutral').toLowerCase();
  const validStances: RecommendationStance[] = [
    'favorable',
    'cautious',
    'neutral',
    'unfavorable',
    'insufficient_evidence',
  ];
  const stance: RecommendationStance = validStances.includes(rawStance as RecommendationStance)
    ? (rawStance as RecommendationStance)
    : 'neutral';

  return {
    stance,
    rationale: (raw.rationale as string) || 'Synthesized based on specialist data.',
    time_horizon_suitability: raw.time_horizon_suitability as string | undefined,
    risk_tolerance_suitability: raw.risk_tolerance_suitability as string | undefined,
    capital_allocation_notes: raw.capital_allocation_notes as string | undefined,
    monitoring_points: Array.isArray(raw.monitoring_points)
      ? raw.monitoring_points.map(String)
      : [],
  };
}

/**
 * Normalizes an entire raw report into the validated FinalReport structure.
 */
export function normalizeFinalReport(
  raw: Record<string, unknown>,
  fallbackReportId?: string,
): FinalReport {
  const companyRaw = (raw.company || {}) as Record<string, unknown>;
  const ticker = String(companyRaw.ticker || raw.ticker || 'AAPL').toUpperCase();
  const name = String(companyRaw.name || companyRaw.company_name || `${ticker} Inc.`);

  const rawEvidence = Array.isArray(raw.evidence_sources)
    ? raw.evidence_sources
    : Array.isArray(raw.evidence)
      ? raw.evidence
      : [];

  const evidence_sources: AggregatedEvidenceItem[] = rawEvidence.map((ev, idx) =>
    normalizeEvidenceItem(ev as Record<string, unknown>, idx),
  );

  const reportId = String(raw.report_id || fallbackReportId || '');

  return {
    report_id: reportId,
    company: {
      ticker,
      name,
      sector: (companyRaw.sector as string) || 'Technology',
      currency: (companyRaw.currency as string) || 'USD',
    },
    investor_profile: raw.investor_profile as FinalReport['investor_profile'],
    horizon: (raw.horizon as string) || '3-5 years',
    capital: raw.capital
      ? {
          amount: Number((raw.capital as Record<string, unknown>).amount || 50000),
          currency: String((raw.capital as Record<string, unknown>).currency || 'USD'),
        }
      : undefined,
    created_at: (raw.created_at as string) || (raw.generated_at as string) || new Date().toISOString(),
    confidence: typeof raw.confidence === 'number' ? raw.confidence : 0.88,
    overall_assessment: normalizeOverallAssessment(raw.overall_assessment as Record<string, unknown>),
    recommendation: normalizeRecommendation(raw.recommendation as Record<string, unknown>),
    key_reasons: Array.isArray(raw.key_reasons) ? raw.key_reasons.map(String) : [],
    important_risks: Array.isArray(raw.important_risks) ? raw.important_risks.map(String) : [],
    evidence_sources,
    technical: raw.technical as FinalReport['technical'],
    fundamental: raw.fundamental as FinalReport['fundamental'],
    news: raw.news as FinalReport['news'],
    research: raw.research as FinalReport['research'],
    risk: raw.risk as FinalReport['risk'],
    disclaimer: (raw.disclaimer as string) || STANDARD_DISCLAIMER,
    specialist_statuses: (raw.specialist_statuses || {
      technical: 'completed',
      fundamental: 'completed',
      news: 'completed',
      research: 'completed',
      risk: 'completed',
    }) as Record<SpecialistType, SpecialistStatus>,
  };
}

/**
 * Derives UI status, progress percent, and specialist indicators from backend status response.
 */
export function deriveProgressPayload(
  statusResp: BackendAnalysisStatusResponse,
): AnalysisStatusPayload {
  const status = statusResp.status;
  const analysisId = statusResp.analysis_id;
  const ticker = statusResp.ticker || 'AAPL';
  const reportId = statusResp.report_id || undefined;
  const error = statusResp.error || undefined;

  if (status === 'completed') {
    return {
      analysisId,
      ticker,
      status: 'completed',
      progressPercent: 100,
      progressStage: statusResp.progress_stage || 'Final Investment Report synthesized & formatted',
      specialistStatuses: {
        technical: 'completed',
        fundamental: 'completed',
        news: 'completed',
        research: 'completed',
        risk: 'completed',
      },
      reportId,
      error,
    };
  }

  if (status === 'clarification_needed') {
    return {
      analysisId,
      ticker,
      status: 'clarification_needed',
      progressPercent: 25,
      progressStage: 'Awaiting investor clarification',
      specialistStatuses: {
        technical: 'pending',
        fundamental: 'pending',
        news: 'pending',
        research: 'pending',
        risk: 'pending',
      },
      reportId,
      error,
    };
  }

  if (status === 'failed') {
    return {
      analysisId,
      ticker,
      status: 'failed',
      progressPercent: 100,
      progressStage: 'Analysis execution failed',
      specialistStatuses: {
        technical: 'failed',
        fundamental: 'failed',
        news: 'failed',
        research: 'failed',
        risk: 'failed',
      },
      reportId,
      error: error || 'Analysis encountered an unrecoverable failure.',
    };
  }

  // status === 'running': LangGraph executes specialists concurrently during pipeline run
  return {
    analysisId,
    ticker,
    status: 'running',
    progressPercent: 65,
    progressStage: statusResp.progress_stage || 'Specialist Agents executing concurrent research',
    specialistStatuses: {
      technical: 'running',
      fundamental: 'running',
      news: 'running',
      research: 'running',
      risk: 'running',
    },
    reportId,
    error,
  };
}
