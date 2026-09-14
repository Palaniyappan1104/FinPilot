/**
 * FinPilot API Service Interfaces & DTOs (Phase 17.1).
 * Defines strong typing for HTTP transport, backend contracts, and normalizations.
 */

import {
  AnalysisStatus,
  AnalysisSummary,
  ChatMessage,
  CompanyInfo,
  DocumentItem,
  FinalReport,
  InvestorProfile,
  ResearchQAResult,
  SpecialistStatus,
  SpecialistType,
} from '../../types';

// ---------------------------------------------------------------------------
// Standardized API Error Abstraction
// ---------------------------------------------------------------------------
export interface ApiErrorDetail {
  code: string;
  message: string;
  details?: Array<Record<string, unknown>>;
}

export class ApiError extends Error {
  public readonly code: string;
  public readonly status: number;
  public readonly details?: Array<Record<string, unknown>>;

  constructor(message: string, status: number, code: string = 'HTTP_ERROR', details?: Array<Record<string, unknown>>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

// ---------------------------------------------------------------------------
// Backend Phase 15 Request & Response Payloads
// ---------------------------------------------------------------------------
export interface BackendInvestorProfileInput {
  ticker?: string | null;
  investment_goal?: string | null;
  time_horizon?: string | null;
  capital_amount?: number | null;
  risk_tolerance?: string | null;
}

export interface BackendChatQueryRequest {
  query: string;
  investor_profile?: BackendInvestorProfileInput | null;
  documents_available?: boolean;
  trace_id?: string | null;
}

export interface BackendCompanyAnalysisRequest {
  ticker: string;
  target_company?: string | null;
  query?: string | null;
  investor_profile?: BackendInvestorProfileInput | null;
  documents_available?: boolean;
  clarification_answers?: Record<string, unknown> | null;
  trace_id?: string | null;
}

export interface BackendClarificationSubmitRequest {
  clarification_answers: Record<string, unknown>;
  analysis_id?: string | null;
  ticker?: string | null;
  query?: string | null;
  investor_profile?: BackendInvestorProfileInput | null;
  documents_available?: boolean;
  trace_id?: string | null;
}

export interface BackendResearchQueryRequest {
  query: string;
  ticker?: string | null;
  documents_available?: boolean;
  trace_id?: string | null;
}

export interface BackendAnalysisExecutionResponse {
  analysis_id: string;
  trace_id: string;
  status: 'completed' | 'clarification_needed' | 'failed' | 'running';
  clarification_needed: boolean;
  clarification_questions: string[];
  investor_profile?: Record<string, unknown> | null;
  report?: Record<string, unknown> | null;
  report_id?: string | null;
  error?: string | null;
}

export interface BackendAnalysisStatusResponse {
  analysis_id: string;
  trace_id: string;
  status: 'completed' | 'clarification_needed' | 'failed' | 'running';
  ticker?: string | null;
  clarification_questions: string[];
  report_id?: string | null;
  created_at: string;
  completed_at?: string | null;
  error?: string | null;
  progress_stage?: string | null;
}

export interface BackendReportRetrievalResponse {
  report_id: string;
  format: 'json' | 'markdown' | 'summary';
  report?: Record<string, unknown> | null;
  rendered_content?: string | null;
}

export interface BackendDocumentUploadResponse {
  document_id: string;
  ticker: string;
  document_type: string;
  original_filename: string;
  storage_path: string;
  file_size_bytes: number;
  upload_status: string;
  uploaded_at: string;
}

// ---------------------------------------------------------------------------
// UI-Optimized Progress & Chat Response Models
// ---------------------------------------------------------------------------
export interface AnalysisStatusPayload {
  analysisId: string;
  ticker: string;
  status: AnalysisStatus;
  progressPercent: number;
  progressStage: string;
  specialistStatuses: Record<SpecialistType, SpecialistStatus>;
  reportId?: string;
  error?: string;
}

export interface ChatResponsePayload {
  message: ChatMessage;
  clarificationNeeded: boolean;
  clarificationQuestions?: string[];
  analysisId?: string;
  reportId?: string;
}

// ---------------------------------------------------------------------------
// Unified ApiService Interface
// ---------------------------------------------------------------------------
export interface ApiService {
  /** Search companies in directory or lookup cache */
  searchCompanies(query: string): Promise<CompanyInfo[]>;

  /** Fetch company details by ticker */
  getCompany(ticker: string): Promise<CompanyInfo | null>;

  /** Fetch recent analyses */
  getRecentAnalyses(): Promise<AnalysisSummary[]>;

  /** Conversational query endpoint (POST /api/v1/chat) */
  sendChatMessage(query: string, profile?: InvestorProfile): Promise<ChatResponsePayload>;

  /** Trigger company analysis (POST /api/v1/analysis?background=true) */
  startAnalysis(ticker: string, profile?: InvestorProfile): Promise<{ analysisId: string; reportId?: string }>;

  /** Submit clarification answers (POST /api/v1/clarification?background=true) */
  submitClarification(
    answers: Record<string, string | number>,
    existingTicker?: string,
    analysisId?: string,
  ): Promise<{ analysisId: string; reportId?: string; status: AnalysisStatus }>;

  /** Poll analysis session status (GET /api/v1/analysis/{analysis_id}/status) */
  getAnalysisStatus(analysisId: string): Promise<AnalysisStatusPayload>;

  /** Retrieve completed report (GET /api/v1/reports/{report_id}?format=json) */
  getReport(reportId: string): Promise<FinalReport>;

  /** List indexed documents in Research Vault */
  getDocuments(): Promise<DocumentItem[]>;

  /** Upload PDF document to Research Vault (POST /api/v1/documents/upload) */
  uploadDocument(file: File, ticker: string, docType: DocumentItem['doc_type']): Promise<DocumentItem>;

  /** Ask grounded research question against documents (POST /api/v1/research/query) */
  queryResearchDocuments(query: string, ticker?: string): Promise<ResearchQAResult>;
}
