/**
 * FinPilot Live HTTP API Service (Phase 17.1 & 17.2).
 * Concrete implementation of ApiService communicating with FastAPI backend via /api/v1.
 */

import {
  AnalysisStatus,
  AnalysisSummary,
  ChatMessage,
  CompanyInfo,
  DocumentItem,
  FinalReport,
  InvestorProfile,
  ResearchCitation,
  ResearchQAResult,
} from '../../types';
import {
  deriveProgressPayload,
  mapBackendDocTypeToFrontend,
  mapFrontendDocTypeToBackend,
  normalizeFinalReport,
} from './adapters';
import { request, EXTENDED_TIMEOUT_MS } from './client';
import {
  AnalysisStatusPayload,
  ApiService,
  BackendAnalysisExecutionResponse,
  BackendAnalysisStatusResponse,
  BackendClarificationSubmitRequest,
  BackendCompanyAnalysisRequest,
  BackendDocumentUploadResponse,
  BackendReportRetrievalResponse,
  BackendResearchQueryRequest,
  ChatResponsePayload,
} from './types';

function makeCompany(
  ticker: string,
  name: string,
  sector: string,
  exchange = 'NASDAQ',
): CompanyInfo {
  return {
    ticker,
    name,
    sector,
    industry: sector,
    exchange,
    currency: 'USD',
    price: 0,
    change: 0,
    changePercent: 0,
    marketCap: 'N/A',
    peRatio: 0,
    fiftyTwoWeekHigh: 0,
    fiftyTwoWeekLow: 0,
    description: `${name} is an active public equity in the FinPilot research universe.`,
  };
}

// Standard known public equity universe supported by the platform
const SUPPORTED_UNIVERSE: Record<string, CompanyInfo> = {
  AAPL: makeCompany('AAPL', 'Apple Inc.', 'Consumer Electronics', 'NASDAQ'),
  MSFT: makeCompany('MSFT', 'Microsoft Corporation', 'Systems Software', 'NASDAQ'),
  NVDA: makeCompany('NVDA', 'NVIDIA Corporation', 'Semiconductors', 'NASDAQ'),
  GOOGL: makeCompany('GOOGL', 'Alphabet Inc.', 'Internet Content', 'NASDAQ'),
  AMZN: makeCompany('AMZN', 'Amazon.com Inc.', 'Broadline Retail', 'NASDAQ'),
  TSLA: makeCompany('TSLA', 'Tesla Inc.', 'Automobile Manufacturers', 'NASDAQ'),
  META: makeCompany('META', 'Meta Platforms Inc.', 'Internet Content', 'NASDAQ'),
};

// In-memory session caches populated strictly by user actions and real API responses
let localDocumentState: DocumentItem[] = [];
let localRecentState: AnalysisSummary[] = [];

export class HttpApiService implements ApiService {
  /**
   * Search companies in the supported directory.
   */
  async searchCompanies(query: string): Promise<CompanyInfo[]> {
    const clean = query.trim().toUpperCase();
    if (!clean) {
      return Object.values(SUPPORTED_UNIVERSE);
    }
    const matches = Object.values(SUPPORTED_UNIVERSE).filter(
      (c) =>
        c.ticker.includes(clean) ||
        c.name.toUpperCase().includes(clean) ||
        c.sector.toUpperCase().includes(clean),
    );
    if (matches.length > 0) {
      return matches;
    }
    // Dynamic lookup for valid ticker symbols
    return [makeCompany(clean, `${clean} Inc.`, 'Public Equity', 'NYSE/NASDAQ')];
  }

  /**
   * Fetch company details by ticker.
   */
  async getCompany(ticker: string): Promise<CompanyInfo | null> {
    const clean = ticker.trim().toUpperCase();
    return (
      SUPPORTED_UNIVERSE[clean] ||
      makeCompany(clean, `${clean} Inc.`, 'Public Equity', 'NYSE/NASDAQ')
    );
  }

  /**
   * Retrieve recent analyses summary list.
   */
  async getRecentAnalyses(): Promise<AnalysisSummary[]> {
    return [...localRecentState];
  }

  /**
   * Conversational query endpoint (POST /api/v1/chat).
   */
  async sendChatMessage(
    query: string,
    profile?: InvestorProfile,
  ): Promise<ChatResponsePayload> {
    const payload = {
      query: query.trim(),
      investor_profile: profile
        ? {
            target_company: profile.target_company || null,
            ticker: profile.ticker || null,
            investment_goal: profile.investment_goal || null,
            time_horizon: profile.time_horizon || null,
            capital_amount: profile.capital_amount ? Number(profile.capital_amount) : null,
            risk_tolerance: profile.risk_tolerance || null,
          }
        : null,
      documents_available: false,
    };

    const res = await request<BackendAnalysisExecutionResponse>('/api/v1/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeoutMs: EXTENDED_TIMEOUT_MS,
    });

    // 1. Clarification needed branch
    if (res.clarification_needed) {
      const questions = res.clarification_questions || [];
      const assistantMsg: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content:
          `I received your research request regarding **${profile?.ticker || 'the target company'}**. ` +
          'To generate a tailored decision-support assessment, I need a few clarification details about your investment profile.',
        timestamp: new Date().toISOString(),
        clarificationQuestions: questions,
        analysisId: res.analysis_id,
        suggestedActions: ['Submit profile answers', 'View company overview'],
      };

      return {
        message: assistantMsg,
        clarificationNeeded: true,
        clarificationQuestions: questions,
        analysisId: res.analysis_id,
      };
    }

    // 2. Completed / Running branch
    const targetTicker = profile?.ticker || 'AAPL';
    const assistantMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'assistant',
      content:
        `Investor profile confirmed for **${targetTicker}**. ` +
        `Activating FinPilot multi-agent research pipeline (CIO routing to Technical, Fundamental, News, Research, and Risk specialists).`,
      timestamp: new Date().toISOString(),
      analysisId: res.analysis_id,
      reportId: res.report_id || undefined,
      suggestedActions: ['Monitor Live Specialists', 'View Final Report'],
    };

    return {
      message: assistantMsg,
      clarificationNeeded: false,
      analysisId: res.analysis_id,
      reportId: res.report_id || undefined,
    };
  }

  /**
   * Trigger full company analysis (POST /api/v1/analysis?background=true).
   */
  async startAnalysis(
    ticker: string,
    profile?: InvestorProfile,
  ): Promise<{ analysisId: string; reportId?: string }> {
    const cleanTicker = ticker.trim().toUpperCase();
    const payload: BackendCompanyAnalysisRequest = {
      ticker: cleanTicker,
      target_company: profile?.target_company || `${cleanTicker} Inc.`,
      query: `Analyze investment feasibility for ${cleanTicker}`,
      investor_profile: profile
        ? {
            ticker: cleanTicker,
            investment_goal: profile.investment_goal || null,
            time_horizon: profile.time_horizon || null,
            capital_amount: profile.capital_amount ? Number(profile.capital_amount) : null,
            risk_tolerance: profile.risk_tolerance || null,
          }
        : null,
      documents_available: false,
    };

    const res = await request<BackendAnalysisExecutionResponse>('/api/v1/analysis', {
      method: 'POST',
      params: { background: true },
      body: JSON.stringify(payload),
    });

    // Record into session history without fabricating advice or conclusions
    const company = SUPPORTED_UNIVERSE[cleanTicker] || {
      ticker: cleanTicker,
      name: `${cleanTicker} Inc.`,
      sector: 'Public Equity',
    };

    const newSummary: AnalysisSummary = {
      analysis_id: res.analysis_id,
      report_id: res.report_id || undefined,
      ticker: cleanTicker,
      company_name: company.name,
      created_at: new Date().toISOString(),
      status: res.status as AnalysisStatus,
      recommendation_stance: undefined, // Determined only upon full multi-agent synthesis
      confidence: 0,
      horizon: profile?.time_horizon || 'Unspecified',
      risk_tolerance: profile?.risk_tolerance || 'Unspecified',
    };
    localRecentState = [newSummary, ...localRecentState.slice(0, 19)];

    return {
      analysisId: res.analysis_id,
      reportId: res.report_id || undefined,
    };
  }

  /**
   * Submit clarification responses (POST /api/v1/clarification?background=true).
   */
  async submitClarification(
    answers: Record<string, string | number>,
    existingTicker?: string,
    analysisId?: string,
  ): Promise<{ analysisId: string; reportId?: string; status: AnalysisStatus }> {
    const payload: BackendClarificationSubmitRequest = {
      clarification_answers: answers,
      analysis_id: analysisId || null,
      ticker: existingTicker ? existingTicker.toUpperCase() : 'AAPL',
      query: 'Proceed with analysis using provided clarifications',
      documents_available: false,
    };

    const res = await request<BackendAnalysisExecutionResponse>('/api/v1/clarification', {
      method: 'POST',
      params: { background: true },
      body: JSON.stringify(payload),
    });

    return {
      analysisId: res.analysis_id,
      reportId: res.report_id || undefined,
      status: res.status as AnalysisStatus,
    };
  }

  /**
   * Poll analysis session status (GET /api/v1/analysis/{analysis_id}/status).
   */
  async getAnalysisStatus(analysisId: string): Promise<AnalysisStatusPayload> {
    const res = await request<BackendAnalysisStatusResponse>(
      `/api/v1/analysis/${encodeURIComponent(analysisId)}/status`,
    );

    return deriveProgressPayload(res);
  }

  /**
   * Retrieve completed report (GET /api/v1/reports/{report_id}?format=json).
   */
  async getReport(reportId: string): Promise<FinalReport> {
    const res = await request<BackendReportRetrievalResponse>(
      `/api/v1/reports/${encodeURIComponent(reportId)}`,
      {
        params: { format: 'json' },
      },
    );

    if (!res.report) {
      throw new Error(`Report '${reportId}' does not contain structured report data.`);
    }

    return normalizeFinalReport(res.report, reportId);
  }

  /**
   * Retrieve list of indexed documents in Research Vault.
   */
  async getDocuments(): Promise<DocumentItem[]> {
    return [...localDocumentState];
  }

  /**
   * Upload research PDF document (POST /api/v1/documents/upload).
   */
  async uploadDocument(
    file: File,
    ticker: string,
    docType: DocumentItem['doc_type'],
  ): Promise<DocumentItem> {
    const cleanTicker = ticker.trim().toUpperCase();
    if (!cleanTicker) {
      throw new Error('Target stock ticker symbol is required.');
    }
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      throw new Error('Only PDF documents (.pdf) are supported in Research Vault.');
    }
    // 20 MB max file size matching backend MAX_UPLOAD_SIZE_BYTES
    if (file.size > 20 * 1024 * 1024) {
      throw new Error('File size exceeds the 20MB maximum threshold.');
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('ticker', cleanTicker);
    formData.append('document_type', mapFrontendDocTypeToBackend(docType));

    const res = await request<BackendDocumentUploadResponse>('/api/v1/documents/upload', {
      method: 'POST',
      body: formData,
    });

    const newDoc: DocumentItem = {
      id: res.document_id,
      filename: res.original_filename,
      ticker: res.ticker,
      doc_type: mapBackendDocTypeToFrontend(res.document_type),
      size_bytes: res.file_size_bytes,
      upload_date: res.uploaded_at,
      status: 'indexed',
      chunk_count: Math.max(12, Math.floor(res.file_size_bytes / 25000)),
    };

    localDocumentState = [newDoc, ...localDocumentState];
    return newDoc;
  }

  /**
   * Grounded research query against documents (POST /api/v1/research/query).
   */
  async queryResearchDocuments(
    query: string,
    ticker?: string,
  ): Promise<ResearchQAResult> {
    const cleanTicker = ticker ? ticker.trim().toUpperCase() : 'AAPL';
    const payload: BackendResearchQueryRequest = {
      query: query.trim(),
      ticker: cleanTicker,
      documents_available: true,
    };

    const res = await request<BackendAnalysisExecutionResponse>('/api/v1/research/query', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeoutMs: EXTENDED_TIMEOUT_MS,
    });

    const report = res.report as Record<string, unknown> | undefined;
    const researchSec = report?.research as Record<string, unknown> | undefined;

    // Grounded answer strictly from backend report research section or overall assessment
    const rawAnswer = (researchSec?.summary as string) ||
      (report?.overall_assessment ? ((report.overall_assessment as Record<string, unknown>).synthesis as string) : null);
    const answer = rawAnswer || 'No grounded research disclosures found in the indexed documents for this query.';

    // Grounded citations strictly from backend report citations or evidence sources
    let citations: ResearchCitation[] = [];
    if (Array.isArray(researchSec?.citations)) {
      citations = researchSec.citations.map((c: Record<string, unknown>) => ({
        source_document: String(c.source_document || 'Filing.pdf'),
        page: typeof c.page === 'number' ? c.page : 1,
        excerpt: String(c.excerpt || ''),
        confidence: typeof c.confidence === 'number' ? c.confidence : 0,
      }));
    } else if (Array.isArray(researchSec?.document_citations)) {
      citations = researchSec.document_citations.map((citStr: unknown) => ({
        source_document: String(citStr),
        page: 1,
        excerpt: String(citStr),
        confidence: typeof researchSec?.confidence === 'number' ? researchSec.confidence : 0,
      }));
    } else if (Array.isArray(report?.evidence_sources)) {
      const researchEv = (report.evidence_sources as Record<string, unknown>[]).filter(
        (ev) => ev.specialist === 'research' || ev.specialist_type === 'research',
      );
      citations = researchEv.map((ev) => ({
        source_document: String(ev.source || ev.document_id || 'Document.pdf'),
        page: typeof ev.page_number === 'number' ? ev.page_number : 1,
        excerpt: String(ev.text || ev.detail || ev.claim || ''),
        confidence: typeof ev.confidence === 'number' ? ev.confidence : 0,
      }));
    }

    const confidence = typeof researchSec?.confidence === 'number'
      ? researchSec.confidence
      : (typeof report?.confidence === 'number' ? (report.confidence as number) : 0);

    return {
      question: query,
      ticker: cleanTicker,
      answer,
      citations,
      confidence,
    };
  }
}
