/**
 * Isolated Mock API service for FinPilot Phase 16 React Frontend.
 * Simulates async operations matching Phase 15 API contracts.
 * Preserves strict Phase 16 boundary: No live Phase 17 backend wiring.
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
} from '../types';
import {
  MOCK_COMPANIES,
  MOCK_DOCUMENTS,
  MOCK_RECENT_ANALYSES,
  MOCK_REPORTS,
} from './mockData';

// In-memory session state for mocked execution
let recentAnalysesState: AnalysisSummary[] = [...MOCK_RECENT_ANALYSES];
let documentsState: DocumentItem[] = [...MOCK_DOCUMENTS];
const reportsState: Record<string, FinalReport> = { ...MOCK_REPORTS };

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

export const mockApi = {
  /**
   * Search companies by ticker or name.
   */
  async searchCompanies(query: string): Promise<CompanyInfo[]> {
    await new Promise((resolve) => setTimeout(resolve, 80));
    const clean = query.trim().toUpperCase();
    if (!clean) {
      return Object.values(MOCK_COMPANIES);
    }
    return Object.values(MOCK_COMPANIES).filter(
      (c) =>
        c.ticker.includes(clean) ||
        c.name.toUpperCase().includes(clean) ||
        c.sector.toUpperCase().includes(clean),
    );
  },

  /**
   * Fetch company details by ticker symbol.
   */
  async getCompany(ticker: string): Promise<CompanyInfo | null> {
    await new Promise((resolve) => setTimeout(resolve, 60));
    const clean = ticker.trim().toUpperCase();
    return MOCK_COMPANIES[clean] || null;
  },

  /**
   * Fetch recent analyses for the dashboard.
   */
  async getRecentAnalyses(): Promise<AnalysisSummary[]> {
    await new Promise((resolve) => setTimeout(resolve, 80));
    return [...recentAnalysesState];
  },

  /**
   * Process a conversational query through the Conversation Agent.
   * Checks whether investor profile parameters are complete or need clarification.
   */
  async sendChatMessage(
    query: string,
    profile?: InvestorProfile,
  ): Promise<{
    message: ChatMessage;
    clarificationNeeded: boolean;
    clarificationQuestions?: string[];
    analysisId?: string;
    reportId?: string;
  }> {
    await new Promise((resolve) => setTimeout(resolve, 200));

    const cleanQuery = query.trim();
    // Detect ticker in query or profile
    let detectedTicker = profile?.ticker;
    if (!detectedTicker) {
      const match = cleanQuery.match(/\b(AAPL|MSFT|NVDA|INFY|GOOGL|Apple|Microsoft|Nvidia|Infosys|Google)\b/i);
      if (match) {
        const found = match[0].toUpperCase();
        if (found.startsWith('APP') || found === 'AAPL') detectedTicker = 'AAPL';
        else if (found.startsWith('MIC') || found === 'MSFT') detectedTicker = 'MSFT';
        else if (found.startsWith('NVI') || found === 'NVDA') detectedTicker = 'NVDA';
        else if (found.startsWith('INF') || found === 'INFY') detectedTicker = 'INFY';
        else if (found.startsWith('GOO') || found === 'GOOGL') detectedTicker = 'GOOGL';
      }
    }

    // If query doesn't specify horizon or risk tolerance, check if profile has it
    const missingHorizon = !profile?.time_horizon && !cleanQuery.toLowerCase().includes('year');
    const missingRisk = !profile?.risk_tolerance;

    if (missingHorizon || missingRisk) {
      const questions: string[] = [];
      if (missingHorizon) {
        questions.push('What is your intended investment time horizon (e.g. 1-3 years, 3-5 years, or 5+ years)?');
      }
      if (missingRisk) {
        questions.push('What is your risk tolerance profile (Conservative, Moderate, or Aggressive)?');
      }

      const assistantMsg: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content:
          `I received your research request regarding **${detectedTicker || 'the target company'}**. ` +
          'To generate a tailored decision-support assessment, I need a few clarification details about your investment profile.',
        timestamp: new Date().toISOString(),
        clarificationQuestions: questions,
        suggestedActions: ['Submit profile answers', 'View company overview'],
      };

      return {
        message: assistantMsg,
        clarificationNeeded: true,
        clarificationQuestions: questions,
      };
    }

    // Profile has constraints; initiate analysis
    const targetTicker = detectedTicker || 'AAPL';
    const analysisId = `an-${Date.now()}`;
    const reportId = targetTicker === 'INFY' ? 'rep-infy-002' : 'rep-aapl-001';

    const assistantMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'assistant',
      content:
        `Investor profile confirmed for **${targetTicker}**. ` +
        `Activating FinPilot multi-agent research pipeline (CIO routing to Technical, Fundamental, News, Research, and Risk specialists).`,
      timestamp: new Date().toISOString(),
      analysisId,
      reportId,
      suggestedActions: ['Monitor Live Specialists', 'View Final Report'],
    };

    return {
      message: assistantMsg,
      clarificationNeeded: false,
      analysisId,
      reportId,
    };
  },

  /**
   * Submit clarification responses and advance workflow.
   */
  async submitClarification(
    answers: Record<string, string | number>,
    existingTicker?: string,
  ): Promise<{
    analysisId: string;
    reportId: string;
    status: AnalysisStatus;
  }> {
    await new Promise((resolve) => setTimeout(resolve, 150));
    const analysisId = `an-${Date.now()}`;
    const ticker = existingTicker || 'AAPL';
    const reportId = ticker === 'INFY' ? 'rep-infy-002' : 'rep-aapl-001';

    // Record into recent analyses
    const company = MOCK_COMPANIES[ticker] || MOCK_COMPANIES.AAPL;
    const summary: AnalysisSummary = {
      analysis_id: analysisId,
      report_id: reportId,
      ticker: company.ticker,
      company_name: company.name,
      created_at: new Date().toISOString(),
      status: 'completed',
      recommendation_stance: 'favorable',
      confidence: 0.88,
      horizon: String(answers.time_horizon || '3-5 years'),
      risk_tolerance: String(answers.risk_tolerance || 'moderate'),
    };
    recentAnalysesState = [summary, ...recentAnalysesState.slice(0, 9)];

    return {
      analysisId,
      reportId,
      status: 'completed',
    };
  },

  /**
   * Trigger a fresh company analysis.
   */
  async startAnalysis(
    ticker: string,
    profile?: InvestorProfile,
  ): Promise<{ analysisId: string; reportId: string }> {
    await new Promise((resolve) => setTimeout(resolve, 120));
    const cleanTicker = ticker.toUpperCase();
    const analysisId = `an-${Date.now()}`;
    const reportId = cleanTicker === 'INFY' ? 'rep-infy-002' : 'rep-aapl-001';

    const company = MOCK_COMPANIES[cleanTicker] || {
      ticker: cleanTicker,
      name: `${cleanTicker} Corp`,
      sector: 'Technology',
    };

    const newSummary: AnalysisSummary = {
      analysis_id: analysisId,
      report_id: reportId,
      ticker: company.ticker,
      company_name: company.name,
      created_at: new Date().toISOString(),
      status: 'completed',
      recommendation_stance: 'favorable',
      confidence: 0.87,
      horizon: profile?.time_horizon || '3-5 years',
      risk_tolerance: profile?.risk_tolerance || 'moderate',
    };
    recentAnalysesState = [newSummary, ...recentAnalysesState];

    return { analysisId, reportId };
  },

  /**
   * Poll status of an analysis session (16.6).
   */
  async getAnalysisStatus(
    analysisId: string,
    stepIndex = 4,
  ): Promise<AnalysisStatusPayload> {
    await new Promise((resolve) => setTimeout(resolve, 50));
    // Determine stages based on stepIndex
    const stages = [
      'Validating user query & constraints',
      'CIO Agent allocating specialist domains',
      'Specialist Agents executing concurrent research',
      'Aggregating cross-domain findings & checking conflicts',
      'Final Investment Report synthesized & formatted',
    ];

    const currentStage = stages[Math.min(stepIndex, stages.length - 1)];
    const isDone = stepIndex >= 4;

    const specStatus: Record<SpecialistType, SpecialistStatus> = {
      technical: stepIndex >= 2 ? 'completed' : stepIndex === 1 ? 'running' : 'pending',
      fundamental: stepIndex >= 2 ? 'completed' : stepIndex === 1 ? 'running' : 'pending',
      news: stepIndex >= 3 ? 'completed' : stepIndex >= 1 ? 'running' : 'pending',
      research: stepIndex >= 3 ? 'completed' : stepIndex >= 2 ? 'running' : 'pending',
      risk: stepIndex >= 3 ? 'completed' : stepIndex >= 2 ? 'running' : 'pending',
    };

    return {
      analysisId,
      ticker: 'AAPL',
      status: isDone ? 'completed' : 'running',
      progressPercent: Math.min(100, Math.round(((stepIndex + 1) / stages.length) * 100)),
      progressStage: currentStage,
      specialistStatuses: specStatus,
      reportId: isDone ? 'rep-aapl-001' : undefined,
    };
  },

  /**
   * Retrieve a completed report by ID (16.9).
   */
  async getReport(reportId: string): Promise<FinalReport> {
    await new Promise((resolve) => setTimeout(resolve, 100));
    const report = reportsState[reportId] || reportsState['rep-aapl-001'];
    return report;
  },

  /**
   * Retrieve all uploaded documents in Research Vault (16.8).
   */
  async getDocuments(): Promise<DocumentItem[]> {
    await new Promise((resolve) => setTimeout(resolve, 80));
    return [...documentsState];
  },

  /**
   * Upload a research filing with client-side validation (16.8.1).
   */
  async uploadDocument(
    file: File,
    ticker: string,
    docType: DocumentItem['doc_type'],
  ): Promise<DocumentItem> {
    await new Promise((resolve) => setTimeout(resolve, 300));

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      throw new Error('Only PDF documents (.pdf) are supported in Research Vault.');
    }
    if (file.size > 50 * 1024 * 1024) {
      throw new Error('File size exceeds the 50MB maximum threshold.');
    }
    if (!ticker.trim()) {
      throw new Error('Target stock ticker symbol is required.');
    }

    const newDoc: DocumentItem = {
      id: `doc-${Date.now()}`,
      filename: file.name,
      ticker: ticker.trim().toUpperCase(),
      doc_type: docType,
      size_bytes: file.size,
      upload_date: new Date().toISOString(),
      status: 'indexed',
      chunk_count: Math.max(12, Math.floor(file.size / 25000)),
    };

    documentsState = [newDoc, ...documentsState];
    return newDoc;
  },

  /**
   * Ask a question against uploaded filings (16.8.2).
   */
  async queryResearchDocuments(
    query: string,
    ticker?: string,
  ): Promise<ResearchQAResult> {
    await new Promise((resolve) => setTimeout(resolve, 250));
    const targetTicker = ticker ? ticker.toUpperCase() : 'AAPL';

    return {
      question: query,
      ticker: targetTicker,
      answer:
        `Grounded research analysis for **${targetTicker}**: Review of the indexed 10-K/10-Q SEC filings ` +
        `indicates that operating expenditures and capital commitments remain aligned with core guidance. ` +
        `Management maintains disciplined cash allocation with no structural liquidity constraints disclosed.`,
      citations: [
        {
          source_document: `${targetTicker}_2026_Q3_10Q.pdf`,
          page: 14,
          excerpt:
            'Liquidity and capital resources remain adequate to satisfy ongoing working capital requirements and scheduled capital expenditures.',
          confidence: 0.96,
        },
        {
          source_document: `${targetTicker}_Earnings_Call_Transcript.pdf`,
          page: 6,
          excerpt:
            'We continue to prioritize organic reinvestment into proprietary model development and customer delivery infrastructure.',
          confidence: 0.93,
        },
      ],
      confidence: 0.94,
    };
  },
};
