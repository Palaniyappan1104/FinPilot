/**
 * FinPilot Mock API Service Adapter (Phase 17.1).
 * Wraps mockApi to fulfill the ApiService interface for offline testing and fallback.
 */

import {
  AnalysisStatus,
  AnalysisSummary,
  CompanyInfo,
  DocumentItem,
  FinalReport,
  InvestorProfile,
  ResearchQAResult,
} from '../../types';
import { mockApi } from '../mockApi';
import { AnalysisStatusPayload, ApiService, ChatResponsePayload } from './types';

export class MockApiService implements ApiService {
  async searchCompanies(query: string): Promise<CompanyInfo[]> {
    return mockApi.searchCompanies(query);
  }

  async getCompany(ticker: string): Promise<CompanyInfo | null> {
    return mockApi.getCompany(ticker);
  }

  async getRecentAnalyses(): Promise<AnalysisSummary[]> {
    return mockApi.getRecentAnalyses();
  }

  async sendChatMessage(
    query: string,
    profile?: InvestorProfile,
  ): Promise<ChatResponsePayload> {
    return mockApi.sendChatMessage(query, profile);
  }

  async startAnalysis(
    ticker: string,
    profile?: InvestorProfile,
  ): Promise<{ analysisId: string; reportId?: string }> {
    return mockApi.startAnalysis(ticker, profile);
  }

  async submitClarification(
    answers: Record<string, string | number>,
    existingTicker?: string,
  ): Promise<{ analysisId: string; reportId?: string; status: AnalysisStatus }> {
    return mockApi.submitClarification(answers, existingTicker);
  }

  async getAnalysisStatus(analysisId: string): Promise<AnalysisStatusPayload> {
    return mockApi.getAnalysisStatus(analysisId, 4);
  }

  async getReport(reportId: string): Promise<FinalReport> {
    return mockApi.getReport(reportId);
  }

  async getDocuments(): Promise<DocumentItem[]> {
    return mockApi.getDocuments();
  }

  async uploadDocument(
    file: File,
    ticker: string,
    docType: DocumentItem['doc_type'],
  ): Promise<DocumentItem> {
    return mockApi.uploadDocument(file, ticker, docType);
  }

  async queryResearchDocuments(
    query: string,
    ticker?: string,
  ): Promise<ResearchQAResult> {
    return mockApi.queryResearchDocuments(query, ticker);
  }
}
