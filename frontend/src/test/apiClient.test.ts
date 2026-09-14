import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  ApiError,
  request,
  mapFrontendDocTypeToBackend,
  mapBackendDocTypeToFrontend,
  normalizeEvidenceItem,
  normalizeOverallAssessment,
  normalizeRecommendation,
  normalizeFinalReport,
  deriveProgressPayload,
} from '../services/api';

describe('Phase 17.1 API Client & Adapters Unit Tests', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe('ApiError model', () => {
    it('properly encapsulates status code, machine code, message, and details', () => {
      const details = [{ loc: ['body', 'query'], msg: 'Field required', type: 'missing' }];
      const err = new ApiError('Validation error', 422, 'VALIDATION_ERROR', details);

      expect(err).toBeInstanceOf(Error);
      expect(err).toBeInstanceOf(ApiError);
      expect(err.message).toBe('Validation error');
      expect(err.status).toBe(422);
      expect(err.code).toBe('VALIDATION_ERROR');
      expect(err.details).toEqual(details);
    });
  });

  describe('HTTP Client request wrapper', () => {
    it('resolves JSON on successful 200 response', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: 'healthy', version: '0.1.0' }),
      });

      const data = await request<{ status: string }>('/api/v1/health');
      expect(data.status).toBe('healthy');
      expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    });

    it('unwraps Phase 15 APIErrorEnvelope on 4xx/5xx failure', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        json: async () => ({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Ticker symbol is required.',
            details: [{ loc: ['body', 'ticker'], msg: 'Field required' }],
          },
        }),
      });

      await expect(request('/api/v1/analysis', { method: 'POST' })).rejects.toThrow(
        'Ticker symbol is required.',
      );

      try {
        await request('/api/v1/analysis', { method: 'POST' });
      } catch (err) {
        expect(err).toBeInstanceOf(ApiError);
        const apiErr = err as ApiError;
        expect(apiErr.status).toBe(422);
        expect(apiErr.code).toBe('VALIDATION_ERROR');
        expect(apiErr.details?.[0].msg).toBe('Field required');
      }
    });

    it('aborts and throws ApiError 408 on timeout', async () => {
      globalThis.fetch = vi.fn().mockImplementation((_url, options) => {
        return new Promise((_resolve, reject) => {
          options.signal.addEventListener('abort', () => {
            const abortErr = new DOMException('The user aborted a request.', 'AbortError');
            reject(abortErr);
          });
        });
      });

      await expect(
        request('/api/v1/chat', { timeoutMs: 50 }),
      ).rejects.toThrow(/timed out after 0.05s/);
    });
  });

  describe('Schema Adapters & Normalizers', () => {
    it('correctly maps document types bidirectionally', () => {
      expect(mapFrontendDocTypeToBackend('10-K')).toBe('annual_report');
      expect(mapFrontendDocTypeToBackend('10-Q')).toBe('company_report');
      expect(mapFrontendDocTypeToBackend('Earnings Transcript')).toBe('earnings_transcript');
      expect(mapFrontendDocTypeToBackend('Investor Presentation')).toBe('investor_presentation');

      expect(mapBackendDocTypeToFrontend('annual_report')).toBe('10-K');
      expect(mapBackendDocTypeToFrontend('company_report')).toBe('10-Q');
      expect(mapBackendDocTypeToFrontend('earnings_transcript')).toBe('Earnings Transcript');
      expect(mapBackendDocTypeToFrontend('investor_presentation')).toBe('Investor Presentation');
    });

    it('normalizes evidence items with backend reference_id and detail fields', () => {
      const rawBackendEvidence = {
        reference_id: 'EV-TECH-101',
        detail: 'RSI(14) reading of 58.4 reflects sustained accumulation.',
        specialist: 'technical',
        source_tool: 'calculate_rsi',
        confidence: 0.95,
      };

      const normalized = normalizeEvidenceItem(rawBackendEvidence, 0);
      expect(normalized.id).toBe('EV-TECH-101');
      expect(normalized.claim).toBe('RSI(14) reading of 58.4 reflects sustained accumulation.');
      expect(normalized.specialist_type).toBe('technical');
      expect(normalized.source_tool).toBe('calculate_rsi');
      expect(normalized.confidence).toBe(0.95);
    });

    it('normalizes overall assessment synthesis and signal conflicts', () => {
      const rawAssessment = {
        synthesis: 'Unified analytical assessment confirms strong fundamentals.',
        data_completeness_ratio: 0.85,
        specialist_consensus: 'favorable',
        signal_conflicts: [
          { description: 'Short-term momentum slightly overbought vs value rating.' },
        ],
      };

      const normalized = normalizeOverallAssessment(rawAssessment);
      expect(normalized.summary).toBe('Unified analytical assessment confirms strong fundamentals.');
      expect(normalized.completeness_ratio).toBe(0.85);
      expect(normalized.specialist_consensus).toBe('favorable');
      expect(normalized.signal_conflicts).toEqual([
        'Short-term momentum slightly overbought vs value rating.',
      ]);
    });

    it('normalizes recommendation stance casing', () => {
      const rec = normalizeRecommendation({
        stance: 'FAVORABLE',
        rationale: 'Solid balance sheet.',
        monitoring_points: ['Margin expansion'],
      });

      expect(rec).toBeDefined();
      expect(rec?.stance).toBe('favorable');
      expect(rec?.rationale).toBe('Solid balance sheet.');
    });

    it('normalizes full FinalReport payload', () => {
      const rawReport = {
        report_id: 'rep-msft-999',
        company: { ticker: 'MSFT', name: 'Microsoft Corporation' },
        recommendation: { stance: 'favorable', rationale: 'Cloud growth' },
        overall_assessment: { synthesis: 'Strong performance across domains.' },
        evidence_sources: [
          { reference_id: 'EV-1', detail: 'Gross margin expanded to 69%' },
        ],
      };

      const report = normalizeFinalReport(rawReport);
      expect(report.report_id).toBe('rep-msft-999');
      expect(report.company.ticker).toBe('MSFT');
      expect(report.recommendation?.stance).toBe('favorable');
      expect(report.overall_assessment.summary).toBe('Strong performance across domains.');
      expect(report.evidence_sources).toHaveLength(1);
      expect(report.evidence_sources[0].id).toBe('EV-1');
      expect(report.evidence_sources[0].claim).toBe('Gross margin expanded to 69%');
    });

    it('derives progress payload correctly based on backend status states', () => {
      // Completed state
      const completed = deriveProgressPayload({
        analysis_id: 'an-1',
        trace_id: 'tr-1',
        status: 'completed',
        ticker: 'NVDA',
        clarification_questions: [],
        report_id: 'rep-nvda-123',
        created_at: '2026-09-14T00:00:00Z',
      });
      expect(completed.status).toBe('completed');
      expect(completed.progressPercent).toBe(100);
      expect(completed.specialistStatuses.technical).toBe('completed');
      expect(completed.reportId).toBe('rep-nvda-123');

      // Clarification needed state
      const clarification = deriveProgressPayload({
        analysis_id: 'an-2',
        trace_id: 'tr-2',
        status: 'clarification_needed',
        clarification_questions: ['What is your time horizon?'],
        created_at: '2026-09-14T00:00:00Z',
      });
      expect(clarification.status).toBe('clarification_needed');
      expect(clarification.progressPercent).toBe(25);
      expect(clarification.specialistStatuses.technical).toBe('pending');

      // Failed state
      const failed = deriveProgressPayload({
        analysis_id: 'an-3',
        trace_id: 'tr-3',
        status: 'failed',
        error: 'Network connectivity lost',
        clarification_questions: [],
        created_at: '2026-09-14T00:00:00Z',
      });
      expect(failed.status).toBe('failed');
      expect(failed.error).toBe('Network connectivity lost');
      expect(failed.specialistStatuses.technical).toBe('failed');

      // Running state
      const running = deriveProgressPayload({
        analysis_id: 'an-4',
        trace_id: 'tr-4',
        status: 'running',
        progress_stage: 'CIO Agent allocating specialist domains',
        clarification_questions: [],
        created_at: '2026-09-14T00:00:00Z',
      });
      expect(running.status).toBe('running');
      expect(running.progressPercent).toBe(65);
      expect(running.progressStage).toBe('CIO Agent allocating specialist domains');
    });
  });
});
