/**
 * FinPilot Unified API Service Layer (Phase 17.1).
 * Explicit named exports to guarantee clean module initialization.
 */

export {
  ApiError,
  type ApiErrorDetail,
  type ApiService,
  type BackendChatQueryRequest,
  type BackendCompanyAnalysisRequest,
  type BackendClarificationSubmitRequest,
  type BackendResearchQueryRequest,
  type BackendAnalysisExecutionResponse,
  type BackendAnalysisStatusResponse,
  type BackendReportRetrievalResponse,
  type BackendDocumentUploadResponse,
  type AnalysisStatusPayload,
  type ChatResponsePayload,
} from './types';

export {
  DEFAULT_TIMEOUT_MS,
  EXTENDED_TIMEOUT_MS,
  getApiBaseUrl,
  request,
  type RequestOptions,
} from './client';

export {
  STANDARD_DISCLAIMER,
  mapFrontendDocTypeToBackend,
  mapBackendDocTypeToFrontend,
  normalizeEvidenceItem,
  normalizeOverallAssessment,
  normalizeRecommendation,
  normalizeFinalReport,
  deriveProgressPayload,
} from './adapters';

export { HttpApiService } from './httpService';
export { MockApiService } from './mockService';

import { HttpApiService } from './httpService';
export const apiService = new HttpApiService();
