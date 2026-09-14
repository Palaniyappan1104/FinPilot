/**
 * Centralized API client service (Phase 17).
 * Re-exports everything from the api/ subsystem and maintains health check backward compatibility.
 */

export * from './api/index';

import { request } from './api/client';

export interface HealthResponse {
  status: string;
  service?: string;
  version: string;
  environment: string;
}

export const fetchHealth = async (): Promise<HealthResponse> => {
  return request<HealthResponse>('/health');
};
