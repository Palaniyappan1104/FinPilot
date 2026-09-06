/**
 * Centralized API client service.
 * Base URL is dynamically resolved from VITE_API_BASE_URL.
 */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
}

export const getApiBaseUrl = (): string => {
  return import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
};

export const fetchHealth = async (): Promise<HealthResponse> => {
  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/health`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }

  return response.json();
};
