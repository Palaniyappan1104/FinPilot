/**
 * FinPilot HTTP Client (Phase 17.1).
 * Handles typed HTTP fetch operations, timeouts via AbortController,
 * and centralized extraction of Phase 15 APIErrorEnvelope error payloads.
 */

import { ApiError } from './types';

export const DEFAULT_TIMEOUT_MS = 15000;
export const EXTENDED_TIMEOUT_MS = 60000;

/**
 * Resolves the API Base URL.
 * Prefers VITE_API_BASE_URL if set; otherwise defaults to empty string for relative proxying.
 */
export function getApiBaseUrl(): string {
  const envUrl = import.meta.env?.VITE_API_BASE_URL;
  if (typeof envUrl === 'string' && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  if (typeof window !== 'undefined' && window.location?.origin && window.location.origin !== 'null') {
    return window.location.origin;
  }
  return 'http://localhost:8000';
}

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
  params?: Record<string, string | number | boolean | undefined>;
}

/**
 * Standard fetch wrapper with timeout and structured error extraction.
 */
export async function request<T>(
  endpoint: string,
  options: RequestOptions = {},
): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  let url = `${baseUrl}${normalizedEndpoint}`;

  // Append search parameters if provided
  if (options.params) {
    const searchParams = new URLSearchParams();
    Object.entries(options.params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    if (queryString) {
      url += (url.includes('?') ? '&' : '?') + queryString;
    }
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const headers = new Headers(options.headers || {});
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json');
  }
  // If body is NOT FormData, set application/json default
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // Handle non-2xx responses
    if (!response.ok) {
      let errorCode = 'HTTP_ERROR';
      let errorMessage = `HTTP ${response.status}: ${response.statusText || 'Request failed'}`;
      let errorDetails: Array<Record<string, unknown>> | undefined;

      try {
        const errorJson = await response.json();
        if (errorJson && typeof errorJson === 'object') {
          if (errorJson.error && typeof errorJson.error === 'object') {
            errorCode = errorJson.error.code || errorCode;
            errorMessage = errorJson.error.message || errorMessage;
            errorDetails = errorJson.error.details;
          } else if (errorJson.detail) {
            // Standard FastAPI HTTPException
            errorMessage = typeof errorJson.detail === 'string'
              ? errorJson.detail
              : JSON.stringify(errorJson.detail);
          }
        }
      } catch {
        // Body was not JSON; use default status message
      }

      throw new ApiError(errorMessage, response.status, errorCode, errorDetails);
    }

    // Return parsed JSON payload (or empty object for 204 No Content)
    if (response.status === 204) {
      return {} as T;
    }

    return (await response.json()) as T;
  } catch (err: unknown) {
    clearTimeout(timeoutId);

    if (err instanceof ApiError) {
      throw err;
    }

    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError(
        `Request timed out after ${timeoutMs / 1000}s. The service may be busy processing.`,
        408,
        'TIMEOUT_ERROR',
      );
    }

    const nativeMessage = err instanceof Error ? err.message : 'Network failure';
    throw new ApiError(
      `Network request failed: ${nativeMessage}`,
      0,
      'NETWORK_ERROR',
    );
  }
}
