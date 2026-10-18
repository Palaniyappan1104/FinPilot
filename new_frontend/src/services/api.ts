import { ClarifyRequest, ClarifyResponse, AnalyzeRequest, AnalyzeResponse } from '../types';

const API_BASE = '/api/v1';

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: any,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const errorText = await res.text();
    let errorDetail = errorText;
    try {
      const errJson = JSON.parse(errorText);
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      } else if (errJson.error?.message) {
        errorDetail = errJson.error.message;
      }
    } catch {
      // errorText was not JSON, use errorText directly
    }
    throw new ApiError(res.status, errorDetail || `Request failed with HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const apiService = {
  /**
   * Clarify a user query to extract questions and detected stock/company
   */
  async clarify(query: string): Promise<ClarifyResponse> {
    const res = await fetch(`${API_BASE}/demo/clarify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query } as ClarifyRequest),
    });
    return handleResponse<ClarifyResponse>(res);
  },

  /**
   * Run full multi-stage AI research pipeline
   */
  async analyze(payload: AnalyzeRequest, signal?: AbortSignal): Promise<AnalyzeResponse> {
    const res = await fetch(`${API_BASE}/demo/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });
    return handleResponse<AnalyzeResponse>(res);
  },

  /**
   * Health check endpoint
   */
  async getHealth(): Promise<{ status: string }> {
    try {
      const res = await fetch(`${API_BASE}/health`);
      return handleResponse<{ status: string }>(res);
    } catch {
      return { status: 'unavailable' };
    }
  },
};
