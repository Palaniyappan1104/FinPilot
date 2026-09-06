import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import App from '../App';

describe('Frontend Smoke Test - App', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders application layout and dashboard successfully', async () => {
    // Mock health check API response
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          status: 'healthy',
          service: 'finpilot-backend',
          version: '0.1.0',
          environment: 'development',
        }),
      } as unknown as Response),
    );

    render(<App />);

    // Header brand and phase badge render
    expect(screen.getByText('FinPilot')).toBeInTheDocument();
    expect(screen.getByText('Phase 1 Foundation')).toBeInTheDocument();

    // Welcome banner and dashboard header render
    expect(screen.getByText('FinPilot System Dashboard')).toBeInTheDocument();

    // Backend status transitions to healthy with mocked data
    await waitFor(() => {
      expect(screen.getByText('Backend Operational')).toBeInTheDocument();
    });

    // Verification baseline list is rendered
    expect(
      screen.getByText('FastAPI modular backend with /health endpoint'),
    ).toBeInTheDocument();
  });
});
