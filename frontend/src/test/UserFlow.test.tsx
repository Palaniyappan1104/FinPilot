import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';

describe('Main User Flow Integration Test (Phase 17.4)', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();

    globalThis.fetch = vi.fn().mockImplementation(async (url: string | URL | Request) => {
      const urlStr = String(url);

      if (urlStr.includes('/api/v1/chat')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            analysis_id: 'an-flow-001',
            trace_id: 'tr-flow-001',
            status: 'completed',
            clarification_needed: false,
            clarification_questions: [],
            report_id: 'rep-aapl-001',
          }),
        };
      }

      if (urlStr.includes('/api/v1/reports/')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            report_id: 'rep-aapl-001',
            format: 'json',
            report: {
              report_id: 'rep-aapl-001',
              company: { ticker: 'AAPL', name: 'Apple Inc.', currency: 'USD' },
              recommendation: {
                stance: 'favorable',
                rationale: 'Exceptional profitability and ecosystem retention.',
              },
              overall_assessment: {
                synthesis: 'Holistic multi-agent consensus indicates strong financial health.',
                data_completeness_ratio: 1.0,
              },
              evidence_sources: [
                {
                  reference_id: 'EV-FLOW-01',
                  detail: 'Operating margins hold steady above 30%.',
                  specialist: 'fundamental',
                },
              ],
              important_risks: ['Regulatory scrutiny over App Store practices.'],
              disclaimer:
                'FinPilot provides automated financial research for informational purposes only. Past performance does not guarantee future results. All investments carry risk of loss.',
            },
          }),
        };
      }

      return {
        ok: true,
        status: 200,
        json: async () => ({ status: 'ok' }),
      } as Response;
    }) as typeof globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('completes the full research journey from query to report against mocked API', async () => {
    render(<App />);

    // 1. Land on Dashboard
    expect(
      screen.getByText(
        'Institutional-Grade Research & Investment Decision Support',
      ),
    ).toBeInTheDocument();

    // 2. Click "New Analysis" in sidebar navigation
    const newAnalysisLink = screen.getByRole('link', { name: /New Analysis/i });
    fireEvent.click(newAnalysisLink);

    // Verify on New Analysis page
    await waitFor(() => {
      expect(
        screen.getByText('Initiate Financial Research'),
      ).toBeInTheDocument();
    });

    // 3. Search and select Apple Inc. (AAPL)
    const companySearchInput = screen.getByPlaceholderText(
      /search by ticker/i,
    );
    fireEvent.focus(companySearchInput);
    fireEvent.change(companySearchInput, { target: { value: 'AAPL' } });

    await waitFor(() => {
      expect(screen.getByTestId('company-search-dropdown')).toBeInTheDocument();
    });

    const dropdown = screen.getByTestId('company-search-dropdown');
    const aaplOption = dropdown.querySelector('button');
    expect(aaplOption).not.toBeNull();
    fireEvent.click(aaplOption!);

    // Verify company overview card renders
    expect(screen.getByTestId('company-overview-card')).toBeInTheDocument();
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument();

    // 4. Submit query into Conversation Agent
    const chatInput = screen.getByPlaceholderText(
      /Ask a financial research question/i,
    );
    fireEvent.change(chatInput, {
      target: { value: 'Analyze Apple growth and services profitability' },
    });

    const sendBtn = screen.getByLabelText('Send query');
    fireEvent.click(sendBtn);

    // 5. Verify assistant response with multi-agent pipeline activation
    await waitFor(() => {
      expect(
        screen.getByText(/Activating FinPilot multi-agent research pipeline/i),
      ).toBeInTheDocument();
      expect(screen.getByText('View Final Report')).toBeInTheDocument();
    });

    // 8. Navigate to Final Report
    const viewReportBtn = screen.getByText('View Final Report');
    fireEvent.click(viewReportBtn);

    // 9. Verify full report view renders with all required sections
    await waitFor(() => {
      expect(screen.getByTestId('report-view')).toBeInTheDocument();
      expect(screen.getByTestId('report-header')).toBeInTheDocument();
      expect(
        screen.getByTestId('executive-recommendation'),
      ).toBeInTheDocument();
      expect(screen.getByTestId('critical-risks-section')).toBeInTheDocument();
      expect(
        screen.getByTestId('evidence-provenance-table'),
      ).toBeInTheDocument();
      expect(screen.getByTestId('report-disclaimer-card')).toBeInTheDocument();
    });
  });
});
