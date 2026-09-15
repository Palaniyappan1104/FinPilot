import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from '../App';
import { AppContextProvider, SESSION_ANALYSIS_STORAGE_KEY } from '../context/AppContext';
import { NewAnalysisPage } from '../pages/NewAnalysisPage';
import { apiService } from '../services/api';

describe('App Shell & Navigation (Phase 16.1)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
  });

  it('renders application layout, navigation sidebar, and landing dashboard', async () => {
    render(<App />);

    // Brand and logo in sidebar
    expect(screen.getAllByText('FinPilot').length).toBeGreaterThan(0);
    expect(screen.getByText('AI Financial Research')).toBeInTheDocument();

    // Primary navigation links in sidebar
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('New Analysis')).toBeInTheDocument();
    expect(screen.getByText('Research Vault')).toBeInTheDocument();
    expect(screen.getByText('Multi-Agent Analysis')).toBeInTheDocument();
    expect(screen.getByText('Investor Profile')).toBeInTheDocument();

    // Landing dashboard hero title
    expect(
      screen.getByText(
        'Institutional-Grade Research & Investment Decision Support',
      ),
    ).toBeInTheDocument();

    // Primary Start New Analysis button on dashboard
    expect(screen.getByTestId('hero-start-analysis-btn')).toBeInTheDocument();

    // Non-dismissible regulatory disclosure footer is rendered
    await waitFor(() => {
      expect(screen.getByTestId('non-dismissible-disclaimer')).toBeInTheDocument();
    });
  });
});

describe('New Analysis SessionStorage Resume Behavior', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
  });

  it('resumes polling on mount for a running analysis without creating a duplicate analysis', async () => {
    sessionStorage.setItem(
      SESSION_ANALYSIS_STORAGE_KEY,
      JSON.stringify({
        analysisId: 'an-resume-running-001',
        ticker: 'MSFT',
        companyName: 'Microsoft Corporation',
      }),
    );

    const getStatusSpy = vi.spyOn(apiService, 'getAnalysisStatus').mockResolvedValue({
      analysisId: 'an-resume-running-001',
      ticker: 'MSFT',
      status: 'running',
      progressPercent: 60,
      progressStage: 'Specialist Execution',
      specialistStatuses: {
        technical: 'running',
        fundamental: 'running',
        news: 'pending',
        research: 'pending',
        risk: 'pending',
      },
    });

    const startAnalysisSpy = vi.spyOn(apiService, 'startAnalysis');

    render(
      <MemoryRouter initialEntries={['/analysis/new']}>
        <AppContextProvider>
          <NewAnalysisPage />
        </AppContextProvider>
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('analysis-progress-bar')).toBeInTheDocument();
      expect(screen.getAllByText('Specialist Execution').length).toBeGreaterThan(0);
    });

    // Verify status was queried for the persisted ID
    expect(getStatusSpy).toHaveBeenCalledWith('an-resume-running-001');

    // Verify NO duplicate analysis creation call was dispatched
    expect(startAnalysisSpy).not.toHaveBeenCalled();
  });

  it('retrieves existing report when resumed analysis is completed with report_id', async () => {
    sessionStorage.setItem(
      SESSION_ANALYSIS_STORAGE_KEY,
      JSON.stringify({
        analysisId: 'an-resume-completed-002',
        ticker: 'AAPL',
        companyName: 'Apple Inc.',
      }),
    );

    vi.spyOn(apiService, 'getAnalysisStatus').mockResolvedValue({
      analysisId: 'an-resume-completed-002',
      ticker: 'AAPL',
      status: 'completed',
      progressPercent: 100,
      progressStage: 'Complete',
      reportId: 'rep-resumed-001',
      specialistStatuses: {
        technical: 'completed',
        fundamental: 'completed',
        news: 'completed',
        research: 'completed',
        risk: 'completed',
      },
    });

    const getReportSpy = vi.spyOn(apiService, 'getReport').mockResolvedValue({
      report_id: 'rep-resumed-001',
      company: { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Technology', currency: 'USD' },
      recommendation: {
        stance: 'favorable',
        rationale: 'Strong ecosystem.',
        monitoring_points: ['Services margin expansion'],
      },
      overall_assessment: {
        summary: 'Robust profitability across segments.',
        specialist_consensus: 'favorable',
        signal_conflicts: [],
        completeness_ratio: 1.0,
      },
      key_reasons: ['Expanding services margins'],
      important_risks: ['Antitrust inquiries.'],
      evidence_sources: [
        {
          id: 'EV-1',
          claim: 'Gross margin expanded to 46%.',
          specialist_type: 'fundamental',
          source_tool: 'Financial Statements / Q3 10-Q',
          confidence: 0.9,
          timestamp: '2026-09-15T00:00:00Z',
        },
      ],
      created_at: '2026-09-15T00:00:00Z',
      confidence: 0.92,
      disclaimer: 'Institutional research only.',
      specialist_statuses: {
        technical: 'completed',
        fundamental: 'completed',
        news: 'completed',
        research: 'completed',
        risk: 'completed',
      },
    });

    render(
      <MemoryRouter initialEntries={['/analysis/new']}>
        <AppContextProvider>
          <NewAnalysisPage />
        </AppContextProvider>
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('report-view')).toBeInTheDocument();
      expect(screen.getByText('Strong ecosystem.')).toBeInTheDocument();
    });

    expect(getReportSpy).toHaveBeenCalledWith('rep-resumed-001');

    // Resume token must be cleared upon completion
    expect(sessionStorage.getItem(SESSION_ANALYSIS_STORAGE_KEY)).toBeNull();
  });

  it('safely clears malformed or stale resume token without crashing', async () => {
    sessionStorage.setItem(SESSION_ANALYSIS_STORAGE_KEY, 'not-valid-json{{');

    render(
      <MemoryRouter initialEntries={['/analysis/new']}>
        <AppContextProvider>
          <NewAnalysisPage />
        </AppContextProvider>
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('new-analysis-page')).toBeInTheDocument();
    });

    // Malformed token was discarded cleanly
    expect(sessionStorage.getItem(SESSION_ANALYSIS_STORAGE_KEY)).toBeNull();
  });
});

