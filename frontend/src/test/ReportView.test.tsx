import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ReportView } from '../components/report/ReportView';
import { MOCK_REPORTS } from '../services/mockData';

const report = MOCK_REPORTS['rep-aapl-001'];

describe('Final Investment Report View (Phase 16.9)', () => {
  it('renders complete report with header, recommendation stance, and reasons', () => {
    render(<ReportView report={report} />);

    expect(screen.getByTestId('report-view')).toBeInTheDocument();
    expect(screen.getByTestId('report-header')).toBeInTheDocument();
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    expect(screen.getByText('88% Statistical')).toBeInTheDocument();

    // Recommendation stance badge
    expect(screen.getByTestId('executive-recommendation')).toBeInTheDocument();
    expect(screen.getAllByText(/favorable/i).length).toBeGreaterThan(0);

    // Core analytical reasons
    expect(
      screen.getByText(/Core Analytical Reasons/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Services revenue now represents >25% of total revenue/i,
      ),
    ).toBeInTheDocument();
  });

  it('renders critical risks section with adverse scenario warnings', () => {
    render(<ReportView report={report} />);

    expect(screen.getByTestId('critical-risks-section')).toBeInTheDocument();
    expect(
      screen.getByText(
        /Critical Risk Factors & Adverse Scenarios/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Regulatory scrutiny targeting App Store fees/i,
      ),
    ).toBeInTheDocument();
  });

  it('renders evidence provenance table linking claims to specialist source tools', () => {
    render(<ReportView report={report} />);

    expect(screen.getByTestId('evidence-provenance-table')).toBeInTheDocument();
    expect(
      screen.getByText(
        /Evidence & Grounded Sources/i,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('SEC Edgar 10-Q Filing')).toBeInTheDocument();
    expect(screen.getByText('ChromaDB Filing Vault')).toBeInTheDocument();
  });

  it('renders mandatory non-dismissible regulatory disclaimer', () => {
    render(<ReportView report={report} />);

    const disclaimerBox = screen.getByTestId('report-disclaimer-card');
    expect(disclaimerBox).toBeInTheDocument();
    expect(
      screen.getByText(
        /Mandatory Regulatory Disclosure & Decision Support Limits/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/FinPilot provides automated financial research/i),
    ).toBeInTheDocument();
  });
});
