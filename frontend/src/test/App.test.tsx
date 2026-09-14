import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import App from '../App';

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
    expect(screen.getByText('Analysis Monitor')).toBeInTheDocument();
    expect(screen.getByText('Specialist Insights')).toBeInTheDocument();
    expect(screen.getByText('Research Reports')).toBeInTheDocument();
    expect(screen.getByText('Research Vault')).toBeInTheDocument();
    expect(screen.getByText('Investor Profile')).toBeInTheDocument();

    // Landing dashboard hero title
    expect(
      screen.getByText(
        'Institutional-Grade Research & Investment Decision Support',
      ),
    ).toBeInTheDocument();

    // Primary query form on dashboard
    expect(
      screen.getByPlaceholderText(
        "Ask an investment research question (e.g., 'Should I invest in Infosys for 5 years?')...",
      ),
    ).toBeInTheDocument();

    // Non-dismissible regulatory disclosure footer is rendered
    await waitFor(() => {
      expect(screen.getByTestId('non-dismissible-disclaimer')).toBeInTheDocument();
    });
  });
});
