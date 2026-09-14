import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AnalysisProgressBar } from '../components/analysis/AnalysisProgressBar';
import { SpecialistStatusGrid } from '../components/analysis/SpecialistStatusGrid';

describe('Analysis Progress & Specialist Status (Phase 16.6)', () => {
  it('renders progress bar with stage and stepper completion status', () => {
    render(
      <AnalysisProgressBar
        progressPercent={65}
        currentStage="Specialist Agents executing concurrent research"
      />,
    );

    expect(screen.getByTestId('analysis-progress-bar')).toBeInTheDocument();
    expect(screen.getByText('65%')).toBeInTheDocument();
    expect(
      screen.getByText('Specialist Agents executing concurrent research'),
    ).toBeInTheDocument();
    expect(screen.getByText('Query & Profile')).toBeInTheDocument();
    expect(screen.getByText('CIO Allocation')).toBeInTheDocument();
    expect(screen.getByText('Specialist Execution')).toBeInTheDocument();
  });

  it('renders per-specialist status cards across all domains', () => {
    const onSelect = vi.fn();
    render(
      <SpecialistStatusGrid
        specialistStatuses={{
          technical: 'completed',
          fundamental: 'completed',
          news: 'running',
          research: 'pending',
          risk: 'pending',
        }}
        onSelectSpecialist={onSelect}
      />,
    );

    expect(screen.getByTestId('specialist-status-grid')).toBeInTheDocument();
    expect(screen.getByText('Technical Analyst')).toBeInTheDocument();
    expect(screen.getByText('Fundamental Analyst')).toBeInTheDocument();
    expect(screen.getByText('News & Sentiment Analyst')).toBeInTheDocument();
    expect(screen.getByText('Research Vault Analyst')).toBeInTheDocument();
    expect(screen.getByText('Risk Analyst')).toBeInTheDocument();

    // Verify completed, running, pending badges
    expect(screen.getAllByText('completed').length).toBe(2);
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getAllByText('pending').length).toBe(2);
  });
});
