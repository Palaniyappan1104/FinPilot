import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CompanySearch } from '../components/company/CompanySearch';
import { CompanyOverviewCard } from '../components/company/CompanyOverviewCard';
import { MOCK_COMPANIES } from '../services/mockData';

describe('Company Selection & Display (Phase 16.4)', () => {
  it('searches equities and allows selection via autocomplete', async () => {
    const handleSelect = vi.fn();
    render(<CompanySearch onSelectCompany={handleSelect} />);

    const input = screen.getByPlaceholderText(/search by ticker/i);
    expect(input).toBeInTheDocument();

    // Focus input to open dropdown
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: 'NVDA' } });

    await waitFor(() => {
      expect(screen.getByTestId('company-search-dropdown')).toBeInTheDocument();
    });

    // Click NVDA from dropdown
    const nvdaOption = screen.getByText('NVIDIA Corporation');
    fireEvent.click(nvdaOption);

    expect(handleSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        ticker: 'NVDA',
        name: 'NVIDIA Corporation',
      }),
    );
  });

  it('renders detailed company overview card with financial metrics', () => {
    const onAnalyze = vi.fn();
    render(
      <CompanyOverviewCard
        company={MOCK_COMPANIES.AAPL}
        onAnalyzeClick={onAnalyze}
      />,
    );

    expect(screen.getByTestId('company-overview-card')).toBeInTheDocument();
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText(/Consumer Electronics/i)).toBeInTheDocument();

    // Market cap and P/E
    expect(screen.getByText('$3.48T')).toBeInTheDocument();
    expect(screen.getByText('34.2x')).toBeInTheDocument();

    // CTA button click
    const btn = screen.getByText(/Launch Multi-Agent Analysis/i);
    fireEvent.click(btn);
    expect(onAnalyze).toHaveBeenCalled();
  });
});
