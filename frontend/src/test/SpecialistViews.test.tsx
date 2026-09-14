import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TechnicalView } from '../components/specialists/TechnicalView';
import { FundamentalView } from '../components/specialists/FundamentalView';
import { NewsView } from '../components/specialists/NewsView';
import { RiskView } from '../components/specialists/RiskView';
import { MOCK_REPORTS } from '../services/mockData';

const report = MOCK_REPORTS['rep-aapl-001'];

describe('Specialist Results Views (Phase 16.7)', () => {
  it('renders technical analysis metrics, moving averages, and RSI', () => {
    render(<TechnicalView data={report.technical} />);

    expect(screen.getByTestId('technical-view')).toBeInTheDocument();
    expect(screen.getByText('RSI (14-Day)')).toBeInTheDocument();
    expect(screen.getByText('56.4')).toBeInTheDocument();
    expect(screen.getByText('SMA Alignment')).toBeInTheDocument();
    expect(screen.getByText('Primary Support')).toBeInTheDocument();
    expect(screen.getByText('$218.50')).toBeInTheDocument();
  });

  it('renders fundamental multiples, margins, and cash flow', () => {
    render(<FundamentalView data={report.fundamental} />);

    expect(screen.getByTestId('fundamental-view')).toBeInTheDocument();
    expect(screen.getByText('P/E Ratio (TTM)')).toBeInTheDocument();
    expect(screen.getByText('34.2x')).toBeInTheDocument();
    expect(screen.getByText('Free Cash Flow')).toBeInTheDocument();
    expect(screen.getByText('$104.8B')).toBeInTheDocument();
    expect(screen.getByText('Gross Margin')).toBeInTheDocument();
    expect(screen.getByText('46.2%')).toBeInTheDocument();
  });

  it('renders news sentiment breakdown and curated headlines', () => {
    render(<NewsView data={report.news} />);

    expect(screen.getByTestId('news-view')).toBeInTheDocument();
    expect(screen.getByText('Sentiment Distribution')).toBeInTheDocument();
    expect(screen.getByText(/Positive: 68%/i)).toBeInTheDocument();
    expect(
      screen.getByText(
        'Apple Expands Private Cloud Compute Architecture for Enterprise Workflow',
      ),
    ).toBeInTheDocument();
  });

  it('renders risk assessment, VaR, and categorized risk factors', () => {
    render(<RiskView data={report.risk} />);

    expect(screen.getByTestId('risk-view')).toBeInTheDocument();
    expect(screen.getByText(/38 \/ 100/i)).toBeInTheDocument();
    expect(screen.getByText(/Low Risk/i)).toBeInTheDocument();
    expect(screen.getByText('Value at Risk (95% 1-Month VaR)')).toBeInTheDocument();
    expect(screen.getByText('6.2%')).toBeInTheDocument();
    expect(screen.getByText('Market & Beta Risk')).toBeInTheDocument();
  });
});
