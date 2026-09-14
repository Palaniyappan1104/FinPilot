import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricCard } from '../components/common/MetricCard';

describe('MetricCard Component', () => {
  it('renders metric label and value correctly', () => {
    render(<MetricCard label="P/E Ratio (TTM)" value="34.2x" />);

    expect(screen.getByText('P/E Ratio (TTM)')).toBeInTheDocument();
    expect(screen.getByText('34.2x')).toBeInTheDocument();
  });

  it('renders optional unit, subtext, and badge', () => {
    render(
      <MetricCard
        label="RSI (14-Day)"
        value="56.4"
        subtext="Momentum within normal neutral range"
        badge={<span data-testid="test-badge">Neutral</span>}
      />,
    );

    expect(screen.getByText('RSI (14-Day)')).toBeInTheDocument();
    expect(screen.getByText('56.4')).toBeInTheDocument();
    expect(screen.getByText('Momentum within normal neutral range')).toBeInTheDocument();
    expect(screen.getByTestId('test-badge')).toBeInTheDocument();
  });

  it('supports custom child nodes for compound metric values', () => {
    render(
      <MetricCard label="SMA Alignment" size="lg">
        <div data-testid="compound-sma">
          <span>20d: $220.50</span>
          <span>50d: $215.00</span>
        </div>
      </MetricCard>,
    );

    expect(screen.getByText('SMA Alignment')).toBeInTheDocument();
    expect(screen.getByTestId('compound-sma')).toBeInTheDocument();
    expect(screen.getByText('20d: $220.50')).toBeInTheDocument();
  });
});
