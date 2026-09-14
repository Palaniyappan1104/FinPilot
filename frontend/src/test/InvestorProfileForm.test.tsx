import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { InvestorProfileForm } from '../components/profile/InvestorProfileForm';
import { AppContextProvider } from '../context/AppContext';

describe('Investor Profile Inputs & Session Persistence (Phase 16.5)', () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it('renders form inputs for goal, horizon, capital, and risk tolerance', () => {
    render(
      <AppContextProvider>
        <InvestorProfileForm />
      </AppContextProvider>,
    );

    expect(
      screen.getByLabelText(/Investment Objective/i),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Holding Period \/ Time Horizon/i),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Available Investment Capital/i),
    ).toBeInTheDocument();
    expect(screen.getByText('Moderate')).toBeInTheDocument();
    expect(screen.getByText('Conservative')).toBeInTheDocument();
    expect(screen.getByText('Aggressive')).toBeInTheDocument();
  });

  it('validates capital amount to prevent invalid or zero values', async () => {
    render(
      <AppContextProvider>
        <InvestorProfileForm />
      </AppContextProvider>,
    );

    const capitalInput = screen.getByLabelText(
      /Available Investment Capital/i,
    );
    fireEvent.change(capitalInput, { target: { value: '0' } });

    const form = screen.getByTestId('investor-profile-form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(
        screen.getByText(/Capital amount must be a positive finite number/i),
      ).toBeInTheDocument();
    });
  });

  it('persists validated profile inputs to sessionStorage', async () => {
    const onSaved = vi.fn();
    render(
      <AppContextProvider>
        <InvestorProfileForm onSaved={onSaved} />
      </AppContextProvider>,
    );

    const capitalInput = screen.getByLabelText(
      /Available Investment Capital/i,
    );
    fireEvent.change(capitalInput, { target: { value: '150000' } });

    const aggressiveRadio = screen.getByRole('radio', { name: /aggressive/i });
    fireEvent.click(aggressiveRadio);

    const form = screen.getByTestId('investor-profile-form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText('Profile Saved')).toBeInTheDocument();
    });

    expect(onSaved).toHaveBeenCalled();

    // Verify sessionStorage has updated data
    const savedRaw = sessionStorage.getItem('finpilot_investor_profile');
    expect(savedRaw).not.toBeNull();
    const parsed = JSON.parse(savedRaw!);
    expect(parsed.capital_amount).toBe(150000);
    expect(parsed.risk_tolerance).toBe('aggressive');
  });
});
