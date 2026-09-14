import React, { useState, useEffect } from 'react';
import { ShieldCheck, Check, AlertCircle, RefreshCw } from 'lucide-react';
import { InvestorProfile } from '../../types';
import { useApp } from '../../context/AppContext';

interface InvestorProfileFormProps {
  onSaved?: () => void;
  className?: string;
}

export const InvestorProfileForm: React.FC<InvestorProfileFormProps> = ({
  onSaved,
  className = '',
}) => {
  const { profile, updateProfile, resetProfile } = useApp();

  const [formData, setFormData] = useState<InvestorProfile>(profile);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    setFormData(profile);
  }, [profile]);

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.investment_goal || !formData.investment_goal.trim()) {
      newErrors.investment_goal = 'Investment objective is required.';
    }

    if (!formData.time_horizon) {
      newErrors.time_horizon = 'Please select an investment time horizon.';
    }

    const capital = Number(formData.capital_amount);
    if (!capital || isNaN(capital) || capital <= 0) {
      newErrors.capital_amount = 'Capital amount must be a positive finite number.';
    } else if (capital > 10000000000) {
      newErrors.capital_amount = 'Capital amount exceeds allowable threshold.';
    }

    if (!formData.risk_tolerance) {
      newErrors.risk_tolerance = 'Please specify your risk tolerance profile.';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    updateProfile(formData);
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2500);
    if (onSaved) onSaved();
  };

  const handlePresetCapital = (amt: number, curr?: string) => {
    setFormData((prev) => ({
      ...prev,
      capital_amount: amt,
      currency: curr || prev.currency || 'USD',
    }));
    if (errors.capital_amount) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next.capital_amount;
        return next;
      });
    }
  };

  return (
    <form
      data-testid="investor-profile-form"
      onSubmit={handleSubmit}
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 ${className}`}
    >
      <div className="flex items-center justify-between pb-4 border-b border-slate-100">
        <div>
          <h3 className="text-base font-bold text-slate-900">
            Investor Constraints & Preferences
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Calibrate portfolio boundaries. Persisted across your session.
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            resetProfile();
            setFormData(profile);
          }}
          className="text-xs text-slate-400 hover:text-slate-600 flex items-center"
        >
          <RefreshCw className="w-3 h-3 mr-1" />
          Reset Default
        </button>
      </div>

      <div className="space-y-4 pt-4">
        {/* Investment Goal */}
        <div>
          <label
            htmlFor="investment_goal"
            className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1"
          >
            Investment Objective
          </label>
          <select
            id="investment_goal"
            value={formData.investment_goal || ''}
            onChange={(e) =>
              setFormData({ ...formData, investment_goal: e.target.value })
            }
            className={`w-full px-3 py-2 text-sm bg-white border rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 ${
              errors.investment_goal ? 'border-rose-300 ring-1 ring-rose-300' : 'border-slate-300'
            }`}
          >
            <option value="Capital Appreciation & Moderate Growth">
              Capital Appreciation & Growth
            </option>
            <option value="Wealth Preservation & Conservative Return">
              Wealth Preservation & Safety
            </option>
            <option value="High Dividend Yield & Cash Flow">
              High Dividend Yield & Cash Flow
            </option>
            <option value="Balanced Multi-Asset Growth">Balanced Growth & Income</option>
            <option value="Speculative High-Growth Opportunity">
              Speculative High-Growth / Thematic
            </option>
          </select>
          {errors.investment_goal && (
            <p className="text-xs text-rose-600 mt-1 flex items-center">
              <AlertCircle className="w-3 h-3 mr-1" />
              {errors.investment_goal}
            </p>
          )}
        </div>

        {/* Time Horizon */}
        <div>
          <label
            htmlFor="time_horizon"
            className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1"
          >
            Holding Period / Time Horizon
          </label>
          <select
            id="time_horizon"
            value={formData.time_horizon || ''}
            onChange={(e) =>
              setFormData({ ...formData, time_horizon: e.target.value })
            }
            className={`w-full px-3 py-2 text-sm bg-white border rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 ${
              errors.time_horizon ? 'border-rose-300 ring-1 ring-rose-300' : 'border-slate-300'
            }`}
          >
            <option value="<1 year">Short-Term (&lt; 1 year)</option>
            <option value="1-3 years">Medium-Term (1–3 years)</option>
            <option value="3-5 years">Long-Term (3–5 years)</option>
            <option value="5+ years">Ultra Long-Term (5+ years)</option>
          </select>
          {errors.time_horizon && (
            <p className="text-xs text-rose-600 mt-1 flex items-center">
              <AlertCircle className="w-3 h-3 mr-1" />
              {errors.time_horizon}
            </p>
          )}
        </div>

        {/* Capital Amount & Currency */}
        <div>
          <label
            htmlFor="capital_amount"
            className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1"
          >
            Available Investment Capital
          </label>
          <div className="flex space-x-2">
            <select
              aria-label="Currency Selector"
              value={formData.currency || 'USD'}
              onChange={(e) =>
                setFormData({ ...formData, currency: e.target.value })
              }
              className="w-24 px-2 py-2 text-sm bg-slate-50 border border-slate-300 rounded-lg text-slate-800 font-semibold focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="USD">USD ($)</option>
              <option value="INR">INR (₹)</option>
              <option value="EUR">EUR (€)</option>
              <option value="GBP">GBP (£)</option>
            </select>
            <input
              id="capital_amount"
              type="number"
              min="1"
              step="100"
              value={formData.capital_amount ?? ''}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  capital_amount: e.target.value ? Number(e.target.value) : undefined,
                })
              }
              placeholder="e.g. 50000"
              className={`flex-1 px-3 py-2 text-sm bg-white border rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 ${
                errors.capital_amount ? 'border-rose-300 ring-1 ring-rose-300' : 'border-slate-300'
              }`}
            />
          </div>
          {errors.capital_amount && (
            <p className="text-xs text-rose-600 mt-1 flex items-center">
              <AlertCircle className="w-3 h-3 mr-1" />
              {errors.capital_amount}
            </p>
          )}

          {/* Quick presets */}
          <div className="flex items-center space-x-2 mt-2">
            <span className="text-[11px] text-slate-400">Presets:</span>
            <button
              type="button"
              onClick={() => handlePresetCapital(10000, 'USD')}
              className="text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 hover:bg-slate-200"
            >
              $10,000
            </button>
            <button
              type="button"
              onClick={() => handlePresetCapital(50000, 'USD')}
              className="text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 hover:bg-slate-200"
            >
              $50,000
            </button>
            <button
              type="button"
              onClick={() => handlePresetCapital(100000, 'INR')}
              className="text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 hover:bg-slate-200"
            >
              ₹1,00,000
            </button>
            <button
              type="button"
              onClick={() => handlePresetCapital(500000, 'INR')}
              className="text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 hover:bg-slate-200"
            >
              ₹5,00,000
            </button>
          </div>
        </div>

        {/* Risk Tolerance */}
        <div>
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
            Risk Tolerance
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              {
                value: 'conservative',
                label: 'Conservative',
                desc: 'Priority on capital protection and low volatility',
              },
              {
                value: 'moderate',
                label: 'Moderate',
                desc: 'Balanced risk-return trade-off with standard drawdown tolerance',
              },
              {
                value: 'aggressive',
                label: 'Aggressive',
                desc: 'Maximized growth potential with tolerance for elevated beta',
              },
            ].map((opt) => (
              <label
                key={opt.value}
                className={`flex flex-col p-3 rounded-lg border cursor-pointer transition-all ${
                  formData.risk_tolerance?.toLowerCase() === opt.value
                    ? 'border-emerald-500 bg-emerald-50/50 ring-1 ring-emerald-500'
                    : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <input
                    id={`risk-${opt.value}`}
                    aria-label={opt.label}
                    type="radio"
                    name="risk_tolerance"
                    value={opt.value}
                    checked={formData.risk_tolerance?.toLowerCase() === opt.value}
                    onChange={(e) =>
                      setFormData({ ...formData, risk_tolerance: e.target.value })
                    }
                    className="text-emerald-600 focus:ring-emerald-500 h-3.5 w-3.5"
                  />
                  <span className="text-xs font-bold text-slate-800">
                    {opt.label}
                  </span>
                </div>
                <span className="text-[11px] text-slate-500 mt-1 leading-snug">
                  {opt.desc}
                </span>
              </label>
            ))}
          </div>
          {errors.risk_tolerance && (
            <p className="text-xs text-rose-600 mt-1 flex items-center">
              <AlertCircle className="w-3 h-3 mr-1" />
              {errors.risk_tolerance}
            </p>
          )}
        </div>
      </div>

      {/* Save Button & Feedback */}
      <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between">
        <div className="flex items-center text-xs text-slate-500">
          <ShieldCheck className="w-4 h-4 text-emerald-600 mr-1.5" />
          <span>Used by Aggregator Agent to personalize recommendations.</span>
        </div>

        <div className="flex items-center space-x-3">
          {saveSuccess && (
            <span className="text-xs font-semibold text-emerald-600 flex items-center">
              <Check className="w-3.5 h-3.5 mr-1" />
              Profile Saved
            </span>
          )}
          <button
            type="submit"
            className="inline-flex items-center px-4 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold transition-colors shadow-sm"
          >
            Save Constraints
          </button>
        </div>
      </div>
    </form>
  );
};
