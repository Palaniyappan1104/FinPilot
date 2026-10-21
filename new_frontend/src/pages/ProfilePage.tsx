import React, { useState } from 'react';
import { UserCheck, Check } from 'lucide-react';
import { useApp } from '../context/AppContext';

export const ProfilePage: React.FC = () => {
  const { profile, updateProfile } = useApp();

  const [timeHorizon, setTimeHorizon] = useState(profile.time_horizon || '3-5 years');
  const [riskTolerance, setRiskTolerance] = useState(profile.risk_tolerance || 'moderate');
  const [capitalAmount, setCapitalAmount] = useState(profile.capital_amount || 50000);
  const [investmentGoal, setInvestmentGoal] = useState(
    profile.investment_goal || 'Capital Appreciation & Long-Term Compounding',
  );
  const [savedBanner, setSavedBanner] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    updateProfile({
      time_horizon: timeHorizon,
      risk_tolerance: riskTolerance,
      capital_amount: Number(capitalAmount) || 50000,
      investment_goal: investmentGoal,
    });
    setSavedBanner(true);
    setTimeout(() => setSavedBanner(false), 3000);
  };

  return (
    <div className="space-y-8 max-w-3xl">
      {/* Header Banner */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-850 to-blue-950 border border-slate-800 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-sm text-white">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-blue-300">
            <UserCheck className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">Investor Mandate</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            Investor Profile & Risk Calibration
          </h1>
          <p className="text-xs text-slate-200">
            Configure portfolio constraints and risk appetite used by the Conversation Agent and Chief Strategist.
          </p>
        </div>
      </div>

      {savedBanner && (
        <div className="p-4 rounded-xl bg-green-50 border border-green-200 text-xs text-green-700 flex items-center space-x-2 shadow-xs animate-fadeIn">
          <Check className="w-4 h-4 text-green-600 flex-shrink-0" />
          <span>Investor constraints saved successfully and applied to upcoming analyses!</span>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSave} className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 space-y-6 shadow-xs">
        {/* Risk Tolerance */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-700 block">
            Risk Tolerance Appetite
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { id: 'conservative', label: 'Conservative', desc: 'Capital preservation, low beta' },
              { id: 'moderate', label: 'Moderate', desc: 'Balanced growth & volatility tolerance' },
              { id: 'aggressive', label: 'Aggressive', desc: 'Maximum capital appreciation, high vol' },
            ].map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setRiskTolerance(opt.id)}
                className={`p-4 rounded-xl border text-left transition-all ${
                  riskTolerance === opt.id
                    ? 'bg-blue-50 border-blue-500 text-blue-900 shadow-xs'
                    : 'bg-slate-50 border-slate-200 text-slate-700 hover:border-slate-300'
                }`}
              >
                <div className="text-xs font-bold capitalize">{opt.label}</div>
                <div className="text-[11px] text-slate-500 mt-1">{opt.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Investment Horizon */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-700 block">
            Investment Time Horizon
          </label>
          <select
            value={timeHorizon}
            onChange={(e) => setTimeHorizon(e.target.value)}
            className="w-full px-4 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-slate-900 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
          >
            <option value="Under 6 months">Under 6 months (Short-term tactical)</option>
            <option value="1-2 years">1-2 years (Medium-term cyclical)</option>
            <option value="3-5 years">3-5 years (Long-term compounder)</option>
            <option value="5+ years">5+ years (Multi-cycle secular)</option>
          </select>
        </div>

        {/* Capital Amount */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-700 block">
            Capital Allocation Pool ($ USD)
          </label>
          <input
            type="number"
            min="1000"
            step="1000"
            value={capitalAmount}
            onChange={(e) => setCapitalAmount(Number(e.target.value))}
            className="w-full px-4 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-slate-900 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
          />
        </div>

        {/* Primary Goal */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-700 block">
            Primary Investment Objective
          </label>
          <input
            type="text"
            value={investmentGoal}
            onChange={(e) => setInvestmentGoal(e.target.value)}
            placeholder="e.g. Compound capital, dividend income, tech growth"
            className="w-full px-4 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-slate-900 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
          />
        </div>

        <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
          <span className="text-[11px] text-slate-400">
            Stored locally in browser session
          </span>

          <button
            type="submit"
            className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs transition-colors shadow-xs"
          >
            Save Profile Constraints
          </button>
        </div>
      </form>
    </div>
  );
};
