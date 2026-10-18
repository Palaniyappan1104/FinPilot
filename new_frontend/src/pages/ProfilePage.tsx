import React, { useState } from 'react';
import { UserCheck, ShieldCheck, Check, RotateCcw } from 'lucide-react';
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
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xl">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-emerald-400">
            <UserCheck className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">Investor Mandate</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            Investor Profile & Risk Calibration
          </h1>
          <p className="text-xs text-slate-400">
            Configure portfolio constraints and risk appetite used by the Conversation Agent and Chief Strategist.
          </p>
        </div>
      </div>

      {savedBanner && (
        <div className="p-4 rounded-xl bg-emerald-950/80 border border-emerald-800 text-xs text-emerald-300 flex items-center space-x-2 animate-fadeIn">
          <Check className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span>Investor constraints saved successfully and applied to upcoming analyses!</span>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSave} className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6 shadow-lg">
        {/* Risk Tolerance */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-300 block">
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
                    ? 'bg-emerald-950/60 border-emerald-500 text-white shadow-sm'
                    : 'bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="text-xs font-bold capitalize">{opt.label}</div>
                <div className="text-[11px] text-slate-400 mt-1">{opt.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Investment Horizon */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-300 block">
            Investment Time Horizon
          </label>
          <select
            value={timeHorizon}
            onChange={(e) => setTimeHorizon(e.target.value)}
            className="w-full px-4 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-200 text-xs focus:outline-none focus:ring-2 focus:ring-emerald-500"
          >
            <option value="Under 6 months">Under 6 months (Short-term tactical)</option>
            <option value="1-2 years">1-2 years (Medium-term cyclical)</option>
            <option value="3-5 years">3-5 years (Long-term compounder)</option>
            <option value="5+ years">5+ years (Multi-cycle secular)</option>
          </select>
        </div>

        {/* Capital Amount */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-300 block">
            Capital Allocation Pool ($ USD)
          </label>
          <input
            type="number"
            min="1000"
            step="1000"
            value={capitalAmount}
            onChange={(e) => setCapitalAmount(Number(e.target.value))}
            className="w-full px-4 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-200 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
        </div>

        {/* Primary Goal */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-300 block">
            Primary Investment Objective
          </label>
          <input
            type="text"
            value={investmentGoal}
            onChange={(e) => setInvestmentGoal(e.target.value)}
            placeholder="e.g. Compound capital, dividend income, tech growth"
            className="w-full px-4 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-200 text-xs focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
        </div>

        <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
          <span className="text-[11px] text-slate-500">
            Stored locally in browser session
          </span>

          <button
            type="submit"
            className="px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs transition-colors shadow-sm"
          >
            Save Profile Constraints
          </button>
        </div>
      </form>
    </div>
  );
};
