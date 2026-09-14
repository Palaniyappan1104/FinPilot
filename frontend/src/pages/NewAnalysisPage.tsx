import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { SlidersHorizontal } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { CompanySearch } from '../components/company/CompanySearch';
import { CompanyOverviewCard } from '../components/company/CompanyOverviewCard';
import { InvestorProfileForm } from '../components/profile/InvestorProfileForm';
import { ChatWindow } from '../components/chat/ChatWindow';
import { apiService } from '../services/api';
import { CompanyInfo } from '../types';

export const NewAnalysisPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const {
    selectedCompany,
    setSelectedCompany,
    profile,
    updateProfile,
    startNewAnalysis,
  } = useApp();

  const [showProfileConfig, setShowProfileConfig] = useState(false);

  // Sync with URL query or ticker
  useEffect(() => {
    let active = true;
    const tickerParam = searchParams.get('ticker');
    if (tickerParam) {
      apiService.getCompany(tickerParam).then((comp) => {
        if (active && comp) {
          setSelectedCompany(comp);
          updateProfile({ ticker: comp.ticker, target_company: comp.name });
        }
      });
    }
    return () => {
      active = false;
    };
  }, [searchParams, setSelectedCompany, updateProfile]);

  const handleSelectCompany = (comp: CompanyInfo) => {
    setSelectedCompany(comp);
    updateProfile({ ticker: comp.ticker, target_company: comp.name });
  };

  const handleLaunchAnalysis = async () => {
    if (!selectedCompany) return;
    const analysisId = await startNewAnalysis(selectedCompany.ticker, profile);
    navigate(`/analysis/${analysisId}/progress`);
  };

  return (
    <div data-testid="new-analysis-page" className="space-y-6">
      {/* Page Heading */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
        <div>
          <h2 className="text-xl font-bold text-slate-900">
            Initiate Financial Research
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Select a target company, calibrate investor profile boundaries, and engage the Conversation Agent.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setShowProfileConfig(!showProfileConfig)}
          className="inline-flex items-center px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors self-start sm:self-auto"
        >
          <SlidersHorizontal className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
          <span>{showProfileConfig ? 'Hide Profile Settings' : 'Customize Profile Constraints'}</span>
        </button>
      </div>

      {/* Optional Profile Drawer / Card */}
      {showProfileConfig && (
        <div className="mb-4">
          <InvestorProfileForm
            onSaved={() => setShowProfileConfig(false)}
          />
        </div>
      )}

      {/* Main Grid: Left Side Company Info, Right Side Chat Agent */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Company Search & Overview */}
        <div className="lg:col-span-5 space-y-4">
          <CompanySearch
            selectedTicker={selectedCompany?.ticker}
            onSelectCompany={handleSelectCompany}
          />

          {selectedCompany && (
            <CompanyOverviewCard
              company={selectedCompany}
              onAnalyzeClick={handleLaunchAnalysis}
            />
          )}

          {/* Quick Profile Summary Badge */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 text-xs space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-900">
                Active Investor Profile:
              </span>
              <span className="capitalize font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 text-[10px]">
                {profile.risk_tolerance || 'Moderate'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-slate-600 text-[11px] pt-1 border-t border-slate-100">
              <div>
                <span className="text-slate-400 block">Horizon:</span>
                <span className="font-medium text-slate-800">
                  {profile.time_horizon || '3-5 years'}
                </span>
              </div>
              <div>
                <span className="text-slate-400 block">Capital:</span>
                <span className="font-medium text-slate-800">
                  {profile.currency || 'USD'} {Number(profile.capital_amount || 50000).toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Conversational Query Agent & Clarification UI */}
        <div className="lg:col-span-7">
          <ChatWindow />
        </div>
      </div>
    </div>
  );
};
