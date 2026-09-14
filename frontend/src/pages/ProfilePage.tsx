import React from 'react';
import { UserCheck } from 'lucide-react';
import { InvestorProfileForm } from '../components/profile/InvestorProfileForm';

export const ProfilePage: React.FC = () => {
  return (
    <div data-testid="profile-page" className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="pb-4 border-b border-slate-200">
        <div className="flex items-center space-x-2">
          <UserCheck className="w-5 h-5 text-emerald-600" />
          <h2 className="text-xl font-bold text-slate-900">
            Investor Constraints & Preferences Configuration
          </h2>
        </div>
        <p className="text-xs text-slate-500 mt-0.5">
          Configure capital constraints, time horizons, and risk tolerance. These constraints are passed to the Chief Investment Officer and Aggregator agents to calibrate recommendation suitability.
        </p>
      </div>

      <InvestorProfileForm />
    </div>
  );
};
