import React, { useState } from 'react';
import {
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  ShieldAlert,
} from 'lucide-react';
import {
  FundamentalReportSection,
  NewsReportSection,
  RiskReportSection,
  SpecialistType,
  TechnicalReportSection,
} from '../../types';
import { TechnicalView } from './TechnicalView';
import { FundamentalView } from './FundamentalView';
import { NewsView } from './NewsView';
import { RiskView } from './RiskView';

interface SpecialistTabContainerProps {
  technical?: TechnicalReportSection;
  fundamental?: FundamentalReportSection;
  news?: NewsReportSection;
  risk?: RiskReportSection;
  initialTab?: SpecialistType;
  className?: string;
}

export const SpecialistTabContainer: React.FC<SpecialistTabContainerProps> = ({
  technical,
  fundamental,
  news,
  risk,
  initialTab = 'technical',
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<SpecialistType>(initialTab);

  const tabs: Array<{
    id: SpecialistType;
    label: string;
    icon: React.ReactNode;
  }> = [
    {
      id: 'technical',
      label: 'Technical Analysis',
      icon: <TrendingUp className="w-4 h-4 mr-2" />,
    },
    {
      id: 'fundamental',
      label: 'Fundamental Analysis',
      icon: <FileSpreadsheet className="w-4 h-4 mr-2" />,
    },
    {
      id: 'news',
      label: 'News & Sentiment',
      icon: <Newspaper className="w-4 h-4 mr-2" />,
    },
    {
      id: 'risk',
      label: 'Risk & Volatility',
      icon: <ShieldAlert className="w-4 h-4 mr-2" />,
    },
  ];

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Tab Navigation */}
      <div className="flex border-b border-slate-200 overflow-x-auto bg-white rounded-t-xl px-2 pt-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center px-4 py-3 text-xs font-semibold border-b-2 whitespace-nowrap transition-colors ${
              activeTab === tab.id
                ? 'border-slate-900 text-slate-900 bg-slate-50/50'
                : 'border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Body */}
      <div>
        {activeTab === 'technical' && <TechnicalView data={technical} />}
        {activeTab === 'fundamental' && <FundamentalView data={fundamental} />}
        {activeTab === 'news' && <NewsView data={news} />}
        {activeTab === 'risk' && <RiskView data={risk} />}
      </div>
    </div>
  );
};
