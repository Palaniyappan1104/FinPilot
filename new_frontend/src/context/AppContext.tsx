import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { InvestorProfile, AnalyzeResponse, StoredReport } from '../types';

interface AppContextType {
  profile: InvestorProfile;
  updateProfile: (updated: Partial<InvestorProfile>) => void;
  storedReports: StoredReport[];
  saveReport: (report: AnalyzeResponse, query: string) => StoredReport;
  deleteReport: (id: string) => void;
  activeReport: AnalyzeResponse | null;
  setActiveReport: (report: AnalyzeResponse | null) => void;
  selectedTicker: string;
  setSelectedTicker: (ticker: string) => void;
}

const DEFAULT_PROFILE: InvestorProfile = {
  investment_goal: 'Capital Appreciation & Long-Term Compounding',
  time_horizon: '3-5 years',
  capital_amount: 50000,
  risk_tolerance: 'moderate',
};

const AppContext = createContext<AppContextType | undefined>(undefined);

const PROFILE_STORAGE_KEY = 'finpilot_investor_profile';
const REPORTS_STORAGE_KEY = 'finpilot_stored_reports';

export const AppContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // 1. Investor profile
  const [profile, setProfile] = useState<InvestorProfile>(() => {
    try {
      const saved = localStorage.getItem(PROFILE_STORAGE_KEY);
      return saved ? JSON.parse(saved) : DEFAULT_PROFILE;
    } catch {
      return DEFAULT_PROFILE;
    }
  });

  const updateProfile = useCallback((updated: Partial<InvestorProfile>) => {
    setProfile((prev) => {
      const next = { ...prev, ...updated };
      try {
        localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(next));
      } catch {
        // storage disabled
      }
      return next;
    });
  }, []);

  // 2. Stored Reports History
  const [storedReports, setStoredReports] = useState<StoredReport[]>(() => {
    try {
      const saved = localStorage.getItem(REPORTS_STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const saveReport = useCallback((report: AnalyzeResponse, query: string): StoredReport => {
    const newReport: StoredReport = {
      id: `rep_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      timestamp: new Date().toISOString(),
      ticker: report.ticker,
      company: report.company,
      query,
      confidence: report.confidence || 'Medium',
      executive_summary: report.executive_summary || '',
      verdict: report.final_verdict || '',
      data: report,
    };

    setStoredReports((prev) => {
      // Prepend the new report
      const next = [newReport, ...prev.filter((r) => r.ticker !== newReport.ticker || r.timestamp !== newReport.timestamp)];
      try {
        localStorage.setItem(REPORTS_STORAGE_KEY, JSON.stringify(next));
      } catch {
        // quota exceeded / disabled
      }
      return next;
    });

    return newReport;
  }, []);

  const deleteReport = useCallback((id: string) => {
    setStoredReports((prev) => {
      const next = prev.filter((r) => r.id !== id);
      try {
        localStorage.setItem(REPORTS_STORAGE_KEY, JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
  }, []);

  // 3. Active Report & Selected Ticker
  const [activeReport, setActiveReport] = useState<AnalyzeResponse | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string>('AAPL');

  return (
    <AppContext.Provider
      value={{
        profile,
        updateProfile,
        storedReports,
        saveReport,
        deleteReport,
        activeReport,
        setActiveReport,
        selectedTicker,
        setSelectedTicker,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = (): AppContextType => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppContextProvider');
  }
  return context;
};
