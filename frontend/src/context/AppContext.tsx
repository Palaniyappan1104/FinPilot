/**
 * FinPilot Global Application Context (Phase 16.1.2).
 * Centralizes investor profile state with session persistence,
 * active analysis tracking, report state, and chat history.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';
import {
  AnalysisStatus,
  AnalysisSummary,
  ChatMessage,
  CompanyInfo,
  DocumentItem,
  FinalReport,
  InvestorProfile,
  SpecialistStatus,
  SpecialistType,
} from '../types';
import { mockApi } from '../services/mockApi';
import { MOCK_COMPANIES, MOCK_RECENT_ANALYSES } from '../services/mockData';

const SESSION_PROFILE_STORAGE_KEY = 'finpilot_investor_profile';

export interface ActiveAnalysisState {
  analysisId: string;
  ticker: string;
  companyName: string;
  status: AnalysisStatus;
  progressPercent: number;
  progressStage: string;
  specialistStatuses: Record<SpecialistType, SpecialistStatus>;
  reportId?: string;
  error?: string;
}

export interface AppContextType {
  // Investor Profile (16.5)
  profile: InvestorProfile;
  updateProfile: (updated: Partial<InvestorProfile>) => void;
  resetProfile: () => void;
  isProfileConfigured: boolean;

  // Company Selection (16.4)
  selectedCompany: CompanyInfo | null;
  setSelectedCompany: (company: CompanyInfo | null) => void;

  // Active Analysis & Status (16.6)
  activeAnalysis: ActiveAnalysisState | null;
  setActiveAnalysis: React.Dispatch<React.SetStateAction<ActiveAnalysisState | null>>;
  startNewAnalysis: (ticker: string, customProfile?: InvestorProfile) => Promise<string>;

  // Reports (16.9)
  activeReport: FinalReport | null;
  setActiveReport: (report: FinalReport | null) => void;
  recentAnalyses: AnalysisSummary[];
  refreshRecentAnalyses: () => Promise<void>;

  // Conversational Chat & Clarifications (16.3)
  chatMessages: ChatMessage[];
  addChatMessage: (msg: ChatMessage) => void;
  clearChat: () => void;

  // Research Vault Documents (16.8)
  documents: DocumentItem[];
  refreshDocuments: () => Promise<void>;

  // Global loading state
  isLoading: boolean;
}

const DEFAULT_PROFILE: InvestorProfile = {
  investment_goal: 'Capital Appreciation & Moderate Growth',
  time_horizon: '3-5 years',
  capital_amount: 50000,
  currency: 'USD',
  risk_tolerance: 'moderate',
};

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppContextProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  // 1. Investor profile with sessionStorage persistence (16.5.2)
  const [profile, setProfile] = useState<InvestorProfile>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_PROFILE_STORAGE_KEY);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch {
      // Fallback if sessionStorage is disabled or invalid
    }
    return DEFAULT_PROFILE;
  });

  const updateProfile = useCallback((updated: Partial<InvestorProfile>) => {
    setProfile((prev) => {
      const next = { ...prev, ...updated };
      try {
        sessionStorage.setItem(
          SESSION_PROFILE_STORAGE_KEY,
          JSON.stringify(next),
        );
      } catch {
        // Safe fail
      }
      return next;
    });
  }, []);

  const resetProfile = useCallback(() => {
    setProfile(DEFAULT_PROFILE);
    try {
      sessionStorage.removeItem(SESSION_PROFILE_STORAGE_KEY);
    } catch {
      // Safe fail
    }
  }, []);

  const isProfileConfigured = Boolean(
    profile.time_horizon &&
      profile.risk_tolerance &&
      profile.capital_amount &&
      profile.capital_amount > 0,
  );

  // 2. Selected Company (16.4)
  const [selectedCompany, setSelectedCompany] = useState<CompanyInfo | null>(
    MOCK_COMPANIES.AAPL,
  );

  // 3. Active Analysis & Pipeline Status (16.6)
  const [activeAnalysis, setActiveAnalysis] =
    useState<ActiveAnalysisState | null>(null);

  // 4. Reports & Recent Analyses (16.2, 16.9)
  const [activeReport, setActiveReport] = useState<FinalReport | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisSummary[]>(
    MOCK_RECENT_ANALYSES,
  );

  const refreshRecentAnalyses = useCallback(async () => {
    try {
      const data = await mockApi.getRecentAnalyses();
      setRecentAnalyses(data);
    } catch {
      // Keep existing
    }
  }, []);

  // 5. Chat History (16.3)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'initial-welcome',
      role: 'assistant',
      content:
        'Welcome to **FinPilot**. I am your Multi-Agent Financial Research Assistant. ' +
        'Ask a question about an equity, provide an investment goal, or request deep multi-agent analysis.',
      timestamp: new Date().toISOString(),
      suggestedActions: [
        'Analyze Apple (AAPL)',
        'Should I invest ₹1,00,000 in Infosys for 5 years?',
        'Review NVIDIA valuation vs semiconductors',
      ],
    },
  ]);

  const addChatMessage = useCallback((msg: ChatMessage) => {
    setChatMessages((prev) => [...prev, msg]);
  }, []);

  const clearChat = useCallback(() => {
    setChatMessages([]);
  }, []);

  // 6. Research Vault Documents (16.8)
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await mockApi.getDocuments();
      setDocuments(docs);
    } catch {
      // Keep existing
    }
  }, []);

  useEffect(() => {
    refreshRecentAnalyses();
    refreshDocuments();
  }, [refreshRecentAnalyses, refreshDocuments]);

  // Trigger new analysis helper
  const [isLoading, setIsLoading] = useState(false);

  const startNewAnalysis = useCallback(
    async (ticker: string, customProfile?: InvestorProfile): Promise<string> => {
      setIsLoading(true);
      const targetProfile = customProfile || profile;
      const comp = MOCK_COMPANIES[ticker.toUpperCase()] || {
        ticker: ticker.toUpperCase(),
        name: `${ticker.toUpperCase()} Corporation`,
        sector: 'Technology',
      };

      const { analysisId, reportId } = await mockApi.startAnalysis(
        ticker,
        targetProfile,
      );

      const initialActive: ActiveAnalysisState = {
        analysisId,
        ticker: comp.ticker,
        companyName: comp.name,
        status: 'running',
        progressPercent: 20,
        progressStage: 'CIO Agent allocating specialist domains',
        specialistStatuses: {
          technical: 'running',
          fundamental: 'running',
          news: 'pending',
          research: 'pending',
          risk: 'pending',
        },
        reportId,
      };

      setActiveAnalysis(initialActive);
      setIsLoading(false);
      return analysisId;
    },
    [profile],
  );

  return (
    <AppContext.Provider
      value={{
        profile,
        updateProfile,
        resetProfile,
        isProfileConfigured,
        selectedCompany,
        setSelectedCompany,
        activeAnalysis,
        setActiveAnalysis,
        startNewAnalysis,
        activeReport,
        setActiveReport,
        recentAnalyses,
        refreshRecentAnalyses,
        chatMessages,
        addChatMessage,
        clearChat,
        documents,
        refreshDocuments,
        isLoading,
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
