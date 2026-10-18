import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppContextProvider } from './context/AppContext';
import { AppLayout } from './layouts/AppLayout';

import { DashboardPage } from './pages/DashboardPage';
import { NewAnalysisPage } from './pages/NewAnalysisPage';
import { MultiAgentPage } from './pages/MultiAgentPage';
import { ReportsPage } from './pages/ReportsPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { ProfilePage } from './pages/ProfilePage';

export const App: React.FC = () => {
  return (
    <AppContextProvider>
      <BrowserRouter>
        <AppLayout>
          <Routes>
            {/* Dashboard */}
            <Route path="/" element={<DashboardPage />} />

            {/* Streamlit-driven New Analysis Flow */}
            <Route path="/analysis/new" element={<NewAnalysisPage />} />
            <Route path="/analysis" element={<Navigate to="/analysis/new" replace />} />

            {/* 3-Agent Templated Specialist Pipeline */}
            <Route path="/multi-agent" element={<MultiAgentPage />} />

            {/* Generated Reports & Full Dossier Archive */}
            <Route path="/reports/:id" element={<ReportsPage />} />
            <Route path="/reports" element={<ReportsPage />} />

            {/* Research Vault (Filings & Storage Standby) */}
            <Route path="/documents" element={<DocumentsPage />} />

            {/* Investor Profile */}
            <Route path="/profile" element={<ProfilePage />} />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AppLayout>
      </BrowserRouter>
    </AppContextProvider>
  );
};

export default App;
