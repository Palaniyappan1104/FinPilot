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

            {/* Unified Research & Analysis Flow */}
            <Route path="/analysis/new" element={<NewAnalysisPage />} />
            <Route path="/analysis/:id/progress" element={<NewAnalysisPage />} />
            <Route path="/analysis/progress" element={<NewAnalysisPage />} />

            {/* Dedicated Multi-Agent Analysis Section */}
            <Route path="/multi-agent" element={<MultiAgentPage />} />

            {/* Final Investment Report (Direct Link / History) */}
            <Route path="/reports/:id" element={<ReportsPage />} />
            <Route path="/reports" element={<ReportsPage />} />

            {/* Research Document Upload & Filing Q&A */}
            <Route path="/documents" element={<DocumentsPage />} />

            {/* Investor Profile */}
            <Route path="/profile" element={<ProfilePage />} />

            {/* Legacy redirect */}
            <Route path="/specialists" element={<Navigate to="/multi-agent" replace />} />
            <Route path="/analysis/:id/specialists" element={<Navigate to="/multi-agent" replace />} />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AppLayout>
      </BrowserRouter>
    </AppContextProvider>
  );
};

export default App;
