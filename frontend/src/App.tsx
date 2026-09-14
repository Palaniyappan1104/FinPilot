import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppContextProvider } from './context/AppContext';
import { AppLayout } from './layouts/AppLayout';

import { DashboardPage } from './pages/DashboardPage';
import { NewAnalysisPage } from './pages/NewAnalysisPage';
import { AnalysisProgressPage } from './pages/AnalysisProgressPage';
import { SpecialistsPage } from './pages/SpecialistsPage';
import { ReportsPage } from './pages/ReportsPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { ProfilePage } from './pages/ProfilePage';

export const App: React.FC = () => {
  return (
    <AppContextProvider>
      <BrowserRouter>
        <AppLayout>
          <Routes>
            {/* 16.2 Dashboard */}
            <Route path="/" element={<DashboardPage />} />

            {/* 16.3, 16.4, 16.5 New Analysis & Chat Query */}
            <Route path="/analysis/new" element={<NewAnalysisPage />} />

            {/* 16.6 Analysis Progress & Status */}
            <Route
              path="/analysis/:id/progress"
              element={<AnalysisProgressPage />}
            />
            <Route
              path="/analysis/progress"
              element={<AnalysisProgressPage />}
            />

            {/* 16.7 Specialist Results */}
            <Route
              path="/analysis/:id/specialists"
              element={<SpecialistsPage />}
            />
            <Route path="/specialists" element={<SpecialistsPage />} />

            {/* 16.9 Final Investment Report */}
            <Route path="/reports/:id" element={<ReportsPage />} />
            <Route path="/reports" element={<ReportsPage />} />

            {/* 16.8 Research Document Upload & Filing Q&A */}
            <Route path="/documents" element={<DocumentsPage />} />

            {/* 16.5 Investor Profile */}
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
