import React, { useState } from 'react';
import { Sidebar } from '../components/navigation/Sidebar';
import { Header } from '../components/navigation/Header';
import { FooterDisclaimer } from '../components/navigation/FooterDisclaimer';

interface AppLayoutProps {
  children: React.ReactNode;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen flex font-sans antialiased" style={{ background: '#ffffff', color: '#0f172a' }}>
      {/* Desktop Persistent Sidebar */}
      <div className="hidden md:flex md:flex-shrink-0 print:hidden">
        <Sidebar />
      </div>

      {/* Mobile Drawer Backdrop & Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex print:hidden">
          {/* Backdrop */}
          <div
            className="fixed inset-0 backdrop-blur-sm transition-opacity"
            style={{ background: 'rgba(15,23,42,0.45)' }}
            onClick={() => setMobileMenuOpen(false)}
            aria-hidden="true"
          />

          {/* Drawer Sidebar */}
          <div className="relative flex-1 flex flex-col max-w-xs w-full z-50" style={{ background: '#ffffff' }}>
            <Sidebar onClose={() => setMobileMenuOpen(false)} />
          </div>
        </div>
      )}

      {/* Main Column */}
      <div className="flex-1 flex flex-col min-w-0 print:block print:w-full" style={{ background: '#ffffff' }}>
        <div className="print:hidden">
          <Header onToggleMobileMenu={() => setMobileMenuOpen(!mobileMenuOpen)} />
        </div>

        <main className="flex-1 p-4 sm:p-6 md:p-8 max-w-7xl w-full mx-auto overflow-y-auto print:p-0 print:m-0 print:max-w-none print:w-full print:overflow-visible">
          {children}
        </main>

        <div className="print:hidden">
          <FooterDisclaimer />
        </div>
      </div>
    </div>
  );
};
