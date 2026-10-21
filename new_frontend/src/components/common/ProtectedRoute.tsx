/**
 * ProtectedRoute — wraps routes that require authentication in new_frontend.
 * Redirects unauthenticated users to /login.
 */
import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { Loader2 } from 'lucide-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { session, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="flex flex-col items-center space-y-4">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center font-black text-lg text-white"
            style={{ background: '#1e40af' }}
          >
            FP
          </div>
          <Loader2 className="w-5 h-5 animate-spin" style={{ color: '#1e40af' }} />
          <span className="text-sm text-gray-500 font-medium">Loading FinPilot…</span>
        </div>
      </div>
    );
  }

  if (!session) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};
