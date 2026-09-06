import React, { useEffect, useState, useCallback } from 'react';
import { fetchHealth, getApiBaseUrl, HealthResponse } from '../services/api';
import { CheckCircle2, AlertCircle, RefreshCw, Server, Globe } from 'lucide-react';

export const HomePage: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<string | null>(null);

  const checkBackendHealth = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHealth();
      setHealth(data);
      setLastChecked(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to backend');
      setHealth(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkBackendHealth();
  }, [checkBackendHealth]);

  const apiBaseUrl = getApiBaseUrl();

  return (
    <div className="space-y-6">
      {/* Welcome Banner */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h1 className="text-2xl font-bold text-slate-900">FinPilot System Dashboard</h1>
        <p className="mt-1 text-sm text-slate-600">
          A Multi-Agent AI Platform for Financial Research and Investment Decision Support.
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-xs">
          <span className="inline-flex items-center px-2.5 py-1 rounded-full font-medium bg-slate-100 text-slate-700">
            <Globe className="w-3.5 h-3.5 mr-1" />
            API Base: {apiBaseUrl}
          </span>
          <span className="inline-flex items-center px-2.5 py-1 rounded-full font-medium bg-teal-50 text-teal-700 border border-teal-200">
            Phase 1 Foundation Active
          </span>
        </div>
      </div>

      {/* Backend Health Check Card */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Server className="w-5 h-5 text-slate-600" />
            <h2 className="text-lg font-semibold text-slate-800">Backend Health Status</h2>
          </div>
          <button
            onClick={checkBackendHealth}
            disabled={loading}
            className="inline-flex items-center px-3 py-1.5 text-xs font-medium rounded-lg text-slate-700 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        <div className="p-6">
          {loading && (
            <div className="flex items-center justify-center py-8 text-slate-500 text-sm">
              <RefreshCw className="w-5 h-5 animate-spin mr-2 text-teal-600" />
              Connecting to backend service...
            </div>
          )}

          {!loading && error && (
            <div className="rounded-lg bg-red-50 border border-red-200 p-4">
              <div className="flex items-start">
                <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                <div>
                  <h3 className="text-sm font-semibold text-red-800">Connection Error</h3>
                  <p className="text-xs text-red-700 mt-1">{error}</p>
                  <p className="text-xs text-red-600 mt-2">
                    Ensure the FastAPI backend is running at <code className="bg-red-100 px-1 py-0.5 rounded">{apiBaseUrl}</code>.
                  </p>
                </div>
              </div>
            </div>
          )}

          {!loading && health && (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 rounded-lg bg-emerald-50 border border-emerald-200">
                <div className="flex items-center space-x-3">
                  <CheckCircle2 className="w-6 h-6 text-emerald-600 flex-shrink-0" />
                  <div>
                    <div className="font-semibold text-emerald-900 text-sm">Backend Operational</div>
                    <div className="text-xs text-emerald-700">All baseline systems running normally</div>
                  </div>
                </div>
                <span className="px-2.5 py-1 text-xs font-bold rounded-full bg-emerald-200 text-emerald-800 uppercase tracking-wide">
                  {health.status}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-xs font-medium text-slate-500">Service</div>
                  <div className="text-sm font-bold text-slate-800 mt-1">{health.service}</div>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-xs font-medium text-slate-500">Version</div>
                  <div className="text-sm font-bold text-slate-800 mt-1">{health.version}</div>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-xs font-medium text-slate-500">Environment</div>
                  <div className="text-sm font-bold text-slate-800 mt-1 uppercase">{health.environment}</div>
                </div>
              </div>

              {lastChecked && (
                <div className="text-right text-xs text-slate-400">
                  Last checked: {lastChecked}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Phase 1 Verification Overview */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h3 className="text-base font-semibold text-slate-800 mb-3">Phase 1 Foundation Baseline</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="flex items-center space-x-2 text-slate-700">
            <CheckCircle2 className="w-4 h-4 text-teal-600" />
            <span>FastAPI modular backend with /health endpoint</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-700">
            <CheckCircle2 className="w-4 h-4 text-teal-600" />
            <span>Pydantic centralized configuration & safe settings</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-700">
            <CheckCircle2 className="w-4 h-4 text-teal-600" />
            <span>Structured logging & request monitoring middleware</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-700">
            <CheckCircle2 className="w-4 h-4 text-teal-600" />
            <span>React + Vite + TypeScript + Tailwind CSS skeleton</span>
          </div>
        </div>
      </div>
    </div>
  );
};
