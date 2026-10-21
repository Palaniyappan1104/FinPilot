/**
 * FinPilot Login Page (new_frontend)
 * High-visibility branded login & signup form using Supabase Auth.
 */
import React, { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  TrendingUp,
  Mail,
  Lock,
  Eye,
  EyeOff,
  Loader2,
  ShieldCheck,
  BarChart2,
  BrainCircuit,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';

type Tab = 'login' | 'signup';

export const LoginPage: React.FC = () => {
  const { session, signIn, signUp, isLoading: authLoading } = useAuth();

  const [tab, setTab] = useState<Tab>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // If already authenticated, redirect to home
  if (!authLoading && session) {
    return <Navigate to="/" replace />;
  }

  const resetMessages = () => {
    setErrorMsg(null);
    setSuccessMsg(null);
  };

  const handleTabSwitch = (t: Tab) => {
    setTab(t);
    resetMessages();
    setPassword('');
  };

  const validateForm = (): boolean => {
    if (!email.trim()) {
      setErrorMsg('Please enter your email address.');
      return false;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      setErrorMsg('Please enter a valid email address.');
      return false;
    }
    if (password.length < 6) {
      setErrorMsg('Password must be at least 6 characters.');
      return false;
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    resetMessages();

    if (!validateForm()) return;

    setIsSubmitting(true);
    try {
      if (tab === 'login') {
        const { error } = await signIn(email.trim(), password);
        if (error) {
          setErrorMsg(error);
        }
      } else {
        const { error, needsConfirmation } = await signUp(email.trim(), password);
        if (error) {
          setErrorMsg(error);
        } else if (needsConfirmation) {
          setSuccessMsg(
            'Account created! Please check your email and click the confirmation link before logging in.',
          );
          setPassword('');
        } else {
          setSuccessMsg(
            'Account created and logged in successfully! Redirecting…',
          );
        }
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const features = [
    {
      icon: <BrainCircuit className="w-5 h-5 text-blue-400" />,
      title: 'Groq-Accelerated Multi-Agent AI',
      desc: 'High-throughput specialist agents for deep equity research',
    },
    {
      icon: <BarChart2 className="w-5 h-5 text-blue-400" />,
      title: 'Live Market Intelligence',
      desc: 'Real-time quotes, financials, and technical setups via yfinance',
    },
    {
      icon: <ShieldCheck className="w-5 h-5 text-blue-400" />,
      title: 'Strict Non-Advisory Guardrails',
      desc: 'Objective evidence dossiers calibrated to investor mandates',
    },
  ];

  return (
    <div
      className="min-h-screen flex"
      style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}
    >
      {/* ── Left Branding Panel ── */}
      <div
        className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12"
        style={{
          background: 'linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e40af 100%)',
        }}
      >
        <div className="flex items-center space-x-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center font-black text-lg"
            style={{ background: '#3b82f6', color: '#fff' }}
          >
            FP
          </div>
          <div>
            <div className="text-white font-bold text-xl tracking-tight">FinPilot</div>
            <div className="text-blue-300 text-xs font-medium">Groq Multi-Agent Research Platform</div>
          </div>
        </div>

        <div>
          <h2 className="text-4xl font-bold text-white leading-tight mb-4">
            Institutional-Grade<br />
            <span style={{ color: '#60a5fa' }}>AI Research Engine</span><br />
            powered by Groq
          </h2>
          <p className="text-blue-200 text-base mb-10 leading-relaxed max-w-lg">
            Deploy specialized autonomous analysts in parallel to synthesize
            fundamental, technical, sentiment, and risk dossiers in seconds.
          </p>

          <div className="space-y-5">
            {features.map((f) => (
              <div key={f.title} className="flex items-start space-x-4">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)' }}
                >
                  {f.icon}
                </div>
                <div>
                  <div className="text-white font-semibold text-sm">{f.title}</div>
                  <div className="text-blue-300 text-xs mt-0.5">{f.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="text-blue-400 text-xs">
          Decision support only · Not financial advice · FinPilot v0.2
        </div>
      </div>

      {/* ── Right Auth Form (Crisp White Background) ── */}
      <div
        className="flex-1 flex flex-col items-center justify-center p-8 bg-white"
        style={{ background: '#ffffff' }}
      >
        <div className="lg:hidden flex items-center space-x-3 mb-10">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center font-black text-base"
            style={{ background: '#1e40af', color: '#fff' }}
          >
            FP
          </div>
          <span className="text-slate-900 font-bold text-xl">FinPilot</span>
        </div>

        <div className="w-full max-w-md">
          <div className="mb-8 text-center">
            <h1 className="text-2xl font-bold text-slate-900 mb-1">
              {tab === 'login' ? 'Welcome back' : 'Create your account'}
            </h1>
            <p className="text-slate-500 text-sm">
              {tab === 'login'
                ? 'Sign in to access your FinPilot research workspace'
                : 'Start your AI-powered investment research'}
            </p>
          </div>

          {/* Tab Selector */}
          <div
            className="flex mb-7 rounded-xl p-1 bg-slate-100 border border-slate-200"
          >
            {(['login', 'signup'] as Tab[]).map((t) => (
              <button
                key={t}
                type="button"
                id={`auth-tab-${t}`}
                onClick={() => handleTabSwitch(t)}
                className="flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all duration-200"
                style={
                  tab === t
                    ? {
                        background: '#ffffff',
                        color: '#1e40af',
                        boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
                      }
                    : { color: '#64748b' }
                }
              >
                {t === 'login' ? 'Sign In' : 'Sign Up'}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            {/* Email */}
            <div>
              <label
                htmlFor="auth-email"
                className="block text-sm font-semibold text-slate-700 mb-1.5"
              >
                Email address
              </label>
              <div className="relative">
                <Mail
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400"
                />
                <input
                  id="auth-email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => { setEmail(e.target.value); resetMessages(); }}
                  placeholder="you@example.com"
                  className="w-full pl-10 pr-4 py-3 rounded-xl text-sm text-slate-900 bg-slate-50 border border-slate-200 outline-none focus:border-blue-700 focus:bg-white focus:ring-2 focus:ring-blue-100 transition-all"
                  disabled={isSubmitting}
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label
                htmlFor="auth-password"
                className="block text-sm font-semibold text-slate-700 mb-1.5"
              >
                Password
              </label>
              <div className="relative">
                <Lock
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400"
                />
                <input
                  id="auth-password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); resetMessages(); }}
                  placeholder={tab === 'signup' ? 'Min. 6 characters' : '••••••••'}
                  className="w-full pl-10 pr-11 py-3 rounded-xl text-sm text-slate-900 bg-slate-50 border border-slate-200 outline-none focus:border-blue-700 focus:bg-white focus:ring-2 focus:ring-blue-100 transition-all"
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  id="toggle-password-visibility"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 rounded text-slate-400 hover:text-slate-600"
                  tabIndex={-1}
                >
                  {showPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
              {tab === 'signup' && (
                <p className="text-xs text-slate-400 mt-1.5">
                  Must be at least 6 characters.
                </p>
              )}
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div
                className="flex items-start space-x-2.5 rounded-xl p-3.5 text-sm bg-red-50 border border-red-200 text-red-700"
              >
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Success Message */}
            {successMsg && (
              <div
                className="flex items-start space-x-2.5 rounded-xl p-3.5 text-sm bg-green-50 border border-green-200 text-green-700"
              >
                <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{successMsg}</span>
              </div>
            )}

            {/* Submit Button */}
            <button
              id="auth-submit-btn"
              type="submit"
              disabled={isSubmitting || authLoading}
              className="w-full py-3 rounded-xl text-sm font-bold text-white transition-all duration-200 flex items-center justify-center space-x-2 shadow-md hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-50 disabled:translate-y-0"
              style={{
                background: isSubmitting
                  ? '#93c5fd'
                  : 'linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%)',
              }}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>{tab === 'login' ? 'Signing in…' : 'Creating account…'}</span>
                </>
              ) : (
                <>
                  <TrendingUp className="w-4 h-4" />
                  <span>{tab === 'login' ? 'Sign In' : 'Create Account'}</span>
                </>
              )}
            </button>
          </form>

          {/* Switch Tab Link */}
          <p className="text-center text-sm text-slate-500 mt-6">
            {tab === 'login' ? "Don't have an account? " : 'Already have an account? '}
            <button
              type="button"
              id={`switch-to-${tab === 'login' ? 'signup' : 'login'}`}
              onClick={() => handleTabSwitch(tab === 'login' ? 'signup' : 'login')}
              className="font-semibold text-blue-700 hover:text-blue-800"
            >
              {tab === 'login' ? 'Sign up free' : 'Sign in'}
            </button>
          </p>

          <div className="mt-8 flex items-center justify-center space-x-1.5 text-xs text-slate-400">
            <ShieldCheck className="w-3.5 h-3.5 text-slate-400" />
            <span>For decision support only · Non-advisory</span>
          </div>
        </div>
      </div>
    </div>
  );
};
