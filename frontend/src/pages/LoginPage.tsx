/**
 * FinPilot Login Page
 * Professional navy/white branded login & signup form using Supabase Auth.
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

  // Already authenticated → go to dashboard
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
        // On success, AuthContext updates session → Navigate kicks in
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
      title: 'Multi-Agent AI Research',
      desc: 'Parallel specialist agents for deep investment analysis',
    },
    {
      icon: <BarChart2 className="w-5 h-5 text-blue-400" />,
      title: 'Real-Time Market Data',
      desc: 'Live price feeds and technical indicator overlays',
    },
    {
      icon: <ShieldCheck className="w-5 h-5 text-blue-400" />,
      title: 'Regulatory Guardrails',
      desc: 'Built-in compliance filters and risk constraints',
    },
  ];

  return (
    <div
      className="min-h-screen flex"
      style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}
    >
      {/* ── Left Panel: Branding ── */}
      <div
        className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12"
        style={{
          background: 'linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e40af 100%)',
        }}
      >
        {/* Logo */}
        <div className="flex items-center space-x-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center font-black text-lg"
            style={{ background: '#3b82f6', color: '#fff' }}
          >
            FP
          </div>
          <div>
            <div className="text-white font-bold text-xl tracking-tight">FinPilot</div>
            <div className="text-blue-300 text-xs font-medium">AI Financial Research Platform</div>
          </div>
        </div>

        {/* Hero Copy */}
        <div>
          <h2 className="text-4xl font-bold text-white leading-tight mb-4">
            Institutional-Grade<br />
            <span style={{ color: '#60a5fa' }}>AI Research</span><br />
            at your fingertips
          </h2>
          <p className="text-blue-200 text-base mb-10 leading-relaxed">
            FinPilot deploys a team of specialist AI agents to deliver deep
            investment analysis — from fundamental valuation to risk modelling.
          </p>

          {/* Feature List */}
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

        {/* Footer disclaimer */}
        <div className="text-blue-400 text-xs">
          Decision support only · Not financial advice · FinPilot v0.1
        </div>
      </div>

      {/* ── Right Panel: Auth Form ── */}
      <div
        className="flex-1 flex flex-col items-center justify-center p-8"
        style={{ background: '#ffffff' }}
      >
        {/* Mobile Logo */}
        <div className="lg:hidden flex items-center space-x-3 mb-10">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center font-black text-base"
            style={{ background: '#1e40af', color: '#fff' }}
          >
            FP
          </div>
          <span className="text-gray-900 font-bold text-xl">FinPilot</span>
        </div>

        <div className="w-full max-w-md">
          {/* Heading */}
          <div className="mb-8 text-center">
            <h1 className="text-2xl font-bold text-gray-900 mb-1">
              {tab === 'login' ? 'Welcome back' : 'Create your account'}
            </h1>
            <p className="text-gray-500 text-sm">
              {tab === 'login'
                ? 'Sign in to continue to FinPilot'
                : 'Start your AI-powered investment research'}
            </p>
          </div>

          {/* Tabs */}
          <div
            className="flex mb-7 rounded-xl p-1"
            style={{ background: '#f1f5f9' }}
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
                        boxShadow: '0 1px 4px rgba(0,0,0,0.12)',
                      }
                    : { color: '#64748b' }
                }
              >
                {t === 'login' ? 'Sign In' : 'Sign Up'}
              </button>
            ))}
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            {/* Email */}
            <div>
              <label
                htmlFor="auth-email"
                className="block text-sm font-semibold text-gray-700 mb-1.5"
              >
                Email address
              </label>
              <div className="relative">
                <Mail
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
                  style={{ color: '#94a3b8' }}
                />
                <input
                  id="auth-email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => { setEmail(e.target.value); resetMessages(); }}
                  placeholder="you@example.com"
                  className="w-full pl-10 pr-4 py-3 rounded-xl text-sm text-gray-900 outline-none transition-all"
                  style={{
                    border: '1.5px solid #e2e8f0',
                    background: '#f8fafc',
                  }}
                  onFocus={(e) => {
                    e.currentTarget.style.borderColor = '#1e40af';
                    e.currentTarget.style.background = '#fff';
                    e.currentTarget.style.boxShadow = '0 0 0 3px rgba(30,64,175,0.1)';
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = '#e2e8f0';
                    e.currentTarget.style.background = '#f8fafc';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                  disabled={isSubmitting}
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label
                htmlFor="auth-password"
                className="block text-sm font-semibold text-gray-700 mb-1.5"
              >
                Password
              </label>
              <div className="relative">
                <Lock
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
                  style={{ color: '#94a3b8' }}
                />
                <input
                  id="auth-password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); resetMessages(); }}
                  placeholder={tab === 'signup' ? 'Min. 6 characters' : '••••••••'}
                  className="w-full pl-10 pr-11 py-3 rounded-xl text-sm text-gray-900 outline-none transition-all"
                  style={{
                    border: '1.5px solid #e2e8f0',
                    background: '#f8fafc',
                  }}
                  onFocus={(e) => {
                    e.currentTarget.style.borderColor = '#1e40af';
                    e.currentTarget.style.background = '#fff';
                    e.currentTarget.style.boxShadow = '0 0 0 3px rgba(30,64,175,0.1)';
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = '#e2e8f0';
                    e.currentTarget.style.background = '#f8fafc';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  id="toggle-password-visibility"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 rounded"
                  style={{ color: '#94a3b8' }}
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
                <p className="text-xs text-gray-400 mt-1.5">
                  Use at least 6 characters.
                </p>
              )}
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div
                className="flex items-start space-x-2.5 rounded-xl p-3.5 text-sm"
                style={{
                  background: '#fef2f2',
                  border: '1px solid #fecaca',
                  color: '#b91c1c',
                }}
              >
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Success Message */}
            {successMsg && (
              <div
                className="flex items-start space-x-2.5 rounded-xl p-3.5 text-sm"
                style={{
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  color: '#166534',
                }}
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
              className="w-full py-3 rounded-xl text-sm font-bold text-white transition-all duration-200 flex items-center justify-center space-x-2"
              style={{
                background: isSubmitting
                  ? '#93c5fd'
                  : 'linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%)',
                boxShadow: isSubmitting
                  ? 'none'
                  : '0 4px 14px rgba(30,64,175,0.35)',
              }}
              onMouseEnter={(e) => {
                if (!isSubmitting) {
                  e.currentTarget.style.background =
                    'linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%)';
                  e.currentTarget.style.transform = 'translateY(-1px)';
                }
              }}
              onMouseLeave={(e) => {
                if (!isSubmitting) {
                  e.currentTarget.style.background =
                    'linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }
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
          <p className="text-center text-sm text-gray-500 mt-6">
            {tab === 'login' ? "Don't have an account? " : 'Already have an account? '}
            <button
              type="button"
              id={`switch-to-${tab === 'login' ? 'signup' : 'login'}`}
              onClick={() => handleTabSwitch(tab === 'login' ? 'signup' : 'login')}
              className="font-semibold"
              style={{ color: '#1e40af' }}
            >
              {tab === 'login' ? 'Sign up free' : 'Sign in'}
            </button>
          </p>

          {/* Disclaimer */}
          <div className="mt-8 flex items-center justify-center space-x-1.5 text-xs text-gray-400">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>For decision support only · Not financial advice</span>
          </div>
        </div>
      </div>
    </div>
  );
};
