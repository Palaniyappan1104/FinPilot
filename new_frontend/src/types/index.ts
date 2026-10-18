/**
 * Core type definitions for FinPilot Frontend.
 * Directly mirrors backend models from demo.py and supports rich reporting.
 */

export interface AgentStatus {
  name: string;
  status: 'completed' | 'partial' | 'failed' | 'running' | 'queued';
  summary: string;
}

export interface ClarifyRequest {
  query: string;
}

export interface ClarifyResponse {
  questions: string[];
  detected_ticker?: string | null;
  detected_company?: string | null;
}

export interface AnalyzeRequest {
  query: string;
  clarification_answers?: Record<string, string> | null;
  ticker?: string | null;
  target_company?: string | null;
}

export interface AnalyzeResponse {
  ticker: string;
  company: string;
  executive_summary: string;
  technical_outlook: string;
  fundamental_view: string;
  news_sentiment: string;
  risk_assessment: string;
  final_verdict: string;
  confidence: 'High' | 'Medium' | 'Low' | string;
  disclaimer: string;
  agent_statuses: AgentStatus[];
  raw_markdown: string;
  investor_profile?: string;
  valuation_view?: string;
  scenario_analysis?: string;
  catalysts_to_watch?: string;
  data_availability?: string;
  key_metrics?: Record<string, string>;
  sources?: string[];
  research_briefs?: Record<string, string>;
}

export interface StoredReport {
  id: string;
  timestamp: string;
  ticker: string;
  company: string;
  query: string;
  confidence: string;
  executive_summary: string;
  verdict: string;
  data: AnalyzeResponse;
}

export interface InvestorProfile {
  investment_goal: string;
  time_horizon: string;
  capital_amount: number;
  risk_tolerance: 'conservative' | 'moderate' | 'aggressive' | string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  questions?: string[];
  detectedTicker?: string | null;
  detectedCompany?: string | null;
}
