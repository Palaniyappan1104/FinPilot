/**
 * Domain TypeScript interfaces and types for FinPilot Phase 16.
 * Formulated to strictly align with Phase 12 and Phase 15 backend contracts.
 */

export type RecommendationStance =
  | 'favorable'
  | 'cautious'
  | 'neutral'
  | 'unfavorable'
  | 'insufficient_evidence';

export type SpecialistType =
  | 'technical'
  | 'fundamental'
  | 'news'
  | 'research'
  | 'risk';

export type SpecialistStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'insufficient_evidence'
  | 'omitted';

export type AnalysisStatus =
  | 'running'
  | 'completed'
  | 'clarification_needed'
  | 'failed';

export interface InvestorProfile {
  target_company?: string;
  ticker?: string;
  investment_goal?: string;
  time_horizon?: string;
  capital_amount?: number;
  currency?: string;
  risk_tolerance?: string;
}

export interface CompanyInfo {
  ticker: string;
  name: string;
  sector: string;
  industry: string;
  exchange: string;
  currency: string;
  price: number;
  change: number;
  changePercent: number;
  marketCap: string;
  peRatio: number;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
  description: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  clarificationQuestions?: string[];
  analysisId?: string;
  reportId?: string;
  suggestedActions?: string[];
}

export interface TechnicalReportSection {
  trend: 'bullish' | 'bearish' | 'neutral';
  rsi_14: number;
  rsi_status: 'overbought' | 'neutral' | 'oversold';
  sma_20: number;
  sma_50: number;
  sma_200: number;
  macd_signal: 'bullish' | 'bearish' | 'neutral';
  volatility_30d_pct: number;
  support_level: number;
  resistance_level: number;
  summary: string;
  key_findings: string[];
}

export interface FundamentalReportSection {
  pe_ratio: number;
  pb_ratio: number;
  ev_ebitda: number;
  gross_margin_pct: number;
  operating_margin_pct: number;
  net_margin_pct: number;
  roe_pct: number;
  debt_to_equity: number;
  current_ratio: number;
  free_cash_flow: string;
  revenue_growth_yoy_pct: number;
  summary: string;
  key_findings: string[];
}

export interface NewsArticleItem {
  id: string;
  title: string;
  source: string;
  published_at: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  relevance_score: number;
  url?: string;
}

export interface NewsReportSection {
  overall_sentiment: 'positive' | 'neutral' | 'negative';
  sentiment_score: number; // -1.0 to 1.0
  positive_pct: number;
  neutral_pct: number;
  negative_pct: number;
  summary: string;
  top_headlines: NewsArticleItem[];
  key_themes: string[];
}

export interface ResearchCitation {
  source_document: string;
  page?: number;
  excerpt: string;
  confidence: number;
}

export interface ResearchReportSection {
  documents_analyzed: number;
  summary: string;
  key_filing_insights: string[];
  citations: ResearchCitation[];
}

export interface RiskReportSection {
  composite_risk_score: number; // 0 to 100
  overall_risk_level: 'Low' | 'Moderate' | 'High' | 'Severe';
  var_95_pct: number;
  max_drawdown_pct: number;
  market_risk: string;
  valuation_risk: string;
  operational_risk: string;
  regulatory_risk: string;
  macro_risk: string;
  summary: string;
  mitigating_factors: string[];
}

export interface OverallAssessmentSection {
  summary: string;
  specialist_consensus: string;
  signal_conflicts: string[];
  completeness_ratio: number;
}

export interface ReportRecommendation {
  stance: RecommendationStance;
  rationale: string;
  time_horizon_suitability?: string;
  risk_tolerance_suitability?: string;
  capital_allocation_notes?: string;
  monitoring_points: string[];
}

export interface AggregatedEvidenceItem {
  id: string;
  claim: string;
  specialist_type: SpecialistType;
  source_tool: string;
  confidence: number;
  timestamp: string;
}

export interface FinalReport {
  report_id: string;
  company: {
    ticker: string;
    name: string;
    sector: string;
    currency: string;
  };
  investor_profile?: InvestorProfile;
  horizon?: string;
  capital?: {
    amount: number;
    currency: string;
  };
  created_at: string;
  confidence: number;
  overall_assessment: OverallAssessmentSection;
  recommendation?: ReportRecommendation;
  key_reasons: string[];
  important_risks: string[];
  evidence_sources: AggregatedEvidenceItem[];
  technical?: TechnicalReportSection;
  fundamental?: FundamentalReportSection;
  news?: NewsReportSection;
  research?: ResearchReportSection;
  risk?: RiskReportSection;
  disclaimer: string;
  specialist_statuses?: Record<SpecialistType, SpecialistStatus>;
}

export interface AnalysisSummary {
  analysis_id: string;
  report_id?: string;
  ticker: string;
  company_name: string;
  created_at: string;
  status: AnalysisStatus;
  recommendation_stance?: RecommendationStance;
  confidence: number;
  horizon?: string;
  risk_tolerance?: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  ticker: string;
  doc_type: '10-K' | '10-Q' | '8-K' | 'Earnings Transcript' | 'Investor Presentation';
  size_bytes: number;
  upload_date: string;
  status: 'indexed' | 'processing' | 'failed';
  chunk_count: number;
}

export interface ResearchQAResult {
  question: string;
  answer: string;
  ticker?: string;
  citations: ResearchCitation[];
  confidence: number;
}
