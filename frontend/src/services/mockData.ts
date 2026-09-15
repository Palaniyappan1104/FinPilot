/**
 * Deterministic mock datasets for FinPilot Phase 16 React Frontend.
 * Clearly marked as mock/demo data.
 * Adheres strictly to Phase 12 and Phase 15 backend contract schemas.
 */

import {
  AnalysisSummary,
  CompanyInfo,
  DocumentItem,
  FinalReport,
  ResearchQAResult,
} from '../types';

export const STANDARD_REGULATORY_DISCLAIMER =
  'FinPilot provides automated financial research and decision support for informational purposes only. ' +
  'FinPilot is not a registered investment advisor, broker-dealer, or financial planner. This report does not ' +
  'constitute personalized investment advice, an endorsement, or a recommendation to buy, sell, or hold any ' +
  'security. Past performance does not guarantee future results. All investments carry risk of loss. ' +
  'Investors must conduct independent research and consult licensed financial professionals before making investment decisions.';

export const MOCK_COMPANIES: Record<string, CompanyInfo> = {
  AAPL: {
    ticker: 'AAPL',
    name: 'Apple Inc.',
    sector: 'Technology',
    industry: 'Consumer Electronics',
    exchange: 'NASDAQ',
    currency: 'USD',
    price: 228.45,
    change: 1.82,
    changePercent: 0.8,
    marketCap: '$3.48T',
    peRatio: 34.2,
    fiftyTwoWeekHigh: 237.23,
    fiftyTwoWeekLow: 164.08,
    description:
      'Designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories, alongside a high-margin services ecosystem.',
  },
  MSFT: {
    ticker: 'MSFT',
    name: 'Microsoft Corporation',
    sector: 'Technology',
    industry: 'Software - Infrastructure',
    exchange: 'NASDAQ',
    currency: 'USD',
    price: 432.1,
    change: -2.15,
    changePercent: -0.5,
    marketCap: '$3.21T',
    peRatio: 36.8,
    fiftyTwoWeekHigh: 468.35,
    fiftyTwoWeekLow: 309.45,
    description:
      'Develops and supports software, services, devices, and solutions, led by Azure cloud infrastructure, AI copilots, and enterprise productivity software.',
  },
  NVDA: {
    ticker: 'NVDA',
    name: 'NVIDIA Corporation',
    sector: 'Technology',
    industry: 'Semiconductors',
    exchange: 'NASDAQ',
    currency: 'USD',
    price: 118.9,
    change: 4.35,
    changePercent: 3.8,
    marketCap: '$2.92T',
    peRatio: 58.4,
    fiftyTwoWeekHigh: 140.76,
    fiftyTwoWeekLow: 39.23,
    description:
      'Pioneers accelerated computing through GPU hardware, CUDA programming platform, and full-stack AI data center infrastructure.',
  },
  INFY: {
    ticker: 'INFY',
    name: 'Infosys Limited',
    sector: 'Technology',
    industry: 'Information Technology Services',
    exchange: 'NSE / NYSE',
    currency: 'INR',
    price: 1942.5,
    change: 15.2,
    changePercent: 0.79,
    marketCap: '₹8.05T ($96.5B)',
    peRatio: 28.6,
    fiftyTwoWeekHigh: 1991.45,
    fiftyTwoWeekLow: 1358.35,
    description:
      'Global leader in next-generation digital services and consulting, providing AI, cloud transformation, enterprise software development, and systems integration.',
  },
  GOOGL: {
    ticker: 'GOOGL',
    name: 'Alphabet Inc.',
    sector: 'Communication Services',
    industry: 'Internet Content & Information',
    exchange: 'NASDAQ',
    currency: 'USD',
    price: 158.75,
    change: 0.95,
    changePercent: 0.6,
    marketCap: '$1.97T',
    peRatio: 23.5,
    fiftyTwoWeekHigh: 191.75,
    fiftyTwoWeekLow: 120.21,
    description:
      'Holding company that encompasses Google Services (Search, YouTube, Android), Google Cloud enterprise infrastructure, and Other Bets.',
  },
};

export const MOCK_REPORTS: Record<string, FinalReport> = {
  'rep-aapl-001': {
    report_id: 'rep-aapl-001',
    company: {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      sector: 'Technology',
      currency: 'USD',
    },
    investor_profile: {
      target_company: 'Apple Inc.',
      ticker: 'AAPL',
      investment_goal: 'Capital Appreciation & Moderate Dividend',
      time_horizon: '3-5 years',
      capital_amount: 50000,
      risk_tolerance: 'moderate',
    },
    horizon: '3-5 years',
    capital: {
      amount: 50000,
      currency: 'USD',
    },
    created_at: '2026-09-12T14:30:00Z',
    confidence: 0.88,
    overall_assessment: {
      summary:
        'Apple exhibits robust cash flow dynamics and enduring ecosystem retention, though current valuation multiples trade at a premium relative to 10-year historical medians.',
      specialist_consensus:
        'Fundamental and News analysts maintain constructive views supported by high-margin Services expansion, while Technical analyst indicates short-term momentum consolidation.',
      signal_conflicts: [
        'Fundamental growth outlook remains defensive and steady, whereas Technical RSI shows near-term range-bound resistance at $235.',
      ],
      completeness_ratio: 0.95,
    },
    recommendation: {
      stance: 'favorable',
      rationale:
        'Strong alignment with a 3-5 year moderate-risk investment profile. While multiple expansion is limited in the near term, structural growth in Services and disciplined share buybacks provide capital resilience.',
      time_horizon_suitability:
        'Favorable for multi-year horizons allowing ecosystem monetization and hardware replacement cycles to compound.',
      risk_tolerance_suitability:
        'Consistent with moderate risk parameters given conservative leverage ratios and dominant free cash flow generation.',
      capital_allocation_notes:
        'Dollar-cost averaging into position over 3-6 months is suggested to mitigate entry at technical resistance.',
      monitoring_points: [
        'Services gross margin trajectory above 70%',
        'Greater China consumer demand and device upgrade velocity',
        'Regulatory developments regarding App Store monetization in the EU and US',
      ],
    },
    key_reasons: [
      'Services revenue now represents >25% of total revenue with industry-leading >73% gross margin.',
      'Massive installed active base of over 2.2 billion devices creating high switching barriers.',
      'Annual free cash flow exceeding $100B backing aggressive share repurchases and stable dividend growth.',
      'Strong institutional sponsorship and pristine balance sheet with net-neutral cash target.',
    ],
    important_risks: [
      'Regulatory scrutiny targeting App Store fees and search revenue sharing agreements.',
      'Hardware upgrade cycle elongation if consumer AI features show slow adoption.',
      'Geopolitical dependency on cross-border supply chains and assembly operations in East Asia.',
    ],
    evidence_sources: [
      {
        id: 'ev-aapl-01',
        claim: 'Services gross margin expanded to 74.1% in the latest quarter.',
        specialist_type: 'fundamental',
        source_tool: 'SEC Edgar 10-Q Filing',
        confidence: 0.96,
        timestamp: '2026-09-12T14:28:10Z',
      },
      {
        id: 'ev-aapl-02',
        claim: 'Active device installed base reached all-time high across all product categories.',
        specialist_type: 'research',
        source_tool: 'ChromaDB Filing Vault',
        confidence: 0.94,
        timestamp: '2026-09-12T14:28:15Z',
      },
      {
        id: 'ev-aapl-03',
        claim: 'RSI at 56.4 confirms neutral momentum with established support at $218.50.',
        specialist_type: 'technical',
        source_tool: 'Technical Indicators (yfinance)',
        confidence: 0.9,
        timestamp: '2026-09-12T14:28:20Z',
      },
      {
        id: 'ev-aapl-04',
        claim: 'News sentiment stands at 72% positive driven by enterprise AI partnerships.',
        specialist_type: 'news',
        source_tool: 'Finnhub News Sentiment Engine',
        confidence: 0.85,
        timestamp: '2026-09-12T14:28:22Z',
      },
      {
        id: 'ev-aapl-05',
        claim: 'Value at Risk (95% 1-month) calculated at 6.2%, within moderate risk bounds.',
        specialist_type: 'risk',
        source_tool: 'Risk Analytics Model',
        confidence: 0.89,
        timestamp: '2026-09-12T14:28:25Z',
      },
    ],
    technical: {
      trend: 'bullish',
      rsi_14: 56.4,
      rsi_status: 'neutral',
      sma_20: 226.1,
      sma_50: 221.8,
      sma_200: 198.4,
      macd_signal: 'bullish',
      volatility_30d_pct: 18.2,
      support_level: 218.5,
      resistance_level: 236.0,
      summary:
        'AAPL maintains a healthy intermediate uptrend above the 50-day and 200-day moving averages, currently consolidating below all-time highs with balanced momentum indicators.',
      key_findings: [
        'Price trading +15.1% above 200-day SMA, validating positive secular trend.',
        'MACD line remains above the signal line with positive histogram slope.',
        'Primary support established at $218.50, overhead resistance tested at $236.00.',
      ],
    },
    fundamental: {
      pe_ratio: 34.2,
      pb_ratio: 48.5,
      ev_ebitda: 25.1,
      gross_margin_pct: 46.2,
      operating_margin_pct: 31.4,
      net_margin_pct: 26.3,
      roe_pct: 154.2,
      debt_to_equity: 1.45,
      current_ratio: 1.05,
      free_cash_flow: '$104.8B',
      revenue_growth_yoy_pct: 5.8,
      summary:
        'Superior profitability profile supported by expanding Services contribution. Capital structure remains optimized through consistent share retirements.',
      key_findings: [
        'Net margin of 26.3% ranks in the 95th percentile of global mega-cap hardware/software.',
        'Free cash flow conversion exceeds 100% of net income.',
        'Valuation at 34.2x P/E reflects a growth premium compared to S&P 500 average.',
      ],
    },
    news: {
      overall_sentiment: 'positive',
      sentiment_score: 0.44,
      positive_pct: 68,
      neutral_pct: 22,
      negative_pct: 10,
      summary:
        'Media and financial press coverage focuses on edge AI software capabilities, developer ecosystem adoption, and services subscription expansion.',
      top_headlines: [
        {
          id: 'news-1',
          title: 'Apple Expands Private Cloud Compute Architecture for Enterprise Workflow',
          source: 'Bloomberg Technology',
          published_at: '2026-09-11T10:15:00Z',
          sentiment: 'positive',
          relevance_score: 0.92,
        },
        {
          id: 'news-2',
          title: 'Morgan Stanley Reaffirms Overweight Rating on Multi-Year Device Replacement Cycle',
          source: 'Reuters Financial',
          published_at: '2026-09-10T16:40:00Z',
          sentiment: 'positive',
          relevance_score: 0.88,
        },
        {
          id: 'news-3',
          title: 'European Regulators Review Third-Party Marketplace Fee Compliance',
          source: 'Wall Street Journal',
          published_at: '2026-09-09T08:20:00Z',
          sentiment: 'negative',
          relevance_score: 0.81,
        },
      ],
      key_themes: [
        'On-device machine learning adoption',
        'App Store anti-steering regulatory scrutiny in the EU',
        'Services ARR expansion and customer lifetime value',
      ],
    },
    research: {
      documents_analyzed: 4,
      summary:
        'Cross-filing research synthesis confirms durable capital return discipline and incremental hardware cycle upgrades.',
      key_filing_insights: [
        'Management noted in Form 10-Q that Services gross margin increased +320 bps year-over-year.',
        'R&D expenditure increased to 8.2% of revenue, prioritized towards custom silicon and on-device model architectures.',
        'Cash and marketable securities totaled $153B as of the latest SEC filing.',
      ],
      citations: [
        {
          source_document: 'AAPL_10Q_Q3_2026.pdf',
          page: 18,
          excerpt:
            'Services gross margin was 74.1% compared to 70.9% in the prior year period, driven by favorable portfolio mix.',
          confidence: 0.97,
        },
        {
          source_document: 'AAPL_Earnings_Call_Transcript_Q3.pdf',
          page: 7,
          excerpt:
            'We returned over $28 billion to shareholders during the quarter, including $3.9 billion in dividends and $24 billion in open-market share repurchases.',
          confidence: 0.95,
        },
      ],
    },
    risk: {
      composite_risk_score: 38,
      overall_risk_level: 'Low',
      var_95_pct: 6.2,
      max_drawdown_pct: 15.4,
      market_risk:
        'Beta of 1.05 against S&P 500; moderate sensitivity to broader tech equity drawdowns.',
      valuation_risk:
        'P/E multiple of 34.2x leaves modest margin of safety if revenue acceleration moderates.',
      operational_risk:
        'Concentration of advanced semiconductor manufacturing with single-source foundries.',
      regulatory_risk:
        'Ongoing DOJ antitrust lawsuit and European Digital Markets Act compliance audits.',
      macro_risk:
        'Discretionary consumer spending sensitivity in non-core international markets.',
      summary:
        'Risk profile is predominantly characterized by valuation multiple sensitivity and regulatory inquiries rather than balance sheet fragility or solvency risk.',
      mitigating_factors: [
        'Unmatched liquidity profile with >$100B annual operating cash generation.',
        'High consumer lock-in with average active device tenure exceeding 4 years.',
        'Diversification into recurrent high-margin digital subscriptions.',
      ],
    },
    disclaimer: STANDARD_REGULATORY_DISCLAIMER,
    specialist_statuses: {
      technical: 'completed',
      fundamental: 'completed',
      news: 'completed',
      research: 'completed',
      risk: 'completed',
    },
  },
  'rep-infy-002': {
    report_id: 'rep-infy-002',
    company: {
      ticker: 'INFY',
      name: 'Infosys Limited',
      sector: 'Technology',
      currency: 'INR',
    },
    investor_profile: {
      target_company: 'Infosys Limited',
      ticker: 'INFY',
      investment_goal: 'Long-term Wealth Creation & Dividend',
      time_horizon: '5 years',
      capital_amount: 100000,
      risk_tolerance: 'moderate',
    },
    horizon: '5 years',
    capital: {
      amount: 100000,
      currency: 'INR',
    },
    created_at: '2026-09-13T10:00:00Z',
    confidence: 0.86,
    overall_assessment: {
      summary:
        'Infosys demonstrates steady execution in enterprise AI modernization and cloud consulting, with attractive dividend yield and cash conversion.',
      specialist_consensus:
        'Specialists agree that large enterprise digital transformation contracts provide revenue predictability over a 5-year period.',
      signal_conflicts: [
        'Discretionary IT consulting demand in the US exhibits temporary lumpiness vs strong European BFSI vertical signings.',
      ],
      completeness_ratio: 0.92,
    },
    recommendation: {
      stance: 'favorable',
      rationale:
        'Appropriate for a 5-year holding horizon with ₹1,00,000 capital. Healthy return on equity (>30%) and consistent free cash flow yield support steady capital appreciation alongside regular dividends.',
      time_horizon_suitability:
        'Excellent suitability for a 5-year investment horizon spanning global enterprise IT refresh cycles.',
      risk_tolerance_suitability:
        'Matches moderate risk criteria due to zero net debt and disciplined operating margin management.',
      capital_allocation_notes:
        'Full deployment of ₹1,00,000 across 2-3 staggered tranches is prudent given current near-peak valuation levels.',
      monitoring_points: [
        'Quarterly Large Deal TCV (Total Contract Value) above $3 billion',
        'Operating margin stabilization within the 20-22% guidance band',
        'Voluntary attrition rate trends among specialized cloud & AI engineers',
      ],
    },
    key_reasons: [
      'Strong pipeline of generative AI enterprise implementations under Topaz suite.',
      'Industry-leading cash generation with >80% free cash flow returned via dividends and buybacks.',
      'Robust corporate governance and debt-free balance sheet.',
      'High client retention in core Financial Services, Retail, and Manufacturing segments.',
    ],
    important_risks: [
      'Slower decision-making cycles on non-critical discretionary enterprise software projects.',
      'Foreign exchange volatility (USD/INR and EUR/INR cross-currency movements).',
      'Wage inflation and competition for senior machine learning architects.',
    ],
    evidence_sources: [
      {
        id: 'ev-infy-01',
        claim: 'Large deal TCV for the latest quarter reached $3.4 billion.',
        specialist_type: 'fundamental',
        source_tool: 'BSE/NSE Regulatory Filings',
        confidence: 0.95,
        timestamp: '2026-09-13T09:50:00Z',
      },
      {
        id: 'ev-infy-02',
        claim: 'Operating margin maintained at 21.1% despite wage adjustments.',
        specialist_type: 'fundamental',
        source_tool: 'Audited Financial Statements',
        confidence: 0.93,
        timestamp: '2026-09-13T09:51:00Z',
      },
      {
        id: 'ev-infy-03',
        claim: 'Technical setup displays bullish golden cross (50-day crossing above 200-day SMA).',
        specialist_type: 'technical',
        source_tool: 'NSE Market Data Engine',
        confidence: 0.88,
        timestamp: '2026-09-13T09:52:00Z',
      },
    ],
    technical: {
      trend: 'bullish',
      rsi_14: 61.2,
      rsi_status: 'neutral',
      sma_20: 1910.0,
      sma_50: 1845.5,
      sma_200: 1620.0,
      macd_signal: 'bullish',
      volatility_30d_pct: 16.4,
      support_level: 1820.0,
      resistance_level: 2000.0,
      summary:
        'Sustained upward trajectory supported by expanding volume on earnings breakouts. Consolidation right under psychological resistance of ₹2,000.',
      key_findings: [
        '50-day moving average firmly above 200-day SMA.',
        'RSI at 61.2 exhibits constructive momentum without extreme overbought conditions.',
        'Strong buyer support at ₹1,820 pivot zone.',
      ],
    },
    fundamental: {
      pe_ratio: 28.6,
      pb_ratio: 8.9,
      ev_ebitda: 19.4,
      gross_margin_pct: 34.8,
      operating_margin_pct: 21.1,
      net_margin_pct: 17.5,
      roe_pct: 31.8,
      debt_to_equity: 0.08,
      current_ratio: 1.85,
      free_cash_flow: '₹22,400 Cr',
      revenue_growth_yoy_pct: 6.5,
      summary:
        'Clean capital structure with zero net debt, premium ROE, and disciplined operational margins.',
      key_findings: [
        'Return on Equity of 31.8% demonstrates superior capital efficiency.',
        'Consistent dividend payout ratio exceeding 75% of net profit.',
        'Net cash balance provides strategic flexibility for bolt-on AI acquisitions.',
      ],
    },
    news: {
      overall_sentiment: 'positive',
      sentiment_score: 0.38,
      positive_pct: 62,
      neutral_pct: 30,
      negative_pct: 8,
      summary:
        'Sentiment bolstered by multimillion-dollar multi-year AI contract renewals with global European automotive and banking leaders.',
      top_headlines: [
        {
          id: 'news-infy-1',
          title: 'Infosys Expands Strategic AI Collaboration with European Financial Group',
          source: 'Economic Times',
          published_at: '2026-09-12T07:30:00Z',
          sentiment: 'positive',
          relevance_score: 0.94,
        },
        {
          id: 'news-infy-2',
          title: 'Indian IT Firms See Stable Deal Wins Amid Cautious Tech Budgets',
          source: 'LiveMint',
          published_at: '2026-09-11T12:00:00Z',
          sentiment: 'neutral',
          relevance_score: 0.82,
        },
      ],
      key_themes: [
        'Enterprise generative AI delivery contracts',
        'BFSI vertical recovery and cloud migrations',
      ],
    },
    research: {
      documents_analyzed: 2,
      summary:
        'Management commentary emphasizes accelerating adoption of enterprise AI platforms across global client engagements.',
      key_filing_insights: [
        'Topaz platform integrated into over 300 client transformation engagements.',
        'Free cash flow conversion to net profit reached 102% during the fiscal year.',
      ],
      citations: [
        {
          source_document: 'Infosys_Annual_Report_2026.pdf',
          page: 42,
          excerpt:
            'Free cash flow for the year stood at ₹22,400 crore, reflecting our focus on working capital discipline.',
          confidence: 0.96,
        },
      ],
    },
    risk: {
      composite_risk_score: 34,
      overall_risk_level: 'Low',
      var_95_pct: 5.8,
      max_drawdown_pct: 12.1,
      market_risk: 'Low correlation with high-beta US equities; beta of 0.72.',
      valuation_risk: 'Trading near the higher end of 5-year historic valuation band (28x P/E).',
      operational_risk: 'Retention of specialized technical talent in tier-1 delivery centers.',
      regulatory_risk: 'Visa compliance and local hiring mandates in overseas target markets.',
      macro_risk: 'Slowdown in North American corporate IT spending.',
      summary:
        'Low risk profile supported by net cash reserves, stable blue-chip client relationships, and recurring services revenue.',
      mitigating_factors: [
        'Zero financial debt on balance sheet.',
        'Geographic diversification across US, Europe, and Asia-Pacific.',
        'Long-term client relationships with high renewal rates.',
      ],
    },
    disclaimer: STANDARD_REGULATORY_DISCLAIMER,
    specialist_statuses: {
      technical: 'completed',
      fundamental: 'completed',
      news: 'completed',
      research: 'completed',
      risk: 'completed',
    },
  },
};

export const MOCK_RECENT_ANALYSES: AnalysisSummary[] = [
  {
    analysis_id: 'an-001',
    report_id: 'rep-aapl-001',
    ticker: 'AAPL',
    company_name: 'Apple Inc.',
    created_at: '2026-09-12T14:30:00Z',
    status: 'completed',
    recommendation_stance: 'favorable',
    confidence: 0.88,
    horizon: '3-5 years',
    risk_tolerance: 'moderate',
  },
  {
    analysis_id: 'an-002',
    report_id: 'rep-infy-002',
    ticker: 'INFY',
    company_name: 'Infosys Limited',
    created_at: '2026-09-13T10:00:00Z',
    status: 'completed',
    recommendation_stance: 'favorable',
    confidence: 0.86,
    horizon: '5 years',
    risk_tolerance: 'moderate',
  },
  {
    analysis_id: 'an-003',
    ticker: 'NVDA',
    company_name: 'NVIDIA Corporation',
    created_at: '2026-09-14T08:15:00Z',
    status: 'completed',
    recommendation_stance: 'cautious',
    confidence: 0.82,
    horizon: '1-3 years',
    risk_tolerance: 'aggressive',
  },
];

export const MOCK_DOCUMENTS: DocumentItem[] = [
  {
    id: 'doc-001',
    filename: 'AAPL_2026_Q3_10Q.pdf',
    ticker: 'AAPL',
    doc_type: '10-Q',
    size_bytes: 4259840,
    upload_date: '2026-09-11T12:00:00Z',
    status: 'indexed',
    chunk_count: 142,
  },
  {
    id: 'doc-002',
    filename: 'AAPL_Earnings_Call_Transcript_Q3.pdf',
    ticker: 'AAPL',
    doc_type: 'Earnings Transcript',
    size_bytes: 845210,
    upload_date: '2026-09-11T12:05:00Z',
    status: 'indexed',
    chunk_count: 38,
  },
  {
    id: 'doc-003',
    filename: 'INFY_Annual_Report_FY26.pdf',
    ticker: 'INFY',
    doc_type: '10-K',
    size_bytes: 12450120,
    upload_date: '2026-09-12T09:15:00Z',
    status: 'indexed',
    chunk_count: 420,
  },
  {
    id: 'doc-004',
    filename: 'NVDA_10K_Annual_Filing.pdf',
    ticker: 'NVDA',
    doc_type: '10-K',
    size_bytes: 6184000,
    upload_date: '2026-09-13T16:20:00Z',
    status: 'indexed',
    chunk_count: 215,
  },
];

export const MOCK_RESEARCH_QA_DATABASE: Record<string, ResearchQAResult> = {
  default: {
    question: 'General research inquiry',
    answer:
      'According to the indexed SEC filings and transcripts, capital expenditure trends are increasingly oriented towards generative AI infrastructure, while operational cash flows remain sufficient to fund both internal investments and shareholder returns.',
    citations: [
      {
        source_document: 'AAPL_2026_Q3_10Q.pdf',
        page: 22,
        excerpt:
          'Capital expenditures totaled $2.8B, driven by investments in data center infrastructure and specialized tooling.',
        confidence: 0.94,
      },
    ],
    confidence: 0.91,
  },
};
