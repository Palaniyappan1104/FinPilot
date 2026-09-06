# FinPilot — Master Implementation Roadmap

**Project:** FinPilot — A Multi-Agent AI Platform for Financial Research and Investment Decision Support
**Document purpose:** Complete phase-by-phase implementation plan. This is a planning document only — no application code is implemented as part of this document.
**Type of project:** Final-year engineering project (AI-powered financial research / decision-support platform, not a trading bot).

---

## How to Read This Plan

- The project is split into **21 major phases** (Phase 0–20).
- Every major phase is split into **subphases** (X.1, X.2, …).
- Every subphase is split into **tasks** (X.Y.1, X.Y.2, …) small enough to implement, test, and commit independently.
- Every major phase ends with a **Testing** subphase and a **Phase Completion Review** subphase.
- Every major phase ends with **one Git checkpoint** (commit + push). Large phases may have smaller intermediate commits, but the final phase-completion commit is mandatory.
- Financial values and technical indicators are always computed from real data/tools. LLMs interpret data — they never invent numbers.

---

## PHASE 0 — Project Planning & Repository Setup

**Estimated time:** 1–2 days | **Difficulty:** Easy | **Dependency:** None
**Deliverables:** Initialized repo, project skeleton, tooling baseline, this plan committed.

### 0.1 Repository Initialization
0.1.1 Create GitHub repository (`finpilot`)
0.1.2 Initialize local git repo, set default branch, add `.gitignore` (Python, Node, IDE, OS files)
0.1.3 Add `LICENSE` and initial `README.md` stub
0.1.4 Set up branch strategy (`main` + feature branches, e.g. `phase/xx-name`)

### 0.2 Monorepo Structure
0.2.1 Create top-level folders: `backend/`, `frontend/`, `docs/`, `scripts/`
0.2.2 Add root `plan.md` (this document) and `CONTRIBUTING.md`
0.2.3 Decide commit message convention (`Phase X: <name>` for checkpoints, `feat/fix/chore: ...` for intermediate commits)

### 0.3 Tooling Baseline
0.3.1 Add `.editorconfig`
0.3.2 Add pre-commit hooks placeholder (formatting/linting to be wired in Phase 1)
0.3.3 Add issue/PR templates (optional, for project management realism)

### 0.4 Project Management Setup
0.4.1 Create a lightweight backlog (GitHub Projects/Issues) mirroring phases 0–20
0.4.2 Record phase dependency map (see end of this document) as a pinned issue/wiki page

### 0.5 Phase Completion Review
0.5.1 Verify repo structure exists and is pushed
0.5.2 **Git checkpoint:** `git commit -m "Phase 0: Project Planning & Repository Setup"`

---

## PHASE 1 — Project Foundation

**Estimated time:** 2–3 days | **Difficulty:** Easy–Medium | **Dependency:** Phase 0
**Deliverables:** Runnable backend/frontend skeletons, config management, logging, CI baseline.

### 1.1 Backend Skeleton
1.1.1 Initialize Python project (`pyproject.toml` or `requirements.txt`), virtual environment
1.1.2 Install core deps: FastAPI, Uvicorn, Pydantic, python-dotenv, pytest
1.1.3 Create `backend/app/` package structure (`api/`, `core/`, `agents/`, `services/`, `models/`, `db/`, `tests/`)
1.1.4 Add minimal FastAPI app with `/health` endpoint

### 1.2 Configuration Management
1.2.1 Define `.env.example` with all expected environment variables (LLM keys, data provider keys, DB URL, ChromaDB path)
1.2.2 Implement a `Settings` class (Pydantic `BaseSettings`) to load env vars centrally
1.2.3 Ensure no secrets are hardcoded anywhere; verify `.env` is gitignored

### 1.3 Logging & Error Handling Baseline
1.3.1 Configure structured logging (module-level loggers, log levels via env var)
1.3.2 Add global FastAPI exception handlers (validation errors, unhandled errors) returning structured JSON
1.3.3 Add request logging middleware (method, path, latency, status)

### 1.4 Frontend Skeleton
1.4.1 Initialize React + Vite project in `frontend/`
1.4.2 Install and configure Tailwind CSS
1.4.3 Add base layout, routing library, and a placeholder home page that calls `/health`
1.4.4 Add `.env.example` for frontend (API base URL)

### 1.5 Dev Tooling & CI
1.5.1 Add formatting/linting: `black`, `ruff`/`flake8` (backend), `eslint`/`prettier` (frontend)
1.5.2 Add a basic GitHub Actions workflow: install deps, run lint, run tests on push/PR
1.5.3 Document local dev setup in `README.md`

### 1.6 Testing
1.6.1 Unit test for `Settings` loading (env var overrides)
1.6.2 API test for `/health` endpoint
1.6.3 Smoke test that frontend build succeeds

### 1.7 Phase Completion Review
1.7.1 Verify backend runs (`uvicorn`), frontend runs (`npm run dev`), CI passes
1.7.2 **Git checkpoint:** `git commit -m "Phase 1: Project Foundation"`

---

## PHASE 2 — LangGraph & Agent Infrastructure

**Estimated time:** 3–4 days | **Difficulty:** Medium | **Dependency:** Phase 1
**Deliverables:** LLM provider abstraction, base agent interface, LangGraph skeleton graph, shared state schema.

### 2.1 LLM Provider Abstraction
2.1.1 Define an `LLMProvider` interface (e.g. `generate(prompt, schema=None)`)
2.1.2 Implement `GeminiProvider` as the first concrete implementation
2.1.3 Add provider factory/config switch (`LLM_PROVIDER=gemini`) so a second provider can be added later without touching agents
2.1.4 Add retry/backoff and error handling for LLM calls

### 2.2 Structured Output Handling
2.2.1 Define a base Pydantic pattern for structured agent outputs
2.2.2 Implement a helper to request/validate structured (JSON) responses from the LLM
2.2.3 Handle malformed/invalid LLM JSON gracefully (retry once, then raise a typed error)

### 2.3 Shared Graph State
2.3.1 Define the LangGraph shared state schema (user query, investor profile, clarified request, specialist outputs, aggregated result, report)
2.3.2 Define state update conventions (how each node reads/writes state)

### 2.4 Base Agent Interface
2.4.1 Define an abstract `Agent` base class (name, input schema, output schema, `run(state)` method)
2.4.2 Define a common `AgentResult` wrapper (success/failure, data, error, confidence)
2.4.3 Add a `Tool` abstraction so agents call external data via tools, not APIs directly

### 2.5 Minimal LangGraph Skeleton
2.5.1 Build a trivial LangGraph graph with a single pass-through node to validate wiring
2.5.2 Add a CLI/script entry point to invoke the graph manually for debugging

### 2.6 Testing
2.6.1 Unit tests for `LLMProvider` abstraction (mocked Gemini responses)
2.6.2 Unit tests for structured-output validation (valid/invalid JSON cases)
2.6.3 Unit test for the trivial graph running end-to-end

### 2.7 Phase Completion Review
2.7.1 Verify a mocked agent can run through the graph and return structured output
2.7.2 **Git checkpoint:** `git commit -m "Phase 2: LangGraph & Agent Infrastructure"`

---

## PHASE 3 — Conversation Agent

**Estimated time:** 2–3 days | **Difficulty:** Medium | **Dependency:** Phase 2
**Deliverables:** Agent that receives raw user input and produces a normalized intent/query representation.

### 3.1 Conversation Schema
3.1.1 Define input schema (raw user message, conversation history)
3.1.2 Define output schema (normalized query, extracted entities: company, capital, horizon, risk tolerance if present, intent type)

### 3.2 Conversation Logic
3.2.1 Design prompt for intent extraction and entity extraction
3.2.2 Implement the Conversation Agent using the base `Agent` interface
3.2.3 Handle multi-turn context (append to conversation history in state)

### 3.3 Guardrails
3.3.1 Detect out-of-scope requests (non-financial queries) and produce a graceful response path
3.3.2 Ensure the agent does not fabricate financial data at this stage (it only extracts/normalizes)

### 3.4 Testing
3.4.1 Unit tests with varied phrasings ("Should I invest ₹1,00,000 in Infosys for 5 years?")
3.4.2 Edge cases: missing entities, ambiguous company names, irrelevant queries
3.4.3 Structured output schema validation tests

### 3.5 Phase Completion Review
3.5.1 Verify the agent correctly normalizes at least 10 varied sample queries
3.5.2 **Git checkpoint:** `git commit -m "Phase 3: Conversation Agent"`

---

## PHASE 4 — Clarification Agent

**Estimated time:** 2–3 days | **Difficulty:** Medium | **Dependency:** Phase 3
**Deliverables:** Agent that identifies missing investor-profile information and asks targeted follow-up questions.

### 4.1 Required-Fields Schema
4.1.1 Define the investor profile schema (goal, horizon, capital, risk tolerance, target company)
4.1.2 Define "completeness" rules (which fields are mandatory before routing to CIO)

### 4.2 Clarification Logic
4.2.1 Design prompt/logic to detect missing fields from the normalized query + state
4.2.2 Generate targeted clarification questions (one batch, not one-by-one spam)
4.2.3 Implement loop-back handling: user answers → merge into investor profile → re-check completeness

### 4.3 Conversation Flow Integration
4.3.1 Wire Clarification Agent after Conversation Agent in the graph
4.3.2 Add a state flag (`profile_complete: bool`) to control graph branching

### 4.4 Testing
4.4.1 Unit tests: fully specified query (no clarification needed)
4.4.2 Unit tests: partially specified query (some fields missing)
4.4.3 Unit tests: multi-turn clarification resolves to a complete profile

### 4.5 Phase Completion Review
4.5.1 Verify a partial query correctly triggers clarification questions and resolves after user input
4.5.2 **Git checkpoint:** `git commit -m "Phase 4: Clarification Agent"`

---

## PHASE 5 — CIO / Router Agent

**Estimated time:** 3–4 days | **Difficulty:** Medium–Hard | **Dependency:** Phase 3, Phase 4
**Deliverables:** Central router that decides which specialist agents to invoke and coordinates their execution.

### 5.1 Routing Schema
5.1.1 Define CIO input schema (clarified request + investor profile)
5.1.2 Define CIO output schema (list of specialist agents to invoke, task parameters per agent)

### 5.2 Routing Logic
5.2.1 Design decision logic/prompt: which specialists are relevant to this request (e.g. skip Research Analyst if no documents uploaded)
5.2.2 Implement structured routing decision (no free-form autonomous loops)
5.2.3 Add a fallback default: if uncertain, route to all core specialists (Technical, Fundamental, News, Risk)

### 5.3 Orchestration in LangGraph
5.3.1 Implement conditional edges in LangGraph based on CIO's routing decision
5.3.2 Implement fan-out (parallel specialist invocation) and fan-in (collect results) pattern
5.3.3 Add per-agent timeout/error isolation (one specialist failing shouldn't crash the whole graph)

### 5.4 Testing
5.4.1 Unit tests for routing decisions across different query types
5.4.2 Integration test: CIO output correctly drives graph branching with stub specialist agents
5.4.3 Failure-injection test: one specialist fails, graph still completes with partial results

### 5.5 Phase Completion Review
5.5.1 Verify routing + fan-out/fan-in works with stub agents
5.5.2 **Git checkpoint:** `git commit -m "Phase 5: CIO / Router Agent"`

---

## PHASE 6 — Fundamental Analyst

**Estimated time:** 4–5 days | **Difficulty:** Medium–Hard | **Dependency:** Phase 2 (independent of Phases 3–5 internals)
**Deliverables:** Specialist agent producing structured fundamental analysis from real financial data.

### 6.1 Financial Data Provider Abstraction
6.1.1 Define provider interface (`get_fundamentals(ticker)`)
6.1.2 Define normalized financial data schema (revenue, net income, EPS, P/E, ROE, debt, cash flow, growth)
6.1.3 Implement first provider (e.g. Yahoo Finance)
6.1.4 Add validation (missing fields, malformed responses)
6.1.5 Handle API failures (timeouts, rate limits, invalid ticker)

### 6.2 Financial Metrics
6.2.1 Revenue (current + historical trend)
6.2.2 Net income
6.2.3 EPS
6.2.4 P/E
6.2.5 ROE
6.2.6 Debt (debt-to-equity or similar)
6.2.7 Cash flow (operating/free cash flow)
6.2.8 Growth metrics (YoY revenue/profit growth)

### 6.3 Fundamental Analysis Logic
6.3.1 Create fundamental analysis output schema (financial health, growth, valuation, profitability, financial risks, fundamental score, evidence, confidence)
6.3.2 Create fundamental analyst prompt (interpret, don't invent)
6.3.3 Pass structured financial data into the prompt context
6.3.4 Generate structured analysis via LLM
6.3.5 Validate output against schema; reject/retry on invalid structure

### 6.4 Testing
6.4.1 Unit tests for metric calculations/normalization
6.4.2 API/data tests (mocked provider responses, including malformed data)
6.4.3 Agent tests (given fixed financial data, output stays grounded in that data)
6.4.4 Edge cases: missing metrics, negative earnings, very new companies with sparse data

### 6.5 Phase Completion Review
6.5.1 Verify the agent produces a valid structured report for at least 3 real tickers
6.5.2 **Git checkpoint:** `git commit -m "Phase 6: Fundamental Analyst"`

---

## PHASE 7 — Technical Analyst

**Estimated time:** 4–5 days | **Difficulty:** Medium–Hard | **Dependency:** Phase 2 (independent of Phase 6)
**Deliverables:** Specialist agent producing structured technical analysis from real price/indicator data.

### 7.1 Market Data Provider Abstraction
7.1.1 Define provider interface (`get_ohlcv(ticker, period, interval)`)
7.1.2 Implement first provider (e.g. Yahoo Finance / Twelve Data)
7.1.3 Add validation for OHLCV data (gaps, non-trading days, malformed rows)
7.1.4 Handle API failures/rate limits

### 7.2 Technical Indicator Calculations
7.2.1 SMA
7.2.2 EMA
7.2.3 RSI
7.2.4 MACD
7.2.5 Volume analysis
7.2.6 Support/resistance detection
7.2.7 Trend detection (uptrend/downtrend/sideways)
7.2.8 Technical scoring function (deterministic, not LLM-generated)

### 7.3 Technical Analysis Logic
7.3.1 Define output schema (trend, indicators, support/resistance, technical score, reasoning/evidence, risks, confidence)
7.3.2 Design prompt for LLM interpretation of pre-calculated indicators
7.3.3 Ensure LLM only narrates/explains calculated values, never computes or invents them
7.3.4 Validate structured output

### 7.4 Testing
7.4.1 Unit tests for each indicator against known reference values
7.4.2 Data validation tests (missing candles, holidays, illiquid stocks)
7.4.3 Agent tests: interpretation stays consistent with the underlying calculated values
7.4.4 Edge cases: insufficient history for indicator window, flat/no-volume periods

### 7.5 Phase Completion Review
7.5.1 Verify indicator calculations match a trusted reference (e.g. cross-check with a known charting tool)
7.5.2 **Git checkpoint:** `git commit -m "Phase 7: Technical Analyst"`

---

## PHASE 8 — News Analyst

**Estimated time:** 3–4 days | **Difficulty:** Medium | **Dependency:** Phase 2 (independent of Phases 6–7)
**Deliverables:** Specialist agent producing structured news/sentiment analysis from real news data.

### 8.1 News Data Provider Abstraction
8.1.1 Define provider interface (`get_news(ticker/company, date_range)`)
8.1.2 Implement first provider (e.g. Finnhub news endpoint)
8.1.3 Add deduplication and recency filtering
8.1.4 Handle API failures/empty results

### 8.2 News Processing
8.2.1 Extract headline + summary per article
8.2.2 Classify sentiment per article (positive/negative/neutral)
8.2.3 Identify "important events" (earnings, management changes, regulatory actions, M&A)

### 8.3 News Analysis Logic
8.3.1 Define output schema (recent news, sentiment, important events, positive factors, negative factors, confidence)
8.3.2 Design prompt to synthesize article-level signals into an overall view
8.3.3 Preserve source/article references for evidence display
8.3.4 Validate structured output

### 8.4 Testing
8.4.1 Unit tests for provider parsing/normalization
8.4.2 Sentiment classification sanity tests on sample articles
8.4.3 Agent tests: no articles found → graceful "insufficient news" output rather than fabrication
8.4.4 Edge cases: conflicting sentiment across articles, stale/old-only news

### 8.5 Phase Completion Review
8.5.1 Verify the agent produces a grounded, source-referenced news summary for a real company
8.5.2 **Git checkpoint:** `git commit -m "Phase 8: News Analyst"`

---

## PHASE 9 — Research Vault / RAG

**Estimated time:** 5–6 days | **Difficulty:** Hard | **Dependency:** Phase 2 (independent of Phases 6–8)
**Deliverables:** Document upload, retrieval pipeline, and a Research Analyst agent answering from retrieved evidence.

### 9.1 Document Upload
9.1.1 Define upload API contract (file, company/ticker association, document type)
9.1.2 Implement file storage (local/dev, abstracted for future cloud storage)
9.1.3 Enforce upload validation (file type, size limits, virus/format sanity checks)

### 9.2 Text Extraction
9.2.1 PDF text extraction
9.2.2 Plain text/Word extraction (if in scope)
9.2.3 Handle extraction failures (scanned/image-only PDFs) with a clear error path

### 9.3 Document Validation
9.3.1 Validate extracted text is non-empty and reasonably well-formed
9.3.2 Detect and reject unsupported/corrupted files

### 9.4 Chunking
9.4.1 Implement chunking strategy (size + overlap) tuned for financial documents
9.4.2 Preserve section/page metadata per chunk

### 9.5 Metadata
9.5.1 Define chunk metadata schema (source document, page, company, document type, upload date)

### 9.6 Embeddings
9.6.1 Choose and integrate an embedding model
9.6.2 Implement embedding generation pipeline for chunks

### 9.7 ChromaDB Storage
9.7.1 Set up ChromaDB collection(s), keyed by company/document
9.7.2 Implement insert/update pipeline for chunk embeddings + metadata

### 9.8 Query Embedding
9.8.1 Implement embedding of incoming user research questions

### 9.9 Similarity Search
9.9.1 Implement similarity search against ChromaDB
9.9.2 Add filtering (by company/document type) alongside semantic search

### 9.10 Top-K Retrieval
9.10.1 Implement top-K retrieval with configurable K
9.10.2 Add relevance-score thresholding to filter weak matches

### 9.11 Context Construction
9.11.1 Assemble retrieved chunks into a structured context block with citations

### 9.12 Research Analyst Agent
9.12.1 Define output schema (retrieved documents, relevant findings, evidence/source metadata, research conclusions, confidence)
9.12.2 Design prompt requiring answers to be grounded in retrieved context only
9.12.3 Validate structured output

### 9.13 Source/Evidence Tracking
9.13.1 Ensure every finding maps back to a specific document + chunk/page reference

### 9.14 Retrieval Testing
9.14.1 Unit tests for chunking, embeddings, and similarity search
9.14.2 Precision/recall spot-checks on a small labeled test set of Q&A pairs

### 9.15 Handling Irrelevant Queries
9.15.1 Detect low-relevance retrieval results and respond with "insufficient evidence" instead of guessing
9.15.2 Handle the "no documents uploaded yet" case gracefully

### 9.16 Testing (Agent-Level)
9.16.1 End-to-end test: upload → chunk → embed → store → query → retrieve → answer
9.16.2 Edge cases: very large documents, duplicate uploads, mixed-language documents

### 9.17 Phase Completion Review
9.17.1 Verify a sample annual report can be uploaded and correctly answers a factual question with citations
9.17.2 **Git checkpoint:** `git commit -m "Phase 9: Research Vault / RAG"`

---

## PHASE 10 — Risk Analyst

**Estimated time:** 3–4 days | **Difficulty:** Medium | **Dependency:** Phases 6, 7, 8 outputs (schemas), independent implementation possible in parallel
**Deliverables:** Specialist agent producing a structured, investor-aware risk assessment.

### 10.1 Risk Input Aggregation
10.1.1 Define which upstream outputs feed the Risk Analyst (technical, fundamental, news signals + investor profile)
10.1.2 Define a lightweight interface to consume partial results if some specialists were skipped/failed

### 10.2 Risk Categories
10.2.1 Market risk
10.2.2 Company-specific risk
10.2.3 Sector risk
10.2.4 Financial risk (leverage, liquidity)
10.2.5 Volatility-based risk (from technical data)
10.2.6 Investor-specific risk (mismatch with horizon/risk tolerance)

### 10.3 Risk Analysis Logic
10.3.1 Define output schema (market risks, company risks, sector risks, financial risks, investor-specific risks, overall risk level, confidence)
10.3.2 Design prompt combining quantitative signals with investor context
10.3.3 Implement deterministic overall-risk-level scoring where possible, LLM explains reasoning
10.3.4 Validate structured output

### 10.4 Testing
10.4.1 Unit tests with varying investor profiles against the same company (risk should differ)
10.4.2 Unit tests with missing upstream data (graceful degradation)
10.4.3 Edge cases: highly volatile stock + low risk tolerance; stable stock + high risk tolerance

### 10.5 Phase Completion Review
10.5.1 Verify risk output changes appropriately across different investor profiles for the same company
10.5.2 **Git checkpoint:** `git commit -m "Phase 10: Risk Analyst"`

---

## PHASE 11 — Report Aggregator

**Estimated time:** 3–4 days | **Difficulty:** Medium–Hard | **Dependency:** Phases 6, 7, 8, 9, 10 (specialist output schemas)
**Deliverables:** Component that merges all specialist outputs into one structured, evidence-preserving analysis.

### 11.1 Aggregation Schema
11.1.1 Define the unified analysis schema combining all specialist outputs + investor profile
11.1.2 Ensure schema preserves per-specialist evidence/source references

### 11.2 Aggregation Logic
11.2.1 Implement merging logic (not concatenation) — cross-referencing signals across specialists
11.2.2 Handle partial input (missing/failed specialist outputs) without breaking aggregation
11.2.3 Resolve conflicting signals (e.g. strong fundamentals but negative news) into a coherent unified view

### 11.3 Consistency Checks
11.3.1 Validate that aggregated output doesn't drop or alter specialist evidence
11.3.2 Add sanity checks (e.g. overall stance shouldn't contradict a unanimous specialist signal without explanation)

### 11.4 Testing
11.4.1 Unit tests combining stub specialist outputs into expected aggregate shapes
11.4.2 Tests for conflicting-signal scenarios
11.4.3 Tests for missing/failed specialist scenarios

### 11.5 Phase Completion Review
11.5.1 Verify aggregation produces a coherent, evidence-preserving structure across multiple test scenarios
11.5.2 **Git checkpoint:** `git commit -m "Phase 11: Report Aggregator"`

---

## PHASE 12 — Report Generator

**Estimated time:** 3–4 days | **Difficulty:** Medium | **Dependency:** Phase 11
**Deliverables:** Final explainable investment report generation from the aggregated analysis.

### 12.1 Report Schema
12.1.1 Define final report schema (company, investor profile, horizon, capital, technical, fundamental, news, research, risk, overall assessment, recommendation, key reasons, important risks, evidence/sources, disclaimer)

### 12.2 Report Generation Logic
12.2.1 Design prompt to synthesize the aggregated analysis into a narrative, explainable report
12.2.2 Ensure the recommendation cites specific reasons tied to the aggregated evidence
12.2.3 Always attach a clear disclaimer (not guaranteed financial advice)

### 12.3 Explainability
12.3.1 Ensure every major claim in the report links back to a specialist finding or evidence source
12.3.2 Avoid single-word verdicts ("BUY"/"SELL") without justification

### 12.4 Output Formats
12.4.1 Structured JSON report (for frontend rendering)
12.4.2 Optional human-readable text/markdown rendering of the same report

### 12.5 Testing
12.5.1 Unit tests validating report schema completeness
12.5.2 Tests verifying disclaimer is always present
12.5.3 Tests checking recommendation differs across differing investor profiles for the same aggregated data

### 12.6 Phase Completion Review
12.6.1 Verify a full sample report reads coherently and cites evidence correctly
12.6.2 **Git checkpoint:** `git commit -m "Phase 12: Report Generator"`

---

## PHASE 13 — End-to-End LangGraph Integration

**Estimated time:** 4–5 days | **Difficulty:** Hard | **Dependency:** Phases 3–12 all stable
**Deliverables:** Fully wired LangGraph pipeline from raw user input to final report.

### 13.1 Full Graph Assembly
13.1.1 Wire Conversation → Clarification → CIO → Specialists (parallel) → Aggregator → Report Generator
13.1.2 Ensure state schema supports all node inputs/outputs without leakage or clobbering

### 13.2 Conditional Branching
13.2.1 Implement clarification loop-back branch
13.2.2 Implement CIO's dynamic specialist selection branch
13.2.3 Implement Research Analyst conditional inclusion (only if documents exist)

### 13.3 Error Handling & Resilience
13.3.1 Ensure a single specialist failure doesn't halt the entire graph
13.3.2 Add graph-level timeouts and fallback messaging for total failure cases
13.3.3 Add structured logging across the full graph execution (trace per request)

### 13.4 Testing
13.4.1 End-to-end tests with fully specified queries
13.4.2 End-to-end tests requiring clarification
13.4.3 End-to-end tests with documents uploaded (Research Analyst included)
13.4.4 Failure-injection tests (simulate provider outages) verifying graceful degradation

### 13.5 Phase Completion Review
13.5.1 Verify at least 5 diverse end-to-end scenarios produce correct, complete reports
13.5.2 **Git checkpoint:** `git commit -m "Phase 13: End-to-End LangGraph Integration"`

---

## PHASE 14 — PostgreSQL & Persistence

**Estimated time:** 3–4 days | **Difficulty:** Medium | **Dependency:** Phase 1 (can be developed in parallel with Phases 3–13, integrated here)
**Deliverables:** Persistent storage for users, profiles, requests, and reports.

### 14.1 Schema Design
14.1.1 Define entities: Users, Investment Profiles, Analysis Requests, Companies, Analysis Results, Reports, Documents Metadata, Conversations
14.1.2 Clearly separate PostgreSQL (structured/app data) from ChromaDB (vector/document retrieval)

### 14.2 Database Setup
14.2.1 Set up PostgreSQL locally/dev (Docker recommended)
14.2.2 Choose and configure ORM/migration tool (e.g. SQLAlchemy + Alembic)
14.2.3 Write initial migration for all core tables

### 14.3 Persistence Layer
14.3.1 Implement repository/service layer for each entity (CRUD)
14.3.2 Wire conversation history and analysis requests to persist during graph execution
14.3.3 Persist final reports with a retrievable ID

### 14.4 Documents Metadata Linkage
14.4.1 Store document metadata (filename, company, upload date, status) in PostgreSQL, with actual content/embeddings staying in ChromaDB

### 14.5 Testing
14.5.1 Unit tests for repository/service layer (CRUD correctness)
14.5.2 Migration tests (apply/rollback)
14.5.3 Integration test: full pipeline run persists a retrievable report row

### 14.6 Phase Completion Review
14.6.1 Verify a report generated end-to-end is correctly persisted and retrievable by ID
14.6.2 **Git checkpoint:** `git commit -m "Phase 14: PostgreSQL & Persistence"`

---

## PHASE 15 — FastAPI Backend Integration

**Estimated time:** 3–4 days | **Difficulty:** Medium | **Dependency:** Phases 13, 14
**Deliverables:** Complete REST API surface exposing the full pipeline to the frontend.

### 15.1 API Design
15.1.1 Health check endpoint
15.1.2 Chat/query endpoint (submit user message, get conversation/clarification response)
15.1.3 Company analysis endpoint (trigger full pipeline for a company + profile)
15.1.4 Clarification endpoint (submit missing-field answers)
15.1.5 Document upload endpoint
15.1.6 Research query endpoint (ask a question against uploaded documents)
15.1.7 Analysis status endpoint (poll long-running analysis progress)
15.1.8 Report retrieval endpoint (fetch a completed report by ID)

### 15.2 Request/Response Contracts
15.2.1 Define Pydantic request/response models for every endpoint
15.2.2 Add input validation and clear error responses

### 15.3 Async Execution
15.3.1 Decide sync vs background-task/queue execution for long-running analysis
15.3.2 Implement status tracking for in-progress analyses

### 15.4 API Documentation
15.4.1 Verify FastAPI auto-generated OpenAPI docs are accurate and complete

### 15.5 Testing
15.5.1 API tests for every endpoint (happy path)
15.5.2 API tests for validation/error cases
15.5.3 Integration test: full user journey via API only (no direct graph calls)

### 15.6 Phase Completion Review
15.6.1 Verify the entire user journey (query → clarification → analysis → report) works purely through the API
15.6.2 **Git checkpoint:** `git commit -m "Phase 15: FastAPI Backend Integration"`

---

## PHASE 16 — React Frontend

**Estimated time:** 5–6 days | **Difficulty:** Medium–Hard | **Dependency:** Phase 1 (UI can be built against mocked APIs in parallel with backend phases; full wiring happens in Phase 17)
**Deliverables:** Functional React UI covering the full user journey.

### 16.1 App Shell
16.1.1 Layout, navigation, routing
16.1.2 Global state/context setup (investor profile, active analysis)

### 16.2 Dashboard
16.2.1 Landing dashboard summarizing recent analyses/reports

### 16.3 Chat/Query Interface
16.3.1 Chat-style input for the Conversation Agent
16.3.2 Clarification question/answer UI flow

### 16.4 Company Selection
16.4.1 Company search/select input
16.4.2 Basic company info display

### 16.5 Investor Profile Inputs
16.5.1 Form for goal, horizon, capital, risk tolerance
16.5.2 Validation and persistence of profile inputs across the session

### 16.6 Analysis Progress/Status
16.6.1 Progress indicator while specialists run
16.6.2 Per-specialist status display (pending/complete/failed)

### 16.7 Specialist Results Display
16.7.1 Technical analysis view
16.7.2 Fundamental analysis view
16.7.3 News analysis view
16.7.4 Risk analysis view

### 16.8 Research Document Upload
16.8.1 Upload UI with progress/validation feedback
16.8.2 Research Q&A interface against uploaded documents

### 16.9 Final Investment Report
16.9.1 Full report rendering (all sections)
16.9.2 Evidence/source display (linking claims to specialist/document evidence)
16.9.3 Risk display section
16.9.4 Disclaimer display (always visible, non-dismissible-by-default)

### 16.10 Testing
16.10.1 Component tests (e.g. React Testing Library) for key components
16.10.2 Mocked-API integration tests for the main user flow

### 16.11 Phase Completion Review
16.11.1 Verify the full UI flow works end-to-end against a mocked/stub API
16.11.2 **Git checkpoint:** `git commit -m "Phase 16: React Frontend"`

---

## PHASE 17 — Frontend-Backend Integration

**Estimated time:** 3–4 days | **Difficulty:** Medium | **Dependency:** Phase 15, Phase 16
**Deliverables:** Fully wired application with real API calls end-to-end.

### 17.1 API Client Layer
17.1.1 Implement a typed API client in the frontend
17.1.2 Centralize error handling and loading states

### 17.2 Wiring Real Endpoints
17.2.1 Replace mocked chat/clarification calls with real endpoints
17.2.2 Replace mocked analysis trigger/status polling with real endpoints
17.2.3 Replace mocked document upload/research query with real endpoints
17.2.4 Replace mocked report retrieval with real endpoint

### 17.3 CORS & Environment Config
17.3.1 Configure CORS on the backend for the frontend origin
17.3.2 Wire frontend `.env` API base URL for dev/prod

### 17.4 Testing
17.4.1 End-to-end manual test of the full journey using the real backend
17.4.2 Automated end-to-end test (e.g. Playwright/Cypress) covering the core happy path

### 17.5 Phase Completion Review
17.5.1 Verify a real user journey (query → clarification → analysis → report) works fully through the deployed UI
17.5.2 **Git checkpoint:** `git commit -m "Phase 17: Frontend-Backend Integration"`

---

## PHASE 18 — Evaluation & Testing

**Estimated time:** 4–5 days | **Difficulty:** Medium–Hard | **Dependency:** Phase 17
**Deliverables:** Systematic evaluation of system quality across defined, measurable criteria.

### 18.1 Evaluation Criteria & Metrics
18.1.1 Routing accuracy (does CIO select the correct specialists for a labeled test set of queries?)
18.1.2 Financial data correctness (spot-check against source provider/reference)
18.1.3 Technical indicator correctness (compare against reference calculations)
18.1.4 RAG retrieval relevance (precision@K on a labeled Q&A set)
18.1.5 Citation/source coverage (% of claims with a traceable source)
18.1.6 Agent output consistency (schema validity rate across N runs)
18.1.7 Report quality (rubric-based scoring: clarity, explainability, completeness)
18.1.8 Response latency (per-agent and end-to-end timing)
18.1.9 API reliability (error rate under repeated calls)
18.1.10 Error handling (behavior under induced provider failures)
18.1.11 Personalization (does output differ appropriately across investor profiles?)
18.1.12 End-to-end functionality (full journey success rate across a test query set)

### 18.2 Test Data & Harness
18.2.1 Build a labeled test set of sample queries, expected routing, and expected profile variations
18.2.2 Build a small labeled RAG Q&A set from sample documents

### 18.3 Automated Evaluation Scripts
18.3.1 Script to run the full test set through the pipeline and collect metrics
18.3.2 Script to compute and report each metric from 18.1

### 18.4 Manual Review
18.4.1 Human review pass on generated reports for coherence/explainability
18.4.2 Document known failure modes and limitations

### 18.5 Phase Completion Review
18.5.1 Compile an evaluation report summarizing all metrics with pass/fail thresholds
18.5.2 **Git checkpoint:** `git commit -m "Phase 18: Evaluation & Testing"`

---

## PHASE 19 — Security, Error Handling & Production Improvements

**Estimated time:** 3–4 days | **Difficulty:** Medium | **Dependency:** Phase 17 (informed by Phase 18 findings)
**Deliverables:** Hardened application ready for demonstration/limited deployment.

### 19.1 Secrets & Configuration
19.1.1 Audit all API keys/secrets are sourced from `.env`, never hardcoded
19.1.2 Verify `.gitignore` covers all secret-bearing files
19.1.3 Scan git history for accidentally committed secrets (rotate any found)

### 19.2 Input & Upload Validation
19.2.1 Re-verify all API input validation is comprehensive (types, ranges, required fields)
19.2.2 Harden file upload validation (type, size, malicious file protection)

### 19.3 Error Handling
19.3.1 Ensure all external calls (LLM, data providers, DB) have consistent error handling and user-facing messages
19.3.2 Add global rate-limiting/backoff for external API calls

### 19.4 Authentication/Authorization (as appropriate for project scope)
19.4.1 Decide scope: basic auth vs no-auth demo mode (document the decision)
19.4.2 If included: implement basic user authentication and authorization on report/document endpoints

### 19.5 Safe Logging
19.5.1 Ensure logs never contain API keys, secrets, or sensitive personal data
19.5.2 Add log redaction for sensitive fields if needed

### 19.6 Testing
19.6.1 Security-focused tests (invalid/malicious inputs, oversized uploads, injection attempts)
19.6.2 Regression test suite run in full to confirm no breakage from hardening changes

### 19.7 Phase Completion Review
19.7.1 Verify a security/quality checklist is fully satisfied
19.7.2 **Git checkpoint:** `git commit -m "Phase 19: Security, Error Handling & Production Improvements"`

---

## PHASE 20 — Documentation & Final Project Preparation

**Estimated time:** 3–4 days | **Difficulty:** Easy–Medium | **Dependency:** Phase 19
**Deliverables:** Complete project documentation and final-year submission materials.

### 20.1 Technical Documentation
20.1.1 `README.md` (overview, setup, run instructions)
20.1.2 Architecture documentation (diagrams of the multi-agent workflow, data flow)
20.1.3 API documentation (link/reference to OpenAPI docs + usage examples)
20.1.4 Agent documentation (per-agent purpose, inputs, outputs, prompts overview)
20.1.5 RAG documentation (pipeline description, chunking/embedding choices)
20.1.6 Environment configuration reference (all env vars explained)
20.1.7 Testing instructions (how to run backend/frontend/e2e tests)
20.1.8 Git workflow documentation (branching, commit conventions)
20.1.9 Project limitations documentation
20.1.10 Future enhancements documentation

### 20.2 Final-Year Project Documentation
20.2.1 Problem statement
20.2.2 Objectives
20.2.3 Methodology
20.2.4 System architecture
20.2.5 Workflow description
20.2.6 Results (from Phase 18 evaluation)
20.2.7 Evaluation summary
20.2.8 Limitations
20.2.9 Future scope

### 20.3 Presentation Materials (optional but recommended)
20.3.1 Slide deck summarizing the project for final review/demo
20.3.2 Demo script covering a representative end-to-end scenario

### 20.4 Final Review
20.4.1 Full read-through of all documentation for accuracy and consistency with the implemented system
20.4.2 Confirm all phases are marked complete and traceable to their Git checkpoints

### 20.5 Phase Completion Review
20.5.1 Verify documentation is complete, accurate, and the project is demo-ready
20.5.2 **Git checkpoint:** `git commit -m "Phase 20: Documentation & Final Project Preparation"`

---

## Complete Phase Dependency Map

```
Phase 0  (none)
Phase 1  → depends on Phase 0
Phase 2  → depends on Phase 1
Phase 3  → depends on Phase 2
Phase 4  → depends on Phase 3
Phase 5  → depends on Phase 3, Phase 4
Phase 6  → depends on Phase 2               (parallel-safe with 7, 8, 9)
Phase 7  → depends on Phase 2               (parallel-safe with 6, 8, 9)
Phase 8  → depends on Phase 2               (parallel-safe with 6, 7, 9)
Phase 9  → depends on Phase 2               (parallel-safe with 6, 7, 8)
Phase 10 → depends on Phase 6, 7, 8 (output schemas)
Phase 11 → depends on Phase 6, 7, 8, 9, 10
Phase 12 → depends on Phase 11
Phase 13 → depends on Phase 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 (all stable)
Phase 14 → depends on Phase 1               (parallel-safe with 3–13; integrated in 13/15)
Phase 15 → depends on Phase 13, Phase 14
Phase 16 → depends on Phase 1               (UI can be built in parallel with 3–15 against mocks)
Phase 17 → depends on Phase 15, Phase 16
Phase 18 → depends on Phase 17
Phase 19 → depends on Phase 17 (informed by Phase 18)
Phase 20 → depends on Phase 19
```

**Parallelization opportunities:**
- Phases 6, 7, 8, 9 (specialist agents) can be built independently and in parallel once Phase 2 is done.
- Phase 14 (database) can be developed in parallel with Phases 3–13, then integrated in Phase 13/15.
- Phase 16 (frontend) can be developed in parallel with backend phases 3–15 using mocked APIs, then wired in Phase 17.

---

## Estimated Total Project Duration

| Phase | Estimated Time |
|---|---|
| 0. Project Planning & Repository Setup | 1–2 days |
| 1. Project Foundation | 2–3 days |
| 2. LangGraph & Agent Infrastructure | 3–4 days |
| 3. Conversation Agent | 2–3 days |
| 4. Clarification Agent | 2–3 days |
| 5. CIO / Router Agent | 3–4 days |
| 6. Fundamental Analyst | 4–5 days |
| 7. Technical Analyst | 4–5 days |
| 8. News Analyst | 3–4 days |
| 9. Research Vault / RAG | 5–6 days |
| 10. Risk Analyst | 3–4 days |
| 11. Report Aggregator | 3–4 days |
| 12. Report Generator | 3–4 days |
| 13. End-to-End LangGraph Integration | 4–5 days |
| 14. PostgreSQL & Persistence | 3–4 days |
| 15. FastAPI Backend Integration | 3–4 days |
| 16. React Frontend | 5–6 days |
| 17. Frontend-Backend Integration | 3–4 days |
| 18. Evaluation & Testing | 4–5 days |
| 19. Security, Error Handling & Production Improvements | 3–4 days |
| 20. Documentation & Final Project Preparation | 3–4 days |

**Sequential worst case (no parallelization):** ~64–83 working days (roughly 13–17 weeks at a steady pace).
**Realistic estimate with parallelization** (specialists 6–9 in parallel; DB and frontend developed alongside backend phases): **~9–12 weeks** of steady part-time-to-full-time work.

These are planning estimates, not strict deadlines — adjust based on actual pace, especially around Phases 9 (RAG) and 13 (integration), which tend to run over.

---

## Git Checkpoint Strategy

- One **mandatory completion commit per major phase**, using the message format `Phase X: <Phase Name>`.
- Intermediate commits within a phase are encouraged for large phases (especially 6, 7, 9, 16, 18) using conventional prefixes: `feat:`, `fix:`, `test:`, `docs:`, `chore:`.
- Use feature branches per phase (`phase/09-research-vault`) merged into `main` via PR at phase completion, to keep history reviewable.
- Never commit `.env`, credentials, database dumps containing secrets, or raw API keys.
- Tag `main` at the end of each phase (`v0.X-phaseX`) for easy rollback/reference during evaluation and demos.

---

## Testing Strategy

- Testing happens **within every phase**, not deferred to the end.
- Test types used throughout:
  - Unit tests (functions, calculations, schema validation)
  - Integration tests (multi-component interactions, e.g. agent + provider)
  - API tests (FastAPI endpoints)
  - Agent tests (input handling, structured output validity, hallucination checks)
  - RAG retrieval tests (relevance, precision@K)
  - Data validation tests (malformed/missing external data)
  - Error handling tests (induced provider/API failures)
  - End-to-end tests (full pipeline, full user journey)
- For every AI agent, explicitly test: valid inputs, invalid inputs, missing data, upstream API failures, hallucination prevention (output must stay grounded in provided data), and output schema validity.
- Phase 18 formalizes measurable evaluation criteria on top of this ongoing testing.

---

## Definition of Done — Per Phase

A phase is considered **done** only when all of the following are true:
1. All subphases/tasks for the phase are implemented.
2. All planned tests for the phase pass (unit/integration/API/agent as applicable).
3. Code has been reviewed (self-review at minimum; peer review if available) against the architecture principles (modularity, provider abstraction, no hardcoded secrets, no LLM-invented financial values).
4. Any issues found during review are fixed.
5. The phase's deliverables (as listed at the top of the phase) are verifiably working, not just "written."
6. Documentation relevant to the phase (docstrings, README notes) is updated.
7. Work is committed with the phase-completion commit message and pushed to GitHub.
8. The phase is marked complete in the project backlog/issue tracker.

---

## Overall Project Definition of Done

FinPilot is considered **complete** when:
1. All 21 phases (0–20) are individually done per the per-phase Definition of Done above.
2. A user can submit a natural-language investment query, go through clarification if needed, and receive a complete, structured, evidence-backed, personalized investment report through the deployed frontend.
3. All specialist agents (Technical, Fundamental, News, Research, Risk) function against real data providers and are individually testable.
4. The Research Vault correctly ingests documents and answers questions with source citations.
5. Financial values and technical indicators are always sourced from real data/tools — verified by test coverage, never fabricated by the LLM.
6. The system correctly personalizes recommendations across differing investor profiles for the same company (verified in Phase 18 evaluation).
7. PostgreSQL and ChromaDB responsibilities remain cleanly separated as designed.
8. Security baseline from Phase 19 is satisfied (no committed secrets, validated inputs, safe logging).
9. Full documentation (technical + final-year project write-up) from Phase 20 is complete and consistent with the implemented system.
10. Every major phase has a corresponding Git checkpoint, giving a clean, reviewable implementation history from Phase 0 to Phase 20.
