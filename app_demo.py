"""FinPilot — Streamlit Demo Frontend

Connects to the FastAPI demo endpoints (/api/v1/demo/clarify and
/api/v1/demo/analyze) for a polished review-ready experience.

Run:  streamlit run app_demo.py
"""

import threading
import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinPilot — AI Financial Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* Hide default header/footer */
header[data-testid="stHeader"] { background: transparent; }
footer { display: none; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #006064 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(13,71,161,0.4);
}
.hero-banner h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: #90caf9;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64b5f6;
    margin-bottom: 0.5rem;
}
.card-body {
    color: #cfd8dc;
    line-height: 1.7;
}

/* Agent status pills */
.agent-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.agent-row:last-child { border-bottom: none; }
.agent-pill {
    background: rgba(0,200,83,0.15);
    border: 1px solid rgba(0,200,83,0.3);
    color: #69f0ae;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
}
.agent-name { font-weight: 500; color: #e2e8f0; font-size: 0.9rem; }
.agent-summary { color: #90a4ae; font-size: 0.82rem; }

/* Metric chips */
.confidence-high   { color: #69f0ae; font-weight: 700; }
.confidence-medium { color: #ffd740; font-weight: 700; }
.confidence-low    { color: #ff5252; font-weight: 700; }

/* Query input override */
textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Primary button override */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0288d1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.8rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(2,136,209,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(2,136,209,0.5) !important;
}

/* Section headers */
h2 { color: #90caf9 !important; font-weight: 600 !important; }
h3 { color: #64b5f6 !important; font-weight: 500 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Success/error banners */
.stAlert { border-radius: 10px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000/api/v1/demo"

# Pipeline pacing (seconds). Each research stage stays visible for at least
# STAGE_SECONDS so the user can actually watch the pipeline progress.
STAGE_SECONDS = 4.0
SYNTHESIS_MIN_SECONDS = 5.0
ANALYZE_TIMEOUT = 420          # backend runs 6 LLM calls + market data
PIPELINE_HARD_CAP = 460        # absolute wall-clock ceiling for the UI loop

AGENT_ICONS = {
    "Conversation Agent": "💬",
    "Technical Analyst": "📈",
    "Fundamental Analyst": "💼",
    "News & Sentiment Agent": "📰",
    "Risk Assessment Agent": "⚠️",
    "Synthesis Agent": "🧠",
}

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-banner">
  <h1>📈 FinPilot</h1>
  <p>Autonomous Multi-Agent AI Financial Research Platform</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "query"          # query | clarify | analyzing | done
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "result" not in st.session_state:
    st.session_state.result = None
if "query" not in st.session_state:
    st.session_state.query = ""
if "detected_ticker" not in st.session_state:
    st.session_state.detected_ticker = None
if "detected_company" not in st.session_state:
    st.session_state.detected_company = None


def reset():
    for k in ["stage", "questions", "answers", "result", "query",
              "detected_ticker", "detected_company"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


# ---------------------------------------------------------------------------
# Stage 1 — Query Input
# ---------------------------------------------------------------------------
if st.session_state.stage == "query":
    st.markdown("### 🔍 What would you like to research?")
    query = st.text_area(
        label="query_input",
        label_visibility="collapsed",
        placeholder='e.g. "Should I invest in Apple for the long term?" or "Analyse TSLA for a growth portfolio"',
        height=100,
        key="query_input_box",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        start_btn = st.button("🚀 Analyse", use_container_width=True)

    if start_btn:
        if not query.strip():
            st.warning("Please enter a query to get started.")
        else:
            st.session_state.query = query.strip()
            with st.spinner("Thinking about your query..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/clarify",
                        json={"query": st.session_state.query},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.questions = data.get("questions", [])
                    st.session_state.detected_ticker = data.get("detected_ticker")
                    st.session_state.detected_company = data.get("detected_company")
                    st.session_state.stage = "clarify"
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure FastAPI is running at http://localhost:8000")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.markdown(
        """
<div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:0.5rem;">
  <div class="card" style="flex:1; min-width:180px">
    <div class="card-title">📈 Technical Analysis</div>
    <div class="card-body">RSI, MACD, moving averages & trend signals</div>
  </div>
  <div class="card" style="flex:1; min-width:180px">
    <div class="card-title">💼 Fundamental Analysis</div>
    <div class="card-body">Revenue, margins, P/E ratio & competitive moat</div>
  </div>
  <div class="card" style="flex:1; min-width:180px">
    <div class="card-title">📰 News Sentiment</div>
    <div class="card-body">Latest catalysts, analyst changes & market mood</div>
  </div>
  <div class="card" style="flex:1; min-width:180px">
    <div class="card-title">⚠️ Risk Assessment</div>
    <div class="card-body">Market, sector & company-specific risk scoring</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Stage 2 — Clarification
# ---------------------------------------------------------------------------
elif st.session_state.stage == "clarify":
    if st.session_state.detected_company or st.session_state.detected_ticker:
        label = st.session_state.detected_company or ""
        ticker = st.session_state.detected_ticker or ""
        tag = f"**{label}**  `{ticker}`" if label and ticker else f"**{label or ticker}**"
        st.success(f"🎯 Detected: {tag}")

    st.markdown(f"**Your Query:** *{st.session_state.query}*")
    st.markdown("### 🤔 A couple of quick questions before we dive in")
    st.markdown("*Help us tailor the analysis to your goals:*")
    st.markdown("")

    answers = {}
    for i, q in enumerate(st.session_state.questions):
        answers[q] = st.text_input(
            label=f"q_{i}",
            label_visibility="collapsed",
            placeholder=q,
            key=f"clarify_ans_{i}",
        )

    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        proceed_btn = st.button("▶ Run Analysis", use_container_width=True)
    with col2:
        skip_btn = st.button("Skip →", use_container_width=True)

    if proceed_btn or skip_btn:
        if proceed_btn:
            # Collect answers (skip empty ones)
            st.session_state.answers = {
                q: a.strip()
                for q, a in answers.items()
                if a.strip()
            }
        else:
            st.session_state.answers = {}
        st.session_state.stage = "analyzing"
        st.rerun()

    st.button("← Back", on_click=reset)

# ---------------------------------------------------------------------------
# Stage 3 — Research Pipeline (paced animation, gated on real completion)
# ---------------------------------------------------------------------------
elif st.session_state.stage == "analyzing":
    st.markdown("### ⚙️ Multi-Agent Research Pipeline Running...")

    agents = [
        ("Conversation Agent", "Parsing query and building the investor mandate"),
        ("Technical Analyst", "Fetching price history and computing RSI / MACD / moving averages"),
        ("Fundamental Analyst", "Pulling valuation ratios, margins and balance-sheet data"),
        ("News & Sentiment Agent", "Searching reputable outlets for recent coverage"),
        ("Risk Assessment Agent", "Scoring market, sector, company and macro risk"),
    ]

    agent_placeholder = st.empty()
    status_placeholder = st.empty()

    def render_agents(completed: int, synthesis_state: str = "queued"):
        """completed = number of finished research agents.
        synthesis_state = queued | running | done"""
        rows = ""
        for idx, (name, _desc) in enumerate(agents):
            icon = AGENT_ICONS.get(name, "🔄")
            if idx < completed:
                pill = '<span class="agent-pill">✓ Done</span>'
                name_style = ""
            elif idx == completed:
                pill = ('<span class="agent-pill" style="background:rgba(255,193,7,0.15);'
                        'border-color:rgba(255,193,7,0.4);color:#ffd740;">⟳ Running</span>')
                name_style = ""
            else:
                pill = ('<span class="agent-pill" style="background:rgba(100,100,100,0.1);'
                        'border-color:rgba(100,100,100,0.2);color:#607d8b;">Queued</span>')
                name_style = "color:#546e7a;"
            rows += (
                f'<div class="agent-row"><span style="{name_style}" class="agent-name">'
                f'{icon} {name}</span>{pill}</div>'
            )

        # Final synthesis row — never shown as done until the API actually returns
        if synthesis_state == "running":
            synth_pill = ('<span class="agent-pill" style="background:rgba(33,150,243,0.15);'
                          'border-color:rgba(33,150,243,0.45);color:#64b5f6;">⟳ Synthesizing</span>')
            synth_style = ""
        elif synthesis_state == "done":
            synth_pill = '<span class="agent-pill">✓ Done</span>'
            synth_style = ""
        else:
            synth_pill = ('<span class="agent-pill" style="background:rgba(100,100,100,0.1);'
                          'border-color:rgba(100,100,100,0.2);color:#607d8b;">Queued</span>')
            synth_style = "color:#546e7a;"
        rows += (
            f'<div class="agent-row"><span style="{synth_style}" class="agent-name">'
            f'🧠 Synthesis Agent</span>{synth_pill}</div>'
        )

        agent_placeholder.markdown(
            f'<div class="card">{rows}</div>', unsafe_allow_html=True
        )

    # --- kick the real backend call off immediately, in the background ----
    payload = {
        "query": st.session_state.query,
        "clarification_answers": st.session_state.answers or None,
        "ticker": st.session_state.detected_ticker,
        "target_company": st.session_state.detected_company,
    }
    outcome = {}

    def _call_backend():
        try:
            resp = requests.post(
                f"{API_BASE}/analyze",
                json=payload,
                timeout=ANALYZE_TIMEOUT,
            )
            resp.raise_for_status()
            outcome["data"] = resp.json()
        except requests.exceptions.ConnectionError:
            outcome["error"] = (
                "Cannot connect to backend. Make sure FastAPI is running at "
                "http://localhost:8000"
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            outcome["error"] = str(exc)

    worker = threading.Thread(target=_call_backend, daemon=True)
    worker.start()

    # --- research stages: each one stays visible for a deliberate moment --
    for idx, (name, desc) in enumerate(agents):
        render_agents(idx)
        status_placeholder.caption(f"{AGENT_ICONS.get(name, '🔄')}  {desc}…")
        time.sleep(STAGE_SECONDS)
        if "error" in outcome:
            break

    if "error" not in outcome:
        render_agents(len(agents), synthesis_state="running")

        # --- synthesis stage: held until the report genuinely exists ------
        started = time.time()
        while True:
            elapsed = time.time() - started
            done = not worker.is_alive()
            if done and elapsed >= SYNTHESIS_MIN_SECONDS:
                break
            if elapsed > PIPELINE_HARD_CAP:
                break
            status_placeholder.caption(
                f"🧠  Preparing final report — synthesizing five research briefs… "
                f"({int(elapsed)}s)"
            )
            time.sleep(0.5)

    # --- resolve -----------------------------------------------------------
    if "data" in outcome:
        render_agents(len(agents), synthesis_state="done")
        status_placeholder.caption("✅  Research note ready.")
        st.session_state.result = outcome["data"]
        st.session_state.stage = "done"
        time.sleep(0.6)
        st.rerun()
    elif "error" in outcome:
        status_placeholder.empty()
        st.error(f"❌ Analysis failed: {outcome['error']}")
        st.button("← Try Again", on_click=reset)
    else:
        status_placeholder.empty()
        st.error("❌ The research pipeline did not finish in time. Please try again.")
        st.button("← Try Again", on_click=reset)

# ---------------------------------------------------------------------------
# Stage 4 — Results
# ---------------------------------------------------------------------------
elif st.session_state.stage == "done":
    r = st.session_state.result

    # Header row
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"## 📊 {r['company']}  `{r['ticker']}`")
    with col_b:
        conf = r.get("confidence", "Medium")
        conf_class = f"confidence-{conf.lower()}"
        st.markdown(
            f"<p style='text-align:right; padding-top:1rem;'>Confidence: <span class='{conf_class}'>{conf}</span></p>",
            unsafe_allow_html=True,
        )

    # Agent pipeline summary
    st.markdown("#### 🤖 Agent Pipeline")
    rows = ""
    for ag in r.get("agent_statuses", []):
        icon = AGENT_ICONS.get(ag["name"], "✅")
        ag_status = ag.get("status", "completed")
        if ag_status == "completed":
            pill = '<span class="agent-pill">✓ Done</span>'
        elif ag_status == "partial":
            pill = ('<span class="agent-pill" style="background:rgba(255,193,7,0.15);'
                    'border-color:rgba(255,193,7,0.4);color:#ffd740;">◐ Partial</span>')
        else:
            pill = ('<span class="agent-pill" style="background:rgba(255,82,82,0.15);'
                    'border-color:rgba(255,82,82,0.4);color:#ff5252;">✕ Failed</span>')
        rows += (
            f'<div class="agent-row">'
            f'{pill}'
            f'<span class="agent-name">{icon} {ag["name"]}</span>'
            f'<span class="agent-summary"> — {ag["summary"]}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tabs for report sections
    tabs = st.tabs([
        "📋 Summary",
        "📈 Technical",
        "💼 Fundamental",
        "📰 News & Sentiment",
        "⚠️ Risk",
        "🎯 Verdict",
        "📄 Full Report",
    ])

    with tabs[0]:
        if r.get("investor_profile"):
            st.markdown(
                f'<div class="card"><div class="card-title">Investor Mandate</div>'
                f'<div class="card-body">{r["investor_profile"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="card"><div class="card-title">Executive Summary</div>'
            f'<div class="card-body">{r["executive_summary"]}</div></div>',
            unsafe_allow_html=True,
        )
        metrics = r.get("key_metrics") or {}
        if metrics:
            metric_html = "".join(
                f'<div class="agent-row"><span class="agent-name">{k}</span>'
                f'<span class="agent-summary">{v}</span></div>'
                for k, v in metrics.items()
            )
            st.markdown(
                f'<div class="card"><div class="card-title">Verified Key Metrics</div>'
                f'{metric_html}</div>',
                unsafe_allow_html=True,
            )
        if r.get("data_availability"):
            st.caption(f"Data availability — {r['data_availability']}")

    with tabs[1]:
        st.markdown(
            f'<div class="card"><div class="card-title">Technical Outlook</div>'
            f'<div class="card-body">{r["technical_outlook"]}</div></div>',
            unsafe_allow_html=True,
        )

    with tabs[2]:
        st.markdown(
            f'<div class="card"><div class="card-title">Fundamental View</div>'
            f'<div class="card-body">{r["fundamental_view"]}</div></div>',
            unsafe_allow_html=True,
        )
        if r.get("valuation_view"):
            st.markdown(
                f'<div class="card"><div class="card-title">Valuation</div>'
                f'<div class="card-body">{r["valuation_view"]}</div></div>',
                unsafe_allow_html=True,
            )

    with tabs[3]:
        st.markdown(
            f'<div class="card"><div class="card-title">News & Market Sentiment</div>'
            f'<div class="card-body">{r["news_sentiment"]}</div></div>',
            unsafe_allow_html=True,
        )
        if r.get("sources"):
            st.caption("Sources used: " + ", ".join(r["sources"]))

    with tabs[4]:
        st.markdown(
            f'<div class="card"><div class="card-title">Risk Assessment</div>'
            f'<div class="card-body">{r["risk_assessment"]}</div></div>',
            unsafe_allow_html=True,
        )

    with tabs[5]:
        st.markdown(
            f'<div class="card"><div class="card-title">Final Verdict</div>'
            f'<div class="card-body">{r["final_verdict"]}</div></div>',
            unsafe_allow_html=True,
        )
        if r.get("scenario_analysis"):
            st.markdown(
                f'<div class="card"><div class="card-title">Scenario Analysis</div>'
                f'<div class="card-body">{r["scenario_analysis"]}</div></div>',
                unsafe_allow_html=True,
            )
        if r.get("catalysts_to_watch"):
            st.markdown(
                f'<div class="card"><div class="card-title">Catalysts To Watch</div>'
                f'<div class="card-body">{r["catalysts_to_watch"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.caption(r.get("disclaimer", ""))

    with tabs[6]:
        st.markdown(r.get("raw_markdown", ""), unsafe_allow_html=False)
        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=r.get("raw_markdown", ""),
            file_name=f"finpilot_{r['ticker']}_report.md",
            mime="text/markdown",
        )
        briefs = r.get("research_briefs") or {}
        if briefs:
            st.markdown("#### 🔬 Raw Specialist Briefs")
            for name, brief in briefs.items():
                with st.expander(f"{AGENT_ICONS.get(name, '🔍')} {name}"):
                    st.markdown(brief)

    st.markdown("---")
    st.button("🔄 New Analysis", on_click=reset)