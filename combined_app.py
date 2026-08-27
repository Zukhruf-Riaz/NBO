"""
combined_app.py
───────────────
Combines the ADS Agent, Segmentation Agent, and Pipeline overview
in one Streamlit app with three top-level tabs.

Tabs
────
1. 🏦 ADS Agent         — build your Analytical Data Set
2. 📊 Segmentation Agent — run customer segmentation on the ADS
3. 🔄 Pipeline          — visual, plain-English overview of the full workflow

Fixes / improvements applied
─────────────────────────────
- Three tabs at the top: ADS Agent | Segmentation Agent | Pipeline
- ADS SQL phase shows only "Proceed to Segmentation" — no Run SQL / download.
  Clicking it resets Segmentation to connection phase and switches to that tab.
- Segmentation algorithm / role phases use RADIO BUTTONS (not tabs).
- LLM feature selection populates the multiselect dropdown for BOTH roles.
- Pipeline tab shows a non-technical, visual flow diagram using HTML/CSS.
- Sidebar renders only the steps for the currently active agent.

Run
───
    streamlit run combined_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="NBO Analytics Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ads_agent import (
    init_ads_state,
    ads_render_sidebar,
    ads_phase_connection,
    ads_phase_context,
    ads_phase_tables_goal,
    ads_phase_features,
    ads_phase_sql,
)
from segmentation_agent import (
    init_seg_state,
    seg_render_sidebar,
    seg_render,
    SEG_DEFAULTS,
)

# ── Constants ─────────────────────────────────────────────────────────────────
_ADS      = "ADS Agent"
_SEG      = "Segmentation Agent"
_PIPELINE = "Pipeline"
_AGENTS   = [_ADS, _SEG, _PIPELINE]

_TAB_LABELS = {
    _ADS:      "🏦  ADS Agent",
    _SEG:      "📊  Segmentation Agent",
    _PIPELINE: "🔄  Pipeline",
}


# ── Session-state boot ────────────────────────────────────────────────────────

def _init():
    if "active_agent" not in st.session_state:
        st.session_state.active_agent = _ADS


# ── Handoff callback ──────────────────────────────────────────────────────────

def _on_proceed_to_seg():
    """
    Called by ads_phase_sql when "Proceed to Segmentation" is clicked.
    Fully resets the Segmentation agent and switches to its tab.
    """
    for k, v in SEG_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.active_agent = _SEG
    st.rerun()


# ── ADS dispatcher ────────────────────────────────────────────────────────────

def _run_ads():
    phase = st.session_state.get("ads_phase", "connection")
    if phase == "sql":
        ads_phase_sql(on_proceed_to_seg=_on_proceed_to_seg)
    else:
        {
            "connection":  ads_phase_connection,
            "context":     ads_phase_context,
            "tables_goal": ads_phase_tables_goal,
            "features":    ads_phase_features,
        }.get(phase, ads_phase_connection)()


# ── Pipeline renderer ─────────────────────────────────────────────────────────

def _render_pipeline():
    st.title("🔄 Analytics Pipeline — How It All Works")
    st.markdown(
        "This page shows you the **complete journey** from raw database tables "
        "to actionable customer segments. Each coloured block is a step you complete "
        "inside one of the two agents above."
    )

    # ── HTML / CSS pipeline diagram ──────────────────────────────────────────
    pipeline_html = """
<style>
  .pipe-wrap {
    font-family: 'Segoe UI', Arial, sans-serif;
    padding: 10px 0 30px 0;
  }

  /* ── Phase label above each group ──────────────────────────────── */
  .phase-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748b;
    margin: 28px 0 6px 0;
  }

  /* ── Agent header banner ────────────────────────────────────────── */
  .agent-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    border-radius: 10px 10px 0 0;
    font-size: 16px;
    font-weight: 700;
    color: #fff;
    margin-top: 32px;
  }
  .banner-ads  { background: linear-gradient(90deg, #0284c7, #0ea5e9); }
  .banner-seg  { background: linear-gradient(90deg, #7c3aed, #a855f7); }

  /* ── Steps container ────────────────────────────────────────────── */
  .steps-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    border: 2px solid #e2e8f0;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 18px 14px 18px 14px;
    background: #f8fafc;
  }

  /* ── Individual step card ────────────────────────────────────────── */
  .step-card {
    position: relative;
    background: #fff;
    border-radius: 10px;
    padding: 14px 16px 12px 16px;
    min-width: 160px;
    max-width: 200px;
    flex: 1 1 160px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin: 6px;
  }
  .step-card.ads-card  { border-top: 4px solid #0284c7; }
  .step-card.seg-card  { border-top: 4px solid #7c3aed; }

  .step-num {
    display: inline-block;
    width: 22px; height: 22px;
    border-radius: 50%;
    font-size: 11px;
    font-weight: 800;
    text-align: center;
    line-height: 22px;
    margin-bottom: 6px;
    color: #fff;
  }
  .num-ads { background: #0284c7; }
  .num-seg { background: #7c3aed; }

  .step-title {
    font-size: 13px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 4px;
    line-height: 1.3;
  }
  .step-desc {
    font-size: 11.5px;
    color: #475569;
    line-height: 1.5;
  }

  /* ── Arrow between steps ─────────────────────────────────────────── */
  .arrow {
    display: flex;
    align-items: center;
    color: #94a3b8;
    font-size: 22px;
    padding: 0 2px;
    align-self: center;
    margin: 6px 0;
  }

  /* ── Handoff connector between ADS and Segmentation ─────────────── */
  .handoff {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin: 18px 0 4px 0;
    padding: 12px 20px;
    background: linear-gradient(90deg, #e0f2fe, #ede9fe);
    border-radius: 10px;
    border: 1.5px dashed #7c3aed;
  }
  .handoff-icon { font-size: 24px; }
  .handoff-text {
    font-size: 13px;
    font-weight: 600;
    color: #1e293b;
  }
  .handoff-sub {
    font-size: 11.5px;
    color: #64748b;
    margin-top: 2px;
  }

  /* ── Output badge ────────────────────────────────────────────────── */
  .output-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #dcfce7;
    border: 1.5px solid #16a34a;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #166534;
    margin: 18px 6px 0 6px;
  }

  /* ── Legend ──────────────────────────────────────────────────────── */
  .legend {
    display: flex;
    gap: 20px;
    margin-top: 28px;
    flex-wrap: wrap;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    color: #475569;
  }
  .legend-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }
</style>

<div class="pipe-wrap">

  <!-- ═══════════════════════════════ ADS AGENT ═══════════════════════════ -->
  <div class="agent-banner banner-ads">
    🏦 &nbsp;Step 1 — ADS Agent &nbsp;·&nbsp; Build Your Analytical Data Set
  </div>
  <div class="steps-container">

    <div class="step-card ads-card">
      <div class="step-num num-ads">1</div>
      <div class="step-title">Connect to Database</div>
      <div class="step-desc">Paste your database connection string (SQLite, SQL Server, Azure Fabric, PostgreSQL, MySQL). The agent lists all available tables and their row counts.</div>
    </div>

    <div class="arrow">›</div>

    <div class="step-card ads-card">
      <div class="step-num num-ads">2</div>
      <div class="step-title">Describe the Database</div>
      <div class="step-desc">Upload a data dictionary, answer a few questions in chat, or skip. This helps the AI understand what your columns mean so it recommends better features.</div>
    </div>

    <div class="arrow">›</div>

    <div class="step-card ads-card">
      <div class="step-num num-ads">3</div>
      <div class="step-title">Select Tables & Goal</div>
      <div class="step-desc">Pick the tables you want to combine and describe what you're trying to achieve — e.g. "identify customers likely to open a deposit in the next 90 days".</div>
    </div>

    <div class="arrow">›</div>

    <div class="step-card ads-card">
      <div class="step-num num-ads">4</div>
      <div class="step-title">AI Feature Recommendations</div>
      <div class="step-desc">The AI suggests the most useful columns and derived features (like ratios, counts, flags) for your goal. You review and approve the final list.</div>
    </div>

    <div class="arrow">›</div>

    <div class="step-card ads-card">
      <div class="step-num num-ads">5</div>
      <div class="step-title">Generate & Refine SQL</div>
      <div class="step-desc">The AI writes a SQL query that extracts all approved features into a single flat table — your ADS. Chat with the AI to refine the query until satisfied.</div>
    </div>

  </div>

  <!-- ══════════════════════════ HANDOFF CONNECTOR ════════════════════════ -->
  <div class="handoff">
    <span class="handoff-icon">🔗</span>
    <div>
      <div class="handoff-text">Proceed to Segmentation →</div>
      <div class="handoff-sub">Click the "Proceed to Segmentation" button at the bottom of Step 5 to hand off your ADS directly to the Segmentation Agent.</div>
    </div>
  </div>

  <!-- ═══════════════════════ SEGMENTATION AGENT ══════════════════════════ -->
  <div class="agent-banner banner-seg">
    📊 &nbsp;Step 2 — Segmentation Agent &nbsp;·&nbsp; Discover Customer Groups
  </div>
  <div class="steps-container">

    <div class="step-card seg-card">
      <div class="step-num num-seg">1</div>
      <div class="step-title">Connect to Database</div>
      <div class="step-desc">Paste the connection string to your ADS database (same or different database as above). Optionally upload column description documents.</div>
    </div>

    <div class="arrow">›</div>

    <div class="step-card seg-card">
      <div class="step-num num-seg">2</div>
      <div class="step-title">Select Table & Goal</div>
      <div class="step-desc">Choose the ADS table you built and describe what segmentation should achieve — e.g. "group customers by spending behaviour". The AI uses this to pick better features.</div>
    </div>

    <div class="arrow">›</div>

    <div class="step-card seg-card">
      <div class="step-num num-seg">3</div>
      <div class="step-title">Choose Algorithm</div>
      <div class="step-desc"><strong>K-Means</strong> — you decide how many groups (simple, fast). <strong>DBSCAN</strong> — the data decides the number of groups; also flags unusual customers automatically.</div>
    </div>

    <div class="arrow">›</div>

    <div class="step-card seg-card">
      <div class="step-num num-seg">4</div>
      <div class="step-title">Select Your Role</div>
      <div class="step-desc"><strong>Data Scientist</strong> — review diagnostic charts, manually tune features and parameters, compare runs. <strong>Business User</strong> — fully automatic; plain English summaries and a Q&amp;A chat.</div>
    </div>

    <div class="arrow">›</div>

    <div class="step-card seg-card">
      <div class="step-num num-seg">5</div>
      <div class="step-title">Review Segments</div>
      <div class="step-desc">The AI names and describes each customer group. You can ask questions in plain English, customise segment boundaries using business rules, and download a labeled Excel file.</div>
    </div>

  </div>

  <!-- ══════════════════════════════ OUTPUTS ══════════════════════════════ -->
  <div style="margin-top:24px;">
    <div class="phase-label">What you get at the end</div>
    <div style="display:flex; flex-wrap:wrap;">
      <div class="output-badge">📄 ADS SQL query ready to run</div>
      <div class="output-badge">📊 Customer segments with names &amp; profiles</div>
      <div class="output-badge">📥 Excel with every customer labeled</div>
      <div class="output-badge">💬 Chat Q&amp;A about your segments</div>
    </div>
  </div>

  <!-- ══════════════════════════════ LEGEND ═══════════════════════════════ -->
  <div class="legend">
    <div class="legend-item">
      <div class="legend-dot" style="background:#0284c7;"></div>
      ADS Agent step
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#7c3aed;"></div>
      Segmentation Agent step
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#16a34a;"></div>
      Output / deliverable
    </div>
    <div class="legend-item" style="gap:7px;">
      <span style="color:#7c3aed; font-size:13px;">- - -</span>
      Handoff between agents
    </div>
  </div>

</div>
"""
    st.components.v1.html(pipeline_html, height=980, scrolling=True)

    # ── Plain-English step guide below the diagram ───────────────────────────
    st.divider()
    st.subheader("Quick-Start Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏦 ADS Agent — 5 Steps")
        st.markdown("""
1. **Connect** — paste your database connection string.
2. **Context** — upload a data dictionary or describe your database in a short chat so the AI understands your columns.
3. **Tables & Goal** — pick which tables to include and write one sentence describing what you want to predict or understand.
4. **Feature Recommendations** — the AI lists the most useful columns and calculated metrics. Tick or untick as you like; you can also ask for custom derived features in the chat.
5. **SQL Generation** — the AI writes the query. Chat with it to adjust joins, filters, or column aliases until the SQL matches your needs, then click **Proceed to Segmentation**.
""")

    with col2:
        st.markdown("#### 📊 Segmentation Agent — 5 Steps")
        st.markdown("""
1. **Connect** — paste the connection string to the database containing your ADS table (often the same one).
2. **Table & Goal** — select your ADS table and describe the segmentation objective in plain language.
3. **Algorithm** — choose **K-Means** if you want a fixed number of groups, or **DBSCAN** if you want the data to decide.
4. **Role** — choose **Data Scientist** for full control over features and parameters, or **Business User** for a fully automatic, jargon-free experience.
5. **Analyse** — review AI-named segments, ask questions in the chat, create custom groups using plain-English rules, and download the labeled Excel file.
""")

    st.info(
        "💡 **Tip:** You don't have to use both agents together. "
        "You can run the Segmentation Agent standalone on any existing table in your database — "
        "no ADS needed."
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar():
    """Render the sidebar for the currently active agent."""
    st.sidebar.title("🏦 NBO Analytics")
    st.sidebar.caption("ADS · Segmentation Platform")
    st.sidebar.divider()

    active = st.session_state.active_agent
    if active == _ADS:
        ads_render_sidebar()
    elif active == _SEG:
        seg_render_sidebar()
    else:
        # Pipeline tab — show a simple info panel
        st.sidebar.subheader("📋 Pipeline Overview")
        st.sidebar.caption(
            "This tab shows the full analytics workflow in a visual, "
            "non-technical format.\n\n"
            "Use the **ADS Agent** tab to build your data set, then the "
            "**Segmentation Agent** tab to discover customer groups."
        )
        st.sidebar.divider()
        st.sidebar.markdown("**ADS Agent** — 5 steps")
        for label in [
            "1. Connect to DB",
            "2. Describe the DB",
            "3. Select Tables & Goal",
            "4. Approve Features",
            "5. Generate SQL",
        ]:
            st.sidebar.caption(f"○ {label}")
        st.sidebar.divider()
        st.sidebar.markdown("**Segmentation Agent** — 5 steps")
        for label in [
            "1. Connect to DB",
            "2. Select Table & Goal",
            "3. Choose Algorithm",
            "4. Select Role",
            "5. Review Segments",
        ]:
            st.sidebar.caption(f"○ {label}")


# ── Tab switcher ───────────────────────────────────────────────────────────────

def _render_tabs():
    """
    Render the three top-level tabs.
    Uses st.radio styled as tabs (same approach as before) so on_change fires
    synchronously before any tab body executes — preventing duplicate widget keys.
    """
    st.markdown(
        """
        <style>
        /* Hide the radio label */
        div[data-testid="stRadio"] > label { display: none !important; }
        /* Style radio options as tab bar */
        div[data-testid="stRadio"] > div {
            display: flex;
            gap: 0;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 1rem;
        }
        div[data-testid="stRadio"] > div > label {
            display: flex !important;
            align-items: center;
            padding: 8px 24px;
            cursor: pointer;
            border: 1px solid #e2e8f0;
            border-bottom: none;
            border-radius: 6px 6px 0 0;
            font-weight: 500;
            background: #f8fafc;
            margin-right: 4px;
            transition: background 0.15s;
        }
        div[data-testid="stRadio"] > div > label:has(input:checked) {
            background: white;
            border-bottom: 2px solid white;
            margin-bottom: -2px;
            color: #0284c7;
        }
        div[data-testid="stRadio"] > div > label > div:first-child {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _on_agent_change():
        st.session_state.active_agent = st.session_state["_agent_radio"]

    st.radio(
        "Switch agent",
        options=_AGENTS,
        index=_AGENTS.index(st.session_state.active_agent),
        key="_agent_radio",
        on_change=_on_agent_change,
        format_func=lambda x: _TAB_LABELS[x],
        horizontal=True,
        label_visibility="collapsed",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init()
    init_ads_state()
    init_seg_state()

    # 1. Sidebar (rendered once, before tab bodies to avoid duplicate widget keys)
    _render_sidebar()

    # 2. Top-level tab switcher
    _render_tabs()

    # 3. Render the active tab
    active = st.session_state.active_agent

    if active == _ADS:
        _run_ads()

    elif active == _SEG:
        # render_sidebar=False — sidebar already rendered above
        seg_render(pre_loaded=None, render_sidebar=False)

    else:
        _render_pipeline()


if __name__ == "__main__":
    main()
