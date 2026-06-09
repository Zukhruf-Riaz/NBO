"""
ads_agent.py
------------
ADS Creation Agent — all phase functions.
All session-state keys are prefixed with "ads_" to avoid collisions with the
Segmentation agent when both run in the same Streamlit app.

Imported and called by combined_app.py.

Fix: ads_phase_sql() shows ONLY the "Proceed to Segmentation" button.
     No "Run SQL", no download button.  Clicking the button calls
     on_proceed_to_seg() with no arguments (combined_app handles the reset).
"""

import re
import io
import json
import os
import requests
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, inspect as sa_inspect
from sqlalchemy.engine import Engine

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL       = "qwen2.5:7b"
MODEL_SMALL = "qwen2.5:3b"
OLLAMA_URL  = "http://localhost:11434/api/chat"
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

SYSTEM_ANALYST = (
    "You are a senior banking data analyst. Be precise and concise. "
    "Always reference actual column names from the schema provided."
)
SYSTEM_SQL = (
    "You are an expert SQL developer for banking analytics. "
    "Write clean, well-commented SQL. You are a collaborator — not the final authority. "
    "The data engineer will refine your queries. "
    "When asked to change something, return the full updated SQL only — no explanation."
)
SYSTEM_CONTEXT = (
    "You are helping build an understanding of a banking database for an analytics project. "
    "Ask focused, one-at-a-time questions to understand what each table represents, "
    "what the key business entities are, and how tables relate. Be conversational."
)

FABRIC_SUFFIXES = (
    ".datawarehouse.fabric.microsoft.com",
    ".database.fabric.microsoft.com",
)

_CHUNK_SIZE = 8000

# ── Default session-state values (all prefixed with ads_) ────────────────────
ADS_DEFAULTS = {
    "ads_phase":             "connection",
    "ads_furthest_phase":    "connection",
    "ads_conn_string":       "",
    "ads_db_path":           "",
    "ads_all_tables":        [],
    "ads_brief_schema":      "",
    "ads_db_context":        "",
    "ads_context_source":    None,
    "ads_context_chat":      [],
    "ads_selected_tables":   [],
    "ads_goal":              "",
    "ads_schema_block":      "",
    "ads_recommendation":    "",
    "ads_approved_features": {},
    "ads_extra_enrichment":  "",
    "ads_sql_messages":           [],
    "ads_current_sql":            "",
    "ads_derived_features":       [],
    "ads_derived_features_chat":  [],
    "ads_business_logic_doc":     "",
}


def init_ads_state():
    for k, v in ADS_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── DB helpers ────────────────────────────────────────────────────────────────

def _engine_from_adonet(adonet: str):
    parts = {}
    for segment in adonet.split(";"):
        segment = segment.strip()
        if "=" in segment:
            key, _, val = segment.partition("=")
            parts[key.strip().lower()] = val.strip()
    server   = parts.get("data source",     parts.get("server", ""))
    database = parts.get("initial catalog", parts.get("database", ""))
    auth     = parts.get("authentication", "")
    encrypt  = "yes" if parts.get("encrypt", "true").lower() == "true" else "no"
    trust    = "yes" if parts.get("trust server certificate", "false").lower() == "true" else "no"
    timeout  = parts.get("connect timeout", "30")
    odbc_parts = [
        "DRIVER={ODBC Driver 17 for SQL Server}",
        f"SERVER={server}", f"DATABASE={database}",
        f"Encrypt={encrypt}", f"TrustServerCertificate={trust}",
        f"Connection Timeout={timeout}",
    ]
    if auth:
        odbc_parts.append(f"Authentication={auth.replace(' ', '')}")
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={quote_plus(';'.join(odbc_parts) + ';')}"
    )


def make_engine(conn_str: str):
    s = conn_str.strip()
    if s.lower().startswith("data source=") or "initial catalog=" in s.lower():
        return _engine_from_adonet(s)
    if not s.lower().startswith(("sqlite", "postgresql", "mysql", "mssql", "oracle")):
        if s.endswith((".db", ".sqlite", ".db3")):
            s = "sqlite:///" + s.replace("\\", "/")
    if s.lower().startswith("sqlite"):
        return create_engine(s, connect_args={"check_same_thread": False})
    return create_engine(s)


_engine_hash = {Engine: lambda e: str(e.url)}


@st.cache_resource
def ads_open_db(conn_str: str):
    return make_engine(conn_str)


@st.cache_data(hash_funcs=_engine_hash)
def ads_db_tables(engine) -> list:
    return sorted(sa_inspect(engine).get_table_names())


@st.cache_data(hash_funcs=_engine_hash)
def ads_db_columns(engine, table: str) -> list:
    return [col["name"] for col in sa_inspect(engine).get_columns(table)]


@st.cache_data(hash_funcs=_engine_hash)
def ads_db_row_count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()


@st.cache_data(hash_funcs=_engine_hash)
def ads_get_sample_rows(engine, table: str, n: int = 2) -> list:
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f'SELECT * FROM "{table}" LIMIT {n}'))
            return [dict(row._mapping) for row in result]
    except Exception:
        return []


@st.cache_data(hash_funcs=_engine_hash)
def ads_get_column_stats(engine, table: str, cols: list) -> dict:
    if not cols:
        return {}
    cols = list(cols[:30])
    try:
        with engine.connect() as conn:
            total = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar() or 1
            select_parts = ", ".join(
                f'COUNT("{c}") AS nn_{i}, COUNT(DISTINCT "{c}") AS dc_{i}'
                for i, c in enumerate(cols)
            )
            row = conn.execute(text(f'SELECT {select_parts} FROM "{table}"')).fetchone()
            stats = {}
            for i, col in enumerate(cols):
                nn = row[i * 2]     or 0
                dc = row[i * 2 + 1] or 0
                stats[col] = {"non_null_pct": round(nn / total * 100), "distinct_count": dc}
        return stats
    except Exception:
        return {}


def ads_validate_formula(engine, formula: str, selected_tables: list) -> str | None:
    test_tables = selected_tables or ads_db_tables(engine)
    last_err = None
    for t in test_tables:
        try:
            with engine.connect() as conn:
                conn.execute(text(f'SELECT {formula} FROM "{t}" LIMIT 1'))
            return None
        except Exception as e:
            last_err = str(e)
    return last_err


def ads_build_brief_schema(engine) -> str:
    lines = []
    for t in ads_db_tables(engine):
        cols    = ads_db_columns(engine, t)
        n       = ads_db_row_count(engine, t)
        preview = cols[:8]
        suffix  = f" +{len(cols)-8} more" if len(cols) > 8 else ""
        lines.append(f"- {t} ({n:,} rows): {', '.join(preview)}{suffix}")
    return "\n".join(lines)


@st.cache_data(hash_funcs=_engine_hash)
def ads_build_schema_block(engine, selected: list) -> str:
    lines = []
    for t in ads_db_tables(engine):
        cols = ads_db_columns(engine, t)
        n    = ads_db_row_count(engine, t)
        if t in selected:
            stats = ads_get_column_stats(engine, t, tuple(cols))
            if stats:
                col_parts = []
                for c in cols:
                    s = stats.get(c)
                    if s:
                        col_parts.append(
                            f"{c} [{s['non_null_pct']}% non-null, {s['distinct_count']} distinct]"
                        )
                    else:
                        col_parts.append(c)
                col_str = ", ".join(col_parts)
            else:
                col_str = ", ".join(cols)
        else:
            preview = cols[:10]
            suffix  = f"  ... +{len(cols)-10} more" if len(cols) > 10 else ""
            col_str = ", ".join(preview) + suffix
        lines.append(f"Table: {t}  ({n:,} rows)\nColumns: {col_str}")
    return "\n\n".join(lines)


# ── Document helpers ──────────────────────────────────────────────────────────

def ads_extract_doc_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith((".txt", ".md")):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            return "[PDF parsing unavailable — run: pip install pypdf]"
    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return "[DOCX parsing unavailable — run: pip install python-docx]"
    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(uploaded_file)
            return df.to_string(index=False)
        except Exception as e:
            return f"[Excel error: {e}]"
    if name.endswith(".csv"):
        return uploaded_file.read().decode("utf-8", errors="ignore")[:6000]
    return "[Unsupported file type]"


def _extract_chunk(chunk_text: str, chunk_num: int, total_chunks: int) -> str:
    prompt = (
        "Extract every feature definition, KPI, business rule, and model requirement "
        "from the document excerpt below as a concise numbered list. "
        "Preserve all distinct metric names, thresholds, and formulas exactly — "
        "do not paraphrase or drop any rule.\n\n"
        f"Document excerpt ({chunk_num}/{total_chunks}):\n{chunk_text}"
    )
    payload = {
        "model":    MODEL_SMALL,
        "messages": [
            {"role": "system", "content": "You are a banking analytics expert. Extract structured rules precisely."},
            {"role": "user",   "content": prompt},
        ],
        "stream":  True,
        "options": {"temperature": 0},
    }
    tokens = []
    placeholder = st.empty()
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=900) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data  = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                tokens.append(token)
                placeholder.markdown("".join(tokens))
            if data.get("done"):
                break
    placeholder.empty()
    return "".join(tokens)


def ads_summarize_doc(raw_text: str) -> str:
    if len(raw_text) <= 4000:
        return raw_text
    chunks = [raw_text[i:i + _CHUNK_SIZE] for i in range(0, len(raw_text), _CHUNK_SIZE)]
    total  = len(chunks)
    chunk_results = []
    try:
        with st.status(
            f"Extracting rules from document ({len(raw_text):,} chars, {total} chunk(s))…",
            expanded=True,
        ) as status:
            for i, chunk in enumerate(chunks, 1):
                status.update(label=f"Processing chunk {i} / {total}…")
                result = _extract_chunk(chunk, i, total)
                chunk_results.append(f"### Chunk {i}/{total}\n{result}")
            if total > 1:
                status.update(label="Merging and deduplicating rules…")
                combined = "\n\n".join(chunk_results)
                merge_prompt = (
                    "Merge these rule extractions into one deduplicated numbered list. "
                    "Keep every distinct rule, KPI, threshold, and formula — "
                    "remove only exact duplicates.\n\n"
                    f"{combined[:14000]}"
                )
                merge_payload = {
                    "model":    MODEL_SMALL,
                    "messages": [
                        {"role": "system", "content": "You are a banking analytics expert. Merge rule lists precisely."},
                        {"role": "user",   "content": merge_prompt},
                    ],
                    "stream":  True, "options": {"temperature": 0},
                }
                merge_tokens = []
                placeholder  = st.empty()
                with requests.post(OLLAMA_URL, json=merge_payload, stream=True, timeout=900) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data  = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            merge_tokens.append(token)
                            placeholder.markdown("".join(merge_tokens))
                        if data.get("done"):
                            break
                placeholder.empty()
                final_rules = "".join(merge_tokens)
            else:
                final_rules = chunk_results[0]
            status.update(label=f"Document fully processed — {total} chunk(s).", state="complete", expanded=False)
        return f"[Extracted from {len(raw_text):,}-char document, {total} chunk(s)]\n\n{final_rules}"
    except Exception:
        return raw_text[:12000]


# ── Derived feature helpers ───────────────────────────────────────────────────

def ads_parse_derived_feature(text: str) -> dict | None:
    name_m    = re.search(r"FEATURE_NAME:\s*(.+)",                          text, re.IGNORECASE)
    formula_m = re.search(r"FORMULA:\s*(.+?)(?:\nDESCRIPTION:|\Z)",         text, re.IGNORECASE | re.DOTALL)
    desc_m    = re.search(r"DESCRIPTION:\s*(.+)",                            text, re.IGNORECASE)
    if name_m and formula_m:
        return {
            "name":        name_m.group(1).strip(),
            "formula":     formula_m.group(1).strip(),
            "description": desc_m.group(1).strip() if desc_m else "",
        }
    return None


def ads_make_features_excel(features_by_table: dict, derived_features: list) -> bytes:
    rows = []
    for table, cols in features_by_table.items():
        for col in cols:
            rows.append({"Type": "Source", "Table": table, "Feature": col,
                         "Formula": "", "Description": ""})
    for df in derived_features:
        rows.append({"Type": "Derived", "Table": "—",
                     "Feature": df["name"], "Formula": df.get("formula", ""),
                     "Description": df.get("description", "")})
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()


# ── Ollama helpers ────────────────────────────────────────────────────────────

def ads_llm_stream(messages: list, system: str):
    payload = {
        "model":    MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream":   True,
        "options":  {"temperature": 0},
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=900) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except Exception as e:
        yield f"\n\n[Error communicating with Ollama: {e}]"


# ── Navigation helpers ────────────────────────────────────────────────────────

def _advance_ads_furthest(phase: str):
    order = ["connection", "context", "tables_goal", "features", "sql"]
    if order.index(phase) > order.index(st.session_state.ads_furthest_phase):
        st.session_state.ads_furthest_phase = phase


def _is_bare_fabric_host(s: str) -> bool:
    sl = s.lower().strip()
    return any(
        sl.endswith(suf) or suf.lstrip(".") in sl
        for suf in FABRIC_SUFFIXES
    ) and not sl.startswith("data source=")


def _build_adonet_from_host(host: str, catalog: str) -> str:
    host = host.strip().rstrip("/")
    if "," in host:
        host = host.split(",")[0].strip()
    return (
        f"Data Source={host},1433;"
        f"Initial Catalog={catalog};"
        f"Authentication=Active Directory Interactive;"
        f"Encrypt=True;Trust Server Certificate=False"
    )


def ads_viewing_past_step_banner():
    order   = ["connection", "context", "tables_goal", "features", "sql"]
    labels  = {
        "connection":  "1. Connect to DB",
        "context":     "2. DB Context",
        "tables_goal": "3. Tables & Goal",
        "features":    "4. Feature Recs",
        "sql":         "5. SQL Generation",
    }
    current  = st.session_state.ads_phase
    furthest = st.session_state.ads_furthest_phase
    if current == furthest:
        return
    cur_idx = order.index(current)
    fur_idx = order.index(furthest)
    if cur_idx >= fur_idx:
        return
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("👁 Viewing a past step. Your work in later steps is still saved.")
    with col2:
        if st.button(f"↷ Back to {labels[furthest]}", type="primary", use_container_width=True,
                     key="ads_back_to_furthest"):
            st.session_state.ads_phase = furthest
            st.rerun()
    st.divider()


def ads_show_history():
    order       = ["connection", "context", "tables_goal", "features", "sql"]
    current_idx = order.index(st.session_state.ads_phase) if st.session_state.ads_phase in order else 0
    if current_idx == 0:
        return
    with st.expander("📋 What was done in previous steps", expanded=False):
        if st.session_state.ads_db_path:
            st.markdown("**Step 1 — Database Connected**")
            st.code(st.session_state.ads_conn_string)
        if current_idx >= 2 and st.session_state.ads_context_source:
            st.markdown("**Step 2 — DB Context**")
            source_label = {
                "document":    "Uploaded document(s)",
                "chat":        "Described via chat",
                "schema_only": "Skipped — schema names only",
            }.get(st.session_state.ads_context_source, st.session_state.ads_context_source)
            st.caption(f"Source: {source_label}")
            if st.session_state.ads_db_context:
                ctx = st.session_state.ads_db_context
                st.text(ctx[:800] + ("…" if len(ctx) > 800 else ""))
        if current_idx >= 3 and st.session_state.ads_selected_tables:
            st.markdown("**Step 3 — Tables & Goal**")
            st.caption(f"Tables: {', '.join(st.session_state.ads_selected_tables)}")
            st.caption(f"Goal: {st.session_state.ads_goal}")
        if current_idx >= 4 and st.session_state.ads_recommendation:
            st.markdown("**Step 4 — Feature Recommendations**")
            st.markdown(st.session_state.ads_recommendation)
            if st.session_state.ads_approved_features:
                st.markdown("*Confirmed features:*")
                for tbl, cols in st.session_state.ads_approved_features.items():
                    st.caption(f"  {tbl}: {', '.join(cols)}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def ads_render_sidebar():
    """Render only the ADS navigation steps in the sidebar (no title/header).
    Called by combined_app.py which owns the sidebar title."""
    phases = [
        ("connection",  "1. Connect to DB"),
        ("context",     "2. DB Context"),
        ("tables_goal", "3. Tables & Goal"),
        ("features",    "4. Feature Recs"),
        ("sql",         "5. SQL Generation"),
    ]
    order    = [p[0] for p in phases]
    current  = st.session_state.ads_phase
    cur_idx  = order.index(current) if current in order else 0
    furthest = st.session_state.ads_furthest_phase
    fur_idx  = order.index(furthest) if furthest in order else 0

    st.sidebar.subheader("ADS Steps")
    for i, (key, label) in enumerate(phases):
        if i == cur_idx:
            st.sidebar.info(f"▶ {label}")
        elif i <= fur_idx:
            icon = "✓" if i < cur_idx else "↷"
            if st.sidebar.button(f"{icon} {label}", key=f"ads_nav_{key}",
                                  use_container_width=True):
                st.session_state.ads_phase = key
                st.rerun()
        else:
            st.sidebar.caption(f"○ {label}")

    if st.session_state.ads_db_path:
        st.sidebar.caption(f"**DB:** `{os.path.basename(st.session_state.ads_db_path)}`")
    if st.session_state.ads_selected_tables:
        st.sidebar.caption(f"**Tables:** {', '.join(st.session_state.ads_selected_tables)}")
    if st.session_state.ads_goal:
        goal_preview = st.session_state.ads_goal[:55]
        st.sidebar.caption(f"**Goal:** {goal_preview}{'…' if len(st.session_state.ads_goal) > 55 else ''}")

    st.sidebar.divider()
    if st.sidebar.button("↩ Reset ADS", use_container_width=True, key="ads_reset"):
        for k, v in ADS_DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ── Phase 1: Connection ───────────────────────────────────────────────────────

def ads_phase_connection():
    st.title("🏦 ADS Agent — Connect to Database")

    with st.expander("Supported connection string formats"):
        st.code("SQLite     :  C:/path/to/file.db")
        st.code("PostgreSQL :  postgresql://user:pass@host:5432/dbname")
        st.code("MySQL      :  mysql+pymysql://user:pass@host:3306/dbname")
        st.code("SQL Server :  mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server")
        st.markdown("**Microsoft Fabric** — paste the ADO.NET string or just the server URL:")
        st.code("ecalzi3o3...datawarehouse.fabric.microsoft.com")

    raw_input = st.text_input(
        "Connection String or Server URL",
        value=st.session_state.ads_conn_string,
        placeholder="Paste server URL, ADO.NET string, or local file path…",
        key="ads_conn_input",
    )

    catalog = ""
    is_bare_fabric = _is_bare_fabric_host(raw_input) if raw_input.strip() else False
    if is_bare_fabric:
        st.info("Fabric server URL detected. Enter the Database / Lakehouse name.")
        catalog = st.text_input("Database / Lakehouse name",
                                placeholder="e.g. NBO_lakehouse",
                                key="ads_catalog_input").strip()

    can_connect = bool(raw_input.strip()) and (not is_bare_fabric or bool(catalog))

    if st.button("Connect →", type="primary", disabled=not can_connect, key="ads_connect_btn"):
        conn_str = (_build_adonet_from_host(raw_input.strip(), catalog)
                    if is_bare_fabric and catalog else raw_input.strip())
        try:
            engine = ads_open_db(conn_str)
            tables = ads_db_tables(engine)
            if not tables:
                st.error("No tables found in this database.")
                return
            brief = ads_build_brief_schema(engine)
            st.session_state.ads_conn_string  = conn_str
            st.session_state.ads_db_path      = conn_str
            st.session_state.ads_all_tables   = tables
            st.session_state.ads_brief_schema = brief
            st.session_state.ads_phase        = "context"
            _advance_ads_furthest("context")
            st.rerun()
        except Exception as e:
            st.error(f"Connection failed: {e}")


# ── Phase 2: DB Context ───────────────────────────────────────────────────────

def ads_phase_context():
    st.title("Database Context")
    ads_viewing_past_step_banner()
    ads_show_history()

    with st.expander("📋 Tables detected in your database", expanded=True):
        st.code(st.session_state.ads_brief_schema)

    st.divider()

    # Business Logic Document
    st.subheader("Business Logic Document")
    st.caption("Upload KPIs, business rules, or feature definitions.")
    bl_file = st.file_uploader(
        "Business Logic Document (optional)",
        type=["txt", "pdf", "docx", "xlsx", "csv", "md"],
        key="ads_bl_uploader",
    )
    if bl_file:
        with st.spinner(f"Reading {bl_file.name}…"):
            bl_text = ads_extract_doc_text(bl_file)
        if len(bl_text) > 4000:
            with st.spinner("Condensing long document…"):
                bl_text = ads_summarize_doc(bl_text)
        st.session_state.ads_business_logic_doc = bl_text
        st.success(f"Business logic document loaded ({len(bl_text):,} chars).")
    elif st.session_state.ads_business_logic_doc:
        st.info("Business logic document already loaded.")

    st.divider()

    tab_doc, tab_chat, tab_skip = st.tabs([
        "📄 Upload Supporting Document",
        "💬 Describe via Chat",
        "⏭️  Skip (schema names only)",
    ])

    with tab_doc:
        st.markdown("Upload a data dictionary, column glossary, or ER diagram notes.")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["txt", "pdf", "docx", "xlsx", "csv", "md"],
            accept_multiple_files=True,
            key="ads_ctx_uploader",
        )
        if uploaded_files:
            with st.spinner(f"Extracting text from {len(uploaded_files)} file(s)…"):
                parts = []
                for f in uploaded_files:
                    extracted = ads_extract_doc_text(f)
                    parts.append(f"=== {f.name} ===\n{extracted}")
                doc_text = "\n\n".join(parts)
            with st.expander(f"Preview ({len(uploaded_files)} file(s))"):
                st.text(doc_text[:3000] + ("…" if len(doc_text) > 3000 else ""))
            if st.button("Use these documents →", type="primary", key="ads_use_docs"):
                ctx_text = doc_text
                if len(ctx_text) > 4000:
                    with st.spinner("Condensing…"):
                        ctx_text = ads_summarize_doc(ctx_text)
                st.session_state.ads_db_context     = ctx_text
                st.session_state.ads_context_source = "document"
                st.session_state.ads_phase          = "tables_goal"
                _advance_ads_furthest("tables_goal")
                st.rerun()

    with tab_chat:
        st.markdown("Answer a few questions and the agent will build context from your answers.")
        for msg in st.session_state.ads_context_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if not st.session_state.ads_context_chat:
            opening = (
                f"I can see your database has **{len(st.session_state.ads_all_tables)} tables**: "
                f"{', '.join(st.session_state.ads_all_tables)}.\n\n"
                "**What is this database about?**"
            )
            st.session_state.ads_context_chat.append({"role": "assistant", "content": opening})
            with st.chat_message("assistant"):
                st.markdown(opening)
        user_input = st.chat_input("Tell the agent about your database…", key="ads_ctx_chat_input")
        if user_input:
            st.session_state.ads_context_chat.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            system_with_schema = (
                f"{SYSTEM_CONTEXT}\n\nDatabase schema:\n{st.session_state.ads_brief_schema}"
            )
            with st.chat_message("assistant"):
                response = st.write_stream(
                    ads_llm_stream(st.session_state.ads_context_chat, system_with_schema)
                )
            st.session_state.ads_context_chat.append({"role": "assistant", "content": response})
            st.rerun()
        user_turns = sum(1 for m in st.session_state.ads_context_chat if m["role"] == "user")
        if user_turns >= 2:
            st.divider()
            if st.button("I've described enough — proceed →", type="primary", key="ads_ctx_proceed"):
                chat_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.ads_context_chat
                )
                st.session_state.ads_db_context     = f"[Context from user conversation]\n{chat_text}"
                st.session_state.ads_context_source = "chat"
                st.session_state.ads_phase          = "tables_goal"
                _advance_ads_furthest("tables_goal")
                st.rerun()

    with tab_skip:
        st.info("The agent will infer meaning from column names only.")
        if st.button("Skip and proceed →", type="secondary", key="ads_ctx_skip"):
            st.session_state.ads_db_context     = ""
            st.session_state.ads_context_source = "schema_only"
            st.session_state.ads_phase          = "tables_goal"
            _advance_ads_furthest("tables_goal")
            st.rerun()


# ── Phase 3: Tables & Goal ────────────────────────────────────────────────────

def ads_phase_tables_goal():
    st.title("Select Tables & Define Goal")
    ads_viewing_past_step_banner()
    ads_show_history()

    engine     = ads_open_db(st.session_state.ads_db_path)
    all_tables = ads_db_tables(engine)

    st.subheader("Available Tables")
    cols_ui = st.columns(4)
    for i, t in enumerate(all_tables):
        n    = ads_db_row_count(engine, t)
        ncol = len(ads_db_columns(engine, t))
        cols_ui[i % 4].metric(label=t, value=f"{n:,} rows", delta=f"{ncol} cols", delta_color="off")

    st.divider()

    selected = st.multiselect(
        "Select reference tables for your ADS",
        options=all_tables,
        default=st.session_state.ads_selected_tables or [],
        key="ads_table_select",
    )
    goal = st.text_area(
        "Describe your ADS goal",
        value=st.session_state.ads_goal,
        height=110,
        placeholder=(
            "e.g. Identify customers likely to open a term deposit in the next 90 days, "
            "based on their transaction behaviour and demographics"
        ),
        key="ads_goal_input",
    )

    goal_stripped  = goal.strip()
    tables_changed = sorted(selected) != sorted(st.session_state.ads_selected_tables)
    goal_changed   = goal_stripped != st.session_state.ads_goal
    has_rec        = bool(st.session_state.ads_recommendation)

    if has_rec and not tables_changed and not goal_changed:
        col_proceed, col_regen = st.columns([2, 1])
        with col_proceed:
            if st.button("Continue with existing recommendation →", type="primary",
                         use_container_width=True, key="ads_tg_continue"):
                st.session_state.ads_phase = "features"
                _advance_ads_furthest("features")
                st.rerun()
        with col_regen:
            if st.button("Re-generate from scratch", type="secondary",
                         use_container_width=True, key="ads_tg_regen"):
                st.session_state.ads_selected_tables       = selected
                st.session_state.ads_goal                  = goal_stripped
                st.session_state.ads_schema_block          = ads_build_schema_block(engine, selected)
                st.session_state.ads_recommendation        = ""
                st.session_state.ads_approved_features     = {}
                st.session_state.ads_sql_messages          = []
                st.session_state.ads_current_sql           = ""
                st.session_state.ads_derived_features      = []
                st.session_state.ads_derived_features_chat = []
                st.session_state.ads_phase = "features"
                _advance_ads_furthest("features")
                st.rerun()
    else:
        if st.button("Get Feature Recommendations →", type="primary",
                     disabled=not (selected and goal_stripped), key="ads_tg_go"):
            st.session_state.ads_selected_tables = selected
            st.session_state.ads_goal            = goal_stripped
            st.session_state.ads_schema_block    = ads_build_schema_block(engine, selected)
            if tables_changed or goal_changed:
                st.session_state.ads_recommendation        = ""
                st.session_state.ads_approved_features     = {}
                st.session_state.ads_sql_messages          = []
                st.session_state.ads_current_sql           = ""
                st.session_state.ads_derived_features      = []
                st.session_state.ads_derived_features_chat = []
            st.session_state.ads_phase = "features"
            _advance_ads_furthest("features")
            st.rerun()


# ── Phase 4: Feature Recommendations ─────────────────────────────────────────

def _build_ads_feature_prompt() -> str:
    engine  = ads_open_db(st.session_state.ads_db_path)
    dialect = engine.dialect.name.upper()
    detail_parts = []
    for t in st.session_state.ads_selected_tables:
        cols  = ads_db_columns(engine, t)
        stats = ads_get_column_stats(engine, t, tuple(cols))
        if stats:
            col_parts = []
            for c in cols:
                s   = stats.get(c)
                ann = f" [{s['non_null_pct']}% non-null, {s['distinct_count']} distinct]" if s else ""
                col_parts.append(f"{c}{ann}")
            col_desc = ", ".join(col_parts)
        else:
            col_desc = ", ".join(cols)
        detail_parts.append(f"### {t}  ({len(cols)} columns)\n{col_desc}")
    sample_parts = []
    for t in st.session_state.ads_selected_tables:
        rows = ads_get_sample_rows(engine, t, n=2)
        if rows:
            headers = list(rows[0].keys())
            sample_parts.append(
                f"### {t} — sample rows\n"
                + " | ".join(headers) + "\n"
                + "\n".join(" | ".join(str(row.get(h, "")) for h in headers) for row in rows)
            )
    sample_block = (
        "\n## Sample Data (2 rows per table)\n" + "\n\n".join(sample_parts)
    ) if sample_parts else ""
    db_context_block = (
        f"\n## Database Context\n{st.session_state.ads_db_context}\n"
        if st.session_state.ads_db_context else ""
    )
    biz_logic_block = (
        f"\n## Business Logic Document\n{st.session_state.ads_business_logic_doc}\n"
        if st.session_state.ads_business_logic_doc else ""
    )
    return f"""\
You are helping a data engineer build an ADS for a bank.
Database dialect: {dialect}

## All tables (non-selected tables for JOIN reference)
{st.session_state.ads_schema_block}
{db_context_block}{biz_logic_block}{sample_block}

## Selected tables — full column lists
{chr(10).join(detail_parts)}

## ADS Goal
"{st.session_state.ads_goal}"

## Task
Recommend features for this ADS.

**Priority 1 — Aggregated / Derived Features** (SUM, AVG, COUNT, ratios, flags, CASE, rolling aggregates)
**Priority 2 — Raw Columns** (stable attributes: demographics, status, segment)

For EACH feature:
  Feature name       : <snake_case_name>
  Type               : Aggregated | Raw
  Source             : <table.column(s)>
  Formula / Value    : <{dialect}-compatible SQL expression or column reference>
  Why it helps       : <one sentence>

### Recommended Features

### Suggested Enrichment from Other Tables (optional)

### Summary
"""


def ads_phase_features():
    st.title("Feature Recommendations")
    ads_viewing_past_step_banner()
    ads_show_history()

    st.info(
        f"**Goal:** {st.session_state.ads_goal}  \n"
        f"**Tables:** {', '.join(st.session_state.ads_selected_tables)}"
    )

    if not st.session_state.ads_recommendation:
        st.subheader("Analysing your tables…")
        messages = [{"role": "user", "content": _build_ads_feature_prompt()}]
        with st.chat_message("assistant"):
            response = st.write_stream(ads_llm_stream(messages, SYSTEM_ANALYST))
        st.session_state.ads_recommendation = response
        st.rerun()

    with st.expander("Full recommendation from agent", expanded=True):
        st.markdown(st.session_state.ads_recommendation)

    st.divider()
    st.subheader("Confirm Your Feature Selection")

    engine    = ads_open_db(st.session_state.ads_db_path)
    rec_lower = st.session_state.ads_recommendation.lower()
    approved  = {}

    for t in st.session_state.ads_selected_tables:
        all_cols = ads_db_columns(engine, t)
        auto_sel = [c for c in all_cols
                    if re.search(r'\b' + re.escape(c.lower()) + r'\b', rec_lower)]
        chosen   = st.multiselect(
            f"Features from **{t}**",
            options=all_cols, default=auto_sel,
            key=f"ads_feat_{t}",
        )
        if chosen:
            approved[t] = chosen

    st.markdown("**Add columns from another table (optional)**")
    other_tables = [t for t in ads_db_tables(engine)
                    if t not in st.session_state.ads_selected_tables]
    extra_table  = st.selectbox("Table", ["— none —"] + other_tables, key="ads_extra_table")
    if extra_table != "— none —":
        extra_cols = st.multiselect("Columns", ads_db_columns(engine, extra_table),
                                    key="ads_extra_cols")
        if extra_cols:
            approved[extra_table] = extra_cols

    extra_enrich = st.text_input(
        "Any free-text JOIN hint? (optional)",
        placeholder="e.g. JOIN nbo_crm ON nbo_crm.cif = nbo_products.client_no",
        key="ads_extra_enrich",
    )

    total = sum(len(v) for v in approved.values())
    st.caption(f"**{total} features** selected across **{len(approved)} table(s)**")

    # Derived Features
    st.divider()
    st.subheader("Derived Features (Optional)")
    if st.session_state.ads_derived_features:
        st.markdown(f"**{len(st.session_state.ads_derived_features)} derived feature(s) approved:**")
        for i, df_feat in enumerate(st.session_state.ads_derived_features):
            col_a, col_b = st.columns([6, 1])
            with col_a:
                st.markdown(f"**`{df_feat['name']}`** — {df_feat['description']}")
                st.code(df_feat["formula"], language="sql")
            with col_b:
                if st.button("Remove", key=f"ads_rm_derived_{i}"):
                    st.session_state.ads_derived_features.pop(i)
                    st.rerun()

    for i, msg in enumerate(st.session_state.ads_derived_features_chat):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("parsed_feature"):
            pf = msg["parsed_feature"]
            already = any(d["name"] == pf["name"] for d in st.session_state.ads_derived_features)
            if pf.get("_validation_error"):
                st.warning(f"⚠️ Formula validation failed: `{pf['_validation_error']}`")
            if not already:
                if st.button(f"➕ Add `{pf['name']}`", key=f"ads_add_derived_{i}"):
                    st.session_state.ads_derived_features.append(pf)
                    st.rerun()
            else:
                st.caption(f"✓ `{pf['name']}` already added")

    derived_input = st.chat_input("Describe a derived feature…", key="ads_derived_chat_input")
    if derived_input:
        schema_for_derived = ads_build_schema_block(engine, st.session_state.ads_selected_tables)
        _dialect     = engine.dialect.name
        _dialect_hint = {"sqlite": "SQLite", "mssql": "T-SQL", "postgresql": "PostgreSQL",
                         "mysql": "MySQL"}.get(_dialect, "standard SQL")
        derived_system = (
            f"You are a banking data analyst helping to engineer derived features. "
            f"Write all formulas as {_dialect_hint}-compatible SQL expressions. "
            "When suggesting a derived feature, ALWAYS end your response with:\n"
            "FEATURE_NAME: <snake_case_name>\n"
            "FORMULA: <SQL expression>\n"
            "DESCRIPTION: <one sentence>\n\n"
            f"## Database schema\n{schema_for_derived}\n"
            f"## ADS Goal\n\"{st.session_state.ads_goal}\""
        )
        st.session_state.ads_derived_features_chat.append(
            {"role": "user", "content": derived_input, "parsed_feature": None}
        )
        with st.chat_message("user"):
            st.markdown(derived_input)
        llm_msgs = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state.ads_derived_features_chat]
        with st.chat_message("assistant"):
            response = st.write_stream(ads_llm_stream(llm_msgs, derived_system))
        parsed = ads_parse_derived_feature(response)
        if parsed and parsed.get("formula"):
            err = ads_validate_formula(engine, parsed["formula"],
                                       st.session_state.ads_selected_tables)
            if err:
                parsed["_validation_error"] = err
        st.session_state.ads_derived_features_chat.append(
            {"role": "assistant", "content": response, "parsed_feature": parsed}
        )
        st.rerun()

    st.divider()
    if approved or st.session_state.ads_derived_features:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        goal_slug = "".join(c if c.isalnum() or c == "_" else ""
                            for c in "_".join(st.session_state.ads_goal.lower().split())[:40])
        try:
            excel_bytes = ads_make_features_excel(approved, st.session_state.ads_derived_features)
            st.download_button(
                label="⬇ Download Feature List (Excel)",
                data=excel_bytes,
                file_name=f"ads_features_{goal_slug}_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="ads_feat_download",
            )
        except Exception:
            st.caption("Excel export requires `pandas` + `openpyxl`.")

    if st.button("Generate SQL →", type="primary", disabled=total == 0, key="ads_gen_sql_btn"):
        st.session_state.ads_approved_features = approved
        st.session_state.ads_extra_enrichment  = extra_enrich
        feat_block = "\n".join(
            f"  {t}.{c}" for t, cols in approved.items() for c in cols
        )
        derived_block = ""
        if st.session_state.ads_derived_features:
            d_lines = [
                f"  {d['name']}  AS  ({d['formula']})  -- {d['description']}"
                for d in st.session_state.ads_derived_features
            ]
            derived_block = "\n## Derived features to compute\n" + "\n".join(d_lines) + "\n"
        db_label = ads_open_db(st.session_state.ads_db_path).dialect.name.upper()
        db_ctx_block = (
            f"\n## Database Context\n{st.session_state.ads_db_context}\n"
            if st.session_state.ads_db_context else ""
        )
        biz_logic_block = (
            f"\n## Business Logic Document\n{st.session_state.ads_business_logic_doc}\n"
            if st.session_state.ads_business_logic_doc else ""
        )
        initial_prompt = f"""\
Generate a SQL query to build an ADS (Analytical Data Set).

## Database: {db_label}
{db_ctx_block}{biz_logic_block}
## All available tables
{st.session_state.ads_schema_block}

## Reference tables: {', '.join(st.session_state.ads_selected_tables)}

## Approved features
{feat_block}
{derived_block}
## Extra JOIN hint
{extra_enrich if extra_enrich.strip() else 'None'}

## ADS Goal
"{st.session_state.ads_goal}"

## Instructions
- Single SELECT statement (or CREATE TABLE ads AS SELECT ...).
- Comment each logical block with -- ...
- Use LEFT JOIN for extra tables, use table aliases.
- Compute every derived feature as a named column alias.
- Output SQL only — no explanation.
"""
        st.session_state.ads_sql_messages = [{"role": "user", "content": initial_prompt}]
        st.session_state.ads_current_sql  = ""
        st.session_state.ads_phase        = "sql"
        _advance_ads_furthest("sql")
        st.rerun()


# ── Phase 5: SQL Generation ───────────────────────────────────────────────────

def ads_phase_sql(on_proceed_to_seg=None):
    """
    SQL generation & refinement phase.

    ONLY action shown: "Proceed to Segmentation" button.
    No "Run SQL", no download button.

    on_proceed_to_seg: callable — called with NO arguments when user clicks
                       the handoff button.  combined_app.py handles the full
                       segmentation state reset and tab switch.
    """
    st.title("SQL Generation")
    ads_viewing_past_step_banner()
    ads_show_history()

    st.info(
        f"**Goal:** {st.session_state.ads_goal}  \n"
        f"**Tables:** {', '.join(st.session_state.ads_approved_features.keys())}"
    )
    st.caption(
        "Use the chat below to refine the generated SQL. "
        "When you are satisfied, click **Proceed to Segmentation** at the bottom."
    )

    # ── Render conversation history ───────────────────────────────────────
    for msg in st.session_state.ads_sql_messages:
        if msg["role"] == "user":
            # Only show short user messages (hide the giant initial prompt)
            if len(msg["content"]) < 400:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(msg["content"].replace(
                        "\n\nReturn the full updated SQL only — no explanation.", ""
                    ))
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.code(msg["content"], language="sql")

    # ── Auto-generate if last message is from user ────────────────────────
    if (st.session_state.ads_sql_messages and
            st.session_state.ads_sql_messages[-1]["role"] == "user"):
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Generating SQL…"):
                response = st.write_stream(
                    ads_llm_stream(st.session_state.ads_sql_messages, SYSTEM_SQL)
                )
        st.session_state.ads_current_sql = response
        st.session_state.ads_sql_messages.append({"role": "assistant", "content": response})
        st.rerun()

    # ── Refinement chat input ─────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask for a change… (e.g. 'add WHERE client_status = active')",
        key="ads_sql_chat_input",
    )
    if user_input:
        st.session_state.ads_sql_messages.append({
            "role":    "user",
            "content": f"{user_input}\n\nReturn the full updated SQL only — no explanation.",
        })
        st.rerun()

    # ── ONLY action button: Proceed to Segmentation ───────────────────────
    if st.session_state.ads_current_sql:
        st.divider()
        st.markdown("#### Ready to segment your data?")
        st.caption(
            "Click below to proceed to the Segmentation Agent. "
            "You will connect to your database there and run the segmentation "
            "on the data set you have built."
        )

        if on_proceed_to_seg is not None:
            if st.button(
                "📊 Proceed to Segmentation →",
                type="primary",
                key="ads_to_seg_btn",
                use_container_width=True,
            ):
                # on_proceed_to_seg takes NO arguments in the new design —
                # combined_app.py resets seg state and switches tabs.
                on_proceed_to_seg()
        else:
            st.info(
                "Connect this agent via `combined_app.py` to enable the "
                "Proceed to Segmentation button."
            )