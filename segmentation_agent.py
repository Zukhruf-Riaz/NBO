"""
segmentation_agent.py
---------------------
NBO Customer Segmentation Agent — all phase functions.
All session-state keys are prefixed with "seg_" to avoid collisions with the
ADS agent when both run in the same Streamlit app.

Imported and called by combined_app.py.

Key fixes applied:
- Radio buttons restored in algorithm and role phases (not tabs).
- LLM feature selection correctly populates the multiselect dropdown for
  BOTH technical and business roles.
- Full functional parity with original_segmentation.py.

Bug fix (v2):
- Removed key="seg_feat_multiselect" from the multiselect widget in
  seg_phase_technical(). When a key is set, Streamlit locks the widget to its
  internal state and ignores the `default` parameter on reruns — meaning the
  LLM-selected features never appeared in the dropdown. Removing the key
  restores the original behaviour: `default` is re-evaluated on every rerun.
- Removed key="seg_chat_input" from st.chat_input to match original.
"""

import io
import json
import warnings
from datetime import datetime
from urllib.parse import quote_plus

import docx as _docx
import pypdf as _pypdf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.engine import Engine

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL         = "http://localhost:11434/api/chat"
MODEL              = "qwen2.5:7b"
MIN_FEATURES       = 6
MAX_FEATURES       = 30
CHUNK_SIZE         = 60
NOMINEES_PER_CHUNK = 8
MAX_DIAG_ROWS      = 25_000
MAX_SIL_ROWS       = 10_000
DBSCAN_MAX_ROWS    = 15_000

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_DATA_SCIENTIST = """\
You are a Senior Data Scientist with experience in retail banking
analytics specialising in customer segmentation, RFM modelling, and behavioural
profiling.

FEATURE SELECTION RULES:
a) ALWAYS EXCLUDE columns whose name contains: 'id','cif','key','code','no','num',
   'number','name','flag','desc','text','type','status','label',
   'ref','created','updated'
b) ALWAYS EXCLUDE columns with more than 50% null values.
c) PREFER numeric columns: transaction amounts, balances, product counts, tenure,
   age, income proxies, frequency metrics.
d) Select BETWEEN 6 AND 15 features. If fewer than 6 valid numeric columns exist
   after exclusions, select ALL available numeric columns and note this in reason.
e) Cover as many dimensions as possible: MONETARY, FREQUENCY, RECENCY/TENURE,
   DEMOGRAPHICS, PRODUCT BREADTH, CHANNEL BEHAVIOUR, TIME AND DATE.

PARAMETER RULES:
- K-Means: K must be an integer 2-10. If an elbow K is provided in the user
  message, use it unless you have a strong reason not to (state why in reason).
- DBSCAN: features will be standardised (mean=0, std=1) before clustering.
  In standardised space, typical eps is 0.3-2.0. min_samples should be
  approximately ln(n_rows), minimum 3. Never output eps < 0.05.

JSON OUTPUT RULES:
- Output ONLY valid JSON - no markdown fences, no preamble, no trailing text.
- K-Means schema:  {"features": [...], "params": {"k": <int>}, "reason": "..."}
- DBSCAN schema:   {"features": [...], "params": {"eps": <float>, "min_samples": <int>}, "reason": "..."}
- reason: 2-3 sentences explaining why these features produce meaningful segments.
"""

SYSTEM_SEGMENT_NAMER = """\
You are a Senior Banking Analyst writing concise segment profiles for a business audience.

OUTPUT RULES:
- Output ONLY a valid JSON array - no markdown fences, no preamble, no trailing text.
- One object per segment_id provided.
- Schema: {"segment_id": <int>, "name": "<short name>", "profile_summary": "<1-2 sentences>", "key_characteristics": "<comma-separated stats>"}
- name: 3-5 words, business-friendly.
- Do NOT mention products, offers, or recommendations.
- Do NOT output anything outside the JSON array.
"""

SYSTEM_TUNING_ADVISOR = """\
You are a Senior Data Scientist providing parameter tuning advice after a
segmentation run for a banking analyst.

FOR K-MEANS: only suggest changing K. State the exact new K value.
FOR DBSCAN: suggest changing eps and/or min_samples with exact numbers.

Assess silhouette score: >0.5 good, 0.25-0.5 moderate, <0.25 needs work.
Keep to 4-5 bullet points.
End with: "RECOMMENDED NEXT STEP: [one clear action with exact parameter values]."
Do NOT suggest products or offers.
"""

SYSTEM_COMPARISON_ANALYST = """\
You are a Senior Data Scientist comparing two consecutive segmentation runs.
Start with a one-sentence verdict.
Give 4-5 bullet points comparing specific metrics.
End with: "FINAL RECOMMENDATION: [keep Run 2 / revert to Run 1 / try a third run with ...]"
Do NOT suggest products or offers.
"""

SYSTEM_BUSINESS_CHAT = """\
You are a friendly banking analytics expert helping a NON-TECHNICAL business user
understand customer segmentation results. Avoid jargon.

CRITICAL RULES:
1. NEVER suggest products, offers, or recommendations.
2. DO NOT change the number of segments.
3. Describe WHO customers are, WHAT characterises them, HOW they differ.
4. If the user mentions a company policy, re-describe relevant segments through
   that lens. Do NOT create new segments or change boundaries.
5. Be conversational. Keep answers 3-6 sentences unless more detail is needed.
6. Stay grounded in the segment data. Do not invent numbers.
"""

SYSTEM_RULE_TRANSLATOR = """\
You are a data analyst. Convert a plain-English customer description into a
valid pandas boolean expression that filters rows of a DataFrame called `df`.

RULES:
- Output ONLY the pandas expression - no explanation, no markdown, no quotes around it.
- Reference columns as df['column_name'].
- Use operators: >, <, >=, <=, ==, !=, &, |, ~
- Wrap each condition in parentheses before combining with & or |.
- String values must be quoted inside the expression.
- Example output: (df['balance'] > 50000) & (df['num_products'] >= 3)
"""

ID_KEYWORDS = [
    "id","cif","key","code","no","num","number","name","date","time",
    "flag","desc","text","type","status","label","created","updated","ref",
]

# ── Default session-state values (all prefixed with seg_) ────────────────────
SEG_DEFAULTS = {
    "seg_phase":          "connection",
    "seg_conn_string":    "",
    "seg_engine":         None,
    "seg_all_tables":     [],
    "seg_table_name":     "",
    "seg_col_docs":       "",
    "seg_goal":           "",
    "seg_algorithm":      "",
    "seg_user_role":      "",
    "seg_df":             None,
    "seg_features":       [],
    "seg_params":         {},
    "seg_df_labeled":     None,
    "seg_cluster_stats":  None,
    "seg_segment_names":  [],
    "seg_sil_score":      None,
    "seg_run_history":    [],
    "seg_iteration":      1,
    "seg_elbow_data":     None,
    "seg_kdist_data":     None,
    "seg_tuning_advice":  "",
    "seg_comparison":     "",
    "seg_X_scaled_cache":    None,
    "seg_X_scaled_features": [],
    "seg_strategy_cache":    {},
    "seg_df_labeled_base":        None,
    "seg_segment_rules":          [],
    "seg_pending_rule":           None,
    "seg_pending_rethreshold":    None,
    "seg_chat_history":   [],
    "seg_chat_context":   "",
    "seg_excel_bytes":    None,
}


def init_seg_state():
    for k, v in SEG_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Connection error helper ───────────────────────────────────────────────────

def _show_conn_error(exc: Exception):
    msg = str(exc)
    low = msg.lower()
    if "timeout" in low or "258" in msg or "delay in login" in low:
        st.error(
            "**Connection timed out** — the server did not respond in time.\n\n"
            "Check firewall rules, VPN, and that port 1433 is open."
        )
    elif "login failed" in low or "18456" in msg:
        st.error("**Authentication failed** — check username/password or AAD Interactive popup.")
    elif "unable to open" in low or "no such file" in low or "does not exist" in low:
        st.error(f"**Database file not found.**\n\n`{msg}`")
    else:
        st.error(f"Connection failed: {exc}")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _engine_from_adonet(adonet: str) -> Engine:
    parts = {}
    for seg in adonet.split(";"):
        seg = seg.strip()
        if "=" in seg:
            k, _, v = seg.partition("=")
            parts[k.strip().lower()] = v.strip()
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


def _make_engine(conn_str: str) -> Engine:
    s = conn_str.strip()
    if s.lower().startswith("data source=") or "initial catalog=" in s.lower():
        return _engine_from_adonet(s)
    if not s.lower().startswith(("sqlite","postgresql","mysql","mssql","oracle")):
        if s.endswith((".db", ".sqlite", ".db3")):
            s = "sqlite:///" + s.replace("\\", "/")
    if s.lower().startswith("sqlite"):
        return create_engine(s, connect_args={"check_same_thread": False})
    return create_engine(s)


@st.cache_resource
def seg_open_engine(conn_str: str) -> Engine:
    return _make_engine(conn_str)


def seg_list_tables(engine: Engine) -> list:
    return sorted(sa_inspect(engine).get_table_names())


def seg_extract_doc_text(uploaded_files) -> str:
    parts = []
    for f in uploaded_files:
        name = f.name.lower()
        try:
            if name.endswith(".pdf"):
                reader = _pypdf.PdfReader(io.BytesIO(f.read()))
                txt = "\n".join(p.extract_text() or "" for p in reader.pages)
                parts.append(f"[{f.name}]\n{txt.strip()}")
            elif name.endswith(".docx"):
                doc = _docx.Document(io.BytesIO(f.read()))
                txt = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                parts.append(f"[{f.name}]\n{txt.strip()}")
            elif name.endswith((".xlsx", ".xls")):
                xf = pd.read_excel(io.BytesIO(f.read()), sheet_name=None)
                lines = []
                for sheet_name, sdf in xf.items():
                    lines.append(f"Sheet: {sheet_name}")
                    lines.append(sdf.to_string(index=False))
                parts.append(f"[{f.name}]\n" + "\n".join(lines))
        except Exception as exc:
            parts.append(f"[{f.name}] (could not parse: {exc})")
    return "\n\n".join(parts)


def seg_get_row_count(engine: Engine, table: str) -> int:
    with engine.connect() as c:
        return c.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()


def seg_load_table(engine: Engine, table: str) -> pd.DataFrame:
    with engine.connect() as c:
        return pd.read_sql_query(text(f'SELECT * FROM "{table}"'), c)


# ── Ollama helpers ────────────────────────────────────────────────────────────

def seg_check_ollama() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def seg_call_llm(messages: list, system: str, temperature: float = 0.0,
                 num_predict: int = 512, num_ctx: int = 4096) -> str:
    payload = {
        "model":    MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream":   False,
        "options":  {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=900)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except Exception as e:
        return f"[LLM error: {e}]"


def seg_llm_stream_generator(messages: list, system: str, temperature: float = 0.7):
    payload = {
        "model":    MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream":   True,
        "options":  {"temperature": temperature},
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=900) as resp:
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
        yield f"\n[Stream error: {e}]"


# ── Feature helpers ───────────────────────────────────────────────────────────

def is_likely_id_column(col: str) -> bool:
    low = col.lower()
    return any(kw in low for kw in ID_KEYWORDS)


def seg_get_fallback_features(df: pd.DataFrame, algorithm: str) -> dict:
    numeric_cols = df.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    eligible = [
        c for c in numeric_cols
        if not is_likely_id_column(c)
        and df[c].isnull().mean() < 0.5
        and df[c].nunique() > 1
    ]
    if not eligible:
        eligible = [c for c in numeric_cols if df[c].nunique() > 1][:MAX_FEATURES]
    if len(eligible) > MIN_FEATURES:
        try:
            corr = df[eligible].fillna(0).corr().abs()
            drop = set()
            cols = list(corr.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if cols[j] not in drop and corr.iloc[i, j] > 0.97:
                        drop.add(cols[j])
            eligible = [c for c in eligible if c not in drop]
        except Exception:
            pass
    feature_cols = eligible[:MAX_FEATURES]
    if len(feature_cols) < MIN_FEATURES:
        extra = [c for c in numeric_cols if c not in feature_cols and df[c].nunique() > 1]
        feature_cols = (feature_cols + extra)[:MAX_FEATURES]
    n = len(df)
    if algorithm == "kmeans":
        params = {"k": max(2, min(6, int((n / 1000) ** 0.5) + 2))}
    else:
        params = {"eps": 0.5, "min_samples": max(3, min(10, n // 500))}
    return {"features": feature_cols, "params": params,
            "reason": f"Auto-selected {len(feature_cols)} numeric non-ID columns as fallback."}


def _seg_build_candidate_feature_block(df: pd.DataFrame) -> tuple:
    DIM_KEYWORDS = {
        "monetary":   ["amt","amount","balance","bal","spend","value","revenue",
                       "income","salary","credit","debit","total_cr","total_dr","net"],
        "frequency":  ["count","cnt","freq","txn","transaction","login","visit","usage","times"],
        "recency":    ["recency","tenure","months","days","age","since","last","duration","vintage"],
        "demographic":["age","gender","segment","income","salary","band","tier","class"],
        "product":    ["product","prod","loan","card","account","acct","mortgage","insurance",
                       "invest","fund","deposit","saving","current","num_prod","n_prod"],
        "channel":    ["digital","mobile","internet","online","branch","atm","pos","web","app","channel"],
    }
    def _dim(col):
        low = col.lower()
        for dim, kws in DIM_KEYWORDS.items():
            if any(kw in low for kw in kws):
                return dim
        return "other"

    candidates = [
        c for c in df.columns
        if not is_likely_id_column(c)
        and df[c].isnull().mean() < 0.5
        and df[c].nunique() > 1
    ]
    groups = {d: [] for d in list(DIM_KEYWORDS.keys()) + ["other"]}
    for col in candidates:
        groups[_dim(col)].append(col)
    dim_labels = {
        "monetary": "MONETARY VALUE", "frequency": "FREQUENCY",
        "recency": "RECENCY / TENURE", "demographic": "DEMOGRAPHICS",
        "product": "PRODUCT BREADTH", "channel": "CHANNEL BEHAVIOUR", "other": "OTHER NUMERIC",
    }
    lines = [f"ELIGIBLE COLUMNS ({len(candidates)} total):", ""]
    for dim, cols in groups.items():
        if cols:
            lines.append(f"  [{dim_labels[dim]}]")
            for c in cols:
                null_pct = round(df[c].isnull().mean() * 100, 1)
                if pd.api.types.is_numeric_dtype(df[c]):
                    lines.append(f"    {c}  (null={null_pct}%  min={df[c].min():.1f}  max={df[c].max():.1f})")
                else:
                    lines.append(f"    {c}  (null={null_pct}%  unique={df[c].nunique()})")
            lines.append("")
    return "\n".join(lines), candidates


def _seg_parse_feature_json(raw: str, df: pd.DataFrame) -> list:
    import re as _re
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = "\n".join(l for l in cleaned.split("\n") if not l.strip().startswith("```")).strip()
    bs, be = cleaned.find("{"), cleaned.rfind("}") + 1
    if bs != -1 and be > bs:
        cleaned = cleaned[bs:be]
    cleaned = _re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = cleaned.replace("'", '"')
    try:
        result = json.loads(cleaned)
    except Exception:
        return []
    col_map = {c.lower(): c for c in df.columns}
    valid = []
    for f in result.get("features", []):
        if f in df.columns:
            valid.append(f)
        elif str(f).lower() in col_map:
            valid.append(col_map[str(f).lower()])
    valid = [f for f in valid if not is_likely_id_column(f)]
    seen, deduped = set(), []
    for f in valid:
        if f not in seen:
            seen.add(f); deduped.append(f)
    return deduped


def _seg_col_stats_line(df: pd.DataFrame, col: str) -> str:
    null_pct = round(df[col].isnull().mean() * 100, 1)
    if pd.api.types.is_numeric_dtype(df[col]):
        return f"  {col}  (null={null_pct}%  min={df[col].min():.1f}  max={df[col].max():.1f})"
    return f"  {col}  (null={null_pct}%  unique={df[col].nunique()})"


# ── Primary feature-selection function ───────────────────────────────────────

def seg_llm_select_features(df: pd.DataFrame, table_name: str,
                             col_docs: str = "", seg_goal: str = "") -> dict:
    """
    Select segmentation features using a multi-pass chunked approach.
    Returns dict with 'features' key populated — used for BOTH technical and
    business roles to pre-fill the multiselect dropdown.
    """
    _, candidates = _seg_build_candidate_feature_block(df)
    n_rows = len(df)
    min_f  = min(MIN_FEATURES, len(candidates))

    col_docs_trimmed = (col_docs.strip() or "")[:800]
    if col_docs.strip() and len(col_docs.strip()) > 800:
        col_docs_trimmed += "\n... (truncated)"
    col_docs_block = (
        f"\nCOLUMN DOCUMENTATION:\n{col_docs_trimmed}\n" if col_docs_trimmed else ""
    )
    goal_block = (
        f"\nSEGMENTATION OBJECTIVE: {seg_goal.strip()}\n"
        if seg_goal.strip() else ""
    )

    # ── Single-pass for narrow tables ────────────────────────────────────────
    if len(candidates) <= CHUNK_SIZE:
        schema_lines = "\n".join(_seg_col_stats_line(df, c) for c in candidates)
        prompt = (
            f'Table: "{table_name}"  Rows: {n_rows:,}\n'
            f"{goal_block}\n"
            f"Eligible columns ({len(candidates)} total):\n{schema_lines}\n"
            f"{col_docs_block}\n"
            f"Select {min_f}-{MAX_FEATURES} columns that best serve the segmentation objective above.\n"
            f"Cover monetary, frequency, recency/tenure, demographics, product, channel dimensions.\n"
            f'Output ONLY JSON (no markdown): {{"features": ["col1", ...], "reason": "2-3 sentences."}}'
        )
        raw = seg_call_llm([{"role": "user", "content": prompt}], SYSTEM_DATA_SCIENTIST,
                           num_predict=1024, num_ctx=8192)
        valid = _seg_parse_feature_json(raw, df)
        if len(valid) >= min_f:
            reason = ""
            try:
                bs, be = raw.find("{"), raw.rfind("}") + 1
                if bs != -1 and be > bs:
                    reason = json.loads(raw[bs:be]).get("reason", "")
            except Exception:
                pass
            return {"features": valid[:MAX_FEATURES], "reason": reason, "_fallback": False}
        fb = seg_get_fallback_features(df, "kmeans")
        return {
            "features": fb["features"], "reason": "",
            "_fallback": True,
            "_fallback_reason": f"LLM returned {len(valid)} valid columns, need {min_f}.",
            "_raw_response": raw,
        }

    # ── Multi-pass for wide tables ────────────────────────────────────────────
    nominees: list = []
    chunks = [candidates[i:i + CHUNK_SIZE] for i in range(0, len(candidates), CHUNK_SIZE)]
    last_raw = ""
    for idx, chunk in enumerate(chunks):
        schema_lines = "\n".join(_seg_col_stats_line(df, c) for c in chunk)
        prompt = (
            f'Table: "{table_name}" - column batch {idx + 1} of {len(chunks)} '
            f'({len(chunk)} columns shown, {len(candidates)} total eligible).\n'
            f"{goal_block}{col_docs_block}\n"
            f"Columns in this batch:\n{schema_lines}\n\n"
            f"From ONLY the columns listed above, pick the {NOMINEES_PER_CHUNK} most useful "
            f"for the segmentation objective above.\n"
            f'Output ONLY JSON (no markdown): {{"features": ["col1", ...]}}'
        )
        raw = seg_call_llm([{"role": "user", "content": prompt}], SYSTEM_DATA_SCIENTIST,
                           num_predict=512, num_ctx=8192)
        last_raw = raw
        picked = _seg_parse_feature_json(raw, df)
        chunk_set = set(chunk)
        nominees.extend(f for f in picked if f in chunk_set)

    seen: set = set()
    nominees = [f for f in nominees if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]

    if not nominees:
        fb = seg_get_fallback_features(df, "kmeans")
        return {
            "features": fb["features"], "reason": "",
            "_fallback": True,
            "_fallback_reason": "All chunk passes returned no valid columns.",
            "_raw_response": last_raw,
        }

    if len(nominees) <= MAX_FEATURES:
        final = nominees
        reason = f"Selected via {len(chunks)}-chunk pass covering all {len(candidates)} eligible columns."
        return {"features": final[:MAX_FEATURES], "reason": reason, "_fallback": False}

    schema_lines = "\n".join(_seg_col_stats_line(df, c) for c in nominees)
    prompt = (
        f'Table: "{table_name}"  Rows: {n_rows:,}\n'
        f"{goal_block}{col_docs_block}\n"
        f"These {len(nominees)} columns were nominated across all {len(chunks)} column batches:\n"
        f"{schema_lines}\n\n"
        f"Choose the final best {min_f}-{MAX_FEATURES} that best serve the segmentation objective.\n"
        f"Ensure good coverage: monetary, frequency, recency, demographics, product, channel.\n"
        f'Output ONLY JSON (no markdown): {{"features": ["col1", ...], "reason": "2-3 sentences."}}'
    )
    raw = seg_call_llm([{"role": "user", "content": prompt}], SYSTEM_DATA_SCIENTIST,
                       num_predict=1024, num_ctx=8192)
    last_raw = raw
    final = _seg_parse_feature_json(raw, df)
    reason = ""
    try:
        bs, be = raw.find("{"), raw.rfind("}") + 1
        if bs != -1 and be > bs:
            reason = json.loads(raw[bs:be]).get("reason", "")
    except Exception:
        pass
    if len(final) < min_f:
        final = nominees[:MAX_FEATURES]
        reason = f"Final ranking pass failed; using top nominees from {len(chunks)} chunks."
    return {"features": final[:MAX_FEATURES], "reason": reason, "_fallback": False}


# ── Strategy function (technical flow — includes parameter recommendation) ────

def seg_llm_decide_strategy(df: pd.DataFrame, table_name: str, algorithm: str,
                             elbow_k=None, kdist_eps=None, col_docs: str = "",
                             seg_goal: str = "") -> dict:
    schema_block, candidates = _seg_build_candidate_feature_block(df)
    n_rows    = len(df)
    n_cands   = len(candidates)
    algo_name = "K-Means" if algorithm == "kmeans" else "DBSCAN"
    k_max     = min(8, max(2, int((n_rows / 1000) ** 0.5) + 2))

    if algorithm == "kmeans":
        if elbow_k is not None:
            param_instruction = f'"params": {{"k": {elbow_k}}}  /* elbow-detected */'
            diag_ctx = f"\nELBOW ANALYSIS RESULT: Optimal K={elbow_k}. Use this value.\n"
        else:
            param_instruction = f'"params": {{"k": <integer 2 to {k_max}>}}'
            diag_ctx = ""
    else:
        if kdist_eps is not None:
            param_instruction = (
                f'"params": {{"eps": {kdist_eps}, "min_samples": <integer 3 to 15>}}'
                f'  /* knee-detected eps */'
            )
            diag_ctx = f"\nK-DISTANCE RESULT: Optimal eps={kdist_eps}. Use this value.\n"
        else:
            param_instruction = '"params": {"eps": <float 0.3-2.0>, "min_samples": <integer 3-15>}'
            diag_ctx = ""

    min_f    = min(MIN_FEATURES, n_cands)
    max_f    = min(MAX_FEATURES, n_cands)
    target_f = max(min_f, min(max_f, max(8, n_cands // 2)))

    col_docs_block = (
        f"\nCOLUMN DOCUMENTATION (provided by user):\n{col_docs.strip()}\n"
        if col_docs.strip() else ""
    )
    goal_block = (
        f"\nSEGMENTATION OBJECTIVE: {seg_goal.strip()}\n"
        if seg_goal.strip() else ""
    )

    prompt = (
        f'Table: "{table_name}"  Rows: {n_rows:,}  Algorithm: {algo_name}\n'
        f"{diag_ctx}\n{schema_block}\n{goal_block}{col_docs_block}\n"
        f"MANDATORY: features array MUST contain {min_f}-{max_f} column names (aim for ~{target_f}).\n"
        f"Cover monetary, frequency, recency/tenure, demographics, product, channel dimensions.\n\n"
        f"Output ONLY this JSON (no markdown):\n"
        f"{{\n"
        f'  "features": ["col1", "col2", ...],\n'
        f"  {param_instruction},\n"
        f'  "reason": "2-3 sentences on why these features produce meaningful segments."\n'
        f"}}"
    )

    raw     = seg_call_llm([{"role": "user", "content": prompt}], SYSTEM_DATA_SCIENTIST,
                           num_predict=1024, num_ctx=8192)
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = "\n".join(l for l in cleaned.split("\n") if not l.strip().startswith("```")).strip()
    bs, be = cleaned.find("{"), cleaned.rfind("}") + 1
    if bs != -1 and be > bs:
        cleaned = cleaned[bs:be]

    try:
        result = json.loads(cleaned)
    except Exception:
        fb = seg_get_fallback_features(df, algorithm)
        if algorithm == "kmeans" and elbow_k is not None:
            fb["params"]["k"] = elbow_k
        if algorithm == "dbscan" and kdist_eps is not None:
            fb["params"]["eps"] = kdist_eps
        return fb

    col_map = {c.lower(): c for c in df.columns}
    valid   = []
    for f in result.get("features", []):
        if f in df.columns:
            valid.append(f)
        elif str(f).lower() in col_map:
            valid.append(col_map[str(f).lower()])
    valid = [f for f in valid if not is_likely_id_column(f)]
    seen, deduped = set(), []
    for f in valid:
        if f not in seen:
            seen.add(f); deduped.append(f)
    valid = deduped

    if len(valid) < MIN_FEATURES:
        fb = seg_get_fallback_features(df, algorithm)
        if algorithm == "kmeans" and elbow_k is not None:
            fb["params"]["k"] = elbow_k
        if algorithm == "dbscan" and kdist_eps is not None:
            fb["params"]["eps"] = kdist_eps
        return fb

    result["features"] = valid[:MAX_FEATURES]

    if algorithm == "kmeans":
        if elbow_k is not None:
            result.setdefault("params", {})["k"] = elbow_k
        else:
            k = result.get("params", {}).get("k")
            if not isinstance(k, int) or k < 2 or k > 15:
                result.setdefault("params", {})["k"] = 4
    else:
        if kdist_eps is not None:
            result.setdefault("params", {})["eps"] = kdist_eps
        else:
            eps = result.get("params", {}).get("eps")
            if not isinstance(eps, (int, float)) or float(eps) <= 0:
                result.setdefault("params", {})["eps"] = 0.5
        ms = result.get("params", {}).get("min_samples")
        if not isinstance(ms, int) or ms < 2:
            result.setdefault("params", {})["min_samples"] = 5

    return result


# ── Preprocessing & Clustering ────────────────────────────────────────────────

def seg_preprocess_features(df: pd.DataFrame, features: list) -> tuple:
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("No valid features available for clustering.")
    X = df[features].copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median() if not X[col].isnull().all() else 0)
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "unknown")
    for col in X.select_dtypes(include=["object","category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] == 0:
        raise ValueError("All selected features are constant — cannot cluster.")
    return StandardScaler().fit_transform(X), X, {}


def _sample_for_silhouette(X: np.ndarray, labels: np.ndarray) -> tuple:
    n = len(X)
    if n <= MAX_SIL_ROWS:
        return X, labels
    rng = np.random.default_rng(42)
    unique_labels = np.unique(labels)
    per_cluster   = max(1, MAX_SIL_ROWS // len(unique_labels))
    idx = []
    for lbl in unique_labels:
        cluster_idx = np.where(labels == lbl)[0]
        take = min(per_cluster, len(cluster_idx))
        idx.append(rng.choice(cluster_idx, size=take, replace=False))
    idx = np.concatenate(idx)
    return X[idx], labels[idx]


def seg_run_kmeans(df: pd.DataFrame, features: list, k: int,
                   X_scaled: np.ndarray = None) -> tuple:
    if X_scaled is None:
        X_scaled, _, __ = seg_preprocess_features(df, features)
    k = max(2, min(int(k), len(df) - 1))
    model  = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(X_scaled)
    sil = -1.0
    if len(set(labels)) > 1:
        try:
            Xs, ls = _sample_for_silhouette(X_scaled, labels)
            sil = round(float(silhouette_score(Xs, ls)), 4)
        except Exception:
            pass
    df_out = df.copy()
    df_out["segment_id"] = labels
    return df_out, sil, model


def seg_run_dbscan(df: pd.DataFrame, features: list, eps: float, min_samples: int,
                   X_scaled: np.ndarray = None) -> tuple:
    if X_scaled is None:
        X_scaled, _, __ = seg_preprocess_features(df, features)
    sampled_n = None
    X_fit = X_scaled
    if len(X_scaled) > DBSCAN_MAX_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_scaled), DBSCAN_MAX_ROWS, replace=False)
        X_fit = X_scaled[idx]
        sampled_n = DBSCAN_MAX_ROWS
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples), algorithm="ball_tree", n_jobs=1)
    try:
        sample_labels = model.fit_predict(X_fit)
    except MemoryError:
        raise MemoryError(
            f"DBSCAN ran out of memory on {len(X_fit):,} rows. "
            "Try reducing eps or selecting fewer features."
        )
    if sampled_n is not None:
        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=1)
        nn.fit(X_fit)
        _, indices = nn.kneighbors(X_scaled)
        labels = sample_labels[indices.ravel()]
    else:
        labels = sample_labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = -1.0
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > n_clusters:
            try:
                Xm, lm = _sample_for_silhouette(X_scaled[mask], labels[mask])
                sil = round(float(silhouette_score(Xm, lm)), 4)
            except Exception:
                pass
    df_out = df.copy()
    df_out["segment_id"] = labels
    return df_out, sil, model, sampled_n


def seg_compute_cluster_stats(df_labeled: pd.DataFrame, features: list) -> pd.DataFrame:
    existing = [f for f in features if f in df_labeled.columns]
    total    = len(df_labeled)
    rows     = []
    for seg_id in sorted(df_labeled["segment_id"].unique()):
        subset = df_labeled[df_labeled["segment_id"] == seg_id]
        row    = {"segment_id": int(seg_id), "customer_count": int(len(subset)),
                  "pct_total": round(len(subset) / total * 100, 1)}
        for col in existing:
            if pd.api.types.is_numeric_dtype(df_labeled[col]):
                row[f"{col}_mean"] = round(float(subset[col].mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def seg_compute_cluster_thresholds(df_labeled: pd.DataFrame, features: list) -> pd.DataFrame:
    existing = [f for f in features if f in df_labeled.columns
                and pd.api.types.is_numeric_dtype(df_labeled[f])]
    seg_ids = sorted(df_labeled["segment_id"].unique())
    rows = []
    for feat in existing:
        row = {"Feature": feat, "Global Mean": round(float(df_labeled[feat].mean()), 2)}
        for seg_id in seg_ids:
            prefix = "Noise" if seg_id == -1 else f"Seg {seg_id}"
            subset = df_labeled[df_labeled["segment_id"] == seg_id][feat].dropna()
            row[f"{prefix} Min"]  = round(float(subset.min()),  2)
            row[f"{prefix} Mean"] = round(float(subset.mean()), 2)
            row[f"{prefix} Max"]  = round(float(subset.max()),  2)
        rows.append(row)
    return pd.DataFrame(rows)


# ── LLM Tasks ─────────────────────────────────────────────────────────────────

def _as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def seg_llm_name_segments(cluster_stats: pd.DataFrame, algorithm: str,
                           params: dict, sil_score: float, seg_goal: str = "") -> list:
    size_cols  = ["segment_id", "customer_count", "pct_total"]
    mean_cols  = [c for c in cluster_stats.columns if c not in size_cols][:8]
    stats_slim = cluster_stats[size_cols + mean_cols]
    goal_line  = f"Segmentation objective: {seg_goal.strip()}\n" if seg_goal.strip() else ""
    n_segs     = len(cluster_stats)
    out_tokens = max(512, n_segs * 150)
    prompt = (
        f'Algorithm: {"K-Means" if algorithm=="kmeans" else "DBSCAN"}\n'
        f"Parameters: {json.dumps(params)}  Silhouette: {sil_score}\n"
        f"{goal_line}\n{stats_slim.to_string(index=False)}\n\n"
        "For EACH segment_id provide a JSON array:\n"
        '[{"segment_id": 0, "name": "...", "profile_summary": "...", "key_characteristics": "..."}]\n'
        "No markdown. No products/offers."
    )
    raw = seg_call_llm([{"role": "user", "content": prompt}], SYSTEM_SEGMENT_NAMER,
                       num_predict=out_tokens, num_ctx=4096)
    if raw.startswith("[LLM error:"):
        return [{"segment_id": int(r["segment_id"]),
                 "name": f"Segment {int(r['segment_id'])}",
                 "profile_summary": f"Naming failed: {raw}",
                 "description": f"Naming failed: {raw}",
                 "key_characteristics": "—"} for _, r in cluster_stats.iterrows()]
    import re as _re
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = "\n".join(l for l in cleaned.split("\n") if not l.strip().startswith("```")).strip()
    cleaned = _re.sub(r",\s*([}\]])", r"\1", cleaned)
    as_, ae = cleaned.find("["), cleaned.rfind("]") + 1
    if as_ != -1 and ae > as_:
        cleaned = cleaned[as_:ae]
    try:
        result = json.loads(cleaned)
        if isinstance(result, list) and result:
            for item in result:
                if "profile_summary" in item and "description" not in item:
                    item["description"] = item["profile_summary"]
            return result
    except Exception:
        pass
    return [{"segment_id": int(r["segment_id"]),
             "name": f"Segment {int(r['segment_id'])}",
             "profile_summary": f"LLM format error. Raw: {raw[:200]}",
             "description": "LLM interpretation unavailable.",
             "key_characteristics": "—"} for _, r in cluster_stats.iterrows()]


def seg_llm_tuning_advice(cluster_stats: pd.DataFrame, algorithm: str,
                           params: dict, sil_score: float) -> str:
    stats_str = cluster_stats[["segment_id","customer_count","pct_total"]].to_string(index=False)
    if algorithm == "kmeans":
        ctx = f"K-Means run. K={params.get('k','?')}. Focus on K adjustment only. Do NOT mention eps."
    else:
        n_noise = 0
        if -1 in cluster_stats["segment_id"].values:
            n_noise = int(cluster_stats[cluster_stats["segment_id"]==-1]["customer_count"].sum())
        ctx = (f"DBSCAN run. eps={params.get('eps','?')} min_samples={params.get('min_samples','?')}. "
               f"Noise points: {n_noise}. Do NOT mention K.")
    prompt = f"{ctx}\nSilhouette: {sil_score}\n{stats_str}\n4-5 bullets. End: RECOMMENDED NEXT STEP: ..."
    return seg_call_llm([{"role": "user", "content": prompt}], SYSTEM_TUNING_ADVISOR)


def seg_llm_compare_runs(stats1, stats2, names1, names2, algorithm, p1, p2, sil1, sil2) -> str:
    def _fmt(stats, names, params, sil):
        nm = {s["segment_id"]: s.get("name","") for s in names}
        lines = [f"Params: {json.dumps(params)}  Silhouette: {sil}"]
        for _, r in stats.iterrows():
            sid = int(r["segment_id"])
            lines.append(f"  Seg {sid} ({nm.get(sid,'?')}): {int(r['customer_count'])} ({r['pct_total']}%)")
        return "\n".join(lines)
    prompt = (
        f'Algorithm: {"K-Means" if algorithm=="kmeans" else "DBSCAN"}\n\n'
        f"RUN 1:\n{_fmt(stats1, names1, p1, sil1)}\n\n"
        f"RUN 2:\n{_fmt(stats2, names2, p2, sil2)}\n\n"
        "One-sentence verdict. 4-5 bullets. End: FINAL RECOMMENDATION: [...]"
    )
    return seg_call_llm([{"role": "user", "content": prompt}], SYSTEM_COMPARISON_ANALYST)


# ── Business override helpers ─────────────────────────────────────────────────

def seg_llm_translate_rule(description: str, df: pd.DataFrame,
                            feature_cols=None) -> str:
    cols = [c for c in (feature_cols or df.columns) if c in df.columns]
    col_lines = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_lines.append(f"  {col}: numeric  min={df[col].min():.2f}  max={df[col].max():.2f}")
        else:
            top = df[col].value_counts().head(4).index.tolist()
            col_lines.append(f"  {col}: categorical  sample_values={top}")
    prompt = f"Available columns:\n{chr(10).join(col_lines)}\n\nRule: {description}"
    raw = seg_call_llm([{"role": "user", "content": prompt}],
                       SYSTEM_RULE_TRANSLATOR, temperature=0.0,
                       num_predict=256, num_ctx=2048)
    if raw.startswith("[LLM error:"):
        raise RuntimeError(f"LLM unavailable: {raw}")
    cleaned = raw.strip().strip("`")
    if "```" in cleaned:
        cleaned = "\n".join(l for l in cleaned.split("\n") if not l.strip().startswith("```")).strip()
    return cleaned


def _safe_eval_mask(expr: str, df: pd.DataFrame) -> pd.Series:
    allowed = {"df": df, "pd": pd, "np": np}
    result = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
    if not isinstance(result, pd.Series):
        raise ValueError("Expression did not return a boolean Series.")
    return result.astype(bool)


def seg_apply_all_rules(df_base: pd.DataFrame, rules: list) -> pd.DataFrame:
    df = df_base.copy()
    for rule in rules:
        try:
            mask = _safe_eval_mask(rule["condition"], df)
            if rule.get("type") == "rethreshold":
                in_target = df["segment_id"] == rule["segment_id"]
                df.loc[in_target & ~mask, "segment_id"] = rule["remainder_id"]
            else:
                df.loc[mask, "segment_id"] = rule["segment_id"]
        except Exception:
            pass
    return df


# ── Elbow / k-distance ────────────────────────────────────────────────────────

def _sample_for_diagnostics(X_scaled: np.ndarray) -> np.ndarray:
    n = len(X_scaled)
    if n <= MAX_DIAG_ROWS:
        return X_scaled
    rng = np.random.default_rng(42)
    return X_scaled[rng.choice(n, size=MAX_DIAG_ROWS, replace=False)]


def seg_compute_elbow_data(X_scaled: np.ndarray, k_max: int = 10, progress_cb=None) -> tuple:
    Xs    = _sample_for_diagnostics(X_scaled)
    k_max = min(k_max, len(Xs) - 1)
    k_values, inertias, sil_scores = [], [], []
    for k in range(2, k_max + 1):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        try:
            sil_scores.append(round(float(silhouette_score(Xs, labels)), 4))
        except Exception:
            sil_scores.append(0.0)
        k_values.append(k)
        if progress_cb:
            progress_cb(k, k_max)
    return k_values, inertias, sil_scores


def seg_find_elbow_k(k_values: list, inertias: list) -> int:
    if len(k_values) < 3:
        return k_values[0]
    x  = np.array(k_values, dtype=float)
    y  = np.array(inertias, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    vec = np.array([xn[-1] - xn[0], yn[-1] - yn[0]])
    vl  = np.linalg.norm(vec)
    dists = [abs((xi-xn[0])*vec[1]-(yi-yn[0])*vec[0])/(vl+1e-12) for xi,yi in zip(xn,yn)]
    return int(k_values[int(np.argmax(dists))])


def seg_find_knee_eps(point_idx: list, distances: list) -> float:
    if len(distances) < 3:
        return round(float(distances[0]), 4)
    x  = np.array(point_idx, dtype=float)
    y  = np.array(distances, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    vec = np.array([xn[-1] - xn[0], yn[-1] - yn[0]])
    vl  = np.linalg.norm(vec)
    dists = [abs((xi-xn[0])*vec[1]-(yi-yn[0])*vec[0])/(vl+1e-12) for xi,yi in zip(xn,yn)]
    return round(float(distances[int(np.argmax(dists))]), 4)


def seg_compute_kdist_data(X_scaled: np.ndarray, min_samples: int) -> tuple:
    Xs   = _sample_for_diagnostics(X_scaled)
    k    = max(2, int(min_samples))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(Xs)
    dists, _ = nbrs.kneighbors(Xs)
    sorted_d = np.sort(dists[:, -1])[::-1]
    return list(range(len(sorted_d))), list(sorted_d)


def seg_make_elbow_fig(k_values, inertias, sil_scores, elbow_k,
                       marked_k=None, run_label="") -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#ffffff"); ax1.set_facecolor("#f8fafc")
    ax1.yaxis.grid(True, color="#e2e8f0", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.plot(k_values, inertias, color="#0284c7", linewidth=2.5, marker="o",
             markersize=7, markerfacecolor="#ffffff", markeredgecolor="#0284c7",
             markeredgewidth=1.8, label="WCSS (Inertia)")
    ax1.fill_between(k_values, inertias, alpha=0.07, color="#0284c7")
    ax1.set_xlabel("K (Number of Clusters)", fontsize=11)
    ax1.set_ylabel("WCSS", color="#0284c7", fontsize=10)
    ax1.tick_params(axis="y", colors="#0284c7"); ax1.set_xticks(k_values)
    ax2 = ax1.twinx()
    ax2.plot(k_values, sil_scores, color="#7c3aed", linewidth=2.2, linestyle="--",
             marker="s", markersize=5, label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color="#7c3aed", fontsize=10)
    ax2.tick_params(axis="y", colors="#7c3aed")
    ei = k_values.index(elbow_k)
    ax1.scatter([elbow_k], [inertias[ei]], marker="*", s=380, color="#d97706",
                zorder=7, label=f"Elbow K={elbow_k}")
    ax1.axvline(elbow_k, color="#d97706", linewidth=1, linestyle=":", alpha=0.5)
    if marked_k is not None and marked_k in k_values:
        mi = k_values.index(marked_k)
        ax1.scatter([marked_k], [inertias[mi]], marker="D", s=200, color="#e11d48",
                    zorder=8, label=f"Selected K={marked_k}")
        ax1.axvline(marked_k, color="#e11d48", linewidth=1, linestyle=":", alpha=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    fig.suptitle(f"K-Means Elbow Analysis" + (f"  ·  {run_label}" if run_label else ""),
                 fontsize=12, fontweight="bold")
    plt.tight_layout(pad=1.4)
    return fig


def seg_make_kdist_fig(point_idx, distances, knee_eps, min_samples,
                       marked_eps=None, run_label="") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#f8fafc")
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.plot(point_idx, distances, color="#0284c7", linewidth=2.2,
            label=f"k-dist (k={min_samples})")
    ax.fill_between(point_idx, distances, alpha=0.07, color="#0284c7")
    ax.set_xlabel("Points (sorted by decreasing k-distance)", fontsize=11)
    ax.set_ylabel(f"{min_samples}-NN Distance", fontsize=10)
    dist_arr = np.array(distances)
    ki = int(np.argmin(np.abs(dist_arr - knee_eps)))
    ax.scatter([point_idx[ki]], [distances[ki]], marker="*", s=380, color="#d97706",
               zorder=7, label=f"Knee eps={knee_eps:.4f}")
    ax.axhline(knee_eps, color="#d97706", linewidth=1, linestyle=":", alpha=0.5)
    if marked_eps is not None:
        si = int(np.argmin(np.abs(dist_arr - float(marked_eps))))
        ax.scatter([point_idx[si]], [distances[si]], marker="D", s=200, color="#e11d48",
                   zorder=8, label=f"Selected eps={marked_eps:.4f}")
        ax.axhline(marked_eps, color="#e11d48", linewidth=1, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"DBSCAN k-Distance Graph" + (f"  ·  {run_label}" if run_label else ""),
                 fontsize=12, fontweight="bold")
    plt.tight_layout(pad=1.4)
    return fig


# ── Excel export ──────────────────────────────────────────────────────────────

def seg_build_excel(df_labeled: pd.DataFrame, segment_names: list,
                    cluster_stats: pd.DataFrame, table_name: str,
                    algorithm: str, params: dict, sil_score: float,
                    features=None) -> bytes:
    name_map = {s["segment_id"]: s.get("name", f"Segment {s['segment_id']}") for s in segment_names}
    df_out   = df_labeled.copy()
    df_out["segment_label"] = df_out["segment_id"].map(name_map).fillna("Unclassified")

    summary_rows = []
    for s in segment_names:
        sid = s["segment_id"]
        r   = cluster_stats[cluster_stats["segment_id"] == sid]
        cnt = int(r["customer_count"].values[0]) if not r.empty else 0
        pct = float(r["pct_total"].values[0])    if not r.empty else 0.0
        summary_rows.append({
            "Segment ID":          sid,
            "Segment Name":        s.get("name",""),
            "Customers":           cnt,
            "% of Total":          round(pct, 1),
            "Profile Summary":     s.get("profile_summary", s.get("description","")),
            "Key Characteristics": s.get("key_characteristics",""),
        })

    df_meta = pd.DataFrame([
        {"Field": "Generated",        "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Field": "Source Table",     "Value": table_name},
        {"Field": "Algorithm",        "Value": "K-Means" if algorithm=="kmeans" else "DBSCAN"},
        {"Field": "Parameters",       "Value": json.dumps(params)},
        {"Field": "Silhouette Score", "Value": sil_score},
        {"Field": "Total Customers",  "Value": len(df_labeled)},
    ])

    _SUMMARY_COLS  = ["Segment ID","Segment Name","Customers","% of Total",
                      "Profile Summary","Key Characteristics"]
    _EXCEL_ROW_LIMIT = 200_000
    _front   = ["segment_id", "segment_label"]
    _rest    = [c for c in df_out.columns if c not in _front]
    df_export  = df_out[_front + _rest].head(_EXCEL_ROW_LIMIT)
    was_capped = len(df_out) > _EXCEL_ROW_LIMIT

    try:
        import xlsxwriter as _xlsxwriter  # noqa: F401
        _engine = "xlsxwriter"
    except ImportError:
        _engine = "openpyxl"

    def _safe_write(df_sheet, writer, sheet_name):
        try:
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception as exc:
            pd.DataFrame({"Sheet": [sheet_name], "Error": [str(exc)]}).to_excel(
                writer, sheet_name=sheet_name, index=False)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine=_engine) as writer:
        _safe_write(df_export, writer, "Labeled Data")
        _safe_write(pd.DataFrame(summary_rows, columns=_SUMMARY_COLS), writer, "Segment Summary")
        _safe_write(cluster_stats, writer, "Cluster Stats")
        if features:
            thresh_df = seg_compute_cluster_thresholds(df_labeled, features)
            if thresh_df.empty:
                thresh_df = pd.DataFrame({"Note": ["No numeric features available"]})
            _safe_write(thresh_df, writer, "Feature Thresholds")
        if was_capped:
            df_meta = pd.concat([df_meta, pd.DataFrame([{
                "Field": "Export Note",
                "Value": f"Labeled Data capped at {_EXCEL_ROW_LIMIT:,} rows."
            }])], ignore_index=True)
        _safe_write(df_meta, writer, "Metadata")
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────

_SEG_PHASE_RESETS = {
    "connection": list(SEG_DEFAULTS.keys()),
    "table": [
        "seg_table_name","seg_col_docs","seg_goal","seg_algorithm","seg_user_role",
        "seg_df","seg_features","seg_params","seg_df_labeled","seg_cluster_stats",
        "seg_segment_names","seg_sil_score","seg_run_history","seg_iteration",
        "seg_elbow_data","seg_kdist_data","seg_tuning_advice","seg_comparison",
        "seg_chat_history","seg_chat_context","seg_X_scaled_cache","seg_X_scaled_features",
    ],
    "algorithm": [
        "seg_algorithm","seg_user_role",
        "seg_df","seg_features","seg_params","seg_df_labeled","seg_cluster_stats",
        "seg_segment_names","seg_sil_score","seg_run_history","seg_iteration",
        "seg_elbow_data","seg_kdist_data","seg_tuning_advice","seg_comparison",
        "seg_chat_history","seg_chat_context","seg_X_scaled_cache","seg_X_scaled_features",
    ],
    "role": [
        "seg_user_role",
        "seg_df","seg_features","seg_params","seg_df_labeled","seg_cluster_stats",
        "seg_segment_names","seg_sil_score","seg_run_history","seg_iteration",
        "seg_elbow_data","seg_kdist_data","seg_tuning_advice","seg_comparison",
        "seg_chat_history","seg_chat_context","seg_X_scaled_cache","seg_X_scaled_features",
        "seg_df_labeled_base","seg_segment_rules","seg_pending_rule","seg_pending_rethreshold",
    ],
}


def _seg_go_to_phase(target: str):
    for key in _SEG_PHASE_RESETS.get(target, []):
        st.session_state[key] = SEG_DEFAULTS[key]
    st.session_state.seg_phase = target


def seg_render_sidebar():
    phases = [
        ("connection", "1. Database Connection"),
        ("table",      "2. Table Selection"),
        ("algorithm",  "3. Clustering Technique"),
        ("role",       "4. User Role"),
        ("running",    "5. Analysis"),
    ]
    phase_order = [p[0] for p in phases]
    current_idx = (phase_order.index(st.session_state.seg_phase)
                   if st.session_state.seg_phase in phase_order else 4)

    st.sidebar.subheader("Segmentation Steps")
    for i, (key, label) in enumerate(phases):
        if i < current_idx:
            if st.sidebar.button(f"✅ {label}", key=f"seg_nav_{key}", use_container_width=True):
                _seg_go_to_phase(key)
                st.rerun()
        elif i == current_idx:
            st.sidebar.markdown(f"**▶ {label}**")
        else:
            st.sidebar.markdown(f"⬜ {label}")

    st.sidebar.divider()
    if st.session_state.seg_phase not in ("connection",):
        if st.sidebar.button("🔄 Reset Segmentation", use_container_width=True,
                             key="seg_reset_btn"):
            for k, v in SEG_DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()
    st.sidebar.caption(f"Model: `{MODEL}`")
    st.sidebar.caption("Ollama: " + ("🟢 Online" if seg_check_ollama() else "🔴 Offline"))


# ── Phase renderers ───────────────────────────────────────────────────────────

def seg_phase_connection():
    st.title("Customer Segmentation Agent")
    st.subheader("Step 1 — Database Connection")
    st.markdown("""
Connect to your database. Supported formats:

| Type | Example |
|------|---------|
| SQLite | `C:/path/to/file.db` |
| PostgreSQL | `postgresql://user:pass@host:5432/dbname` |
| MySQL | `mysql+pymysql://user:pass@host:3306/dbname` |
| SQL Server | `mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server` |
| Azure Fabric | Paste the ADO.NET string from the Fabric portal |
""")
    conn_str = st.text_input("Connection string", type="password",
                              placeholder="e.g. C:/data/nbo_ads.db",
                              key="seg_conn_input")
    if st.button("Connect", type="primary",
                 disabled=not conn_str.strip(), key="seg_connect_btn"):
        with st.spinner("Connecting…"):
            try:
                engine = seg_open_engine(conn_str.strip())
                with engine.connect() as c:
                    c.execute(text("SELECT 1"))
                tables = seg_list_tables(engine)
                st.session_state.seg_conn_string = conn_str.strip()
                st.session_state.seg_engine      = engine
                st.session_state.seg_all_tables  = tables
                st.session_state.seg_phase       = "table"
                st.rerun()
            except Exception as e:
                _show_conn_error(e)


def seg_phase_table():
    st.subheader("Step 2 — Table Selection")
    engine = st.session_state.seg_engine
    tables = st.session_state.seg_all_tables

    if not tables:
        st.error("No tables found in this database.")
        return

    info = []
    for t in tables:
        try:
            n    = seg_get_row_count(engine, t)
            cols = len(sa_inspect(engine).get_columns(t))
        except Exception:
            n, cols = "?", "?"
        info.append({"Table": t, "Rows": n, "Columns": cols})
    st.dataframe(pd.DataFrame(info), use_container_width=True, hide_index=True)

    chosen = st.selectbox("Select a table to segment", tables, key="seg_table_select")

    st.markdown("---")
    st.markdown("#### 🎯 Segmentation Objective *(required)*")
    st.caption(
        "Describe what you want to achieve with this segmentation. "
        "The LLM uses this to decide which columns matter and which to ignore."
    )
    seg_goal = st.text_area(
        "Segmentation goal",
        value=st.session_state.seg_goal,
        placeholder=(
            "e.g. 'Identify high-value customers for a premium credit card campaign', "
            "'Group customers by digital channel behaviour'"
        ),
        height=90,
        label_visibility="collapsed",
        key="seg_goal_input",
    )

    st.markdown("---")
    st.markdown("#### 📄 Column Description Documents *(optional)*")
    st.caption("Upload PDFs, Word docs, or Excel files describing the table columns.")
    uploaded_docs = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "xlsx", "xls"],
        accept_multiple_files=True,
        key="seg_doc_uploader",
        label_visibility="collapsed",
    )
    if uploaded_docs:
        st.success(f"{len(uploaded_docs)} document(s) ready: "
                   + ", ".join(f.name for f in uploaded_docs))

    if st.button("Continue →", type="primary", key="seg_table_continue"):
        if not seg_goal.strip():
            st.error("Please describe your segmentation objective before continuing.")
            st.stop()
        st.session_state.seg_table_name = chosen
        st.session_state.seg_goal       = seg_goal.strip()
        if uploaded_docs:
            with st.spinner("Reading documents…"):
                st.session_state.seg_col_docs = seg_extract_doc_text(uploaded_docs)
        else:
            st.session_state.seg_col_docs = ""
        st.session_state.seg_phase = "algorithm"
        st.rerun()


def seg_phase_algorithm():
    """
    Algorithm selection — RADIO BUTTONS (matching original_segmentation.py).
    NOT tabs.
    """
    st.subheader("Step 3 — Clustering Technique")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### K-Means")
        st.markdown("""
- You specify the **number of clusters** (K)
- Produces equal-sized, spherical groups
- Best when you have an idea how many customer groups exist
- Fast and predictable
""")
    with col2:
        st.markdown("### DBSCAN")
        st.markdown("""
- The **data decides** the number of clusters
- Finds clusters of arbitrary shape
- Automatically flags **outlier** customers as noise
- Best when the number of groups is unknown
""")

    # ── RADIO BUTTONS — same as original_segmentation.py ──────────────────
    algo = st.radio(
        "Choose algorithm",
        ["K-Means", "DBSCAN"],
        horizontal=True,
        label_visibility="collapsed",
        key="seg_algo_radio",
    )

    if st.button("Continue →", type="primary", key="seg_algo_continue"):
        st.session_state.seg_algorithm = "kmeans" if algo == "K-Means" else "dbscan"
        st.session_state.seg_phase     = "role"
        st.rerun()


def seg_phase_role():
    """
    Role selection — RADIO BUTTONS (matching original_segmentation.py).
    NOT tabs.
    """
    st.subheader("Step 4 — User Role")
    st.markdown("Choose your role to get an experience tailored to your needs.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔬 Data Scientist")
        st.markdown("""
- Full schema with statistics
- Elbow / k-distance diagnostic charts
- LLM feature selection with manual override
- Multiple run comparison
- Detailed tuning recommendations
""")
    with col2:
        st.markdown("### 💼 Business User")
        st.markdown("""
- Automatic feature selection and clustering
- Plain-language segment descriptions
- Chat-based Q&A about segments
- Apply company policies to re-interpret segments
- No technical jargon
""")

    # ── RADIO BUTTONS — same as original_segmentation.py ──────────────────
    role = st.radio(
        "I am a",
        ["Data Scientist", "Business User"],
        horizontal=True,
        label_visibility="collapsed",
        key="seg_role_radio",
    )

    if st.button("Start Analysis →", type="primary", key="seg_role_continue"):
        st.session_state.seg_user_role = "technical" if role == "Data Scientist" else "business"
        st.session_state.seg_phase     = "running"
        st.rerun()


# ── Technical analysis ────────────────────────────────────────────────────────

def seg_phase_technical():
    engine     = st.session_state.seg_engine
    table_name = st.session_state.seg_table_name
    algorithm  = st.session_state.seg_algorithm
    algo_label = "K-Means" if algorithm == "kmeans" else "DBSCAN"

    st.subheader(f"Technical Analysis — {table_name}  ·  {algo_label}")

    if st.session_state.seg_df is None:
        with st.spinner("Loading table…"):
            st.session_state.seg_df = seg_load_table(engine, table_name)

    df = st.session_state.seg_df
    st.caption(f"{len(df):,} rows  ·  {len(df.columns)} columns")

    with st.expander("📋 Schema", expanded=False):
        schema_rows = []
        for col in df.columns:
            dtype    = str(df[col].dtype)
            null_pct = round(df[col].isnull().mean() * 100, 1)
            if "int" in dtype or "float" in dtype:
                schema_rows.append({
                    "Column": col, "Type": "numeric",
                    "Min": round(float(df[col].min()), 2) if not df[col].isnull().all() else None,
                    "Max": round(float(df[col].max()), 2) if not df[col].isnull().all() else None,
                    "Median": round(float(df[col].median()), 2) if not df[col].isnull().all() else None,
                    "Unique": int(df[col].nunique()), "Nulls%": f"{null_pct}%",
                })
            else:
                schema_rows.append({
                    "Column": col, "Type": "text/cat",
                    "Min": None, "Max": None, "Median": None,
                    "Unique": int(df[col].nunique()), "Nulls%": f"{null_pct}%",
                })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    # ── Feature Selection ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 1 — Feature Selection")
    if st.session_state.seg_goal:
        st.info(f"**Segmentation objective:** {st.session_state.seg_goal}")

    all_candidates = [
        c for c in df.columns
        if not is_likely_id_column(c)
        and df[c].isnull().mean() < 0.5
        and df[c].nunique() > 1
    ]

    def _feat_label(col: str) -> str:
        null_pct = round(df[col].isnull().mean() * 100, 1)
        if pd.api.types.is_numeric_dtype(df[col]):
            lo = df[col].min(); hi = df[col].max()
            return f"{col} — Numeric, {null_pct}% null, {lo:.1f}–{hi:.1f}"
        return f"{col} — Categorical, {null_pct}% null, {df[col].nunique()} values"

    label_of = {c: _feat_label(c) for c in all_candidates}
    col_of   = {v: k for k, v in label_of.items()}

    # Stable cache key — algorithm-agnostic (feature selection is independent of algo)
    feat_key = (
        "feat", table_name,
        (st.session_state.seg_col_docs or "")[:80],
        (st.session_state.seg_goal or "")[:80],
    )

    c_llm, c_all, c_none, c_reset = st.columns([4, 1.2, 1.2, 1.2])

    with c_llm:
        llm_btn = st.button("🤖 Let LLM select features", use_container_width=True,
                            key="seg_llm_feat_btn")
    with c_all:
        all_btn = st.button("Select All", use_container_width=True, key="seg_all_feat")
    with c_none:
        none_btn = st.button("Clear All", use_container_width=True, key="seg_clear_feat")
    with c_reset:
        reset_btn = st.button("↺ Reset", use_container_width=True, key="seg_reset_feat")

    # Handle button actions — write to session state then rerun so multiselect
    # default re-reads the new value on the next render pass.
    if llm_btn:
        with st.spinner("LLM analysing schema… this may take 30–60 s"):
            if feat_key in st.session_state.seg_strategy_cache:
                result = st.session_state.seg_strategy_cache[feat_key]
            else:
                result = seg_llm_select_features(
                    df, table_name,
                    col_docs=st.session_state.seg_col_docs,
                    seg_goal=st.session_state.seg_goal,
                )
                st.session_state.seg_strategy_cache[feat_key] = result
            st.session_state.seg_features    = result.get("features", [])
            st.session_state.seg_elbow_data  = None
            st.session_state.seg_kdist_data  = None
            st.session_state.seg_params      = {}
        st.rerun()

    if all_btn:
        st.session_state.seg_features   = all_candidates[:]
        st.session_state.seg_elbow_data = None
        st.session_state.seg_kdist_data = None
        st.session_state.seg_params     = {}
        st.rerun()

    if none_btn:
        st.session_state.seg_features   = []
        st.session_state.seg_elbow_data = None
        st.session_state.seg_kdist_data = None
        st.session_state.seg_params     = {}
        st.rerun()

    if reset_btn:
        st.session_state.seg_strategy_cache.pop(feat_key, None)
        st.session_state.seg_features   = []
        st.session_state.seg_elbow_data = None
        st.session_state.seg_kdist_data = None
        st.session_state.seg_params     = {}
        st.rerun()

    # ── Multiselect — NO key= so that `default` is re-evaluated on every rerun ──
    # This is the critical fix: when a key is set on st.multiselect, Streamlit
    # locks the widget to its internal widget-state cache and ignores the
    # `default` parameter after the first render.  The original segmentation app
    # (original_segmentation.py) does NOT use a key here, which is why features
    # appeared correctly there.  Removing the key restores that behaviour.
    valid_current  = [f for f in st.session_state.seg_features if f in label_of]
    default_labels = [label_of[f] for f in valid_current]

    chosen_labels = st.multiselect(
        f"Features to use for clustering  *({len(all_candidates)} eligible columns)*",
        options=list(label_of.values()),
        default=default_labels,
        help="Each option shows: column name — type, null%, range/cardinality. "
             "Changing the selection clears the diagnostic so it re-runs on the new set.",
        # NO key= here — intentional, matches original_segmentation.py
    )
    chosen_features = [col_of[l] for l in chosen_labels]

    # Sync manual edits back to session state
    if chosen_features != st.session_state.seg_features:
        st.session_state.seg_features   = chosen_features
        st.session_state.seg_elbow_data = None
        st.session_state.seg_kdist_data = None
        st.session_state.seg_params     = {}

    # LLM reasoning / fallback warning
    cached_feat = st.session_state.seg_strategy_cache.get(feat_key)
    if cached_feat:
        if cached_feat.get("_fallback"):
            st.warning(
                "LLM feature selection fell back to heuristic keyword matching. "
                "You can edit the list above or click **Let LLM select features** to retry."
            )
            with st.expander("🔍 Why did the LLM fall back?", expanded=True):
                st.markdown(f"**Reason:** {cached_feat.get('_fallback_reason','unknown')}")
                st.code(cached_feat.get("_raw_response","(no response)"), language="text")
        elif cached_feat.get("reason"):
            with st.expander("💡 Why these features?", expanded=False):
                st.caption(cached_feat["reason"])

    if not st.session_state.seg_features:
        st.info("Click **Let LLM select features** or choose columns above to continue.")
        return

    # ── Diagnostic ──────────────────────────────────────────────────────────
    st.markdown("---")
    diag_label = "Elbow" if algorithm == "kmeans" else "k-Distance"
    st.markdown(f"#### Step 2 — {diag_label} Diagnostic  *(on your selected features)*")

    n_rows    = len(df)
    diag_done = (
        (algorithm == "kmeans" and st.session_state.seg_elbow_data is not None) or
        (algorithm == "dbscan" and st.session_state.seg_kdist_data is not None)
    )

    if not diag_done:
        sample_note = (
            f"  (will sample {MAX_DIAG_ROWS:,} rows — table has {n_rows:,})"
            if n_rows > MAX_DIAG_ROWS else ""
        )
        if st.button(f"Run {diag_label} Analysis on selected features{sample_note}",
                     key="seg_run_diag"):
            X_s, *_ = seg_preprocess_features(df, st.session_state.seg_features)
            if algorithm == "kmeans":
                pbar = st.progress(0, text="Starting…")
                def _cb(k, k_max):
                    pbar.progress((k-1)/(k_max-1), text=f"K={k}/{k_max}")
                k_values, inertias, sil_scores = seg_compute_elbow_data(
                    X_s, k_max=10, progress_cb=_cb)
                pbar.progress(1.0, text="Complete!")
                elbow_k = seg_find_elbow_k(k_values, inertias)
                st.session_state.seg_elbow_data = (k_values, inertias, sil_scores, elbow_k)
                st.session_state.seg_params     = {"k": elbow_k}
            else:
                with st.spinner("Computing k-distance graph…"):
                    init_ms = seg_get_fallback_features(df, "dbscan")["params"].get("min_samples", 5)
                    pt_idx, dists = seg_compute_kdist_data(X_s, min_samples=init_ms)
                    knee_eps = seg_find_knee_eps(pt_idx, dists)
                    st.session_state.seg_kdist_data = (pt_idx, dists, knee_eps, init_ms)
                    st.session_state.seg_params     = {"eps": knee_eps, "min_samples": init_ms}
            st.rerun()
        return

    if algorithm == "kmeans":
        k_vals, iner, sils, elbow_k = st.session_state.seg_elbow_data
        fig = seg_make_elbow_fig(k_vals, iner, sils, elbow_k,
                                  marked_k=st.session_state.seg_params.get("k"))
        st.pyplot(fig); plt.close(fig)
        st.caption(f"Elbow suggests **K = {elbow_k}** — auto-applied below. Adjust freely.")
    else:
        pt_idx, dists, knee_eps, init_ms = st.session_state.seg_kdist_data
        fig = seg_make_kdist_fig(pt_idx, dists, knee_eps, init_ms,
                                  marked_eps=st.session_state.seg_params.get("eps"))
        st.pyplot(fig); plt.close(fig)
        st.caption(f"k-distance knee suggests **eps = {knee_eps}** — auto-applied below. Adjust freely.")

    # ── Parameters ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 3 — Parameters")
    if algorithm == "kmeans":
        elbow_k = st.session_state.seg_elbow_data[3]
        k_val   = st.number_input("K (number of clusters)", min_value=2, max_value=15,
                                   value=int(st.session_state.seg_params.get("k", elbow_k)),
                                   help=f"Elbow suggests K={elbow_k}. Change and re-run to compare.",
                                   key="seg_k_input")
        st.session_state.seg_params["k"] = int(k_val)
    else:
        kdist_eps = st.session_state.seg_kdist_data[2]
        init_ms   = st.session_state.seg_kdist_data[3]
        col1, col2 = st.columns(2)
        with col1:
            _eps_default = float(st.session_state.seg_params.get("eps", kdist_eps))
            eps_val = st.number_input("eps", min_value=min(0.001, _eps_default),
                                       max_value=10.0, step=0.001, value=_eps_default,
                                       format="%.4f",
                                       help=f"k-distance knee suggests eps={kdist_eps}",
                                       key="seg_eps_input")
        with col2:
            ms_val = st.number_input("min_samples", min_value=2, max_value=50,
                                      value=int(st.session_state.seg_params.get("min_samples", init_ms)),
                                      key="seg_ms_input")
        st.session_state.seg_params["eps"]         = float(eps_val)
        st.session_state.seg_params["min_samples"] = int(ms_val)

    # ── Run ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    next_iter  = st.session_state.seg_iteration
    has_results = st.session_state.seg_cluster_stats is not None

    if has_results:
        col_run, col_dl = st.columns(2)
        with col_run:
            run_btn = st.button(f"▶ Re-run {algo_label} with new parameters",
                                type="primary", use_container_width=True,
                                key="seg_rerun_btn")
        with col_dl:
            if st.button("📋 Prepare Download", use_container_width=True, key="seg_prep_dl"):
                with st.spinner("Building Excel…"):
                    st.session_state.seg_excel_bytes = seg_build_excel(
                        st.session_state.seg_df_labeled, st.session_state.seg_segment_names,
                        st.session_state.seg_cluster_stats, table_name, algorithm,
                        st.session_state.seg_params, st.session_state.seg_sil_score,
                        features=st.session_state.seg_features,
                    )
            if st.session_state.seg_excel_bytes:
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "⬇ Download current results",
                    data=st.session_state.seg_excel_bytes,
                    file_name=f"segmentation_{table_name}_{algorithm}_{_ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True, key="seg_dl_btn",
                )
    else:
        run_btn = st.button(f"▶ Run {algo_label} — Run {next_iter}",
                            type="primary", key="seg_run_btn")

    if run_btn:
        if (st.session_state.seg_X_scaled_cache is None or
                st.session_state.seg_X_scaled_features != st.session_state.seg_features):
            with st.spinner("Preprocessing…"):
                X_scaled, _, __ = seg_preprocess_features(df, st.session_state.seg_features)
                st.session_state.seg_X_scaled_cache    = X_scaled
                st.session_state.seg_X_scaled_features = st.session_state.seg_features[:]
        X_scaled = st.session_state.seg_X_scaled_cache

        with st.spinner(f"Clustering (Run {next_iter})…"):
            try:
                if algorithm == "kmeans":
                    df_labeled, sil, _ = seg_run_kmeans(
                        df, st.session_state.seg_features,
                        k=st.session_state.seg_params["k"], X_scaled=X_scaled)
                else:
                    df_labeled, sil, _, dbscan_sampled = seg_run_dbscan(
                        df, st.session_state.seg_features,
                        eps=st.session_state.seg_params["eps"],
                        min_samples=st.session_state.seg_params["min_samples"],
                        X_scaled=X_scaled)
                    if dbscan_sampled:
                        st.warning(f"DBSCAN fitted on {dbscan_sampled:,}-row sample; "
                                   "remaining rows assigned to nearest cluster.")
            except (ValueError, MemoryError) as e:
                st.error(f"Clustering failed: {e}"); return

            cluster_stats = seg_compute_cluster_stats(df_labeled, st.session_state.seg_features)
            st.session_state.seg_df_labeled    = df_labeled
            st.session_state.seg_cluster_stats = cluster_stats
            st.session_state.seg_segment_names = []
            st.session_state.seg_sil_score     = sil
            st.session_state.seg_tuning_advice = ""
            st.session_state.seg_excel_bytes   = None
            st.session_state.seg_comparison    = ""
            st.session_state.seg_run_history.append({
                "iteration": next_iter, "params": st.session_state.seg_params.copy(),
                "sil": sil, "stats": cluster_stats.copy(),
                "names": [], "features": st.session_state.seg_features[:],
            })
            st.session_state.seg_iteration += 1
        st.rerun()

    # ── Results ─────────────────────────────────────────────────────────────
    if st.session_state.seg_cluster_stats is not None:
        cluster_stats = st.session_state.seg_cluster_stats
        segment_names = st.session_state.seg_segment_names
        sil           = st.session_state.seg_sil_score
        params        = st.session_state.seg_params
        last_iter     = st.session_state.seg_iteration - 1

        st.markdown(f"### Results — Run {last_iter}")
        if sil > 0.5:
            sil_label = "Good ✅"
        elif sil > 0.25:
            sil_label = "Moderate ⚠️"
        elif sil < 0:
            sil_label = "N/A"
        else:
            sil_label = "Weak — tune recommended"

        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette Score", f"{sil}", delta=sil_label)
        m2.metric("Segments", len(cluster_stats[cluster_stats["segment_id"] != -1]))
        m3.metric("Parameters", f"K={params.get('k','?')}" if algorithm == "kmeans"
                  else f"eps={params.get('eps','?')} ms={params.get('min_samples','?')}")

        if not segment_names:
            with st.spinner("Naming segments…"):
                names = seg_llm_name_segments(cluster_stats, algorithm, params, sil,
                                               seg_goal=st.session_state.seg_goal)
                st.session_state.seg_segment_names = names
                if st.session_state.seg_run_history:
                    st.session_state.seg_run_history[-1]["names"] = names
            st.rerun()

        name_map     = {s["segment_id"]: s for s in segment_names}
        display_rows = []
        for _, row in cluster_stats.iterrows():
            sid  = int(row["segment_id"])
            info = name_map.get(sid, {})
            display_rows.append({
                "Seg ID":              sid,
                "Segment Name":        "⚠ NOISE" if sid == -1 else info.get("name", f"Seg {sid}"),
                "Customers":           int(row["customer_count"]),
                "% Total":             float(row["pct_total"]),
                "Profile Summary":     _as_str(info.get("profile_summary","")),
                "Key Characteristics": _as_str(info.get("key_characteristics","")),
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        with st.expander("📈 Cluster Statistics (feature means)", expanded=False):
            st.dataframe(cluster_stats, use_container_width=True, hide_index=True)

        with st.expander("📊 Feature Thresholds per Cluster", expanded=True):
            if st.session_state.seg_df_labeled is not None and st.session_state.seg_features:
                thresh_df = seg_compute_cluster_thresholds(
                    st.session_state.seg_df_labeled, st.session_state.seg_features)
                st.caption("Min / Mean / Max per feature per segment (original scale).")
                st.dataframe(thresh_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 🔧 Tuning Advice")
        if not st.session_state.seg_tuning_advice:
            with st.spinner("Generating tuning advice…"):
                st.session_state.seg_tuning_advice = seg_llm_tuning_advice(
                    cluster_stats, algorithm, params, sil)
            st.rerun()
        if st.session_state.seg_tuning_advice:
            st.markdown(st.session_state.seg_tuning_advice)

        if len(st.session_state.seg_run_history) >= 2 and not st.session_state.seg_comparison:
            st.markdown("---")
            st.markdown("#### 🔁 Run Comparison")
            if st.button("Compare with previous run", key="seg_compare_btn"):
                with st.spinner("Comparing runs…"):
                    prev = st.session_state.seg_run_history[-2]
                    curr = st.session_state.seg_run_history[-1]
                    st.session_state.seg_comparison = seg_llm_compare_runs(
                        prev["stats"], curr["stats"], prev["names"], curr["names"],
                        algorithm, prev["params"], curr["params"], prev["sil"], curr["sil"])
                st.rerun()
        if st.session_state.seg_comparison:
            st.markdown("---")
            st.markdown("#### 🔁 Run Comparison")
            st.markdown(st.session_state.seg_comparison)

        if algorithm == "kmeans" and st.session_state.seg_elbow_data is not None:
            k_vals, iner, sils, elbow_k = st.session_state.seg_elbow_data
            fig = seg_make_elbow_fig(k_vals, iner, sils, elbow_k,
                                      marked_k=params.get("k"),
                                      run_label=f"Run {last_iter}")
            with st.expander("📉 Elbow chart with selected K", expanded=False):
                st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.caption("Satisfied? Use **Prepare Download** above to export without re-running.")


# ── Business analysis ─────────────────────────────────────────────────────────

def seg_phase_business():
    engine     = st.session_state.seg_engine
    table_name = st.session_state.seg_table_name
    algorithm  = st.session_state.seg_algorithm
    algo_label = "K-Means" if algorithm == "kmeans" else "DBSCAN"

    st.subheader(f"Customer Segment Explorer — {table_name}")

    # ── Auto-cluster on first visit ─────────────────────────────────────────
    if st.session_state.seg_df is None:
        status = st.status("Setting up segmentation…", expanded=True)
        with status:
            st.write("Loading table…")
            df = seg_load_table(engine, table_name)
            st.session_state.seg_df = df
            n_rows = len(df)
            st.write(f"Loaded {n_rows:,} rows.")

            st.write("Asking LLM to select features…")
            feat_result = seg_llm_select_features(
                df, table_name,
                col_docs=st.session_state.seg_col_docs,
                seg_goal=st.session_state.seg_goal,
            )
            features = feat_result["features"]
            # ── Write features to session state so the multiselect is populated ──
            st.session_state.seg_features = features
            # Cache the result so clicking "Let LLM select" later is instant
            feat_key = (
                "feat", table_name,
                (st.session_state.seg_col_docs or "")[:80],
                (st.session_state.seg_goal or "")[:80],
            )
            st.session_state.seg_strategy_cache[feat_key] = feat_result
            st.write(f"Selected {len(features)} features.")

            X_s, *_ = seg_preprocess_features(df, features)

            if algorithm == "kmeans":
                st.write("Running elbow analysis" +
                         (f" on a {MAX_DIAG_ROWS:,}-row sample…" if n_rows > MAX_DIAG_ROWS else "…"))
                pbar = st.progress(0, text="K=2…")
                def _cb(k, k_max):
                    pbar.progress((k-1)/(k_max-1), text=f"Elbow: K={k}/{k_max}")
                k_values, inertias, sil_scores = seg_compute_elbow_data(
                    X_s, k_max=10, progress_cb=_cb)
                pbar.progress(1.0, text="Elbow done!")
                elbow_k = seg_find_elbow_k(k_values, inertias)
                st.session_state.seg_elbow_data = (k_values, inertias, sil_scores, elbow_k)
                st.write(f"Elbow suggests K = {elbow_k}.")
                params = {"k": elbow_k}
            else:
                st.write("Running k-distance analysis" +
                         (f" on a {MAX_DIAG_ROWS:,}-row sample…" if n_rows > MAX_DIAG_ROWS else "…"))
                init_ms = seg_get_fallback_features(df, "dbscan")["params"].get("min_samples", 5)
                pt_idx, dists = seg_compute_kdist_data(X_s, min_samples=init_ms)
                knee_eps = seg_find_knee_eps(pt_idx, dists)
                st.session_state.seg_kdist_data = (pt_idx, dists, knee_eps, init_ms)
                st.write(f"k-distance knee: eps = {knee_eps}.")
                params = {"eps": knee_eps, "min_samples": init_ms}

            st.session_state.seg_params = params
            st.write(f"Running {algo_label}…")

            try:
                if algorithm == "kmeans":
                    df_labeled, sil, _ = seg_run_kmeans(df, features, k=params["k"])
                else:
                    df_labeled, sil, _, dbscan_sampled = seg_run_dbscan(
                        df, features, eps=params["eps"], min_samples=params["min_samples"])
                    if dbscan_sampled:
                        st.warning(f"DBSCAN fitted on {dbscan_sampled:,}-row sample.")
            except (ValueError, MemoryError) as e:
                st.error(f"Clustering failed: {e}"); return

            cluster_stats = seg_compute_cluster_stats(df_labeled, features)
            st.write("Naming segments…")
            segment_names = seg_llm_name_segments(cluster_stats, algorithm, params, sil,
                                                   seg_goal=st.session_state.seg_goal)
            st.session_state.seg_df_labeled      = df_labeled
            st.session_state.seg_df_labeled_base = df_labeled.copy()
            st.session_state.seg_cluster_stats   = cluster_stats
            st.session_state.seg_segment_names   = segment_names
            st.session_state.seg_sil_score       = sil
            st.session_state.seg_excel_bytes     = None
            status.update(label="Segmentation complete!", state="complete", expanded=False)
        st.rerun()

    df            = st.session_state.seg_df
    cluster_stats = st.session_state.seg_cluster_stats
    segment_names = st.session_state.seg_segment_names
    sil           = st.session_state.seg_sil_score
    params        = st.session_state.seg_params

    if cluster_stats is None:
        st.info("Loading segmentation results…")
        return

    n_real = len(cluster_stats[cluster_stats["segment_id"] != -1])
    if algorithm == "dbscan" and n_real < 2:
        noise_pct = 0.0
        if -1 in cluster_stats["segment_id"].values:
            noise_pct = float(cluster_stats[cluster_stats["segment_id"]==-1]["pct_total"].values[0])
        st.warning(
            f"DBSCAN found only {n_real} real cluster ({noise_pct:.1f}% noise). "
            "eps may be too large. Consider switching to K-Means or using the Technical role."
        )

    st.markdown(f"We found **{n_real} customer segments** in **{table_name}** using {algo_label}.")
    name_map     = {s["segment_id"]: s for s in segment_names}
    display_rows = []
    for _, row in cluster_stats.iterrows():
        sid  = int(row["segment_id"])
        info = name_map.get(sid, {})
        display_rows.append({
            "Segment":             "⚠ Outliers / Noise" if sid == -1 else info.get("name", f"Segment {sid}"),
            "Customers":           int(row["customer_count"]),
            "% of Total":          float(row["pct_total"]),
            "Profile":             _as_str(info.get("profile_summary","")),
            "Key Characteristics": _as_str(info.get("key_characteristics","")),
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    with st.expander("🔬 How were these segments created?", expanded=False):
        if algorithm == "kmeans":
            st.markdown(
                f"**Algorithm:** K-Means | **K:** {params.get('k','?')} | "
                f"**Silhouette:** {sil}"
            )
            st.caption(
                f"K-Means split customers into exactly **{params.get('k','?')} groups**. "
                f"Silhouette score of **{sil}** measures cluster separation (>0.5 = good, 0.25–0.5 = moderate)."
            )
        else:
            st.markdown(
                f"**Algorithm:** DBSCAN | **ε:** {params.get('eps','?')} | "
                f"**min_samples:** {params.get('min_samples','?')} | **Silhouette:** {sil}"
            )
            st.caption(
                f"DBSCAN groups customers whose features are within **ε={params.get('eps','?')}** "
                f"of each other, requiring at least **{params.get('min_samples','?')} neighbours**."
            )
        if st.session_state.seg_df_labeled is not None and st.session_state.seg_features:
            thresh_df = seg_compute_cluster_thresholds(
                st.session_state.seg_df_labeled, st.session_state.seg_features)
            st.dataframe(thresh_df, use_container_width=True, hide_index=True)

    # ── Segment override ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("✏️ Customise Segment Assignments", expanded=False):
        tab_new, tab_rethr = st.tabs(["➕ Create New Segment", "🔧 Rethreshold Existing"])

        with tab_new:
            _all_cols  = [c for c in df.columns if c != "segment_id"]
            rule_cols  = st.multiselect(
                "Columns for this rule", options=_all_cols,
                default=[c for c in st.session_state.seg_features if c in _all_cols],
                key="seg_rule_cols",
            )
            rule_desc  = st.text_input("Rule description",
                                        placeholder='e.g. "balance > 50000 and products >= 3"',
                                        key="seg_rule_desc")
            seg_name   = st.text_input("Name for new segment",
                                        placeholder='e.g. "Premium Clients"',
                                        key="seg_new_seg_name")
            c1, c2     = st.columns([3, 1])
            with c1:
                translate_btn = st.button("🔍 Preview matching customers",
                                          disabled=not rule_desc.strip() or not rule_cols,
                                          use_container_width=True, key="seg_translate_btn")
            with c2:
                if st.button("Clear", use_container_width=True, key="seg_clear_rule"):
                    st.session_state.seg_pending_rule = None; st.rerun()
            if translate_btn and rule_desc.strip():
                with st.spinner("Translating rule…"):
                    expr = seg_llm_translate_rule(rule_desc.strip(), df, feature_cols=rule_cols)
                try:
                    mask  = _safe_eval_mask(expr, df)
                    count = int(mask.sum())
                    st.session_state.seg_pending_rule = {
                        "description": rule_desc.strip(), "condition": expr, "count": count}
                except Exception as e:
                    st.error(f"Could not parse: {e}\n\nExpression: `{expr}`")
                    st.session_state.seg_pending_rule = None

            pending = st.session_state.seg_pending_rule
            if pending:
                st.code(pending["condition"], language="python")
                st.info(f"**{pending['count']:,} customers** match.")
                if pending["count"] > 0:
                    if st.button(f'✅ Apply — move {pending["count"]:,} to '
                                 f'"{seg_name or "New Segment"}"',
                                 type="primary", key="seg_confirm_rule"):
                        new_id = max(st.session_state.seg_df_labeled["segment_id"].max() + 1, 100)
                        rule_entry = {
                            "description": pending["description"],
                            "condition":   pending["condition"],
                            "segment_id":  new_id,
                            "name":        seg_name.strip() or f"Custom Segment {new_id}",
                            "count":       pending["count"],
                        }
                        st.session_state.seg_segment_rules.append(rule_entry)
                        new_df    = seg_apply_all_rules(st.session_state.seg_df_labeled_base,
                                                        st.session_state.seg_segment_rules)
                        new_stats = seg_compute_cluster_stats(new_df, st.session_state.seg_features)
                        new_name_entry = {
                            "segment_id": new_id, "name": rule_entry["name"],
                            "profile_summary": f"Custom rule: {rule_entry['description']}",
                            "description":     f"Custom rule: {rule_entry['description']}",
                            "key_characteristics": rule_entry["description"],
                        }
                        st.session_state.seg_segment_names = [
                            n for n in st.session_state.seg_segment_names
                            if n["segment_id"] != new_id
                        ] + [new_name_entry]
                        st.session_state.seg_df_labeled    = new_df
                        st.session_state.seg_cluster_stats = new_stats
                        st.session_state.seg_pending_rule  = None
                        st.session_state.seg_chat_context  = ""
                        st.session_state.seg_chat_history  = []
                        st.rerun()

        with tab_rethr:
            selectable = [s for s in st.session_state.seg_segment_names if s["segment_id"] != -1]
            if not selectable:
                st.info("No segments available yet.")
            else:
                seg_options = {f"{s['name']} (Seg {s['segment_id']})": s["segment_id"]
                               for s in selectable}
                chosen_label = st.selectbox("Select segment", list(seg_options.keys()),
                                            key="seg_rethr_select")
                target_sid   = seg_options[chosen_label]
                rethr_desc   = st.text_input("New threshold rule",
                                              placeholder='e.g. "balance > 100000"',
                                              key="seg_rethr_desc")
                remainder_nm = st.text_input("Name for remainder group",
                                              placeholder='e.g. "Below Threshold"',
                                              key="seg_rethr_remainder_name")
                c1, c2 = st.columns([3, 1])
                with c1:
                    rethr_prev_btn = st.button("🔍 Preview impact",
                                               disabled=not rethr_desc.strip(),
                                               use_container_width=True, key="seg_rethr_prev")
                with c2:
                    if st.button("Clear", use_container_width=True, key="seg_rethr_clear"):
                        st.session_state.seg_pending_rethreshold = None; st.rerun()

                if rethr_prev_btn and rethr_desc.strip():
                    seg_df = st.session_state.seg_df_labeled[
                        st.session_state.seg_df_labeled["segment_id"] == target_sid]
                    if seg_df.empty:
                        st.warning("No customers in this segment.")
                    else:
                        with st.spinner("Translating threshold rule…"):
                            expr = seg_llm_translate_rule(rethr_desc.strip(), seg_df,
                                                          feature_cols=st.session_state.seg_features)
                        try:
                            mask     = _safe_eval_mask(expr, seg_df)
                            in_count  = int(mask.sum())
                            out_count = int((~mask).sum())
                            st.session_state.seg_pending_rethreshold = {
                                "target_sid": target_sid, "description": rethr_desc.strip(),
                                "condition": expr, "in_count": in_count, "out_count": out_count}
                        except Exception as e:
                            st.error(f"Could not parse: {e}\n\n`{expr}`")
                            st.session_state.seg_pending_rethreshold = None

                pending_rt = st.session_state.seg_pending_rethreshold
                if pending_rt and pending_rt.get("target_sid") == target_sid:
                    st.code(pending_rt["condition"], language="python")
                    col_a, col_b = st.columns(2)
                    col_a.metric("Stay in segment", f"{pending_rt['in_count']:,}")
                    col_b.metric("Moved to remainder", f"{pending_rt['out_count']:,}")
                    if pending_rt["out_count"] > 0 and pending_rt["in_count"] > 0:
                        remainder_label = remainder_nm.strip() or f"Remainder of {chosen_label}"
                        if st.button(
                            f'✅ Keep {pending_rt["in_count"]:,}, move {pending_rt["out_count"]:,} to '
                            f'"{remainder_label}"',
                            type="primary", key="seg_rethr_confirm"):
                            remainder_id = max(
                                st.session_state.seg_df_labeled["segment_id"].max() + 1, 100)
                            rule_entry = {
                                "type": "rethreshold",
                                "description": pending_rt["description"],
                                "condition": pending_rt["condition"],
                                "segment_id": target_sid,
                                "remainder_id": remainder_id,
                                "name": chosen_label,
                                "remainder_name": remainder_label,
                                "count": pending_rt["out_count"],
                            }
                            st.session_state.seg_segment_rules.append(rule_entry)
                            new_df    = seg_apply_all_rules(st.session_state.seg_df_labeled_base,
                                                            st.session_state.seg_segment_rules)
                            new_stats = seg_compute_cluster_stats(new_df, st.session_state.seg_features)
                            remainder_name_entry = {
                                "segment_id": remainder_id, "name": remainder_label,
                                "profile_summary": f"Rethreshold remainder: {pending_rt['description']}",
                                "description":     f"Rethreshold remainder: {pending_rt['description']}",
                                "key_characteristics": f"Did not meet: {pending_rt['description']}",
                            }
                            updated_names = []
                            for n in st.session_state.seg_segment_names:
                                if n["segment_id"] == target_sid:
                                    n = dict(n)
                                    n["profile_summary"] = (
                                        f"[Rethresholded] {_as_str(n.get('profile_summary',''))} "
                                        f"| Threshold: {pending_rt['description']}"
                                    ).strip()
                                elif n["segment_id"] == remainder_id:
                                    continue
                                updated_names.append(n)
                            updated_names.append(remainder_name_entry)
                            st.session_state.seg_df_labeled           = new_df
                            st.session_state.seg_cluster_stats        = new_stats
                            st.session_state.seg_segment_names        = updated_names
                            st.session_state.seg_pending_rethreshold  = None
                            st.session_state.seg_chat_context         = ""
                            st.session_state.seg_chat_history         = []
                            st.rerun()

        if st.session_state.seg_segment_rules:
            st.markdown("---")
            st.markdown("**Applied overrides:**")
            for i, rule in enumerate(st.session_state.seg_segment_rules):
                c1, c2 = st.columns([5, 1])
                with c1:
                    if rule.get("type") == "rethreshold":
                        st.markdown(f"**🔧 Rethreshold** {rule['name']}  \n"
                                    f"Threshold: _{rule['description']}_  \n"
                                    f"Remainder → **{rule['remainder_name']}** "
                                    f"({rule['count']:,} moved out)")
                    else:
                        st.markdown(f"**{rule['name']}** — {rule['count']:,} customers  \n"
                                    f"Rule: _{rule['description']}_")
                with c2:
                    if st.button("Undo", key=f"seg_undo_{i}"):
                        popped = st.session_state.seg_segment_rules.pop(i)
                        ids_to_remove = {popped["segment_id"]}
                        if popped.get("type") == "rethreshold":
                            ids_to_remove.add(popped["remainder_id"])
                        st.session_state.seg_segment_names = [
                            n for n in st.session_state.seg_segment_names
                            if n["segment_id"] not in ids_to_remove]
                        new_df    = seg_apply_all_rules(st.session_state.seg_df_labeled_base,
                                                        st.session_state.seg_segment_rules)
                        new_stats = seg_compute_cluster_stats(new_df, st.session_state.seg_features)
                        st.session_state.seg_df_labeled    = new_df
                        st.session_state.seg_cluster_stats = new_stats
                        st.session_state.seg_chat_context  = ""
                        st.session_state.seg_chat_history  = []
                        st.rerun()

    # ── Business chat ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Ask About Your Segments")
    st.caption("Ask questions about any segment. I will not suggest products.")

    if not st.session_state.seg_chat_context:
        ctx_lines = [
            f"Algorithm: {algo_label} | Silhouette: {sil} | Segments: {len(cluster_stats)}",
            f"Features: {', '.join(st.session_state.seg_features)}", "",
        ]
        for _, stat in cluster_stats.iterrows():
            sid  = int(stat["segment_id"])
            info = name_map.get(sid, {})
            ctx_lines.append(
                f"Segment {sid} — {info.get('name','N/A')}: "
                f"{int(stat['customer_count']):,} customers ({stat['pct_total']}%). "
                f"{_as_str(info.get('profile_summary',''))} "
                f"Key: {_as_str(info.get('key_characteristics',''))}"
            )
        st.session_state.seg_chat_context = "\n".join(ctx_lines)
        intro_prompt = (
            f"{st.session_state.seg_chat_context}\n\n"
            "Greet the user warmly. Briefly describe each segment. Invite questions. No products."
        )
        intro = seg_call_llm([{"role": "user", "content": intro_prompt}], SYSTEM_BUSINESS_CHAT)
        st.session_state.seg_chat_history = [{"role": "assistant", "content": intro}]

    for msg in st.session_state.seg_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # NO key= on chat_input — matches original_segmentation.py
    user_input = st.chat_input("Ask about the segments…")
    if user_input:
        st.session_state.seg_chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        augmented = (
            f"[CONTEXT]\n{st.session_state.seg_chat_context}\n\n"
            f"[RULES] Describe profiles only. No products. No new segments.\n\n"
            f"[QUESTION]\n{user_input}"
        )
        messages = st.session_state.seg_chat_history[:-1] + [
            {"role": "user", "content": augmented}]
        with st.chat_message("assistant"):
            response = st.write_stream(seg_llm_stream_generator(messages, SYSTEM_BUSINESS_CHAT))
        st.session_state.seg_chat_history.append({"role": "assistant", "content": response})
        if len(st.session_state.seg_chat_history) > 16:
            st.session_state.seg_chat_history = (
                st.session_state.seg_chat_history[:2] +
                st.session_state.seg_chat_history[-14:])

    # ── Download ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Download Results")
    if st.button("📋 Prepare Download", key="seg_biz_prep_dl"):
        with st.spinner("Building Excel…"):
            st.session_state.seg_excel_bytes = seg_build_excel(
                st.session_state.seg_df_labeled, segment_names, cluster_stats,
                table_name, algorithm, params, sil,
                features=st.session_state.seg_features,
            )
    if st.session_state.seg_excel_bytes:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇ Download Excel (Labeled Data + Segment Summary)",
            data=st.session_state.seg_excel_bytes,
            file_name=f"segments_{table_name}_{algorithm}_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="seg_biz_dl_btn",
        )


# ── Top-level entry point ─────────────────────────────────────────────────────

def seg_render(pre_loaded=None, render_sidebar: bool = True):
    """
    Render the full segmentation agent.

    pre_loaded: ignored — segmentation always starts from the connection phase.
    render_sidebar: set False when called from combined_app.py which manages
                    the sidebar itself.
    """
    init_seg_state()

    if render_sidebar:
        seg_render_sidebar()

    phase = st.session_state.seg_phase
    if phase == "connection":
        seg_phase_connection()
    elif phase == "table":
        seg_phase_table()
    elif phase == "algorithm":
        seg_phase_algorithm()
    elif phase == "role":
        seg_phase_role()
    elif phase == "running":
        if st.session_state.seg_user_role == "technical":
            seg_phase_technical()
        else:
            seg_phase_business()
    else:
        seg_phase_connection()
