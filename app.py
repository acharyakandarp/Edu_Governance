# streamlit_app.py
"""
Edu Governance System — Streamlit single-file (refactored UI, robust)
Preserves PCA/clustering/reporting/synthesis features. Safe checks for optional
clients (Gemini cloud, Ollama local). Improved aesthetics and layout.
"""
import uuid
import os
import io
import json
import re
import random
from datetime import datetime
from glob import glob
from typing import Tuple, Dict, Any, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# st-aggrid (editable grid)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    ST_AGGRID_AVAILABLE = True
except Exception:
    ST_AGGRID_AVAILABLE = False

# sklearn optional
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# scipy & statsmodels optional (for tests & VIF)
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.stats.multicomp as multi
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# Try to import ollama Python client (optional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# =========================================================
# NATIONAL EDUCATION GOVERNANCE INTELLIGENCE PLATFORM
# =========================================================

st.set_page_config(
    page_title="National Education Governance Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GLOBAL DESIGN SYSTEM
# =========================================================

GLOBAL_CSS = """
<style>

/* =====================================================
ROOT
===================================================== */

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #f8fafc;
    color: #0f172a;
}

/* =====================================================
MAIN APP
===================================================== */

.main {
    background-color: #f8fafc;
    padding-top: 0.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* =====================================================
HEADERS
===================================================== */

.big-title {
    font-size: 36px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0px;
    letter-spacing: -1px;
}

.subtitle {
    font-size: 15px;
    color: #475569;
    margin-top: 4px;
    margin-bottom: 22px;
    line-height: 1.7;
}

/* =====================================================
SECTIONS
===================================================== */

.section-title {
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 4px;
    margin-bottom: 8px;
    letter-spacing: -0.7px;
}

.section-desc {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 22px;
    line-height: 1.8;
}

/* =====================================================
CARDS
===================================================== */

.card {
    background: #ffffff;
    padding: 30px;
    border-radius: 24px;
    border: 1px solid #e2e8f0;
    box-shadow:
        0 10px 30px rgba(15, 23, 42, 0.05),
        0 2px 8px rgba(15, 23, 42, 0.03);
    margin-bottom: 24px;
}

/* =====================================================
KPI CARDS
===================================================== */

.kpi-card {
    background: linear-gradient(
        145deg,
        #ffffff 0%,
        #f8fafc 100%
    );

    border-radius: 22px;
    padding: 24px;
    border: 1px solid #dbeafe;

    box-shadow:
        0 4px 14px rgba(37, 99, 235, 0.08),
        0 1px 3px rgba(15, 23, 42, 0.05);

    transition: all 0.25s ease;

    min-height: 135px;

    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-card:hover {
    transform: translateY(-4px);

    box-shadow:
        0 14px 32px rgba(37, 99, 235, 0.14),
        0 3px 10px rgba(15, 23, 42, 0.08);
}

.kpi-value {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.kpi-label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
}

/* =====================================================
ALERTS
===================================================== */

.alert-success {
    background: linear-gradient(
        135deg,
        #f0fdf4,
        #ecfdf5
    );

    border-left: 6px solid #16a34a;

    padding: 18px;
    border-radius: 18px;

    color: #166534;

    margin-bottom: 18px;

    line-height: 1.8;

    box-shadow:
        0 4px 12px rgba(22, 163, 74, 0.06);
}

.alert-warning {
    background: linear-gradient(
        135deg,
        #fff7ed,
        #fffbeb
    );

    border-left: 6px solid #ea580c;

    padding: 18px;
    border-radius: 18px;

    color: #9a3412;

    margin-bottom: 18px;

    line-height: 1.8;

    box-shadow:
        0 4px 12px rgba(234, 88, 12, 0.06);
}

.alert-critical {
    background: linear-gradient(
        135deg,
        #fef2f2,
        #fff1f2
    );

    border-left: 6px solid #dc2626;

    padding: 18px;
    border-radius: 18px;

    color: #991b1b;

    margin-bottom: 18px;

    line-height: 1.8;

    box-shadow:
        0 4px 12px rgba(220, 38, 38, 0.06);
}

.alert-info {
    background: linear-gradient(
        135deg,
        #eff6ff,
        #f0f9ff
    );

    border-left: 6px solid #2563eb;

    padding: 18px;
    border-radius: 18px;

    color: #1e40af;

    margin-bottom: 18px;

    line-height: 1.8;

    box-shadow:
        0 4px 12px rgba(37, 99, 235, 0.06);
}

/* =====================================================
BADGES
===================================================== */

.badge-success {
    background: linear-gradient(
        135deg,
        #dcfce7,
        #bbf7d0
    );

    color: #166534;

    padding: 6px 12px;

    border-radius: 999px;

    font-size: 12px;
    font-weight: 700;
}

.badge-warning {
    background: linear-gradient(
        135deg,
        #fef3c7,
        #fde68a
    );

    color: #92400e;

    padding: 6px 12px;

    border-radius: 999px;

    font-size: 12px;
    font-weight: 700;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #111827 100%
    );

    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

/* =====================================================
BUTTONS
===================================================== */

.stButton > button {

    border-radius: 12px;

    padding: 0.7rem 1.2rem;

    border: none;

    background: linear-gradient(
        135deg,
        #2563eb,
        #1d4ed8
    );

    color: white;

    font-weight: 700;

    transition: all 0.25s ease;

    box-shadow:
        0 4px 12px rgba(37, 99, 235, 0.15);
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 8px 20px rgba(37, 99, 235, 0.2);
}

/* =====================================================
DATAFRAMES
===================================================== */

[data-testid="stDataFrame"] {

    border-radius: 18px;

    border: 1px solid #e2e8f0;

    overflow: hidden;

    box-shadow:
        0 4px 12px rgba(15, 23, 42, 0.03);
}

/* =====================================================
METRICS
===================================================== */

[data-testid="metric-container"] {

    background: linear-gradient(
        145deg,
        #ffffff,
        #f8fafc
    );

    border: 1px solid #e2e8f0;

    padding: 14px;

    border-radius: 18px;

    box-shadow:
        0 4px 12px rgba(15,23,42,0.04);
}

/* =====================================================
TABS
===================================================== */

button[data-baseweb="tab"] {

    font-size: 14px;

    font-weight: 700;

    color: #475569;

    padding: 12px 20px;

    border-radius: 14px 14px 0px 0px;

    transition: all 0.2s ease;
}

button[data-baseweb="tab"]:hover {

    background: #eff6ff;

    color: #1d4ed8;
}

button[data-baseweb="tab"][aria-selected="true"] {

    background: linear-gradient(
        135deg,
        #ffffff,
        #f8fafc
    );

    color: #2563eb;

    border-bottom: 3px solid #2563eb;
}

/* =====================================================
HORIZONTAL RULE
===================================================== */

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin-top: 2rem;
    margin-bottom: 2rem;
}

</style>
"""

st.markdown(
    GLOBAL_CSS,
    unsafe_allow_html=True
)


# =========================================================
# SYSTEM STATUS
# =========================================================

def system_status():

    df = st.session_state.get("active_df")

    if (
        isinstance(df, pd.DataFrame)
        and not df.empty
    ):

        st.markdown(
            '<span class="badge-success">System Ready • Dataset Loaded</span>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<span class="badge-warning">Awaiting Dataset Upload</span>',
            unsafe_allow_html=True
        )
# =========================================================
# SAFE EXCEPTION FORMATTER
# =========================================================
def pretty_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)}"


# ---------------- ENHANCED SAMPLE DATA (60 DISTRICTS) ----------------
@st.cache_data
def load_sample() -> pd.DataFrame:
    np.random.seed(42)

    states = ["S1", "S2", "S3", "S4", "S5"]
    districts = [f"D{i}" for i in range(1, 61)]
    
    # Base coordinates shifted strictly INLAND to prevent ocean-plotting
    state_coords = {
        "S1": (28.6, 77.2), # North (Delhi/Haryana region)
        "S2": (19.5, 75.5), # West (Inland Maharashtra/Marathwada)
        "S3": (13.5, 78.0), # South (Inland Karnataka/Andhra border)
        "S4": (23.5, 86.0), # East (Inland Jharkhand/Bengal border)
        "S5": (23.2, 77.4)  # Central (Madhya Pradesh)
    }

    data = []

    for i, d in enumerate(districts):
        state = states[i % len(states)]
        
        # Reduced random spread from 3.0 down to 1.5 degrees (~150km radius)
        base_lat, base_lon = state_coords[state]
        lat = base_lat + np.random.uniform(-1.5, 1.5)
        lon = base_lon + np.random.uniform(-1.5, 1.5)

        # simulate realistic variation clusters
        base = np.random.choice(["high", "medium", "low"])

        if base == "high":
            evs = np.random.normal(85, 5)
            lang = np.random.normal(83, 5)
            math = np.random.normal(75, 6)
            infra = np.random.uniform(0.7, 0.95)
            ptr = np.random.uniform(20, 30)

        elif base == "medium":
            evs = np.random.normal(65, 7)
            lang = np.random.normal(68, 6)
            math = np.random.normal(60, 7)
            infra = np.random.uniform(0.5, 0.7)
            ptr = np.random.uniform(28, 38)

        else:
            evs = np.random.normal(45, 6)
            lang = np.random.normal(50, 7)
            math = np.random.normal(70, 6)
            infra = np.random.uniform(0.3, 0.5)
            ptr = np.random.uniform(35, 50)

        data.append({
            "state": state,
            "district": d,
            "Latitude": round(lat, 4),
            "Longitude": round(lon, 4),
            "EVS": round(evs, 1),
            "Language": round(lang, 1),
            "Math": round(math, 1),
            "infra": round(infra, 2),
            "ptr": round(ptr, 1)
        })

    return pd.DataFrame(data)


# ---------------- DATA SANITIZATION ----------------
def sanitize_sample(df: Optional[pd.DataFrame], max_rows: int = 20) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    pii_keywords = ['name', 'id', 'email', 'phone', 'mobile', 'address']
    safe_cols = [c for c in df.columns if not any(k in c.lower() for k in pii_keywords)]

    return df[safe_cols].head(max_rows).copy()


def compact_schema_and_examples(df: Optional[pd.DataFrame], max_examples: int = 1) -> str:
    if df is None or df.shape[0] == 0:
        return ""

    buf = io.StringIO()
    df.head(max_examples).to_csv(buf, index=False)
    return buf.getvalue().strip()

# ---------------- Robust GenAI extractor ----------------
def strip_markdown_and_find_json(text: str) -> str:
    if not text:
        return text
    txt = text.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
    txt = re.sub(r"\s*```$", "", txt, flags=re.I)
    m = re.search(r"(\{(?:.|\n)*\})", txt, flags=re.S)
    if m:
        return m.group(1)
    m2 = re.search(r"(\[(?:.|\n)*\])", txt, flags=re.S)
    if m2:
        return m2.group(1)
    return txt

def extract_text_from_genai_response(resp) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    try:
        if hasattr(resp, "text") and resp.text:
            txt = str(resp.text)
            txt = strip_markdown_and_find_json(txt) or txt
            meta["source"] = "resp.text"
            return txt, meta

        if hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            fr = getattr(cand, "finish_reason", None) or (cand.get("finish_reason") if isinstance(cand, dict) else None)
            if fr is not None:
                meta["finish_reason"] = fr
            cand_content = getattr(cand, "content", None) or (cand.get("content") if isinstance(cand, dict) else None)
            if cand_content:
                parts = None
                if hasattr(cand_content, "parts"):
                    parts = cand_content.parts
                elif isinstance(cand_content, dict) and "parts" in cand_content:
                    parts = cand_content.get("parts")
                if parts:
                    collected = []
                    for p in parts:
                        if isinstance(p, str):
                            collected.append(p)
                        elif hasattr(p, "text"):
                            collected.append(p.text)
                        elif isinstance(p, dict):
                            t = p.get("text") or p.get("payload") or ""
                            if t:
                                collected.append(t)
                    res = "\n".join([r for r in collected if r])
                    if res:
                        res = strip_markdown_and_find_json(res)
                        meta["source"] = "candidate.content.parts"
                        return res, meta
                if isinstance(cand_content, str) and cand_content.strip():
                    res = strip_markdown_and_find_json(cand_content)
                    meta["source"] = "candidate.content(str)"
                    return res, meta

            cand_msg = getattr(cand, "message", None) or (cand.get("message") if isinstance(cand, dict) else None)
            if cand_msg and isinstance(cand_msg, dict):
                content = cand_msg.get("content", [])
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        t = p.get("text") or p.get("payload") or ""
                        if t:
                            parts.append(t)
                    elif isinstance(p, str):
                        parts.append(p)
                res = "\n".join([r for r in parts if r])
                if res:
                    res = strip_markdown_and_find_json(res)
                    meta["source"] = "candidate.message.content"
                    return res, meta

        try:
            d = resp.to_dict() if hasattr(resp, "to_dict") else None
            if d:
                def find_text(obj):
                    if isinstance(obj, str) and obj.strip():
                        return obj
                    if isinstance(obj, dict):
                        for k in ("text","output","content","message","cleaned_preview_csv","cleaned_preview"):
                            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                                return obj[k]
                        for v in obj.values():
                            res = find_text(v)
                            if res:
                                return res
                    if isinstance(obj, list):
                        for item in obj:
                            res = find_text(item)
                            if res:
                                return res
                    return None
                candidate = find_text(d)
                if candidate:
                    candidate = strip_markdown_and_find_json(candidate)
                    meta["source"] = "resp.to_dict_scan"
                    return candidate, meta
        except Exception:
            pass

        meta["source"] = "none_found"
        return "", meta
    except Exception as e:
        return "", {"error": str(e)}

# ---------------- Mock extractor ----------------
def mock_gemini_extract_preview(df_preview: pd.DataFrame) -> Tuple[List[Dict[str,Any]], pd.DataFrame]:
    suggestions = []
    cleaned = df_preview.copy()
    for col in df_preview.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ["math", "mth", "mth_pct", "mthpct"]):
            role, dtype = "math_pct", "numeric"
        elif any(k in col_lower for k in ["lang", "eng", "language", "eng_pct"]):
            role, dtype = "lang_pct", "numeric"
        elif any(k in col_lower for k in ["evs", "env", "environment"]):
            role, dtype = "evs_pct", "numeric"
        elif any(k in col_lower for k in ["infra", "facility"]):
            role, dtype = "infra_index", "numeric"
        elif any(k in col_lower for k in ["state", "st"]):
            role, dtype = "state", "categorical"
        elif any(k in col_lower for k in ["dist", "district", "dist_name"]):
            role, dtype = "district", "categorical"
        elif any(k in col_lower for k in ["ptr", "pupil", "teacher_ratio"]):
            role, dtype = "pupil_teacher_ratio", "numeric"
        else:
            role, dtype = col, str(df_preview[col].dtype)
        conf = round(random.uniform(0.6, 0.98), 2)
        suggestions.append({"original": col, "suggested_role": role, "dtype": dtype, "confidence": conf})
        if dtype == "numeric":
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        else:
            cleaned[col] = cleaned[col].astype(str).str.strip()
    return suggestions, cleaned

def _build_local_cleaned_preview_from_suggestions(sanitized_df: pd.DataFrame, suggestions: List[Dict[str,Any]]) -> pd.DataFrame:
    df = sanitized_df.copy().reset_index(drop=True)
    cleaned = pd.DataFrame()
    for s in suggestions:
        orig = s.get("original")
        if orig not in df.columns:
            continue
        role = s.get("suggested_role") or orig
        dtype = (s.get("dtype") or "").lower()
        if any(k in dtype for k in ("num","digit","numeric","numerical","float","int")) or role.lower() in ("math_pct","lang_pct","evs_pct","infra_index","pupil_teacher_ratio","learning_score","evs","math","language","ptr"):
            cleaned[role] = pd.to_numeric(df[orig], errors="coerce")
        else:
            cleaned[role] = df[orig].astype(str).str.strip()
    if cleaned.shape[1] == 0:
        return df
    return cleaned

# ---------------- Statistical building blocks ----------------
def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    if not STATSMODELS_AVAILABLE:
        return pd.DataFrame({"feature": features, "vif": [None]*len(features)}).set_index("feature")
    X = df[features].dropna()
    Xc = sm.add_constant(X, has_constant='add')
    vifs = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(Xc.values, i)
        except Exception:
            v = np.nan
        vifs.append((col, v))
    return pd.DataFrame(vifs, columns=["feature","vif"]).set_index("feature")

def parallel_analysis(df_vals: np.ndarray, n_iter: int = 100, random_state: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    n, p = df_vals.shape
    obs_corr = np.corrcoef(df_vals, rowvar=False)
    obs_eigs = np.linalg.eigvalsh(obs_corr)[::-1]
    rand_eigs_all = np.zeros((n_iter, p))
    for i in range(n_iter):
        rand_data = np.zeros_like(df_vals)
        for j in range(p):
            col = df_vals[:, j]
            rand_data[:, j] = rng.permutation(col)
        r_corr = np.corrcoef(rand_data, rowvar=False)
        r_eigs = np.linalg.eigvalsh(r_corr)[::-1]
        rand_eigs_all[i, :] = r_eigs
    mean_rand = rand_eigs_all.mean(axis=0)
    return {"observed": obs_eigs, "mean_random": mean_rand, "rand_eigs_all": rand_eigs_all}

def gap_statistic(X: np.ndarray, k_max: int = 8, B: int = 20, random_state: int = 42):
    if not SKLEARN_AVAILABLE:
        return None
    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    def Wk(Xi, labels):
        w = 0.0
        for k in np.unique(labels):
            members = Xi[labels==k]
            if members.shape[0] <= 1:
                continue
            d = pairwise_distances(members)
            w += d.sum() / (2.0 * members.shape[0])
        return w
    results = []
    for k in range(1, min(k_max, n-1)+1):
        if k == 1:
            from sklearn.cluster import KMeans as _K
            km = _K(n_clusters=1, n_init=10, random_state=random_state)
            labels = km.fit_predict(X)
            wk = Wk(X, labels)
            results.append({"k":k, "gap":0.0, "sk":0.0, "Wk": wk})
            continue
        from sklearn.cluster import KMeans as _K
        km = _K(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        wk = Wk(X, labels)
        Wk_refs = np.zeros(B)
        for b in range(B):
            Xref = rng.random((n, p)) * (maxs - mins) + mins
            kmr = _K(n_clusters=k, n_init=10, random_state=random_state)
            lblr = kmr.fit_predict(Xref)
            Wk_refs[b] = Wk(Xref, lblr)
        gap = np.log(Wk_refs.mean()) - np.log(wk)
        sdk = np.sqrt(((np.log(Wk_refs) - np.log(Wk_refs).mean())**2).sum() / B) * np.sqrt(1 + 1.0/B)
        results.append({"k":k,"gap":gap,"sk":sdk,"Wk":wk})
    return pd.DataFrame(results).set_index("k")

def evaluate_k_range(X: np.ndarray, k_min=2, k_max=8, random_state=42):
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    rows = []
    for k in range(k_min, min(k_max, X.shape[0]-1)+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k>1 and X.shape[0] > k else np.nan
        ch = calinski_harabasz_score(X, labels) if k>1 else np.nan
        db = davies_bouldin_score(X, labels) if k>1 else np.nan
        rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
    return pd.DataFrame(rows).set_index("k")

def mahalanobis_outlier_mask(X: np.ndarray, threshold_p=0.001):
    try:
        from scipy.stats import chi2
    except Exception:
        # fallback: no outlier detection
        return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0]), np.ones(X.shape[0])
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    mean = X.mean(axis=0)
    diffs = X - mean
    D2 = np.sum(diffs.dot(inv_cov) * diffs, axis=1)
    pvals = 1 - chi2.cdf(D2, df=X.shape[1])
    mask = pvals < threshold_p
    return mask, D2, pvals

def cluster_profiling_tests(df: pd.DataFrame, cols_sel: List[str], cluster_labels_col: str = "_cluster"):
    profile = {"cluster_sizes": df[cluster_labels_col].value_counts().to_dict(), "variables": {}}
    clusters = sorted(df[cluster_labels_col].unique())
    for var in cols_sel:
        groups = [df[df[cluster_labels_col]==cl][var].dropna().values for cl in clusters]
        stats_summary = {"median": df.groupby(cluster_labels_col)[var].median().round(3).to_dict(),
                         "mean": df.groupby(cluster_labels_col)[var].mean().round(3).to_dict(),
                         "std": df.groupby(cluster_labels_col)[var].std().round(3).to_dict()}
        normal_ps = []
        for g in groups:
            try:
                if len(g) >= 3 and SCIPY_AVAILABLE:
                    p = float(stats.shapiro(g)[1])
                else:
                    p = np.nan
            except Exception:
                p = np.nan
            normal_ps.append(p)
        try:
            if all((p>0.05 or np.isnan(p)) for p in normal_ps) and all(len(g) >= 2 for g in groups):
                # ANOVA
                f_stat, pval = stats.f_oneway(*groups)
                test_result = {"test": "ANOVA", "f_stat": float(f_stat), "pvalue": float(pval)}
                # Tukey post-hoc if statsmodels available
                if STATSMODELS_AVAILABLE:
                    data_long = df[[cluster_labels_col, var]].dropna()
                    try:
                        mc = multi.pairwise_tukeyhsd(data_long[var], data_long[cluster_labels_col])
                        test_result["tukey_summary"] = str(mc.summary())
                    except Exception:
                        test_result["tukey_summary"] = None
            else:
                # Kruskal-Wallis
                try:
                    h_stat, pval = stats.kruskal(*groups)
                    test_result = {"test": "Kruskal-Wallis", "h_stat": float(h_stat), "pvalue": float(pval)}
                except Exception as e:
                    test_result = {"test": "Kruskal-Wallis", "error": str(e)}
        except Exception as e:
            test_result = {"error": str(e)}
        profile["variables"][var] = {"summary": stats_summary, "normal_p_group": normal_ps, "test_result": test_result}
    return profile

def clustering_stability_bootstrap(X: np.ndarray, base_labels: np.ndarray, k: int, n_boot=50, method='kmeans', random_state=42):
    if not SKLEARN_AVAILABLE:
        return {"ari_mean": None, "ari_std": None, "aris": []}
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    aris = []
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Xb = X[idx]
        if method == 'kmeans':
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            lblb = km.fit_predict(Xb)
        else:
            hc = AgglomerativeClustering(n_clusters=k)
            lblb = hc.fit_predict(Xb)
        try:
            ari = adjusted_rand_score(base_labels[idx], lblb)
        except Exception:
            ari = np.nan
        aris.append(float(ari) if ari is not None else np.nan)
    aris_arr = np.array(aris, dtype=float)
    return {"ari_mean": float(np.nanmean(aris_arr)), "ari_std": float(np.nanstd(aris_arr)), "aris": aris}

# ---------------- Gemini connectivity & token test ----------------
def gemini_ping_test():
    """Return (text, meta, error_message). Helps diagnose API issues."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, None, "❌ No GEMINI_API_KEY or GOOGLE_API_KEY found in environment."

    try:
        import google.generativeai as genai
    except Exception:
        return None, None, "❌ google-generative-ai library not installed."

    # configure
    try:
        genai.configure(api_key=api_key, transport="rest")
    except Exception:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass

    # Small request (guaranteed to fit token limits)
    prompt = "Return exactly the text: OK"
    try:
        gm = genai.GenerativeModel("models/gemini-2.5-flash")
        resp = gm.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "candidate_count": 1,
                "max_output_tokens": 20
            }
        )
        from_debug, meta = extract_text_from_genai_response(resp)
        return from_debug, meta, None
    except Exception as e:
        return None, None, f"❌ Exception: {pretty_exception(e)}"

# ---------------- Ollama synthesis wrapper ----------------
def call_ollama_for_synthesis(compact_payload: str, compact_csv: str, model: str = "llama3.2", max_tokens: int = 1000) -> Tuple[Optional[str], Dict[str,Any], Optional[str]]:
    """
    Call local Ollama model to synthesize narrative.
    Returns (text, meta, error_message).
    """
    if not OLLAMA_AVAILABLE:
        return None, {}, "❌ Ollama Python client not installed in this environment."
    try:
        # Build prompt (concise, facts only)
        prompt = (
            "You are an expert educational policy analyst.\n"
            "Use ONLY the facts provided in JSON and the CSV sample. Do NOT impute missing values.\n\n"
            "Produce a research-style narrative report in markdown with the following sections:\n"
            "1) Executive summary (2–3 short paragraphs with numeric facts).\n"
            "2) Key quantitative findings (bullet list): cite means, cluster sizes, explained variance.\n"
            "3) Methods (brief): indicate PCA/explained variance and clustering method and diagnostic scores.\n"
            "4) Cluster-by-cluster interpretation (for each cluster, list member districts and their profile).\n"
            "5) Five prioritized recommendations tied to observed facts.\n\n"
            "FACTS_JSON:\n" + compact_payload + "\n\n"
            "CSV_SAMPLE:\n" + compact_csv + "\n\n"
            "Return markdown only."
        )
        # Ollama Python client usage:
        try:
            resp = ollama.generate(model=model, prompt=prompt, options={"num_predict": int(max_tokens)})
            text = ""
            if isinstance(resp, dict):
                text = resp.get("response") or resp.get("text") or ""
            else:
                text = getattr(resp, "response", "") or getattr(resp, "text", "") or str(resp)
            meta = {"backend": "ollama", "model": model, "raw": resp}
            return text, meta, None
        except Exception as e:
            try:
                resp2 = ollama.run(model=model, prompt=prompt, num_predict=int(max_tokens))
                text = resp2.get("response") if isinstance(resp2, dict) else str(resp2)
                meta = {"backend": "ollama", "model": model, "raw": resp2}
                return text, meta, None
            except Exception as e2:
                return None, {}, f"Ollama call failed: {pretty_exception(e)} | fallback failed: {pretty_exception(e2)}"
    except Exception as e:
        return None, {}, f"Ollama invocation error: {pretty_exception(e)}"

# ---------------- Reporting helpers (PCA/kmeans wrapper and report composer) ----------------
def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["rows"] = int(df.shape[0])
    out["cols"] = int(df.shape[1])
    missing = df.isnull().sum()
    out["missing_per_column"] = missing[missing > 0].to_dict()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats_out = {}
    for c in num_cols:
        s = {}
        valid = df[c].dropna()
        s["observed_count"] = int(valid.shape[0])
        s["missing_count"] = int(df[c].isnull().sum())
        if valid.shape[0] > 0:
            s["mean"] = float(valid.mean())
            s["median"] = float(valid.median())
            s["std"] = float(valid.std(ddof=0))
            s["min"] = float(valid.min())
            s["max"] = float(valid.max())
        else:
            s["mean"] = s["median"] = s["std"] = s["min"] = s["max"] = None
        stats_out[c] = s
    out["numeric_stats"] = stats_out

    if len(num_cols) >= 2:
        corr = df[num_cols].corr(method="pearson")
        out["correlation_matrix"] = corr.round(3).to_dict()
        strong_pairs = []
        for i, a in enumerate(num_cols):
            for b in num_cols[i+1:]:
                val = corr.at[a,b]
                if pd.notna(val) and abs(val) >= 0.6:
                    strong_pairs.append({"var1": a, "var2": b, "r": float(round(val,3))})
        out["strong_correlations"] = strong_pairs
    else:
        out["correlation_matrix"] = {}
        out["strong_correlations"] = []

    out["rows_with_any_missing"] = int(df.isnull().any(axis=1).sum())
    return out

def run_advanced_analyses(df: pd.DataFrame, selected_vars: List[str], n_pca_components: int = 2, k_clusters: int = 3) -> Dict[str, Any]:
    result: Dict[str, Any] = {"selected_vars": selected_vars, "pca": None, "kmeans": None, "n_input_rows": int(df.shape[0])}
    if not SKLEARN_AVAILABLE:
        result["error"] = "scikit-learn not available in this environment."
        return result

    df_cc = df.dropna(subset=selected_vars).copy()
    result["n_complete_case_rows"] = int(df_cc.shape[0])
    if df_cc.shape[0] == 0:
        result["error"] = "No complete-case rows available for the selected variables."
        return result

    scaler = StandardScaler()
    X = scaler.fit_transform(df_cc[selected_vars].astype(float))

    # PCA
    n_comp = min(n_pca_components, len(selected_vars), 3)
    pca = PCA(n_components=n_comp, random_state=42)
    pcs = pca.fit_transform(X)
    explained = [float(x) for x in pca.explained_variance_ratio_.round(4).tolist()]
    loadings_df = pd.DataFrame(pca.components_.T, index=selected_vars, columns=[f"PC{i+1}" for i in range(n_comp)]).round(4)
    result["pca"] = {"n_components": n_comp, "explained_variance_ratio": explained, "loadings": loadings_df.to_dict()}

    # KMeans on PCs
    k = min(max(2, k_clusters), max(2, int(df_cc.shape[0])))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    X_cluster = pcs if pcs.shape[1] >= 2 else X
    labels = km.fit_predict(X_cluster)
    df_cc = df_cc.reset_index(drop=True)
    df_cc["_cluster"] = labels

    cluster_sizes = df_cc["_cluster"].value_counts().sort_index().to_dict()
    cluster_medians = df_cc.groupby("_cluster")[selected_vars].median().round(3).to_dict()

    diagnostics: Dict[str, Any] = {}
    try:
        if len(set(labels)) >= 2 and len(set(labels)) < len(labels):
            try:
                sil = float(silhouette_score(X_cluster, labels).round(4))
                diagnostics["silhouette_score"] = sil
            except Exception:
                diagnostics["silhouette_score"] = None
            try:
                ch = float(calinski_harabasz_score(X_cluster, labels))
                diagnostics["calinski_harabasz"] = float(round(ch, 4))
            except Exception:
                diagnostics["calinski_harabasz"] = None
        else:
            diagnostics["silhouette_score"] = None
            diagnostics["calinski_harabasz"] = None
    except Exception:
        diagnostics["silhouette_score"] = None
        diagnostics["calinski_harabasz"] = None

    result["kmeans"] = {
        "k": k,
        "cluster_sizes": cluster_sizes,
        "cluster_medians": cluster_medians,
        "diagnostics": diagnostics
    }

    pcs_df = pd.DataFrame(pcs[:, :min(3, pcs.shape[1])], columns=[f"PC{i+1}" for i in range(min(3, pcs.shape[1]))])
    result["pca_sample_projection_head"] = pcs_df.head(3).round(4).to_dict()
    result["cluster_assignments"] = df_cc.reset_index().to_dict(orient="list")
    return result

def _compose_research_style_report(stats: Dict[str, Any], adv: Optional[Dict[str,Any]] = None, id_col: Optional[str] = None, selected_vars: Optional[List[str]] = None, verbosity: str = "concise") -> str:
    lines: List[str] = []
    rows = stats.get("rows", 0)
    cols = stats.get("cols", 0)
    lines.append(f"# Analytical Report — {rows} rows × {cols} columns\n")
    lines.append("### Executive summary\n")
    numeric_stats = stats.get("numeric_stats", {})
    if numeric_stats:
        valid_means = {k:v["mean"] for k,v in numeric_stats.items() if v["mean"] is not None}
        if valid_means:
            top_var, top_mean = max(valid_means.items(), key=lambda x: x[1])
            low_var, low_mean = min(valid_means.items(), key=lambda x: x[1])
            lines.append(f"This dataset (n={rows}) exhibits a clear performance gradient across the measured indicators. On average, **{top_var}** shows the highest mean ({top_mean:.1f}) while **{low_var}** shows the lowest mean ({low_mean:.1f}).")
        else:
            lines.append("Numeric summaries are present but not enough complete observations for reliable averaging.")
    else:
        lines.append("No numeric variables detected; the rest of the report will summarize available information only.")
    if adv and adv.get("pca"):
        exp = adv["pca"]["explained_variance_ratio"]
        lines.append(f"Principal component analysis indicates the first {adv['pca']['n_components']} component(s) explain the majority of variance (first-component explained variance = {exp[0]:.3f}).")
    if adv and adv.get("kmeans"):
        cs = adv["kmeans"]["cluster_sizes"]
        lines.append(f"K-means clustering (k={adv['kmeans']['k']}) identified {len(cs)} clusters with sizes: {cs}.")
    lines.append("\n### 1. Data & missingness\n")
    if stats.get("missing_per_column"):
        lines.append("- Columns with missing values: " + ", ".join([f"{k} (missing={v})" for k, v in stats["missing_per_column"].items()]))
    else:
        lines.append("- No column-level missingness detected.")
    lines.append(f"- Rows with any missing cell: {stats.get('rows_with_any_missing', 0)}\n")
    lines.append("### 2. Descriptive statistics (selected variables)\n")
    if numeric_stats:
        for var, s in numeric_stats.items():
            lines.append(f"- **{var}** — n={s['observed_count']}, mean={s['mean'] if s['mean'] is not None else 'NA'}, median={s['median'] if s['median'] is not None else 'NA'}, std={s['std'] if s['std'] is not None else 'NA'}, range=[{s['min'] if s['min'] is not None else 'NA'}, {s['max'] if s['max'] is not None else 'NA'}].")
    else:
        lines.append("- No numeric statistics to display.\n")
    if stats.get("strong_correlations"):
        lines.append("\n### 3. Strong correlations (|r| >= 0.6)\n")
        for pair in stats["strong_correlations"]:
            lines.append(f"- {pair['var1']} ↔ {pair['var2']}: r = {pair['r']}")
    else:
        lines.append("\n### 3. Correlations\n- No strong correlations detected.\n")
    if adv:
        lines.append("\n### 4. Advanced analyses (PCA & Clustering)\n")
        if adv.get("pca"):
            p = adv["pca"]
            lines.append(f"- PCA: n_components = {p['n_components']}; explained_variance_ratio = {p['explained_variance_ratio']}.")
            try:
                ld = pd.DataFrame(p["loadings"])
                for pc in ld.columns:
                    top = ld[pc].abs().sort_values(ascending=False).head(3).index.tolist()
                    lines.append(f"  - {pc} top contributors: {', '.join(top)}")
            except Exception:
                pass
        if adv.get("kmeans"):
            k = adv["kmeans"]
            lines.append(f"- KMeans (k={k['k']}) cluster sizes: {k['cluster_sizes']}.")
            try:
                cm = k["cluster_medians"]
                lines.append("Cluster medians (per variable):")
                for var, dct in cm.items():
                    items = [f"cluster {cl}: {val}" for cl, val in dct.items()]
                    lines.append(f"  - {var}: " + "; ".join(items))
            except Exception:
                pass
            diag = k.get("diagnostics", {})
            if diag:
                lines.append(f"- Cluster validity diagnostics: silhouette_score={diag.get('silhouette_score')}, calinski_harabasz={diag.get('calinski_harabasz')}.")
    else:
        lines.append("\n### 4. Advanced analyses\n- None requested or scikit-learn unavailable.\n")
    if adv and adv.get("kmeans") and id_col and selected_vars:
        lines.append("\n### 5. Cluster-level interpretations (automated)\n")
        try:
            assign = adv.get("cluster_assignments", None)
            if assign:
                assign_df = pd.DataFrame(assign)
                if "_cluster" in assign_df.columns:
                    for cl in sorted(assign_df["_cluster"].unique()):
                        members = assign_df[assign_df["_cluster"] == cl][id_col].astype(str).tolist() if id_col in assign_df.columns else assign_df[assign_df["_cluster"] == cl].index.astype(str).tolist()
                        lines.append(f"- **Cluster {cl}** (n={len(members)}): districts = {', '.join(members)}")
                        med_profile = assign_df[assign_df["_cluster"] == cl][selected_vars].median().to_dict()
                        highest = max(med_profile.items(), key=lambda x: x[1])
                        lowest = min(med_profile.items(), key=lambda x: x[1])
                        lines.append(f"  - Profile: higher on `{highest[0]}` (median={highest[1]}), lower on `{lowest[0]}` (median={lowest[1]}).")
            else:
                lines.append("- Cluster assignments not available for narrative.")
        except Exception:
            lines.append("- Could not produce per-cluster detailed narrative due to data shape.")
    lines.append("\n### 6. Quick observations & prioritized recommendations\n")
    try:
        if numeric_stats:
            valid_means = {k:v["mean"] for k,v in numeric_stats.items() if v["mean"] is not None}
            if valid_means:
                top = max(valid_means.items(), key=lambda x: x[1])
                bottom = min(valid_means.items(), key=lambda x: x[1])
                lines.append(f"- Highest average: **{top[0]}** (mean={top[1]:.1f}). Low average: **{bottom[0]}** (mean={bottom[1]:.1f}).")
    except Exception:
        pass
    if "ptr" in numeric_stats:
        ptr_mean = numeric_stats["ptr"]["mean"]
        if ptr_mean and ptr_mean > 35:
            lines.append("- PTR is high on average (recommend targeted teacher hiring/pedagogical support where PTR > local threshold).")
        else:
            lines.append("- PTR is within acceptable average; monitor specific districts exceeding thresholds.")
    if "infra" in numeric_stats:
        infra_mean = numeric_stats["infra"]["mean"]
        if infra_mean and infra_mean < 0.5:
            lines.append("- Average infrastructure index is low; prioritize infrastructure investments in low-infra clusters.")
        else:
            lines.append("- Infrastructure is moderate/high on average; target remaining low-infra districts.")
    lines.append("\n### 7. Conclusion\n")
    lines.append("This analysis is data-driven and uses observed values only. The PCA shows whether a dominant latent factor structures the data; cluster analysis reveals groups of districts with similar strengths/weaknesses. Use cluster-specific interventions rather than a one-size-fits-all approach.\n")
    if verbosity == "detailed":
        lines.append("\n---\n*End of detailed report.*\n")
    else:
        lines.append("\n*End of concise report.*\n")
    return "\n".join(lines)

# ---------------- Utility: choose reporting dataframe safely ----------------
def _choose_reporting_df() -> Optional[pd.DataFrame]:
    """
    Returns the best available DataFrame for reporting:

    Priority Order:
    1. session_state['active_df']
    2. session_state['cleaned_preview']
    3. data/edited_dataset.csv
    4. global df_edited
    5. None
    """
    # 1) active_df
    cand = st.session_state.get("active_df", None)
    if isinstance(cand, pd.DataFrame) and not cand.empty:
        return cand.copy()

    # 2) cleaned_preview
    cand2 = st.session_state.get("cleaned_preview", None)
    if isinstance(cand2, pd.DataFrame) and not cand2.empty:
        return cand2.copy()

    # 3) saved file
    path = "data/edited_dataset.csv"
    if os.path.exists(path):
        try:
            df_loaded = pd.read_csv(path)
            if isinstance(df_loaded, pd.DataFrame) and not df_loaded.empty:
                return df_loaded
        except Exception:
            pass

    # 4) df_edited global
    if "df_edited" in globals():
        try:
            ge = globals().get("df_edited")
            if isinstance(ge, pd.DataFrame) and not ge.empty:
                return ge.copy()
        except Exception:
            pass

    return None

# ---------------- UI LAYOUT (HEADER + NAVIGATION) ----------------

# ===== HEADER =====
header_col1, header_col2, header_col3 = st.columns([5, 2, 1])

with header_col1:
    st.markdown(
        '<div class="big-title">Education Governance Intelligence Platform</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">System-level analytics, district segmentation, and policy intelligence for data-driven governance.</div>',
        unsafe_allow_html=True
    )

with header_col2:
    st.markdown("**System Status**")
    try:
        system_status()
    except Exception:
        st.markdown('<span class="badge-warning">Status Unknown</span>', unsafe_allow_html=True)

with header_col3:
    st.markdown(
        f"<div class='muted'>Run<br>{datetime.utcnow().strftime('%d %b %Y<br>%H:%M UTC')}</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================================================
# EXECUTIVE GOVERNANCE DASHBOARD
# =========================================================

dashboard_df = st.session_state.get("active_df")

if dashboard_df is None or not isinstance(dashboard_df, pd.DataFrame) or dashboard_df.empty:
    dashboard_df = load_sample()

numeric_cols_dash = dashboard_df.select_dtypes(include=[np.number]).columns.tolist()

# ---------------------------------------------------------
# BUILD GOVERNANCE METRICS
# ---------------------------------------------------------

district_count = dashboard_df.shape[0]

avg_learning = None
if "EVS" in dashboard_df.columns and "Language" in dashboard_df.columns and "Math" in dashboard_df.columns:
    avg_learning = round(
        (
            dashboard_df["EVS"].mean() +
            dashboard_df["Language"].mean() +
            dashboard_df["Math"].mean()
        ) / 3,
        1
    )

avg_infra = round(dashboard_df["infra"].mean(), 2) if "infra" in dashboard_df.columns else None
avg_ptr = round(dashboard_df["ptr"].mean(), 1) if "ptr" in dashboard_df.columns else None

# ---------------------------------------------------------
# RISK ENGINE
# ---------------------------------------------------------

critical_districts = []

if all(col in dashboard_df.columns for col in ["EVS", "infra", "ptr"]):

    for _, row in dashboard_df.iterrows():

        try:
            evs = pd.to_numeric(row["EVS"], errors="coerce")
            infra = pd.to_numeric(row["infra"], errors="coerce")
            ptr = pd.to_numeric(row["ptr"], errors="coerce")

            if pd.notna(evs) and pd.notna(infra) and pd.notna(ptr):

                if evs < 50 and infra < 0.45 and ptr > 35:

                    district_name = str(row.get("district", "Unknown"))

                    critical_districts.append(district_name)

        except Exception:
            pass

risk_count = len(critical_districts)

# ---------------------------------------------------------
# GOVERNANCE HEALTH SCORE
# ---------------------------------------------------------

health_score = 100

if avg_learning is not None:
    health_score -= max(0, (70 - avg_learning))

if avg_ptr is not None:
    health_score -= max(0, (avg_ptr - 30))

if avg_infra is not None:
    health_score -= max(0, (0.7 - avg_infra) * 100)

health_score = round(max(0, min(100, health_score)), 1)

if health_score >= 80:
    health_label = "Stable"
    health_class = "alert-success"

elif health_score >= 60:
    health_label = "Moderate Risk"
    health_class = "alert-warning"

else:
    health_label = "Critical"
    health_class = "alert-critical"

# =========================================================
# NATIONAL GOVERNANCE DASHBOARD
# =========================================================

st.title("National Governance Dashboard")

st.caption(
    "Executive overview of district-level educational performance, "
    "infrastructure readiness, staffing pressure, and governance risk."
)

# =========================================================
# KPI STRIP
# =========================================================

# 1. Define a helper function to build the HTML safely
def render_kpi(value, label):
    # .strip() removes any hidden newlines or spaces in the variables
    val_safe = str(value).strip()
    label_safe = str(label).strip()
    
    # Building the string like this stops your IDE from auto-formatting it into multiple lines
    html_str = '<div class="kpi-card">'
    html_str += '<div class="kpi-value">' + val_safe + '</div>'
    html_str += '<div class="kpi-label">' + label_safe + '</div>'
    html_str += '</div>'
    
    st.markdown(html_str, unsafe_allow_html=True)

# 2. Render the columns
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    render_kpi(district_count, "Districts")

with k2:
    val_k2 = avg_learning if avg_learning else "NA"
    render_kpi(val_k2, "Governance Score")

with k3:
    render_kpi(risk_count, "Critical Districts")

with k4:
    val_k4 = avg_infra if avg_infra else "NA"
    render_kpi(val_k4, "Avg Infrastructure")

with k5:
    val_k5 = avg_ptr if avg_ptr else "NA"
    render_kpi(val_k5, "Avg PTR")

st.markdown("---")
# =========================================================
# GOVERNANCE STATUS + INSIGHTS
# =========================================================

left_panel, right_panel = st.columns([2, 1])

# ---------------------------------------------------------
# LEFT PANEL
# ---------------------------------------------------------

with left_panel:

    status_text = (
        f"Governance Health Status: {health_label}\n\n"
        f"National Education Governance Score: {health_score}/100"
    )

    if health_label == "Stable":

        st.success(status_text)

    elif health_label == "Moderate Risk":

        st.warning(status_text)

    else:

        st.error(status_text)

    if risk_count > 0:

        risk_preview = ", ".join(
            critical_districts[:6]
        )

        st.error(
            f"""
Critical Governance Alert

{risk_count} districts exhibit simultaneous
learning deficits, infrastructure stress,
and high PTR burden.

Priority districts:
{risk_preview}
"""
        )

    else:

        st.success(
            """
System Observation

No districts currently exhibit simultaneous
extreme governance stress indicators.
"""
        )

# ---------------------------------------------------------
# RIGHT PANEL
# ---------------------------------------------------------

with right_panel:

    st.info(
        """
Executive Insight

System-level observations generated
from governance indicators.
"""
    )

    insight_lines = []

    if avg_learning is not None:

        if avg_learning < 60:

            insight_lines.append(
                "Learning outcomes indicate systemic performance stress."
            )

        elif avg_learning < 75:

            insight_lines.append(
                "Learning outcomes remain moderate but uneven."
            )

        else:

            insight_lines.append(
                "Overall learning performance remains comparatively strong."
            )

    if avg_ptr is not None and avg_ptr > 35:

        insight_lines.append(
            "Teacher workload pressure remains elevated."
        )

    if avg_infra is not None and avg_infra < 0.55:

        insight_lines.append(
            "Infrastructure readiness gaps remain significant."
        )

    if not insight_lines:

        insight_lines.append(
            "No major governance stress detected."
        )

    for txt in insight_lines:

        st.write("•", txt)

st.markdown("---")

# =========================================================
# PERFORMANCE DISTRIBUTION
# =========================================================

if all(col in dashboard_df.columns for col in ["EVS", "Language", "Math"]):

    st.markdown("<br>", unsafe_allow_html=True)

    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:

        district_scores = (
            dashboard_df[["district", "EVS", "Language", "Math"]]
            .copy()
        )

        district_scores["Overall"] = district_scores[
            ["EVS", "Language", "Math"]
        ].mean(axis=1)

        top_districts = district_scores.sort_values(
            "Overall",
            ascending=False
        ).head(10)

        fig_top = px.bar(
            top_districts,
            x="district",
            y="Overall",
            title="Top Performing Districts"
        )

        st.plotly_chart(
            fig_top,
            use_container_width=True,
            key="top_districts_chart"
        )

    with vis_col2:

        bottom_districts = district_scores.sort_values(
            "Overall",
            ascending=True
        ).head(10)

        fig_bottom = px.bar(
            bottom_districts,
            x="district",
            y="Overall",
            title="Priority Intervention Districts"
        )

        st.plotly_chart(
            fig_bottom,
            use_container_width=True,
            key="bottom_districts_chart"
        )

st.markdown("---")


# ===== WORKFLOW NAVIGATION (FINAL — ALIGNED WITH SYSTEM) =====

tab_data, tab_prep, tab_clean, tab_analysis, tab_policy, tab_ai, tab_chat, tab_debug = st.tabs([
    "📊 Data Ingestion",
    "🧠 Data Preparation",
    "🧹 Clean & Edit",
    "📈 Statistical Analysis",
    "🏛️ Policy Intelligence",
    "🤖 AI Synthesis",
    "💬 AI Assistant",
    "⚙️ System Debug"
])
# ===== SIDEBAR (CONTROL PANEL) =====
st.sidebar.markdown("## ⚙️ Control Panel")

# ---- Workflow Status ----
st.sidebar.markdown("### Workflow Status")
if "active_df" in st.session_state:
    st.sidebar.success("Data Loaded")
else:
    st.sidebar.warning("Load Data First")

if "suggestions" in st.session_state:
    st.sidebar.success("Schema Ready")
else:
    st.sidebar.info("Run Extraction")

# ---- AI SETTINGS ----
st.sidebar.markdown("### 🤖 AI Engine")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Gemini (Cloud)", "LLaMA (Local)"],
    key="model_selector"
)

# ---- CONTEXT INFO ----
st.sidebar.markdown("### 📌 System Info")
st.sidebar.caption(
    "This platform performs PCA, clustering, and policy intelligence generation "
    "for district-level education governance."
)
# ================================
# TAB 1 — DATA INGESTION
# ================================
with tab_data:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Data Ingestion</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Load district-level dataset for analysis. Ensure one row per district with performance indicators.</div>',
        unsafe_allow_html=True
    )

    # KPI Row
    k1, k2, k3 = st.columns(3)
    df_state = st.session_state.get("original_df", pd.DataFrame())

    k1.metric("Rows", df_state.shape[0] if isinstance(df_state, pd.DataFrame) else 0)
    k2.metric("Columns", df_state.shape[1] if isinstance(df_state, pd.DataFrame) else 0)
    k3.metric("Status", "Loaded" if isinstance(df_state, pd.DataFrame) and not df_state.empty else "Not Loaded")

    st.markdown("---")

    # Input Mode
    input_col1, input_col2 = st.columns([2,1])

    with input_col1:
        input_mode = st.selectbox(
            "Select Data Source",
            ["Sample Dataset", "Upload File", "Manual Entry"],
            key="data_input_mode"
        )

    with input_col2:
        st.info("Use sample data to explore system quickly.")

    original_df = None

    # ---- SAMPLE ----
    if input_mode == "Sample Dataset":
        original_df = load_sample()
        st.success(f"Sample dataset loaded ({original_df.shape[0]} districts).")

    # ---- UPLOAD ----
    elif input_mode == "Upload File":
        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    original_df = pd.read_csv(uploaded)
                else:
                    original_df = pd.read_excel(uploaded)

                st.success(f"Loaded {original_df.shape[0]} rows × {original_df.shape[1]} columns")

            except Exception as e:
                st.error("File read failed: " + pretty_exception(e))

    # ---- MANUAL ----
    else:
        template_cols = st.text_input(
            "Define Columns",
            value="state,district,EVS,Language,Math"
        )

        cols = [c.strip() for c in template_cols.split(",") if c.strip()]
        rows = st.number_input("Rows to generate", min_value=1, max_value=200, value=10)

        if st.button("Create Grid"):
            original_df = pd.DataFrame([[""] * len(cols) for _ in range(rows)], columns=cols)
            st.success("Manual dataset created")

    # Save to session
    if isinstance(original_df, pd.DataFrame) and not original_df.empty:
        st.session_state["original_df"] = original_df.copy()

    st.markdown("### Dataset Preview")

    if isinstance(st.session_state.get("original_df"), pd.DataFrame) and not st.session_state["original_df"].empty:
        st.dataframe(st.session_state["original_df"].head(20), use_container_width=True)
    else:
        st.warning("No dataset available yet.")

    st.markdown('</div>', unsafe_allow_html=True)



# ================================
# TAB 2 — DATA PREPARATION (EXTRACTION)
# ================================
with tab_prep:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Data Preparation & Schema Mapping</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Identify column roles and prepare dataset for analysis. System can auto-detect schema or use AI assistance.</div>',
        unsafe_allow_html=True
    )

    df_input = st.session_state.get("original_df", pd.DataFrame())

    if df_input.empty:
        st.warning("Load dataset in Data Ingestion step.")
        st.stop()

    # KPI Row
    k1, k2 = st.columns(2)
    k1.metric("Columns Detected", df_input.shape[1])
    k2.metric("Rows Available", df_input.shape[0])

    st.markdown("---")

    c1, c2 = st.columns(2)

    # ---------------- MOCK EXTRACTION ----------------
    with c1:
        st.markdown("### Local Schema Detection")

        if st.button("Run Automatic Mapping"):
            preview = df_input.head(50).copy()

            suggestions, cleaned_preview = mock_gemini_extract_preview(preview)

            st.session_state["suggestions"] = suggestions
            st.session_state["cleaned_preview"] = cleaned_preview

            st.success("Schema detected successfully")

    # ---------------- GEMINI ----------------
    with c2:
        st.markdown("### AI-Assisted Mapping")

        consent = st.checkbox("Allow external API usage (no PII sent)", value=False)

        gem_model = st.selectbox(
            "Model",
            ["models/gemini-2.5-flash", "models/gemini-2.5-pro"],
            key="gem_model_select"
        )

        if st.button("Run AI Mapping"):

            if not consent:
                st.error("Consent required")
            else:
                sanitized = sanitize_sample(df_input.head(50))

                try:
                    try:
                        import os, google.generativeai as genai

                        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                        if not api_key:
                            raise ValueError("Missing API key")

                        genai.configure(api_key=api_key)

                        model = genai.GenerativeModel(gem_model)

                        prompt = f"""
                        Identify column roles.

                        Dataset:
                        {sanitized.to_csv(index=False)}

                        Return structured JSON mapping.
                        """

                        resp = model.generate_content(prompt)
                        text = getattr(resp, "text", "")

                        import json, re
                        match = re.search(r"\{{.*\}}", text, re.S)

                        if match:
                            parsed = json.loads(match.group(0))
                            suggestions = parsed.get("suggestions", [])
                        else:
                            raise ValueError("Invalid response")

                        cleaned_preview = _build_local_cleaned_preview_from_suggestions(
                            sanitized,
                            suggestions
                        )

                    except Exception:
                        suggestions, cleaned_preview = mock_gemini_extract_preview(sanitized)

                    st.session_state["suggestions"] = suggestions
                    st.session_state["cleaned_preview"] = cleaned_preview

                    st.success("AI extraction complete")

                except Exception as e:
                    st.error("Extraction failed: " + pretty_exception(e))

    # ---------------- DISPLAY ----------------
    st.markdown("---")

    if "suggestions" in st.session_state:
        st.markdown("### Detected Schema")

        try:
            st.dataframe(
                pd.DataFrame(st.session_state["suggestions"]),
                use_container_width=True
            )
        except Exception:
            st.write(st.session_state["suggestions"])

        st.success("Proceed to Data Cleaning step")

    else:
        st.info("Run schema detection to continue")

    st.markdown('</div>', unsafe_allow_html=True)
    
# ---------------- Tab 3 - Clean & Edit ----------------
with tab_clean:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Clean & Edit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-desc">
        Validate mappings, clean datasets, edit values interactively,
        and prepare a governance-ready analytical dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # LOAD BASE DATAFRAME
    # =====================================================

    base_preview = st.session_state.get("cleaned_preview")

    if (
        base_preview is None or
        not isinstance(base_preview, pd.DataFrame) or
        base_preview.empty
    ):
        base_preview = st.session_state.get(
            "original_df",
            pd.DataFrame()
        )

    if (
        base_preview is None or
        not isinstance(base_preview, pd.DataFrame) or
        base_preview.empty
    ):
        st.warning(
            "No dataset available. Complete previous workflow steps first."
        )
        st.stop()

    # =====================================================
    # DATASET STATUS
    # =====================================================

    st.success(
        f"Dataset Ready • {base_preview.shape[0]} rows • "
        f"{base_preview.shape[1]} columns"
    )

    # =====================================================
    # SCHEMA MAPPING SECTION
    # =====================================================

    if "suggestions" in st.session_state:

        st.markdown("---")

        st.subheader("Schema Mapping & Standardization")

        st.markdown(
            """
            Confirm or modify automatically detected schema mappings.
            Standardized mappings improve downstream analytics quality.
            """
        )

        mapping_cols = []

        for i, s in enumerate(st.session_state["suggestions"]):

            orig = s.get("original", "")
            suggested_role = s.get("suggested_role", orig)
            dtype = s.get("dtype", "unknown")
            conf = s.get("confidence", "")

            st.markdown(
                f"""
                <div class="alert-info">
                    <b>Detected Column:</b> {orig}<br>
                    Suggested Role: <b>{suggested_role}</b><br>
                    Data Type: {dtype}<br>
                    Confidence: {conf}
                </div>
                """,
                unsafe_allow_html=True
            )

            colA, colB = st.columns([2, 1])

            with colA:

                options = [
                    "(keep original)",
                    "state",
                    "district",
                    "EVS",
                    "Language",
                    "Math",
                    "math_pct",
                    "lang_pct",
                    "evs_pct",
                    "infra",
                    "ptr",
                    "learning_score",
                    "(drop)"
                ]

                default_idx = (
                    options.index(suggested_role)
                    if suggested_role in options
                    else 0
                )

                sel = st.selectbox(
                    f"Map `{orig}` to:",
                    options,
                    index=default_idx,
                    key=f"map_{i}"
                )

            with colB:

                new_name = st.text_input(
                    f"Rename `{orig}`",
                    value=(
                        suggested_role
                        if sel != "(keep original)"
                        else orig
                    ),
                    key=f"rename_{i}"
                )

            mapping_cols.append((orig, sel, new_name))

        # =================================================
        # APPLY MAPPINGS
        # =================================================

        if st.button(
            "Apply Mapping & Standardization",
            key="apply_mapping_btn"
        ):

            base_df = base_preview.copy()

            transformed = pd.DataFrame()

            numeric_map = [
                "math_pct",
                "lang_pct",
                "infra",
                "ptr",
                "learning_score",
                "EVS",
                "Math",
                "Language",
                "evs_pct"
            ]

            for orig, sel, new_name in mapping_cols:

                if sel == "(drop)":
                    continue

                if orig not in base_df.columns:
                    continue

                target_name = (
                    new_name.strip()
                    if new_name
                    else orig
                )

                if sel in numeric_map:

                    transformed[target_name] = pd.to_numeric(
                        base_df[orig],
                        errors="coerce"
                    )

                else:

                    col_series = base_df[orig]

                    try:

                        coerced = pd.to_numeric(
                            col_series,
                            errors="coerce"
                        )

                        non_na = coerced.notna().sum()

                        total_nonblank = (
                            col_series
                            .replace({np.nan: None})
                            .dropna()
                            .shape[0]
                        )

                        if (
                            total_nonblank > 0 and
                            (non_na / max(1, total_nonblank)) >= 0.6
                        ):
                            transformed[target_name] = coerced

                        else:
                            transformed[target_name] = (
                                col_series
                                .astype(str)
                                .str.strip()
                            )

                    except Exception:

                        transformed[target_name] = (
                            col_series
                            .astype(str)
                            .str.strip()
                        )

            st.session_state["active_df"] = transformed.copy()

            st.session_state["last_mapping"] = {
                "mapping": mapping_cols,
                "applied_at": datetime.utcnow().isoformat()
            }

            st.success(
                "Schema mapping successfully applied."
            )

    else:

        st.info(
            "No schema suggestions available. "
            "Run Data Preparation step first."
        )

    # =====================================================
    # LOAD ACTIVE DATA
    # =====================================================

    st.markdown("---")
    st.subheader("Interactive Dataset Editor")

    if (
        "active_df" in st.session_state and
        isinstance(st.session_state["active_df"], pd.DataFrame) and
        not st.session_state["active_df"].empty
    ):

        df_for_grid = st.session_state["active_df"].copy()

    else:

        df_for_grid = (
            base_preview.copy()
            if isinstance(base_preview, pd.DataFrame)
            else pd.DataFrame()
        )

    if df_for_grid is None:
        df_for_grid = pd.DataFrame()

    # =====================================================
    # SAFE DATA EDITOR
    # =====================================================

    def safe_data_editor(df: pd.DataFrame, key: str):

        try:

            return st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=key
            )

        except Exception:

            st.warning(
                "Advanced editor unavailable. "
                "Using dataframe fallback."
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            return df

    # =====================================================
    # AGGRID / DATA EDITOR
    # =====================================================

    if (
        ST_AGGRID_AVAILABLE and
        not df_for_grid.empty
    ):

        try:

            gb = GridOptionsBuilder.from_dataframe(df_for_grid)

            gb.configure_default_column(
                editable=True,
                groupable=True,
                resizable=True
            )

            gb.configure_grid_options(
                enableRangeSelection=True,
                ensureDomOrder=True
            )

            grid_options = gb.build()

            grid_response = AgGrid(
                df_for_grid,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                fit_columns_on_grid_load=True,
                enable_enterprise_modules=False,
                allow_unsafe_jscode=False
            )

            df_edited = pd.DataFrame(
                grid_response["data"]
            )

        except Exception as e:

            st.warning(
                "AgGrid unavailable. "
                "Using fallback editor."
            )

            st.info(pretty_exception(e))

            df_edited = safe_data_editor(
                df_for_grid,
                key="fallback_editor"
            )

    else:

        df_edited = safe_data_editor(
            df_for_grid,
            key="default_editor"
        )

    # =====================================================
    # VALIDATE EDITED DATA
    # =====================================================

    if not isinstance(df_edited, pd.DataFrame):
        df_edited = pd.DataFrame(df_for_grid)

    # =====================================================
    # AUTO NUMERIC COERCION
    # =====================================================

    def coerce_numeric_like_columns(
        df: pd.DataFrame,
        threshold: float = 0.6
    ) -> pd.DataFrame:

        df = df.copy()

        for col in df.columns:

            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                continue

            coerced = pd.to_numeric(
                series,
                errors="coerce"
            )

            non_na = coerced.notna().sum()

            total_nonblank = (
                series
                .replace({np.nan: None})
                .dropna()
                .shape[0]
            )

            if (
                total_nonblank > 0 and
                (non_na / total_nonblank) >= threshold
            ):
                df[col] = coerced

        return df

    # =====================================================
    # APPLY CLEANING
    # =====================================================

    df_edited = coerce_numeric_like_columns(
        df_edited,
        threshold=0.6
    )

    st.session_state["df_edited"] = df_edited.copy()

    # =====================================================
    # PREVIEW
    # =====================================================

    st.markdown("---")
    st.subheader("Cleaned Dataset Preview")

    st.dataframe(
        df_edited.head(15),
        use_container_width=True
    )

    # =====================================================
    # DATA QUALITY SUMMARY
    # =====================================================

    st.markdown("---")
    st.subheader("Data Quality Diagnostics")

    q1, q2, q3 = st.columns(3)

    missing_cells = int(df_edited.isna().sum().sum())

    numeric_cols = (
        df_edited
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    q1.metric(
        "Rows",
        df_edited.shape[0]
    )

    q2.metric(
        "Numeric Columns",
        len(numeric_cols)
    )

    q3.metric(
        "Missing Cells",
        missing_cells
    )

    # =====================================================
    # EXPORT CONTROLS
    # =====================================================

    st.markdown("---")
    st.subheader("Dataset Persistence")

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "Save Cleaned Dataset",
            key="save_cleaned_dataset"
        ):

            try:

                os.makedirs("data", exist_ok=True)

                df_edited.to_csv(
                    "data/edited_dataset.csv",
                    index=False
                )

                st.success(
                    "Dataset saved successfully."
                )

            except Exception as e:

                st.error(
                    "Save failed: "
                    + pretty_exception(e)
                )

    with c2:

        csv_export = df_edited.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Cleaned Dataset",
            data=csv_export,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="download_cleaned_dataset"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# Action buttons
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Save edited dataset to data/edited_dataset.csv"):
        os.makedirs("data", exist_ok=True)
        try:
            df_to_save = st.session_state.get("df_edited", pd.DataFrame())

            if isinstance(df_to_save, pd.DataFrame) and not df_to_save.empty:
                df_to_save.to_csv("data/edited_dataset.csv", index=False)
                st.success("Saved to data/edited_dataset.csv")
            else:
                st.error("Nothing to save.")

        except Exception as e:
            st.error("Save failed: " + pretty_exception(e))


with c2:
    if st.button("Load data/edited_dataset.csv (if exists)"):
        path = "data/edited_dataset.csv"

        if os.path.exists(path):
            try:
                df_loaded = pd.read_csv(path)
                st.session_state["active_df"] = df_loaded
                st.session_state["df_edited"] = df_loaded
                st.success("Loaded saved dataset into editor")

            except Exception as e:
                st.error("Load failed: " + pretty_exception(e))
        else:
            st.error("No saved dataset found at data/edited_dataset.csv")


with c3:
    df_stats = st.session_state.get("df_edited")

    if isinstance(df_stats, pd.DataFrame) and not df_stats.empty:
        num_cols = df_stats.select_dtypes(include=[np.number]).columns.tolist()

        if num_cols:
            desc = df_stats[num_cols].describe().T
            desc["median"] = df_stats[num_cols].median()

            try:
                desc["mode"] = df_stats[num_cols].mode().iloc[0]
            except Exception:
                desc["mode"] = np.nan

            csv_stats = desc.reset_index().to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download stats summary (CSV)",
                data=csv_stats,
                file_name="dataset_stats.csv",
                mime="text/csv"
            )


st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab 4 - Analysis (ADVANCED) ----------------
with tab_analysis:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Advanced PCA & Clustering Analytics</div>', unsafe_allow_html=True)

    # -------- SAFE DATA FETCH --------
    def safe_get_df(*keys):
        for k in keys:
            df = st.session_state.get(k)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return None

    df_for_analysis = safe_get_df("active_df", "df_edited")

    if not isinstance(df_for_analysis, pd.DataFrame) or df_for_analysis.empty:
        st.info("No data available. Prepare dataset first.")
        st.stop()

    # -------- NUMERIC CHECK --------
    numeric_cols = df_for_analysis.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available.")
        st.stop()

    # -------- VARIABLE SELECTION --------
    cols_sel = st.multiselect(
        "Select indicators",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
        key="tab4_vars"
    )

    if not cols_sel:
        st.warning("Select variables.")
        st.stop()

    # -------- COMPLETE CASE --------
    df_complete = df_for_analysis.dropna(subset=cols_sel)

    if df_complete.shape[0] < 2:
        st.warning("Not enough data after filtering.")
        st.stop()

    st.success(f"Using {df_complete.shape[0]} rows for analysis")

    # -------- PCA --------
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_complete[cols_sel])

        n_comp = min(3, len(cols_sel))
        pca = PCA(n_components=n_comp, random_state=42)
        pcs = pca.fit_transform(X)

        exp_var = pca.explained_variance_ratio_

        st.subheader("PCA Explained Variance")
        st.write({f"PC{i+1}": round(v, 4) for i, v in enumerate(exp_var)})

        loadings = pd.DataFrame(
            pca.components_.T,
            index=cols_sel,
            columns=[f"PC{i+1}" for i in range(n_comp)]
        )

        st.subheader("PCA Loadings")
        st.dataframe(loadings)

        # -------- CLUSTERING --------
        k = min(3, df_complete.shape[0])
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pcs)

        df_result = df_complete.copy()
        df_result["_cluster"] = labels

        st.subheader("Cluster Distribution")
        st.write(df_result["_cluster"].value_counts().to_dict())

        # -------- VISUALIZATION --------
        if pcs.shape[1] >= 2:
            plot_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"])
            plot_df["cluster"] = labels.astype(str)

            fig = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="cluster",
                title="Cluster Visualization"
            )
            import uuid

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=str(uuid.uuid4())
        )
        # -------- SAVE RESULT --------
        st.session_state["analysis_result"] = df_result

    else:
        st.warning("Scikit-learn not available.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PCA SETUP ----------------
    n_samples, n_features = df_complete.shape[0], len(cols_sel)
    max_comp = min(6, n_samples, n_features)

    n_comp = st.slider("PCA Components", 1, max_comp, min(2, max_comp))

    # ---------------- SCALING ----------------
    scaler = StandardScaler()
    X = scaler.fit_transform(df_complete[cols_sel])

    # ---------------- PCA ----------------
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(X)

    exp_var = pca.explained_variance_ratio_

    # 📊 Scree plot
    fig_scree = px.line(
        x=list(range(1, len(exp_var)+1)),
        y=exp_var,
        markers=True,
        title="Scree Plot (Variance Explained)"
    )
    import uuid

    st.plotly_chart(
    fig,
    use_container_width=True,
    key=str(uuid.uuid4())
)

    # 📊 Cumulative variance
    cum_var = np.cumsum(exp_var)
    fig_cum = px.line(
        x=list(range(1, len(cum_var)+1)),
        y=cum_var,
        markers=True,
        title="Cumulative Variance"
    )
    import uuid

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=str(uuid.uuid4())
    )

    st.dataframe(pd.DataFrame({
        "Component": range(1, len(exp_var)+1),
        "Explained Variance": exp_var,
        "Cumulative": cum_var
    }))

    # ---------------- CORRELATION ----------------
    st.markdown("### Correlation Heatmap")
    corr = df_complete[cols_sel].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    import uuid

    st.plotly_chart(
    fig,
    use_container_width=True,
    key=str(uuid.uuid4())
)
    # ---------------- OUTLIERS ----------------
    if st.checkbox("Detect Outliers"):
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        outliers = (z_scores > 3).any(axis=1)
        st.write(f"Outliers detected: {outliers.sum()}")

    # ---------------- CLUSTERING ----------------
    k_max = min(10, n_samples - 1)

    scores = []
    for k in range(2, k_max+1):
        labels = KMeans(n_clusters=k, n_init=10).fit_predict(pcs)
        score = silhouette_score(pcs, labels)
        scores.append((k, score))

    best_k = max(scores, key=lambda x: x[1])[0]

    st.write(f"Recommended K (Silhouette): {best_k}")

    k = st.slider("Clusters", 2, k_max, best_k)

    model = KMeans(n_clusters=k, n_init=10)
    labels = model.fit_predict(pcs)

    df_complete["_cluster"] = labels

    # ---------------- VISUAL ----------------
    if pcs.shape[1] >= 2:
        plot_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"])
        plot_df["cluster"] = labels.astype(str)

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="cluster",
            title="Cluster Visualization"
        )
        import uuid

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=str(uuid.uuid4())
    )

    # ---------------- CLUSTER PROFILE ----------------
    st.markdown("### Cluster Profiles")
    profile = df_complete.groupby("_cluster")[cols_sel].mean().round(2)
    st.dataframe(profile)

    # ---------------- POLICY INSIGHTS ----------------
    st.markdown("### Policy Insights")
    for col in cols_sel:
        mean_val = df_complete[col].mean()
        st.write(f"- {col}: Avg = {round(mean_val,2)}")

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "Download Results",
        df_complete.to_csv(index=False),
        "analysis_results.csv",
        "text/csv"
    )

    st.markdown('</div>', unsafe_allow_html=True)
    
# ---------------- Tab 5 - Governance Intelligence Engine v2 ----------------
with tab_policy:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">
            Governance Intelligence Engine
        </div>

        <div class="section-desc">
            AI-assisted educational governance intelligence platform for
            district diagnostics, systemic risk detection, intervention
            prioritization, and policy decision support.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # SAFE DATA FETCH
    # =========================================================

    df = st.session_state.get("active_df")

    if (
        df is None or
        not isinstance(df, pd.DataFrame) or
        df.empty
    ):
        df = st.session_state.get("df_edited")

    if (
        df is None or
        not isinstance(df, pd.DataFrame) or
        df.empty
    ):
        df = st.session_state.get("cleaned_preview")

    if (
        df is None or
        not isinstance(df, pd.DataFrame) or
        df.empty
    ):
        st.error(
            "No governance-ready dataset available. "
            "Complete previous workflow stages first."
        )
        st.stop()

    # =========================================================
    # NUMERIC VARIABLES
    # =========================================================

    numeric_cols = (
        df.select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    if len(numeric_cols) < 2:
        st.warning(
            "At least 2 numeric indicators are required."
        )
        st.stop()

    selected_vars = st.multiselect(
        "Governance Indicators",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="gov_engine_vars_v2"
    )

    if len(selected_vars) < 2:
        st.warning(
            "Select at least 2 governance indicators."
        )
        st.stop()

    # =========================================================
    # ANALYSIS ENGINE
    # =========================================================

    with st.spinner(
        "Running governance intelligence engine..."
    ):

        stats = compute_basic_stats(df)

        adv = run_advanced_analyses(
            df,
            selected_vars,
            3,
            3
        )

    # =========================================================
    # DISTRICT COLUMN
    # =========================================================

    district_col = next(
        (
            c for c in df.columns
            if "district" in c.lower()
        ),
        None
    )

    # =========================================================
    # GOVERNANCE SCORING ENGINE
    # =========================================================

    df_policy = df.copy()

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        df_policy[selected_vars].fillna(
            df_policy[selected_vars].median()
        )
    )

    scaled_df = pd.DataFrame(
        scaled,
        columns=selected_vars
    )

    for col in scaled_df.columns:

        if "ptr" in col.lower():
            scaled_df[col] *= -1

    df_policy["Education_Health_Index"] = (
        scaled_df.mean(axis=1) * 20 + 50
    ).round(2)

    # =========================================================
    # PRIORITY CLASSIFICATION
    # =========================================================

    def classify_priority(score):

        if score < 35:
            return "Critical"

        elif score < 50:
            return "High Priority"

        elif score < 65:
            return "Moderate"

        return "Stable"

    df_policy["Priority"] = (
        df_policy["Education_Health_Index"]
        .apply(classify_priority)
    )

    # =========================================================
    # POLICY RISK ENGINE
    # =========================================================

    risk_labels = []

    for _, row in df_policy.iterrows():

        risks = []

        try:

            evs = pd.to_numeric(
                row.get("EVS"),
                errors="coerce"
            )

            ptr = pd.to_numeric(
                row.get("ptr"),
                errors="coerce"
            )

            infra = pd.to_numeric(
                row.get("infra"),
                errors="coerce"
            )

            math = pd.to_numeric(
                row.get("Math"),
                errors="coerce"
            )

            lang = pd.to_numeric(
                row.get("Language"),
                errors="coerce"
            )

            if pd.notna(evs) and evs < 45:
                risks.append("Learning Crisis")

            if (
                pd.notna(ptr) and
                ptr > 35
            ):
                risks.append("Teacher Burden")

            if (
                pd.notna(infra) and
                infra < 0.5
            ):
                risks.append("Infrastructure Stress")

            if (
                pd.notna(math) and
                pd.notna(lang)
            ):
                if abs(math - lang) > 20:
                    risks.append(
                        "Subject Performance Imbalance"
                    )

        except Exception:
            pass

        if not risks:
            risks.append("Stable")

        risk_labels.append(", ".join(risks))

    df_policy["Policy_Risk"] = risk_labels

    # =========================================================
    # NATIONAL GOVERNANCE STATUS
    # =========================================================

    avg_ehi = round(
        df_policy["Education_Health_Index"].mean(),
        2
    )

    if avg_ehi >= 70:
        gov_status = "Stable Governance"
        gov_class = "alert-success"

    elif avg_ehi >= 55:
        gov_status = "Moderate Systemic Risk"
        gov_class = "alert-warning"

    else:
        gov_status = "Critical Governance Stress"
        gov_class = "alert-critical"

    st.markdown(
        f"""
        <div class="{gov_class}">
            <b>National Education Governance Status:</b> {gov_status}
            <br><br>
            System-wide educational performance, staffing pressure, and infrastructure readiness were assessed through a composite governance intelligence model.
            <br><br>
            Current National Governance Score: <b>{avg_ehi}/100</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # EXECUTIVE KPI STRIP
    # =========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    district_count = df_policy.shape[0]

    critical_count = int(
        (df_policy["Priority"] == "Critical")
        .sum()
    )

    high_risk_count = int(
        (
            df_policy["Policy_Risk"]
            .str.contains(
                "Crisis|Stress|Burden",
                case=False
            )
        ).sum()
    )

    avg_infra = (
        round(df_policy["infra"].mean(), 2)
        if "infra" in df_policy.columns
        else "NA"
    )

    avg_ptr = (
        round(df_policy["ptr"].mean(), 1)
        if "ptr" in df_policy.columns
        else "NA"
    )

    # Bulletproof helper function to prevent IDE auto-formatting bugs
    def render_exec_kpi(value, label):
        val_safe = str(value).strip()
        label_safe = str(label).strip()
        html_str = '<div class="kpi-card">'
        html_str += '<div class="kpi-value">' + val_safe + '</div>'
        html_str += '<div class="kpi-label">' + label_safe + '</div>'
        html_str += '</div>'
        st.markdown(html_str, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        render_exec_kpi(district_count, "Districts")

    with k2:
        render_exec_kpi(avg_ehi, "Governance Score")

    with k3:
        render_exec_kpi(critical_count, "Critical Districts")

    with k4:
        render_exec_kpi(avg_infra, "Avg Infrastructure")

    with k5:
        render_exec_kpi(avg_ptr, "Avg PTR")
    # =========================================================
    # GEOSPATIAL INTELLIGENCE MAP
    # =========================================================
    
    st.markdown("---")
    st.subheader("Geospatial Risk Distribution")
    
    # Check if we have coordinate data to plot
    lat_col = next((c for c in df_policy.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df_policy.columns if "lon" in c.lower()), None)

    if lat_col and lon_col:
        st.markdown(
            "<div class='section-desc'>Interactive mapping of district priority levels. Hover over points for specific governance metrics.</div>", 
            unsafe_allow_html=True
        )
        
        # Build the Map
        fig_map = px.scatter_mapbox(
            df_policy,
            lat=lat_col,
            lon=lon_col,
            color="Priority",
            size="Education_Health_Index",  # Bubble size based on score
            size_max=15,
            hover_name=district_col if district_col else None,
            hover_data={
                lat_col: False, # Hide raw coords in tooltip
                lon_col: False,
                "Education_Health_Index": True,
                "Policy_Risk": True,
                "ptr": True,
                "infra": True
            },
            color_discrete_map={
                "Critical": "#dc2626",       # Red
                "High Priority": "#ea580c",  # Orange
                "Moderate": "#f59e0b",       # Yellow
                "Stable": "#16a34a"          # Green
            },
            zoom=3.5,
            center={"lat": 22.0, "lon": 78.0}, # Centered roughly on India
            height=550
        )
        
        # carto-positron gives a beautiful, clean base map without needing an API key
        fig_map.update_layout(
            mapbox_style="carto-positron", 
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Wrap in a card for styling consistency
        st.markdown('<div class="card" style="padding: 10px;">', unsafe_allow_html=True)
        st.plotly_chart(fig_map, use_container_width=True, key="geospatial_risk_map")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("💡 Upload data with 'Latitude' and 'Longitude' columns to unlock the Geospatial Intelligence map.")

    
    # =========================================================
    # PRIORITY DISTRIBUTION
    # =========================================================

    st.markdown("---")

    left_viz, right_viz = st.columns(2)

    with left_viz:

        st.subheader("Priority Distribution")

        priority_counts = (
            df_policy["Priority"]
            .value_counts()
            .reset_index()
        )

        priority_counts.columns = [
            "Priority",
            "Count"
        ]

        fig_priority = px.bar(
            priority_counts,
            x="Priority",
            y="Count",
            color="Priority",
            text_auto=True,
            title="District Priority Classification"
        )

        st.plotly_chart(
            fig_priority,
            use_container_width=True,
            key="gov_priority_chart_v2"
        )

    with right_viz:

        st.subheader("Governance Risk Composition")

        risk_counts = (
            df_policy["Policy_Risk"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        risk_counts.columns = [
            "Risk",
            "Count"
        ]

        fig_risk = px.pie(
            risk_counts,
            names="Risk",
            values="Count",
            title="Policy Risk Distribution"
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True,
            key="gov_risk_pie_v2"
        )

    # =========================================================
    # TOP & BOTTOM DISTRICTS
    # =========================================================

    st.markdown("---")

    top_col, bottom_col = st.columns(2)

    if district_col:

        district_view = df_policy[
            [
                district_col,
                "Education_Health_Index",
                "Priority",
                "Policy_Risk"
            ]
        ].copy()

        with top_col:

            st.subheader(
                "Top Performing Districts"
            )

            top_df = (
                district_view
                .sort_values(
                    "Education_Health_Index",
                    ascending=False
                )
                .head(10)
            )

            st.dataframe(
                top_df,
                use_container_width=True
            )

        with bottom_col:

            st.subheader(
                "Priority Intervention Districts"
            )

            bottom_df = (
                district_view
                .sort_values(
                    "Education_Health_Index",
                    ascending=True
                )
                .head(10)
            )

            st.dataframe(
                bottom_df,
                use_container_width=True
            )

    # =========================================================
    # DISTRICT INTELLIGENCE STUDIO
    # =========================================================

    if district_col:

        st.markdown("---")

        st.markdown(
            """
            <div class="section-title">
                District Intelligence Studio
            </div>

            <div class="section-desc">
                District-level governance diagnostics,
                comparative benchmarking,
                and intervention intelligence.
            </div>
            """,
            unsafe_allow_html=True
        )

        district_options = (
            df_policy[district_col]
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        selected_district = st.selectbox(
            "Select District",
            district_options,
            key="district_intelligence_v2"
        )

        district_data = df_policy[
            df_policy[district_col].astype(str)
            == selected_district
        ].iloc[0]

        # =====================================================
        # DISTRICT KPI PROFILE
        # =====================================================

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Governance Score",
            district_data["Education_Health_Index"]
        )

        d2.metric(
            "Priority",
            district_data["Priority"]
        )

        d3.metric(
            "Risk Status",
            district_data["Policy_Risk"]
        )

        cluster_value = "NA"

        try:

            cluster_assignments = adv.get(
                "cluster_assignments"
            )

            if cluster_assignments is not None:

                cluster_df = pd.DataFrame(
                    cluster_assignments
                )

                if district_col in cluster_df.columns:

                    row_cluster = cluster_df[
                        cluster_df[district_col]
                        .astype(str)
                        == selected_district
                    ]

                    if not row_cluster.empty:

                        cluster_value = int(
                            row_cluster["_cluster"]
                            .iloc[0]
                        )

        except Exception:
            pass

        d4.metric(
            "Cluster",
            cluster_value
        )

        # =====================================================
        # DISTRICT BENCHMARK TABLE
        # =====================================================

        st.markdown("### District Benchmarking")

        benchmark_df = pd.DataFrame({
            "Indicator": selected_vars,
            "District": [
                district_data[v]
                for v in selected_vars
            ],
            "National Average": [
                round(df_policy[v].mean(), 2)
                for v in selected_vars
            ]
        })

        st.dataframe(
            benchmark_df,
            use_container_width=True
        )

        # =====================================================
        # RADAR CHART
        # =====================================================

        st.markdown("### Governance Radar")

        radar_df = pd.DataFrame({
            "Indicator": selected_vars,
            "District": [
                district_data[v]
                for v in selected_vars
            ],
            "National": [
                round(df_policy[v].mean(), 2)
                for v in selected_vars
            ]
        })

        fig_radar = go.Figure()

        fig_radar.add_trace(
            go.Scatterpolar(
                r=radar_df["District"],
                theta=radar_df["Indicator"],
                fill="toself",
                name="District"
            )
        )

        fig_radar.add_trace(
            go.Scatterpolar(
                r=radar_df["National"],
                theta=radar_df["Indicator"],
                fill="toself",
                name="National Average"
            )
        )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True
                )
            ),
            showlegend=True,
            height=550
        )

        st.plotly_chart(
            fig_radar,
            use_container_width=True,
            key="district_radar_v2"
        )

        # =====================================================
        # DISTRICT RISK DIAGNOSIS
        # =====================================================

        st.markdown("### District Governance Diagnosis")

        diagnosis = []

        try:

            if (
                "EVS" in district_data and
                district_data["EVS"] < 50
            ):
                diagnosis.append(
                    "Foundational learning outcomes remain substantially below national expectations."
                )

            if (
                "ptr" in district_data and
                district_data["ptr"] > 35
            ):
                diagnosis.append(
                    "Teacher workload pressure indicates potential classroom capacity stress."
                )

            if (
                "infra" in district_data and
                district_data["infra"] < 0.5
            ):
                diagnosis.append(
                    "Infrastructure readiness gaps may be constraining educational delivery quality."
                )

            if (
                "Math" in district_data and
                "Language" in district_data
            ):

                if abs(
                    district_data["Math"] -
                    district_data["Language"]
                ) > 20:

                    diagnosis.append(
                        "Substantial inter-subject performance imbalance detected."
                    )

        except Exception:
            pass

        if not diagnosis:

            diagnosis.append(
                "No major systemic governance stress indicators detected."
            )

        for item in diagnosis:

            st.markdown(
                f"""
                <div class="alert-warning">
                    {item}
                </div>
                """,
                unsafe_allow_html=True
            )

        # =====================================================
        # INTERVENTION ROADMAP
        # =====================================================

        st.markdown("### Strategic Intervention Roadmap")

        roadmap = []

        try:

            if (
                "ptr" in district_data and
                district_data["ptr"] > 35
            ):
                roadmap.append({
                    "Timeline": "Immediate",
                    "Policy Action":
                    "Deploy additional teachers and optimize staffing allocation."
                })

            if (
                "infra" in district_data and
                district_data["infra"] < 0.5
            ):
                roadmap.append({
                    "Timeline": "3–6 Months",
                    "Policy Action":
                    "Accelerate infrastructure modernization and digital readiness investments."
                })

            if (
                "EVS" in district_data and
                district_data["EVS"] < 50
            ):
                roadmap.append({
                    "Timeline": "6–12 Months",
                    "Policy Action":
                    "Implement targeted foundational learning recovery programmes."
                })

        except Exception:
            pass

        if not roadmap:

            roadmap.append({
                "Timeline": "Ongoing",
                "Policy Action":
                "Maintain governance quality and continue continuous monitoring."
            })

        roadmap_df = pd.DataFrame(
            roadmap
        )

        st.dataframe(
            roadmap_df,
            use_container_width=True
        )

    # =========================================================
    # CORRELATION HEATMAP
    # =========================================================

    st.markdown("---")

    st.subheader(
        "Governance Correlation Intelligence"
    )

    heatmap_df = (
        df_policy[selected_vars]
        .corr()
    )

    fig_heat = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        title="Indicator Correlation Structure"
    )

    st.plotly_chart(
        fig_heat,
        use_container_width=True,
        key="gov_heatmap_v2"
    )

    # =========================================================
    # STRATEGIC POLICY ACTIONS
    # =========================================================

    st.markdown("---")

    st.subheader(
        "Strategic Governance Recommendations"
    )

    st.markdown(
        """
        <div class="alert-info">

        <b>Cluster-Based Governance Reform</b><br>
        Replace uniform intervention strategies with differentiated,
        district-sensitive governance models informed by
        clustering and systemic diagnostics.

        <br><br>

        <b>Teacher Workforce Optimization</b><br>
        Introduce dynamic PTR balancing frameworks and
        targeted teacher deployment mechanisms for
        high-burden districts.

        <br><br>

        <b>Infrastructure Equalization Strategy</b><br>
        Prioritize infrastructure-deficit districts for
        accelerated modernization and digital readiness investments.

        <br><br>

        <b>Real-Time Governance Intelligence</b><br>
        Institutionalize continuous educational monitoring through
        AI-assisted governance dashboards and district intelligence systems.

        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # EXPORT CENTER
    # =========================================================

    st.markdown("---")

    st.subheader(
        "Governance Intelligence Export Center"
    )

    export_df = df_policy.copy()

    csv_export = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Governance Intelligence Dataset",
        data=csv_export,
        file_name="governance_intelligence_dataset.csv",
        mime="text/csv",
        key="gov_export_dataset_v2"
    )

    st.markdown("</div>", unsafe_allow_html=True)
# ---------------- Tab 6 - AI Policy Synthesis ----------------
with tab_ai:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Policy Synthesis</div>', unsafe_allow_html=True)

    df_for_report = _choose_reporting_df()

    if not isinstance(df_for_report, pd.DataFrame) or df_for_report.empty:
        st.info("Prepare dataset first.")
        st.stop()

    st.write(f"Dataset ready: {df_for_report.shape[0]} rows × {df_for_report.shape[1]} columns")

    # ---------------- CLEAN OUTPUT ----------------
    def clean_llm_output(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[#*`]", "", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    # ---------------- PDF GENERATOR ----------------
    def generate_pdf(report_text: str):
        try:
            from reportlab.platypus import SimpleDocTemplate, Preformatted
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.pagesizes import letter
            from io import BytesIO

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            story = [Preformatted(report_text, styles["Normal"])]

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        except Exception:
            return None

    # ---------------- POLICY ENGINE ----------------
    def generate_policy_engine(df, adv):
        df = df.copy()

        for col in ["EVS", "Language", "Math", "infra", "ptr"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        insights = []
        priorities = []

        for _, row in df.iterrows():
            name = str(row.get("district", "Unknown"))
            evs = row.get("EVS")
            infra = row.get("infra")
            ptr = row.get("ptr")

            score = 0
            issues = []

            if pd.notna(evs) and evs < 50:
                issues.append(f"learning deficit (EVS={evs})")
                score += 2

            if pd.notna(infra) and infra < 0.4:
                issues.append(f"infrastructure gap (infra={infra})")
                score += 2

            if pd.notna(ptr) and ptr > 35:
                issues.append(f"teacher overload (PTR={ptr})")
                score += 2

            if issues:
                insights.append(f"{name}: " + ", ".join(issues))
                priorities.append((name, score))

        priorities = sorted(priorities, key=lambda x: x[1], reverse=True)

        return insights, priorities

    # ---------------- ENGINE SELECT ----------------
    synth_choice = st.selectbox(
        "Synthesis Engine",
        ["Local Intelligence Engine", "Gemini (Cloud)", "Ollama (Local)"],
        key="tab6_engine"
    )

    # ---------------- ANALYSIS PREP ----------------
    stats = compute_basic_stats(df_for_report)

    numeric_cols = df_for_report.select_dtypes(include=[np.number]).columns.tolist()
    selected_vars = numeric_cols[:min(5, len(numeric_cols))]

    adv = run_advanced_analyses(
        df_for_report,
        selected_vars,
        n_pca_components=3,
        k_clusters=3
    )

    generated_text = ""

    generated_text = ""

    # ================= ENGINE SELECTION =================

if synth_choice == "Local Generator":

    st.success(
        "Using deterministic local policy engine."
    )

    insights, priorities = generate_policy_engine(
        df_for_report,
        adv
    )

    generated_text = "\n\n".join(
        insights[:12]
    )

# ================= GEMINI =================

elif synth_choice == "Gemini (Cloud)":

    consent = st.checkbox(
        "Allow external API call",
        key="tab6_consent"
    )

    if consent:

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:

            st.error("Missing GEMINI_API_KEY")

        else:

            try:

                import google.generativeai as genai

                genai.configure(
                    api_key=api_key
                )

                model = "models/gemini-2.5-flash"

                if st.button(
                    "Generate AI Report",
                    key="tab6_run"
                ):

                    with st.spinner(
                        "Generating AI policy synthesis..."
                    ):

                        prompt = f"""
You are a senior national education policy advisor.

Generate a professional governance intelligence report.

Include:
- Executive Summary
- System Diagnosis
- District Insights
- Cluster Insights
- Governance Risks
- Strategic Recommendations
- Implementation Roadmap

Use professional readable language.
Avoid markdown symbols like # and *.

DATA:
{json.dumps(stats, indent=2)}
"""

                        try:

                            response = (
                                genai.GenerativeModel(model)
                                .generate_content(
                                    prompt,
                                    generation_config={
                                        "temperature": 0.2
                                    }
                                )
                            )

                            raw = getattr(
                                response,
                                "text",
                                ""
                            )

                            generated_text = (
                                clean_llm_output(raw)
                            )

                            if not generated_text.strip():

                                st.warning(
                                    "Gemini returned empty response."
                                )

                                insights, priorities = (
                                    generate_policy_engine(
                                        df_for_report,
                                        adv
                                    )
                                )

                                generated_text = "\n\n".join(
                                    insights[:12]
                                )

                            else:

                                st.success(
                                    "Gemini report generated."
                                )

                        except Exception as e:

                            st.warning(
                                "Gemini unavailable. "
                                "Using local engine."
                            )

                            st.info(str(e))

                            insights, priorities = (
                                generate_policy_engine(
                                    df_for_report,
                                    adv
                                )
                            )

                            generated_text = "\n\n".join(
                                insights[:12]
                            )

            except Exception as e:

                st.error(
                    "Gemini library unavailable."
                )

                st.info(str(e))

# ================= OLLAMA =================

else:

    st.warning(
        "Ollama not supported on Streamlit Cloud."
    )

# ================= OUTPUT =================

if generated_text:

    st.markdown(
        '<div class="section-title">AI Policy Report</div>',
        unsafe_allow_html=True
    )

    st.text_area(
        "Generated Report",
        generated_text,
        height=550
    )

    # ======================================
    # PDF GENERATION
    # ======================================

    pdf_bytes = None

    try:

        pdf_bytes = generate_pdf(
            generated_text
        )

    except Exception as e:

        st.warning(
            "PDF generation failed. "
            "TXT export enabled instead."
        )

        st.info(str(e))

    # ======================================
    # EXPORTS
    # ======================================

    export_col1, export_col2 = st.columns(2)

    with export_col1:

        if pdf_bytes:

            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="policy_report.pdf",
                mime="application/pdf",
                key="download_pdf_report"
            )

    with export_col2:

        st.download_button(
            "Download TXT Report",
            data=generated_text,
            file_name="policy_report.txt",
            mime="text/plain",
            key="download_txt_report"
        )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ---------------- Tab 7 - AI Data Assistant (Dual-Engine) ----------------
with tab_chat:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Interactive Governance Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Consult with the AI regarding the current dataset. Choose the deterministic Local Expert System (based on the Delphi consensus database) or the Generative Gemini Advisor.</div>', unsafe_allow_html=True)

    # --- MOCK DELPHI DATABASE ---
    MOCK_DELPHI_DB = {
        "learning_math": {
            "keywords": ["math", "numeracy", "calculation", "arithmetic", "learning"],
            "interventions": [
                {"id": "INT-M1", "name": "Targeted Foundational Numeracy Remediation", "action": "Intensive 6-week pull-out program focusing on base-10 concepts and practical arithmetic.", "resources": "₹500,000 per district + 2 Master Trainers"}
            ]
        },
        "learning_lang": {
            "keywords": ["language", "reading", "literacy", "comprehension", "english", "learning"],
            "interventions": [
                {"id": "INT-L1", "name": "Phonetics & Reading Fluency Camp", "action": "Daily 45-minute guided reading sessions with leveled texts.", "resources": "Library kits + 1 Reading Coach per cluster"}
            ]
        },
        "ptr_burden": {
            "keywords": ["ptr", "teacher", "ratio", "staff", "workload", "overload"],
            "interventions": [
                {"id": "INT-P1", "name": "Emergency Teacher Deployment Framework", "action": "Re-routing surplus state educators to high-burden schools with PTR > 35.", "resources": "Deployment budget: ₹1.2M + Transport allowance"}
            ]
        },
        "infra_stress": {
            "keywords": ["infra", "infrastructure", "building", "facilities", "water", "electricity"],
            "interventions": [
                {"id": "INT-I1", "name": "Rapid Infrastructure Upgrades", "action": "Emergency repair of WASH facilities, classroom lighting, and digital readiness.", "resources": "Capital grant: ₹2.5M per district"}
            ]
        }
    }

    # --- LOCAL EXPERT ENGINE LOGIC ---
    def run_local_engine(query, df):
        query_lower = query.lower()
        response_lines = ["### 🏛️ Local Expert System Diagnosis\n*Deterministic analysis based on Delphi Consensus Matrix.*\n"]
        
        # 1. Identify specific critical districts from the live data
        if "Priority" in df.columns:
            critical_districts = df[df["Priority"] == "Critical"]
            if not critical_districts.empty:
                names = critical_districts.get("district", critical_districts.index).tolist()
                names_str = ", ".join([str(n) for n in names])
                response_lines.append(f"**Identified Critical Districts:** {names_str}\n")
        
        # 2. Extract Intent and Match with Delphi DB
        matched_interventions = []
        for category, data in MOCK_DELPHI_DB.items():
            if any(kw in query_lower for kw in data["keywords"]):
                matched_interventions.extend(data["interventions"])
        
        # 3. Fallback: If no keywords matched, scan the actual dataframe averages
        if not matched_interventions:
            response_lines.append("> *No specific metric mentioned in query. Scanning dataset for highest systemic risks...*\n")
            if "Math" in df.columns and df["Math"].mean() < 55:
                matched_interventions.extend(MOCK_DELPHI_DB["learning_math"]["interventions"])
            if "ptr" in df.columns and df["ptr"].mean() > 32:
                matched_interventions.extend(MOCK_DELPHI_DB["ptr_burden"]["interventions"])
            if not matched_interventions: # Absolute fallback
                matched_interventions.extend(MOCK_DELPHI_DB["infra_stress"]["interventions"])

        # 4. Format Output
        response_lines.append("#### Recommended Interventions (Delphi Database):")
        # Deduplicate
        seen = set()
        for inv in matched_interventions:
            if inv['id'] not in seen:
                response_lines.append(f"* **[{inv['id']}] {inv['name']}**")
                response_lines.append(f"    * **Action:** {inv['action']}")
                response_lines.append(f"    * **Resource Allocation:** {inv['resources']}")
                seen.add(inv['id'])

        return "\n".join(response_lines)


    # --- UI & STATE MANAGEMENT ---
    chat_df = st.session_state.get("active_df")
    if not isinstance(chat_df, pd.DataFrame) or chat_df.empty:
        chat_df = st.session_state.get("df_edited")

    if not isinstance(chat_df, pd.DataFrame) or chat_df.empty:
        st.warning("Please load and clean your dataset in the previous tabs before chatting with the AI.")
    else:
        # Engine Toggle
        engine_mode = st.radio(
            "Select Intelligence Routing:", 
            ["Local Expert System (Deterministic & Rule-Based)", "Gemini Policy Advisor (Generative RAG)"],
            horizontal=True
        )
        st.markdown("---")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Render chat history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input box
        if user_query := st.chat_input("E.g., Which districts are critical, and what are the pedagogical interventions?"):
            
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # --- ROUTE 1: LOCAL ENGINE ---
                if "Local" in engine_mode:
                    with st.spinner("Querying Local Delphi Database..."):
                        local_response = run_local_engine(user_query, chat_df)
                        message_placeholder.markdown(local_response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": local_response})
                
                # --- ROUTE 2: GEMINI ENGINE ---
                else:
                    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        message_placeholder.error("⚠️ No API Key found. Set GEMINI_API_KEY in your environment, or switch to the Local Engine.")
                    else:
                        try:
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            chat_model = genai.GenerativeModel("models/gemini-2.5-flash")

                            # Convert Delphi DB to string for RAG Context
                            delphi_context = json.dumps(MOCK_DELPHI_DB, indent=2)
                            data_summary = chat_df.describe(include='all').to_csv()
                            
                            # --- SMART CONTEXT INJECTION ---
                            # Dynamically find top and bottom districts so Gemini knows who they are
                            ranking_context = ""
                            if all(c in chat_df.columns for c in ["district", "EVS", "Language", "Math"]):
                                temp_df = chat_df.copy()
                                temp_df["Overall_Score"] = temp_df[["EVS", "Language", "Math"]].mean(axis=1)
                                top_districts = temp_df.sort_values("Overall_Score", ascending=False).head(5)[["district", "Overall_Score", "EVS", "Language", "Math"]].to_csv(index=False)
                                bottom_districts = temp_df.sort_values("Overall_Score", ascending=True).head(5)[["district", "Overall_Score", "EVS", "Language", "Math", "ptr", "infra"]].to_csv(index=False)
                                
                                ranking_context = f"\nTOP 5 PERFORMING DISTRICTS:\n{top_districts}\n\nBOTTOM 5 AT-RISK DISTRICTS:\n{bottom_districts}"
                            else:
                                ranking_context = "DATA SAMPLE:\n" + chat_df.head(15).to_csv(index=False)

                            system_prompt = f"""
                            You are a Senior Education Policy Advisor using RAG (Retrieval-Augmented Generation).
                            Answer the user's query using ONLY the Data Summary, the District Rankings, and the Delphi Database below.
                            
                            DELPHI INTERVENTION DATABASE:
                            {delphi_context}

                            DATA SUMMARY:
                            {data_summary}
                            
                            DISTRICT RANKINGS:
                            {ranking_context}

                            User Question: {user_query}

                            Instructions:
                            1. Be conversational but authoritative.
                            2. If suggesting solutions, you MUST cite the specific Intervention IDs (e.g., INT-M1) from the Delphi Database. Do not invent your own solutions.
                            3. Do not invent data. Use markdown formatting.
                            """

                            with st.spinner("Gemini synthesizing data and expert consensus..."):
                                response = chat_model.generate_content(system_prompt)
                                full_response = response.text

                            message_placeholder.markdown(full_response)
                            st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

                        except Exception as e:
                            message_placeholder.error(f"Error communicating with AI: {pretty_exception(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
    
# ---------------- Tab 8 - Debug ----------------

with tab_debug:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Debug & Provenance</div>',
        unsafe_allow_html=True
    )

    if "last_mapping" in st.session_state:

        st.write("Last mapping applied")

        st.json(
            st.session_state["last_mapping"]
        )

    st.markdown("---")

    st.write("Generated synthesis/debug files")

    dbg_files = sorted(
        glob("data/*genai*") +
        glob("data/*synthesis*") +
        glob("data/*ollama*") +
        glob("data/*genai_*"),
        key=os.path.getmtime,
        reverse=True
    )

    if dbg_files:

        st.write(
            f"Found {len(dbg_files)} debug files."
        )

        for fpath in dbg_files[:10]:

            with st.expander(
                os.path.basename(fpath)
            ):

                try:

                    with open(
                        fpath,
                        "r",
                        encoding="utf-8"
                    ) as fh:

                        obj = json.load(fh)

                    st.json(obj)

                except Exception:

                    st.warning(
                        "Could not render JSON."
                    )

    else:

        st.info("No debug files found.")

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ---------------- Footer ----------------

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(
    [1, 2, 1]
)

with footer_col1:

    if st.button(
        "Run Gemini ping test",
        key="gemini_ping_test_btn"
    ):

        txt, meta, err = gemini_ping_test()

        if err:

            st.error(err)

        else:

            st.success(
                f"Gemini ping OK: {txt}"
            )

with footer_col2:

    st.markdown(
        (
            "<div class='muted'>"
            "If external synthesis fails, "
            "the deterministic local policy engine "
            "will automatically generate a report."
            "</div>"
        ),
        unsafe_allow_html=True
    )

with footer_col3:

    if OLLAMA_AVAILABLE:

        st.success(
            "Ollama available"
        )

    else:

        st.info(
            "Ollama not detected"
        )
