# streamlit_app.py
"""
Edu Governance System — Streamlit single-file (refactored UI, robust)
Preserves PCA/clustering/reporting/synthesis features. Safe checks for optional
clients (Gemini cloud, Ollama local). Improved aesthetics and layout.
"""

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

# ---------------- GLOBAL UI & DESIGN SYSTEM ----------------
st.set_page_config(
    page_title="Edu Governance Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ===================== */
/* 🎨 DESIGN SYSTEM */
/* ===================== */

/* Typography */
html, body, [class*="css"]  {
    font-family: 'Inter', 'Source Sans Pro', sans-serif;
}

/* Titles */
.big-title {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 4px;
    color: #111827;
}

.subtitle {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 12px;
}

/* Section */
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 4px;
    color: #111827;
}

.section-desc {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 14px;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #f1f5f9;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}

/* KPI */
.kpi {
    font-size: 22px;
    font-weight: 600;
    color: #111827;
}

.kpi-label {
    font-size: 12px;
    color: #6b7280;
}

/* Status Badges */
.badge-success {
    background: #dcfce7;
    color: #166534;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
}

.badge-warning {
    background: #fef3c7;
    color: #92400e;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
}

.badge-danger {
    background: #fee2e2;
    color: #991b1b;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 500;
}

/* Inputs */
.stSelectbox, .stMultiselect {
    border-radius: 8px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SYSTEM STATUS ----------------
def system_status():
    if "active_df" in st.session_state:
        st.markdown('<span class="badge-success">Data Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warning">Awaiting Data</span>', unsafe_allow_html=True)


# ---------------- HELPERS ----------------
def pretty_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)}"


# ---------------- ENHANCED SAMPLE DATA (60 DISTRICTS) ----------------
@st.cache_data
def load_sample() -> pd.DataFrame:
    np.random.seed(42)

    states = ["S1", "S2", "S3", "S4", "S5"]
    districts = [f"D{i}" for i in range(1, 61)]

    data = []

    for i, d in enumerate(districts):
        state = states[i % len(states)]

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
    1) session_state['active_df'] if present and non-empty
    2) session_state['cleaned_preview'] if present and non-empty
    3) data/edited_dataset.csv if exists and non-empty
    4) df_edited (global) if present and non-empty
    5) None
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


# ===== WORKFLOW NAVIGATION (FINAL — ALIGNED WITH SYSTEM) =====

tab_data, tab_prep, tab_clean, tab_analysis, tab_policy, tab_ai, tab_debug = st.tabs([
    "📊 Data Ingestion",
    "🧠 Data Preparation",
    "🧹 Clean & Edit",
    "📈 Statistical Analysis",
    "🏛️ Policy Intelligence",
    "🤖 AI Synthesis",
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
    st.markdown('<div class="section-title">Clean & Edit</div>', unsafe_allow_html=True)
    st.write("Apply mappings suggested in Extraction, edit in-grid, coerce numeric-like columns, and save/load cleaned datasets.")

    # pick base df: cleaned_preview > original_df
    base_preview = st.session_state.get("cleaned_preview", None)
    if base_preview is None or (isinstance(base_preview, pd.DataFrame) and base_preview.empty):
        base_preview = st.session_state.get("original_df", pd.DataFrame())

    if base_preview is None or (isinstance(base_preview, pd.DataFrame) and base_preview.empty):
        st.info("No data available. Load a dataset in Tab 1 or run extraction in Tab 2.")
    else:
        if "suggestions" in st.session_state:
            st.markdown("**Manual mapping editor** — confirm or change mappings")
            mapping_cols = []
            for i, s in enumerate(st.session_state["suggestions"]):
                orig = s.get("original", "")
                suggested_role = s.get("suggested_role", orig)
                dtype = s.get("dtype", "unknown")
                conf = s.get("confidence", "")
                st.write(f"**Original column:** `{orig}` — suggested `{suggested_role}` (dtype: {dtype}, conf: {conf})")
                colA, colB = st.columns([2,1])
                with colA:
                    options = ["(keep original)", "state", "district", "EVS", "Language", "Math", "math_pct", "lang_pct", "evs_pct", "infra_index", "pupil_teacher_ratio", "learning_score", "(drop)"]
                    sel = st.selectbox(f"Map `{orig}` to:", options, index=options.index(suggested_role) if suggested_role in options else 0, key=f"map_{i}")
                with colB:
                    new_name = st.text_input(f"Rename `{orig}` to:", value=suggested_role if sel != "(keep original)" else orig, key=f"rename_{i}")
                mapping_cols.append((orig, sel, new_name))
            if st.button("Apply mapping"):
                base_df = base_preview.copy()
                transformed = pd.DataFrame()
                for orig, sel, new_name in mapping_cols:
                    if sel == "(drop)":
                        continue
                    if orig not in base_df.columns:
                        continue
                    target_name = new_name.strip() if new_name else orig
                    numeric_map = ["math_pct", "lang_pct", "infra_index", "pupil_teacher_ratio", "learning_score", "EVS", "Math", "Language", "evs_pct"]
                    if sel in numeric_map:
                        transformed[target_name] = pd.to_numeric(base_df[orig], errors="coerce")
                    else:
                        col_series = base_df[orig]
                        try:
                            coerced = pd.to_numeric(col_series, errors="coerce")
                            non_na = coerced.notna().sum()
                            total_nonblank = col_series.replace({np.nan: None}).dropna().shape[0]
                            if total_nonblank > 0 and (non_na / max(1, total_nonblank)) >= 0.6:
                                transformed[target_name] = coerced
                            else:
                                transformed[target_name] = col_series.astype(str).str.strip()
                        except Exception:
                            transformed[target_name] = col_series.astype(str).str.strip()
                st.session_state["active_df"] = transformed.copy()
                st.session_state["last_mapping"] = {"mapping": mapping_cols, "applied_at": datetime.utcnow().isoformat()}
                st.success("Mapping applied — cleaned & mapped table loaded into the editor below.")
        else:
            st.info("No suggested mappings available. Use 'Extraction' to get suggestions or edit the dataset directly.")

        st.markdown("### Editable grid (final review)")
        if "active_df" in st.session_state and isinstance(st.session_state["active_df"], pd.DataFrame) and not st.session_state["active_df"].empty:
            df_for_grid = st.session_state["active_df"].copy()
        else:
            df_for_grid = base_preview.copy() if isinstance(base_preview, pd.DataFrame) else pd.DataFrame()

        # fallback guards
        if df_for_grid is None:
            df_for_grid = pd.DataFrame()

       # Editable grid (fixed + safe)
if ST_AGGRID_AVAILABLE and not df_for_grid.empty:
    try:
        gb = GridOptionsBuilder.from_dataframe(df_for_grid)
        gb.configure_default_column(editable=True, groupable=True, resizable=True)
        gb.configure_grid_options(enableRangeSelection=True, ensureDomOrder=True)

        grid_options = gb.build()

        grid_response = AgGrid(
            df_for_grid,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            enable_enterprise_modules=False,
            allow_unsafe_jscode=False,
        )

        df_edited = pd.DataFrame(grid_response["data"])

    except Exception as e:
        st.error("AgGrid failed. Using fallback editor.")
        df_edited = st.data_editor(
            df_for_grid,
            num_rows="dynamic",
            use_container_width=True,
            key="fallback_editor"
        )

else:
    df_edited = st.data_editor(
        df_for_grid,
        num_rows="dynamic",
        use_container_width=True,
        key="default_editor"
    )
def safe_data_editor(df: pd.DataFrame, key: str):
    try:
        return st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key=key
        )
    except Exception:
        return st.dataframe(df)
    
# Ensure df_edited always valid
if not isinstance(df_edited, pd.DataFrame):
    df_edited = pd.DataFrame(df_for_grid)


def coerce_numeric_like_columns(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            continue

        coerced = pd.to_numeric(series, errors="coerce")
        non_na = coerced.notna().sum()
        total_nonblank = series.replace({np.nan: None}).dropna().shape[0]

        if total_nonblank > 0 and (non_na / total_nonblank) >= threshold:
            df[col] = coerced

    return df


# Apply coercion safely
if isinstance(df_edited, pd.DataFrame):
    df_edited = coerce_numeric_like_columns(df_edited, threshold=0.6)
    st.session_state["df_edited"] = df_edited.copy()


# Preview section
st.markdown("#### Preview (first 10 rows)")

if isinstance(df_edited, pd.DataFrame) and not df_edited.empty:
    st.dataframe(df_edited.head(10), use_container_width=True)
else:
    st.info("Grid is empty. Fill rows or upload a dataset.")


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
            st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"cluster_plot_{i}"
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
    st.plotly_chart(fig_scree, use_container_width=True)

    # 📊 Cumulative variance
    cum_var = np.cumsum(exp_var)
    fig_cum = px.line(
        x=list(range(1, len(cum_var)+1)),
        y=cum_var,
        markers=True,
        title="Cumulative Variance"
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "Component": range(1, len(exp_var)+1),
        "Explained Variance": exp_var,
        "Cumulative": cum_var
    }))

    # ---------------- CORRELATION ----------------
    st.markdown("### Correlation Heatmap")
    corr = df_complete[cols_sel].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

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
        st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"cluster_plot_{i}"
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
    
# ---------------- Tab 5 - Policy Intelligence ----------------
with tab_policy:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Policy Intelligence Report</div>', unsafe_allow_html=True)

    # ---------- SAFE DATA FETCH ----------
    df = st.session_state.get("active_df")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        df = st.session_state.get("cleaned_preview")

    if df is None or df.empty:
        st.error("❌ No dataset available. Please complete Data Preparation & Clean steps.")
        st.stop()

    st.success(f"Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ---------- VARIABLES ----------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("At least 2 numeric columns required.")
        st.stop()

    selected_vars = st.multiselect(
        "Select Key Indicators",
        numeric_cols,
        default=numeric_cols[:3],
        key="policy_vars"
    )

    if len(selected_vars) < 2:
        st.warning("Select minimum 2 indicators.")
        st.stop()

    # ---------- ANALYSIS ----------
    with st.spinner("Running system diagnostics..."):
        stats = compute_basic_stats(df)
        adv = run_advanced_analyses(df, selected_vars, 3, 3)

    # ---------- EXEC SUMMARY ----------
    st.subheader("Executive Summary")

    means = {k: v["mean"] for k, v in stats["numeric_stats"].items() if v["mean"] is not None}

    if means:
        top = max(means, key=means.get)
        low = min(means, key=means.get)

        st.info(
            f"System imbalance detected. {top} performs strongest ({means[top]:.1f}), "
            f"while {low} lags ({means[low]:.1f}). Indicates structural inequality."
        )

    # ---------- KPI ----------
    st.subheader("Key Indicators")

    cols = st.columns(len(selected_vars))
    for i, var in enumerate(selected_vars):
        val = stats["numeric_stats"][var]["mean"]
        cols[i].metric(var, round(val, 2) if val else "NA")

    # ---------- DISTRIBUTION ----------
    st.subheader("Distribution Analysis")
    fig = px.box(df, y=selected_vars)
    st.plotly_chart(
    fig,
    use_container_width=True,
    key=f"cluster_plot_{i}"
)

    # ---------- PCA ----------
    if adv and adv.get("pca"):
        st.subheader("Structural Drivers (PCA)")

        exp = adv["pca"]["explained_variance_ratio"]

        fig_pca = px.bar(x=[f"PC{i+1}" for i in range(len(exp))], y=exp)
        st.plotly_chart(fig_pca, use_container_width=True)

        st.info(f"Core factor explains {round(exp[0]*100,1)}% system variance.")

    # ---------- CLUSTERS ----------
    if adv and adv.get("kmeans"):

        st.subheader("District Segmentation")

        cluster_sizes = adv["kmeans"]["cluster_sizes"]
        medians = adv["kmeans"]["cluster_medians"]

        result_df = pd.DataFrame(adv.get("cluster_assignments", {}))

        id_col = next((c for c in df.columns if "district" in c.lower()), None)

        fig_cluster = px.bar(x=list(cluster_sizes.keys()), y=list(cluster_sizes.values()))
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.dataframe(pd.DataFrame(medians).round(2))

        # ---------- INTELLIGENCE ----------
        st.subheader("Cluster Intelligence")

        for cl, size in cluster_sizes.items():

            cluster_rows = result_df[result_df["_cluster"] == cl] if "_cluster" in result_df else pd.DataFrame()

            districts = (
                cluster_rows[id_col].astype(str).tolist()
                if id_col and id_col in cluster_rows.columns
                else []
            )

            evs = medians.get("EVS", {}).get(cl)
            ptr = medians.get("ptr", {}).get(cl)
            infra = medians.get("infra", {}).get(cl)

            st.markdown(f"### Cluster {cl} ({size} districts)")

            if districts:
                st.write(f"Districts: {', '.join(districts)}")

            st.write(f"EVS: {evs}, PTR: {ptr}, Infra: {infra}")

            # ---------- SAFE LOGIC ----------
            if evs is not None and ptr is not None and evs < 50 and ptr > 35:
                st.error("Critical Learning Deficit → Immediate teacher redistribution required")

            elif infra is not None and infra < 0.5:
                st.warning("Infrastructure deficit → Target capital investment")

            elif evs is not None and evs > 80:
                st.success("High-performing cluster → Use as benchmark")

            else:
                st.info("Moderate cluster → Balanced interventions needed")

    # ---------- POLICY ----------
    st.subheader("Strategic Policy Actions")

    st.write("""
    - Optimize teacher allocation across districts  
    - Target infrastructure investments selectively  
    - Implement cluster-based governance  
    - Enable real-time performance monitoring  
    - Strengthen institutional capacity  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

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

    # ================= LOCAL ENGINE =================
    if synth_choice == "Local Intelligence Engine":
        st.success("Using deterministic policy intelligence engine")

        insights, priorities = generate_policy_engine(df_for_report, adv)

        generated_text = "National Education Policy Intelligence Report\n\n"

        generated_text += "Executive Summary\n"
        generated_text += "The system exhibits structural disparities driven by infrastructure gaps, teacher load imbalance, and uneven learning outcomes across districts.\n\n"

        generated_text += "District Risk Insights\n"
        if insights:
            generated_text += "\n".join(insights[:15]) + "\n\n"
        else:
            generated_text += "No critical district risks detected.\n\n"

        generated_text += "Priority Districts\n"
        for name, score in priorities[:5]:
            generated_text += f"{name} (severity score: {score})\n"

        generated_text += "\nPolicy Actions\n"
        generated_text += "Target low-infrastructure districts, reduce teacher overload, and implement cluster-specific strategies.\n"

    # ================= GEMINI =================
    elif synth_choice == "Gemini (Cloud)":
        consent = st.checkbox("Allow external API call", key="tab6_consent")

        if consent:
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                st.error("Missing GEMINI_API_KEY")
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)

                    model = "models/gemini-2.5-flash"

                    if st.button("Generate AI Report", key="tab6_run"):

                        prompt = f"""
You are a senior government policy advisor.

Generate a professional policy report with:
- Executive Summary
- System Diagnosis
- District Insights (with names)
- Cluster Insights
- Deep Recommendations
- Implementation Roadmap

DATA:
{json.dumps(stats, indent=2)}

Write in clean professional format. No symbols like # or *.
"""

                        try:
                            response = genai.GenerativeModel(model).generate_content(
                                prompt,
                                generation_config={"temperature": 0.2}
                            )

                            raw = getattr(response, "text", "")
                            generated_text = clean_llm_output(raw)

                            st.success("Gemini report generated")

                        except Exception:
                            st.warning("Gemini unavailable. Switching to local engine.")

                            insights, priorities = generate_policy_engine(df_for_report, adv)

                            generated_text = "\n".join(insights[:10])

                except Exception:
                    st.error("Gemini not available")

    # ================= OLLAMA =================
    else:
        st.warning("Ollama not supported on Streamlit Cloud")

    # ================= OUTPUT =================
    if generated_text:
        st.markdown("### Policy Report")
        st.text_area("Generated Report", generated_text, height=500)

        pdf_bytes = generate_pdf(generated_text)

        if pdf_bytes:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="policy_report.pdf",
                mime="application/pdf"
            )
        else:
            st.download_button(
                "Download TXT",
                data=generated_text,
                file_name="policy_report.txt",
                mime="text/plain"
            )

    st.markdown('</div>', unsafe_allow_html=True)
# ---------------- Tab 7 - Debug ----------------
with tab_debug:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Debug & Provenance</div>', unsafe_allow_html=True)

    if "last_mapping" in st.session_state:
        st.markdown("**Last mapping applied (provenance)**")
        st.json(st.session_state["last_mapping"])

    st.markdown("---")
    st.markdown("**Local debug files (data/)**")

    dbg_files = sorted(
        glob("data/*genai*") +
        glob("data/*synthesis*") +
        glob("data/*ollama*") +
        glob("data/*genai_*"),
        key=os.path.getmtime,
        reverse=True
    )

    if dbg_files:
        st.write(f"Found {len(dbg_files)} debug files (most recent first).")
        for fpath in dbg_files[:10]:
            with st.expander(os.path.basename(fpath)):
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        obj = json.load(fh)
                    st.json(obj)
                except Exception:
                    st.write("Could not render JSON — open file in editor.")
    else:
        st.write("No debug files found yet.")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Footer ----------------
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col1:
    if st.button("Run Gemini ping test"):
        txt, meta, err = gemini_ping_test()
        if err:
            st.error(err)
        else:
            st.success(f"Gemini ping OK: {txt}")

with footer_col2:
    st.markdown(
        "<div class='muted'>If external synthesis fails, use the deterministic report.</div>",
        unsafe_allow_html=True
    )

with footer_col3:
    if OLLAMA_AVAILABLE:
        st.success("Ollama available")
    else:
        st.info("Ollama not detected")
