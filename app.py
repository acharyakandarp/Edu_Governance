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

# ---------------- Styling (small, tasteful) ----------------
st.set_page_config(page_title="Edu Governance — PCA & Clustering", layout="wide")
st.markdown(
    """
    <style>
    /* fonts & base */
    .big-title { font-size:28px; font-weight:700; margin-bottom:4px; }
    .subtitle { color: #6b7280; margin-top:0; margin-bottom:8px; }
    .card { background: #ffffff; padding: 16px; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
    .muted { color: #6b7280; font-size:13px; }
    .compact { padding:6px 8px; border-radius:6px; border:1px solid #eee; background:#fafafa; }
    .section-title { font-size:18px; font-weight:600; margin-bottom:8px; }
    .kbd { background:#f3f4f6; padding:3px 6px; border-radius:4px; font-family:monospace; font-size:12px; }
    .small { font-size:13px; color:#374151; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
def pretty_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)}"

@st.cache_data
def load_sample() -> pd.DataFrame:
    # One row per district (expected): aggregated scores per district
    return pd.DataFrame({
        'state': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S5', 'S5'],
        'district': ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10'],
        'EVS': [82, 78, 45, 50, 90, 88, 40, 42, 76, 79],
        'Language': [80, 77, 60, 62, 91, 85, 58, 55, 75, 73],
        'Math': [55, 52, 78, 80, 50, 48, 81, 79, 57, 59],
        'infra': [0.7, 0.6, 0.4, 0.45, 0.9, 0.85, 0.3, 0.25, 0.65, 0.68],
        'ptr': [28, 30, 40, 38, 22, 24, 45, 46, 29, 27]
    })

def sanitize_sample(df: Optional[pd.DataFrame], max_rows: int = 20) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    pii_keywords = ['name', 'id', 'email', 'phone', 'mobile', 'address']
    safe_cols = [c for c in df.columns if not any(k in c.lower() for k in pii_keywords)]
    return df[safe_cols].head(max_rows).copy()

def compact_schema_and_examples(df: Optional[pd.DataFrame], max_examples: int = 1) -> str:
    if df is None or df.shape[0] == 0:
        return ""
    head = df.head(max_examples)
    buf = io.StringIO()
    head.to_csv(buf, index=False, lineterminator="\n")
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

# ---------------- UI Layout (tabs + guided flow) ----------------
# Header
header_col1, header_col2 = st.columns([4,1])
with header_col1:
    st.markdown('<div class="big-title">Edu Governance System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">District-centric PCA, clustering, diagnostics and policy-ready reporting — with optional LLM synthesis (Gemini / Ollama).</div>', unsafe_allow_html=True)
with header_col2:
    st.markdown(f"<div class='muted'>Run: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>", unsafe_allow_html=True)

tabs = st.tabs(["1 · Data", "2 · Extraction", "3 · Clean & Edit", "4 · Analysis", "5 · Report", "6 · Synthesis", "7 · Debug"])
# -------- Model Selection --------
st.sidebar.title("AI Settings")

model_choice = st.sidebar.selectbox(
    "Choose AI Model",
    ["Gemini (Cloud)", "LLaMA (Local)"]
)
# ---------------- Tab 1 - Data ----------------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data input</div>', unsafe_allow_html=True)
    st.write("Load a sample dataset, upload your CSV/XLSX file, or create a manual grid. The app expects one row per district.")
    input_col1, input_col2 = st.columns([2,1])
    with input_col1:
        input_mode = st.selectbox("Input mode", ["Use sample", "Upload file", "Manual grid"])
    with input_col2:
        st.markdown('<div class="muted">Tip: Use the sample to try features quickly.</div>', unsafe_allow_html=True)

    uploaded = None
    original_df: Optional[pd.DataFrame] = None

    if input_mode == "Use sample":
        original_df = load_sample()
        st.success("Loaded sample dataset (10 districts).")
    elif input_mode == "Upload file":
        uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])
        if uploaded:
            try:
                if str(uploaded.name).lower().endswith(".csv"):
                    original_df = pd.read_csv(uploaded)
                else:
                    original_df = pd.read_excel(uploaded)
                st.success(f"Loaded `{uploaded.name}` ({original_df.shape[0]} rows × {original_df.shape[1]} cols).")
            except Exception as e:
                st.error(f"Error reading file: {pretty_exception(e)}")
                original_df = None
        else:
            st.info("Please upload a file to proceed.")
    else:
        template_cols = st.text_input("Comma-separated column names", value="state,district,EVS,Language,Math")
        cols = [c.strip() for c in template_cols.split(",") if c.strip()]
        rows = st.number_input("Initial empty rows", min_value=1, max_value=200, value=5)
        original_df = pd.DataFrame(columns=cols)
        if st.button("Fill empty rows"):
            original_df = pd.concat([original_df, pd.DataFrame([dict(zip(cols, [""]*len(cols))) for _ in range(rows)])], ignore_index=True)
            st.success("Manual grid initialized.")

    # store the freshly loaded original_df in session for downstream tabs if not None
    if isinstance(original_df, pd.DataFrame) and not original_df.empty:
        st.session_state["original_df"] = original_df.copy()
    else:
        # ensure key exists but possibly empty
        st.session_state["original_df"] = original_df if original_df is not None else pd.DataFrame()

    st.markdown("#### Preview")
    if isinstance(st.session_state.get("original_df"), pd.DataFrame) and not st.session_state["original_df"].empty:
        st.dataframe(st.session_state["original_df"].head(10), use_container_width=True)
    else:
        st.info("No dataset loaded yet. Use sample, upload, or manual grid.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab 2 - Extraction ----------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Extraction (schema suggestion)</div>', unsafe_allow_html=True)

    st.write(
        "Automatic schema suggestions help map raw columns to expected roles "
        "(EVS, Language, Math, infra, ptr, district). "
        "Use mock extractor locally or Gemini (optional)."
    )

    c1, c2 = st.columns([1, 1])

    # ---------------- MOCK EXTRACTION ----------------
    with c1:
        if st.button("Auto-extract (mock)"):
            preview = st.session_state.get("original_df", pd.DataFrame()).head(50).copy()

            if preview.empty:
                st.error("No data available. Load dataset in Tab 1.")
            else:
                suggestions, cleaned_preview = mock_gemini_extract_preview(preview)

                st.session_state["suggestions"] = suggestions
                st.session_state["cleaned_preview"] = cleaned_preview

                st.success("Mock extraction complete.")

    # ---------------- GEMINI EXTRACTION ----------------
    with c2:
        st.write("Real Gemini extraction (optional)")

        consent = st.checkbox(
            "I consent to send sanitized sample (no PII) to Gemini",
            value=False
        )

        gem_model = st.selectbox(
            "Gemini model",
            ["models/gemini-2.5-flash", "models/gemini-2.5-pro"]
        )

        if st.button("Auto-extract with Gemini"):

            if not consent:
                st.error("Consent required.")
            else:
                preview = st.session_state.get("original_df", pd.DataFrame()).head(50).copy()

                if preview.empty:
                    st.error("No data available.")
                else:
                    sanitized = sanitize_sample(preview, max_rows=20)

                    try:
                        # SAFE GEMINI CALL (no undefined function)
                        try:
                            import os
                            import google.generativeai as genai

                            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

                            if not api_key:
                                raise ValueError("Missing API key")

                            genai.configure(api_key=api_key)

                            model = genai.GenerativeModel(gem_model)

                            prompt = f"""
                            Extract column roles from dataset.

                            CSV:
                            {sanitized.to_csv(index=False)}

                            Return JSON:
                            {{
                              "suggestions": [
                                {{
                                  "original": "",
                                  "suggested_role": "",
                                  "dtype": "",
                                  "confidence": 0.0
                                }}
                              ]
                            }}
                            """

                            resp = model.generate_content(prompt)

                            import json, re
                            text = getattr(resp, "text", "")

                            match = re.search(r"\{.*\}", text, re.S)
                            if not match:
                                raise ValueError("Invalid Gemini response")

                            parsed = json.loads(match.group(0))

                            suggestions = parsed.get("suggestions", [])

                            cleaned_preview = _build_local_cleaned_preview_from_suggestions(
                                sanitized,
                                suggestions
                            )

                        except Exception:
                            # FALLBACK ALWAYS WORKS
                            suggestions, cleaned_preview = mock_gemini_extract_preview(sanitized)

                        if not isinstance(cleaned_preview, pd.DataFrame):
                            cleaned_preview = pd.DataFrame(cleaned_preview)

                        st.session_state["suggestions"] = suggestions
                        st.session_state["cleaned_preview"] = cleaned_preview

                        st.success("Extraction complete.")

                    except Exception as e:
                        st.error("Extraction failed: " + pretty_exception(e))

    # ---------------- DISPLAY ----------------
    st.markdown("---")

    if "suggestions" in st.session_state:
        st.markdown("**Detected suggestions**")

        try:
            st.dataframe(
                pd.DataFrame(st.session_state["suggestions"]),
                use_container_width=True
            )
        except Exception:
            st.write(st.session_state["suggestions"])

        st.markdown("Go to **Clean & Edit** tab to apply mappings.")

    else:
        st.info("Run extraction to see results.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab 3 - Clean & Edit ----------------
with tabs[2]:
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

# ---------------- Tab 4 - Analysis ----------------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">PCA & Clustering (district-centric)</div>', unsafe_allow_html=True)
    st.write("Configure preprocessing, run PCA, evaluate clustering diagnostics and run final clustering. The UI guides you step-by-step.")

    df_for_analysis = None
    if "active_df" in st.session_state and isinstance(st.session_state["active_df"], pd.DataFrame) and not st.session_state["active_df"].empty:
        df_for_analysis = st.session_state["active_df"].copy()
    elif "df_edited" in st.session_state and isinstance(st.session_state["df_edited"], pd.DataFrame) and not st.session_state["df_edited"].empty:
        df_for_analysis = st.session_state["df_edited"].copy()
    else:
        df_for_analysis = None

    if df_for_analysis is None or (isinstance(df_for_analysis, pd.DataFrame) and df_for_analysis.empty):
        st.info("No data available. Load or edit a dataset in 'Clean & Edit' tab.")
    else:
        numeric_cols_all = df_for_analysis.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols_all:
            st.warning("No numeric columns found for PCA/KMeans. Make sure important numeric columns are coerced.")
        else:
            st.write(f"Numeric columns available: {', '.join(numeric_cols_all)}")
            default_sel = [c for c in ['EVS', 'Language', 'Math'] if c in numeric_cols_all][:3]
            cols_sel = st.multiselect("Select numeric variables for district feature vectors (complete-case will be used)", numeric_cols_all, default=default_sel)
            if cols_sel:
                # find label column
                def find_entity_column_for_rows(df: pd.DataFrame) -> Optional[str]:
                    if df is None or df.shape[1] == 0:
                        return None
                    cols = list(df.columns)
                    lowered = [c.lower() for c in cols]
                    for key in ["district", "district_name", "dist", "dist_name", "school_district", "area"]:
                        if key in lowered:
                            return cols[lowered.index(key)]
                    for key in ["state", "region"]:
                        if key in lowered:
                            return cols[lowered.index(key)]
                    for c in cols:
                        if not pd.api.types.is_numeric_dtype(df[c]):
                            return c
                    return cols[0] if cols else None

                id_col = find_entity_column_for_rows(df_for_analysis)
                if id_col:
                    st.write(f"Using `{id_col}` as district label for plotting.")
                else:
                    st.warning("Could not find a district ID column.")

                df_complete = df_for_analysis.dropna(subset=cols_sel).copy()
                n_total = len(df_for_analysis)
                n_complete = len(df_complete)
                if n_complete == 0:
                    st.error("No rows have all selected values — pick other variables or clean data.")
                else:
                    st.write(f"Rows with complete values: {n_complete}/{n_total}")

                    with st.expander("Preprocessing & scaling options", expanded=False):
                        scale_method = st.selectbox("Scaling / transformation", options=["z-score (StandardScaler)", "robust (median/IQR)", "yeo-johnson (power-transform)", "none"], index=0)
                        log_transform = st.checkbox("Apply log(1+x) before scaling for skewed variables (optional)", value=False)
                        run_outlier = st.checkbox("Detect Mahalanobis multivariate outliers (report only)", value=False)

                    with st.expander("PCA & components", expanded=True):
                        max_components_possible = min(6, len(cols_sel))
                        n_comp = st.slider("PCA components (for diagnostics & plotting)", min_value=1, max_value=max_components_possible, value=min(2, max_components_possible))

                    Xvals = df_complete[cols_sel].astype(float).values
                    if log_transform:
                        Xvals = np.log1p(Xvals)
                    if scale_method.startswith("z-score"):
                        scaler = StandardScaler(); Xs = scaler.fit_transform(Xvals); scaler_name = "zscore"
                    elif scale_method.startswith("robust"):
                        scaler = RobustScaler(); Xs = scaler.fit_transform(Xvals); scaler_name = "robust"
                    elif scale_method.startswith("yeo"):
                        try:
                            scaler = PowerTransformer(method="yeo-johnson"); Xs = scaler.fit_transform(Xvals); scaler_name = "yeo-johnson"
                        except Exception:
                            scaler = StandardScaler(); Xs = scaler.fit_transform(Xvals); scaler_name = "zscore"
                    else:
                        scaler = None; Xs = Xvals.copy(); scaler_name = "none"

                    if run_outlier:
                        try:
                            mask_out, D2, pvals = mahalanobis_outlier_mask(Xs, threshold_p=0.001)
                            n_out = int(mask_out.sum())
                            st.write(f"Mahalanobis outliers detected (p < 0.001): {n_out} rows.")
                            if n_out > 0:
                                outlier_idx = np.where(mask_out)[0]
                                outlier_districts = df_complete.reset_index(drop=True).iloc[outlier_idx][id_col].astype(str).tolist() if id_col and id_col in df_complete.columns else df_complete.reset_index(drop=True).iloc[outlier_idx].index.tolist()
                                st.write("Outlier districts:", ", ".join(map(str, outlier_districts)))
                        except Exception as e:
                            st.write("Outlier detection failed or SciPy unavailable:", pretty_exception(e))

                    if SKLEARN_AVAILABLE:
                        pca = PCA(n_components=n_comp, random_state=42)
                        pcs = pca.fit_transform(Xs)
                        pc_cols = [f"PC{i+1}" for i in range(n_comp)]
                        exp_var = pd.Series(pca.explained_variance_ratio_, index=pc_cols).round(4)
                        st.write("Explained variance ratio (per component):")
                        st.dataframe(exp_var)
                        loadings = pd.DataFrame(pca.components_.T, index=cols_sel, columns=pc_cols).round(4)
                        st.write("PCA loadings (variables × components):")
                        st.dataframe(loadings)
                        if st.button("Run parallel analysis (100 permutations)"):
                            try:
                                pa = parallel_analysis(Xs, n_iter=100, random_state=42)
                                obs = np.round(pa["observed"], 4).tolist()
                                mean_rand = np.round(pa["mean_random"], 4).tolist()
                                st.write("Observed eigenvalues:", obs)
                                st.write("Mean random eigenvalues (parallel analysis):", mean_rand)
                                fig_pa = px.line(x=list(range(1, len(obs)+1)), y=obs, markers=True, labels={"x":"Component #","y":"Eigenvalue"}, title="Parallel analysis — observed eigenvalues vs random mean")
                                fig_pa.add_scatter(x=list(range(1, len(mean_rand)+1)), y=mean_rand, mode="lines+markers", name="random mean")
                                st.plotly_chart(fig_pa, use_container_width=True)
                            except Exception as e:
                                st.write("Parallel analysis failed:", pretty_exception(e))

                        if STATSMODELS_AVAILABLE and st.checkbox("Compute VIF (multicollinearity check)", value=False):
                            try:
                                vif_df = compute_vif(df_complete, cols_sel)
                                st.write("VIF (higher than 5 indicates multicollinearity concerns):")
                                st.dataframe(vif_df)
                            except Exception as e:
                                st.write("VIF computation failed:", pretty_exception(e))

                        st.markdown("**Clustering diagnostics & selection**")
                        run_gap = st.checkbox("Compute gap statistic approximation (slower)", value=False)
                        k_max = st.slider("K max (for diagnostics)", min_value=2, max_value=min(10, max(2, n_complete)), value=min(6, n_complete))
                        n_pcs_for_clust = min(3, n_comp)
                        X_for_k_eval = pcs[:, :n_pcs_for_clust] if pcs.shape[1] >= n_pcs_for_clust else pcs
                        eval_df = evaluate_k_range(X_for_k_eval, k_min=2, k_max=k_max, random_state=42)
                        st.write("Clustering indices across K:")
                        st.dataframe(eval_df)
                        if run_gap:
                            try:
                                gap_df = gap_statistic(X_for_k_eval, k_max=k_max, B=20, random_state=42)
                                st.write("Gap statistic (approx):")
                                st.dataframe(gap_df)
                            except Exception as e:
                                st.write("Gap statistic failed:", pretty_exception(e))

                        choose_k_method = st.radio("Choose how to set final K:", options=["Manual K", "Silhouette-max recommended", "Use gap statistic (if computed)"], index=1)
                        rec_k = None
                        if (not eval_df.empty) and ("silhouette" in eval_df.columns):
                            try:
                                rec_k = int(eval_df["silhouette"].idxmax())
                            except Exception:
                                rec_k = None
                        if choose_k_method == "Manual K":
                            final_k = st.slider("Final K for clustering", min_value=2, max_value=min(10, max(2, n_complete)), value=3)
                        elif choose_k_method == "Silhouette-max recommended":
                            final_k = rec_k if rec_k is not None else 3
                            st.write(f"Recommended K by silhouette: {final_k}")
                        else:
                            if run_gap and 'gap_df' in locals() and gap_df is not None:
                                try:
                                    final_k = int(gap_df["gap"].idxmax())
                                    st.write(f"Recommended K by gap: {final_k}")
                                except Exception:
                                    final_k = rec_k if rec_k is not None else 3
                            else:
                                final_k = rec_k if rec_k is not None else 3

                        clustering_methods = ["KMeans (on PCs)", "Agglomerative (on standardized raw variables)"]
                        chosen_method = st.selectbox("Clustering algorithm for final run", clustering_methods, index=0)

                        if st.button("Run final clustering with diagnostics"):
                            if chosen_method.startswith("KMeans"):
                                X_cluster = X_for_k_eval
                                km = KMeans(n_clusters=final_k, random_state=42, n_init=20)
                                labels = km.fit_predict(X_cluster)
                                result_df = df_complete.reset_index(drop=True).copy()
                                result_df["_cluster"] = labels
                                pcs_df_plot = pd.DataFrame(pcs, columns=pc_cols).reset_index(drop=True)
                                if id_col and id_col in result_df.columns:
                                    pcs_df_plot[id_col] = result_df[id_col].astype(str).reset_index(drop=True)
                                else:
                                    pcs_df_plot[id_col or "index_label"] = pcs_df_plot.index.astype(str)
                                for v in cols_sel:
                                    pcs_df_plot[v] = result_df[v].reset_index(drop=True)
                                pcs_df_plot["cluster"] = result_df["_cluster"].astype(str)
                                st.session_state["last_kmeans_result"] = result_df
                                hover_info = cols_sel + ([id_col] if id_col else [])
                                if "PC1" in pcs_df_plot.columns and "PC2" in pcs_df_plot.columns:
                                    fig = px.scatter(pcs_df_plot, x="PC1", y="PC2", color="cluster", hover_data=hover_info, title=f"KMeans (k={final_k}) on first two PCs")
                                    st.plotly_chart(fig, use_container_width=True)
                                try:
                                    cluster_sizes = result_df["_cluster"].value_counts().sort_index().to_dict()
                                    st.write("Cluster sizes:", cluster_sizes)
                                    centers_median = result_df.groupby("_cluster")[cols_sel].median().round(3)
                                    st.write("Cluster medians (original variables):")
                                    st.dataframe(centers_median)
                                    st.write("Districts in each cluster:")
                                    for cl in sorted(result_df["_cluster"].unique()):
                                        members = result_df[result_df["_cluster"]==cl][id_col].astype(str).tolist() if id_col and id_col in result_df.columns else result_df[result_df["_cluster"]==cl].index.astype(str).tolist()
                                        st.write(f"- Cluster {cl}: {', '.join(map(str, members))}")
                                except Exception as e:
                                    st.write("Could not compute cluster summary:", pretty_exception(e))
                                if st.checkbox("Run bootstrap stability (ARI, 50 bootstraps)", value=False):
                                    try:
                                        stab = clustering_stability_bootstrap(X_cluster, labels, final_k, n_boot=50, method='kmeans', random_state=42)
                                        st.write(f"Bootstrap ARI mean: {stab['ari_mean']:.3f} (std {stab['ari_std']:.3f})")
                                    except Exception as e:
                                        st.write("Stability computation failed:", pretty_exception(e))
                                if st.checkbox("Run cluster profiling & inferential tests (ANOVA/Kruskal)", value=True):
                                    try:
                                        profile = cluster_profiling_tests(result_df, cols_sel, cluster_labels_col="_cluster")
                                        st.write("Cluster profiling (summary & tests):")
                                        st.json(profile)
                                    except Exception as e:
                                        st.write("Profiling failed:", pretty_exception(e))
                                try:
                                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                                    st.download_button("Download cluster assignments (CSV)", data=csv_bytes, file_name="kmeans_clusters.csv", mime="text/csv")
                                except Exception:
                                    pass
                            else:
                                if scale_method != "none":
                                    scaler2 = StandardScaler()
                                    X_for_hc = scaler2.fit_transform(Xvals)
                                else:
                                    X_for_hc = Xvals.copy()
                                linkage = st.selectbox("Linkage for hierarchical", ["ward","complete","average","single"], index=0)
                                hc = AgglomerativeClustering(n_clusters=final_k, linkage=linkage)
                                labels = hc.fit_predict(X_for_hc)
                                result_df = df_complete.reset_index(drop=True).copy()
                                result_df["_cluster"] = labels
                                st.session_state["last_hc_result"] = result_df
                                st.write("Cluster medians (original variables):")
                                centers_median = result_df.groupby("_cluster")[cols_sel].median()
                                st.dataframe(centers_median.round(3))
                                st.write("Districts in each cluster:")
                                for cl in sorted(result_df["_cluster"].unique()):
                                    members = result_df[result_df["_cluster"]==cl][id_col].astype(str).tolist() if id_col and id_col in result_df.columns else result_df[result_df["_cluster"]==cl].index.astype(str).tolist()
                                    st.write(f"- Cluster {cl}: {', '.join(map(str, members))}")
                                try:
                                    csv_bytes2 = result_df.to_csv(index=False).encode("utf-8")
                                    st.download_button("Download hierarchical cluster assignments (CSV)", data=csv_bytes2, file_name="hc_clusters.csv", mime="text/csv")
                                except Exception:
                                    pass
                    else:
                        st.info("Install scikit-learn to run PCA & clustering.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab 5 - Report ----------------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Deterministic Research-style Report</div>', unsafe_allow_html=True)

    df_for_report = _choose_reporting_df()

    if df_for_report is None or (isinstance(df_for_report, pd.DataFrame) and df_for_report.empty):
        st.info("No data available to summarize. Go to 'Clean & Edit' to prepare data.")
    else:
        st.write(f"Selected dataset: {df_for_report.shape[0]} rows × {df_for_report.shape[1]} columns")
        include_adv = st.checkbox("Include advanced analyses (PCA + KMeans)", value=False)
        numeric_vars = df_for_report.select_dtypes(include=[np.number]).columns.tolist()
        adv_selected_vars: List[str] = []
        if include_adv:
            if not numeric_vars:
                st.warning("No numeric columns available for advanced analyses.")
                include_adv = False
            else:
                adv_selected_vars = st.multiselect("Select numeric variables for PCA & clustering (complete-case will be used)", numeric_vars, default=[c for c in ["EVS","Language","Math"] if c in numeric_vars])
                if not adv_selected_vars:
                    st.info("Select at least one numeric variable to enable advanced analyses.")
                    include_adv = False

        with st.spinner("Computing local stats..."):
            stats = compute_basic_stats(df_for_report)
            adv_result = None
            if include_adv and adv_selected_vars:
                adv_result = run_advanced_analyses(df_for_report, adv_selected_vars, n_pca_components=min(3, len(adv_selected_vars)), k_clusters=3)

        verbosity = st.selectbox("Report verbosity", options=["concise", "detailed"], index=0)
        local_text = _compose_research_style_report(stats, adv_result, id_col=None, selected_vars=adv_selected_vars if adv_selected_vars else numeric_vars, verbosity=verbosity)
        st.subheader("Local deterministic report")
        st.text_area("Automated research-style report", value=local_text, height=520)
        try:
            export_payload = {"stats": stats, "advanced": adv_result}
            export_bytes = json.dumps(export_payload, indent=2, default=str).encode("utf-8")
            st.download_button("Download computed report data (JSON)", data=export_bytes, file_name="computed_report_data.json", mime="application/json")
        except Exception:
            pass
        try:
            txt_bytes = local_text.encode("utf-8")
            st.download_button("Download research-style report (TXT)", data=txt_bytes, file_name="report.txt", mime="text/plain")
        except Exception:
            pass
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab 6 - Synthesis ----------------
with tabs[5]:
    # Model selection control
if model_choice == "Gemini (Cloud)":
    st.success("Using Gemini")

elif model_choice == "LLaMA (Local)":
    st.warning("⚠️ LLaMA works only locally, not in deployed app")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">LLM Synthesis (Gemini or Ollama)</div>', unsafe_allow_html=True)
    st.write("Choose synthesis engine to produce a polished narrative from computed facts. Gemini is cloud-based and requires an API key; Ollama runs locally if installed.")

    df_for_report = _choose_reporting_df()
    if df_for_report is None or (isinstance(df_for_report, pd.DataFrame) and df_for_report.empty):
        st.info("No data available to synthesize from. Prepare a dataset first.")
    else:
        st.write(f"Dataset ready for synthesis: {df_for_report.shape[0]} rows × {df_for_report.shape[1]} cols")
        st.markdown("**Synthesis engine**")
        synth_choice = st.selectbox("Select engine", ["Local deterministic (no external LLM)", "Gemini (cloud)", "Ollama (local)"], index=0)
        compact_payload = json.dumps({"stats": compute_basic_stats(df_for_report), "advanced": None}, default=str, indent=2)[:15000]
        compact_csv = compact_schema_and_examples(sanitize_sample(df_for_report, max_rows=5), max_examples=5)
        if synth_choice == "Local deterministic (no external LLM)":
            st.info("Local deterministic report is available in the 'Report' tab.")
        elif synth_choice == "Gemini (cloud)":
            consent = st.checkbox("I consent to send sanitized facts to Gemini (no PII)", value=False)
            if consent:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    st.error("GEMINI_API_KEY not found in environment.")
                else:
                    try:
                        import google.generativeai as genai
                    except Exception:
                        st.error("google.generativeai client not installed.")
                        genai = None
                    if genai is None:
                        st.warning("Gemini client not available.")
                    else:
                        model_choice = st.selectbox("Gemini model", options=["models/gemini-2.5-pro", "models/gemini-2.5-flash"], index=0)
                        max_tok = st.slider("Max output tokens", min_value=200, max_value=2000, value=800, step=50)
                        if st.button("Run Gemini synthesis"):
                            gem_prompt = (
                                "You are an expert educational policy analyst. Use only the facts provided in the JSON and CSV (do NOT impute missing values).\n\n"
                                "Task: Produce a research-style narrative report containing:\n"
                                "1) Executive summary (2-3 short paragraphs with numeric facts).\n"
                                "2) Key quantitative findings (bullet list): cite means, cluster sizes, explained variance.\n"
                                "3) Methods (brief): indicate PCA/explained variance and clustering method and diagnostic scores.\n"
                                "4) Cluster-by-cluster interpretation (for each cluster, list member districts and their profile).\n"
                                "5) Five prioritized recommendations tied to facts.\n\n"
                                "FACTS_JSON:\n" + compact_payload + "\n\n"
                                "CSV_SAMPLE:\n" + compact_csv + "\n\n"
                                "Return plain markdown text."
                            )
                            try:
                                try:
                                    genai.configure(api_key=api_key, transport="rest")
                                except Exception:
                                    genai.configure(api_key=api_key)
                                gen_cfg = genai.GenerationConfig(temperature=0.15, candidate_count=1, max_output_tokens=int(max_tok))
                                with st.spinner("Calling Gemini..."):
                                    gm = genai.GenerativeModel(model_choice)
                                    resp = gm.generate_content(gem_prompt, generation_config=gen_cfg)
                                text_out, meta = extract_text_from_genai_response(resp)
                                if text_out and text_out.strip():
                                    st.success("Gemini synthesis complete.")
                                    st.text_area("Gemini narrative", value=text_out.strip(), height=520)
                                    # optional debug save
                                    try:
                                        os.makedirs("data", exist_ok=True)
                                        dbg = {"stage": "gemini_synthesis_success", "model": model_choice, "meta": meta}
                                        with open(os.path.join("data", f"genai_synthesis_success_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"), "w", encoding="utf-8") as f:
                                            json.dump(dbg, f, indent=2, default=str)
                                    except Exception:
                                        pass
                                else:
                                    st.error("Gemini returned no usable text. Check debug files (data/).")
                            except Exception as e:
                                st.error("Gemini call failed: " + pretty_exception(e))
        else:  # Ollama
            st.write("Ollama runs locally; be sure Ollama is installed and a model is available.")
            oll_model = st.text_input("Ollama model", value="llama3.2")
            max_tokens_oll = st.slider("Max tokens (Ollama)", min_value=200, max_value=3000, value=1200, step=50)
            if st.button("Run Ollama synthesis"):
                with st.spinner("Calling local Ollama..."):
                    text_out, meta, err = call_ollama_for_synthesis(compact_payload, compact_csv, model=oll_model, max_tokens=max_tokens_oll)
                    if err:
                        st.error("Ollama synthesis failed: " + err)
                        # save debug
                        try:
                            os.makedirs("data", exist_ok=True)
                            dbg = {"stage": "ollama_failure", "error": err, "model": oll_model}
                            with open(os.path.join("data", f"ollama_synthesis_fail_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"), "w", encoding="utf-8") as f:
                                json.dump(dbg, f, indent=2, default=str)
                        except Exception:
                            pass
                    else:
                        st.success("Ollama synthesis complete.")
                        st.text_area("Ollama narrative", value=text_out.strip(), height=520)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab 7 - Debug ----------------
with tabs[6]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Debug & Provenance</div>', unsafe_allow_html=True)
    if "last_mapping" in st.session_state:
        st.markdown("**Last mapping applied (provenance)**")
        st.json(st.session_state["last_mapping"])
    st.markdown("---")
    st.markdown("**Local debug files (data/)**")
    dbg_files = sorted(glob("data/*genai*") + glob("data/*synthesis*") + glob("data/*ollama*") + glob("data/*genai_*"), key=os.path.getmtime, reverse=True)
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
        st.write("No debug files found yet. They will be created automatically if external calls fail.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer / quick checks ----------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1,2,1])
with footer_col1:
    if st.button("Run Gemini ping test"):
        txt, meta, err = gemini_ping_test()
        if err:
            st.error(err)
        else:
            st.success(f"Gemini ping OK: {txt}")
with footer_col2:
    st.markdown("<div class='muted'>If external synthesis fails, the deterministic local report in Tab 'Report' always provides a complete narrative you can download and present.</div>", unsafe_allow_html=True)
with footer_col3:
    if OLLAMA_AVAILABLE:
        st.success("Ollama client available")
    else:
        st.info("Ollama client not detected (local synthesis unavailable)")

