#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          ULTIMATE DATA PROFILER  —  v2.0                                   ║
# ║          The only profiling script a data scientist will ever need          ║
# ║                                                                              ║
# ║  Features:                                                                   ║
# ║  • Multi-format loading (CSV, Excel, JSON, Parquet, TSV, SQL-dump)          ║
# ║  • Full statistical profiling (numerical + categorical + datetime + text)   ║
# ║  • Missing-value deep analysis + imputation recommendations                 ║
# ║  • Outlier detection (IQR, Z-score, Isolation Forest, LOF)                  ║
# ║  • Normality tests (Shapiro-Wilk, D'Agostino-K², Anderson-Darling)          ║
# ║  • Correlation: Pearson, Spearman, Kendall, Cramér's V, Point-Biserial     ║
# ║  • Feature-target relationship analysis                                     ║
# ║  • Datetime decomposition (trend, seasonality, gaps)                        ║
# ║  • Text column analysis (length, word count, patterns, email/URL detection) ║
# ║  • Class imbalance + SMOTE readiness report                                 ║
# ║  • Data leakage detection                                                   ║
# ║  • Pairplot, violin plots, KDE plots, Q-Q plots, ECDF plots                 ║
# ║  • PCA variance explained plot                                              ║
# ║  • t-SNE 2-D scatter (sampled)                                              ║
# ║  • Chi-squared test (categorical independence)                               ║
# ║  • ANOVA (numerical vs categorical)                                         ║
# ║  • Shapley-inspired feature importance proxy                                ║
# ║  • Full ML readiness score                                                  ║
# ║  • Interactive, tabbed HTML report with dark-mode toggle                    ║
# ║  • CSV export of every statistics table                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import sys
import json
import math
import logging
import warnings
import base64
import io
import re
import hashlib
import datetime
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from scipy import stats as scipy_stats
from scipy.stats import (
    shapiro, normaltest, anderson, chi2_contingency,
    pointbiserialr, f_oneway, kendalltau, spearmanr,
)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import missingno as msno
    MISSINGNO_OK = True
except ImportError:
    MISSINGNO_OK = False

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DataProfiler")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG  (tweak thresholds here)
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    HIGH_MISSING_PCT       = 40.0,
    HIGH_CORR              = 0.85,
    HIGH_CARDINALITY_RATIO = 0.50,
    HIGH_SKEW              = 1.0,
    ZSCORE_THRESHOLD       = 3.0,
    SAMPLE_ROWS_PLOTS      = 50_000,
    SAMPLE_ROWS_TSNE       = 5_000,
    TOP_CAT_N              = 20,
    MAX_PAIRPLOT_COLS      = 6,
    NORMALITY_ALPHA        = 0.05,
    ANOVA_ALPHA            = 0.05,
    CHI2_ALPHA             = 0.05,
    ISOLATION_CONTAMINATION= 0.05,
    LOF_N_NEIGHBORS        = 20,
    REPORT_FILENAME        = "data_profile_report.html",
    STATS_CSV_DIR          = "profile_stats",
)


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _fig_to_b64(fig: plt.Figure, dpi: int = 90) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_tag(b64: str, title: str = "", width: str = "100%") -> str:
    if not b64:
        return ""
    return (
        f'<figure class="chart-wrap">'
        f'<figcaption>{title}</figcaption>'
        f'<img src="data:image/png;base64,{b64}" style="max-width:{width}" loading="lazy"/>'
        f'</figure>'
    )


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=42)


def _safe_call(fn, *args, **kwargs):
    """Run fn; return None on any exception and log warning."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("Non-fatal error in %s: %s", fn.__name__, exc)
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  1.  FILE LOADING  (CSV · Excel · JSON · Parquet · TSV · SQL-dump)
# ═════════════════════════════════════════════════════════════════════════════
def load_dataset(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()
    log.info("Loading  %s  (%s)", path.name, suffix)

    loaders = {
        ".csv":     _load_csv,
        ".tsv":     lambda p: pd.read_csv(p, sep="\t", low_memory=False),
        ".txt":     lambda p: pd.read_csv(p, sep=None, engine="python", low_memory=False),
        ".xlsx":    lambda p: pd.read_excel(p, engine="openpyxl"),
        ".xls":     lambda p: pd.read_excel(p),
        ".json":    lambda p: pd.read_json(p),
        ".parquet": lambda p: pd.read_parquet(p),
        ".feather": lambda p: pd.read_feather(p),
        ".pkl":     lambda p: pd.read_pickle(p),
    }

    loader = loaders.get(suffix)
    if loader is None:
        # Last resort: try CSV
        log.warning("Unknown extension %s – attempting CSV parse.", suffix)
        loader = _load_csv

    df = loader(filepath)
    log.info("Loaded  %d rows × %d columns", len(df), len(df.columns))

    # Auto-correct obvious type issues
    df = _auto_correct_types(df)
    return df


def _load_csv(filepath) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252", "utf-16"):
        try:
            return pd.read_csv(filepath, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError("Could not decode CSV with any common encoding.")


def _auto_correct_types(df: pd.DataFrame) -> pd.DataFrame:
    """Silently coerce obvious mis-typed columns."""
    for col in df.select_dtypes(include="object").columns:
        # Try numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.80:
            df[col] = converted
            continue
        # Try datetime
        try:
            converted_dt = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if converted_dt.notna().mean() > 0.80:
                df[col] = converted_dt
        except Exception:
            pass
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  2.  DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
def basic_overview(df: pd.DataFrame) -> dict:
    log.info("Computing overview …")
    mem_bytes = df.memory_usage(deep=True).sum()

    # column type breakdown
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols   = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    overview = dict(
        rows         = len(df),
        columns      = len(df.columns),
        memory_mb    = round(mem_bytes / 1024 / 1024, 3),
        size_cells   = df.size,
        column_names = df.columns.tolist(),
        dtypes       = df.dtypes.astype(str).to_dict(),
        head         = df.head(5),
        tail         = df.tail(5),
        num_cols     = num_cols,
        cat_cols     = cat_cols,
        dt_cols      = dt_cols,
        bool_cols    = bool_cols,
        complete_rows= int((~df.isnull().any(axis=1)).sum()),
        hash_sha256  = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16],
    )

    _print_section("DATASET OVERVIEW")
    print(f"  Rows            : {overview['rows']:,}")
    print(f"  Columns         : {overview['columns']}")
    print(f"  Memory (MB)     : {overview['memory_mb']}")
    print(f"  Total cells     : {overview['size_cells']:,}")
    print(f"  Complete rows   : {overview['complete_rows']:,}")
    print(f"  Numerical cols  : {len(num_cols)}")
    print(f"  Categorical cols: {len(cat_cols)}")
    print(f"  Datetime cols   : {len(dt_cols)}")
    print(f"  Boolean cols    : {len(bool_cols)}")
    print(f"  SHA-256 (16)    : {overview['hash_sha256']}")
    return overview


# ═════════════════════════════════════════════════════════════════════════════
#  3.  MISSING VALUE ANALYSIS  (deep)
# ═════════════════════════════════════════════════════════════════════════════
def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Analysing missing values …")
    n = len(df)
    mv_count  = df.isnull().sum()
    mv_pct    = (mv_count / n * 100).round(2)
    mv_dtype  = df.dtypes.astype(str)

    # Imputation strategy recommendation
    def _recommend(col):
        pct = mv_pct[col]
        if pct == 0:      return "—"
        if pct > 70:      return "DROP column"
        dtype = str(df[col].dtype)
        if "float" in dtype or "int" in dtype:
            skew = abs(df[col].skew()) if df[col].notna().sum() > 1 else 0
            return "Median impute" if skew > 1 else "Mean impute"
        if "datetime" in dtype: return "Forward-fill"
        return "Mode impute"

    mv_df = pd.DataFrame({
        "dtype":        mv_dtype,
        "missing_count": mv_count,
        "missing_pct":   mv_pct,
        "present_count": n - mv_count,
        "impute_rec":    {c: _recommend(c) for c in df.columns},
    })
    mv_df = mv_df[mv_df["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    _print_section("MISSING VALUE ANALYSIS")
    if mv_df.empty:
        print("  ✔ No missing values found.")
    else:
        print(mv_df.to_string())
    return mv_df


def plot_missing_charts(df: pd.DataFrame) -> tuple:
    """Return (bar_b64, heatmap_b64, dendro_b64)."""
    mv_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    mv_pct = mv_pct[mv_pct > 0]

    bar_b64 = heat_b64 = dendro_b64 = ""

    if mv_pct.empty:
        return bar_b64, heat_b64, dendro_b64

    # Bar
    fig, ax = plt.subplots(figsize=(max(8, len(mv_pct) * 0.55), 5))
    colors = ["#e74c3c" if v > 40 else "#e67e22" if v > 10 else "#3498db" for v in mv_pct]
    mv_pct.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.axhline(40, ls="--", color="red",    lw=1.2, label="40% threshold")
    ax.axhline(10, ls="--", color="orange", lw=1.2, label="10% threshold")
    ax.set_title("Missing Values by Column (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Missing %"); ax.set_xlabel("Column")
    ax.tick_params(axis="x", rotation=50)
    ax.legend(fontsize=9)
    plt.tight_layout()
    bar_b64 = _fig_to_b64(fig)

    # Heatmap
    mv_cols = df.columns[df.isnull().any()].tolist()
    sample  = _sample(df[mv_cols].isnull().astype(int), 600)
    fig, ax = plt.subplots(figsize=(max(8, len(mv_cols) * 0.7), 6))
    sns.heatmap(sample, cbar=False, ax=ax, cmap="YlOrRd", yticklabels=False, linewidths=0)
    ax.set_title("Missing Value Pattern  (row-sample)", fontsize=13)
    plt.tight_layout()
    heat_b64 = _fig_to_b64(fig)

    # Missingno dendrogram (if available)
    if MISSINGNO_OK and len(mv_cols) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(max(8, len(mv_cols) * 0.7), 5))
            msno.dendrogram(df[mv_cols], ax=ax, fontsize=11)
            ax.set_title("Missing-value Dendrogram (correlation of nullity)", fontsize=13)
            plt.tight_layout()
            dendro_b64 = _fig_to_b64(fig)
        except Exception:
            pass

    return bar_b64, heat_b64, dendro_b64


# ═════════════════════════════════════════════════════════════════════════════
#  4.  NUMERICAL  — extended stats + normality tests
# ═════════════════════════════════════════════════════════════════════════════
def numerical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Analysing numerical features …")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    stats = num_df.describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99]).T
    stats["variance"]  = num_df.var()
    stats["skewness"]  = num_df.skew()
    stats["kurtosis"]  = num_df.kurtosis()
    stats["iqr"]       = num_df.quantile(.75) - num_df.quantile(.25)
    stats["cv"]        = (num_df.std() / num_df.mean().replace(0, np.nan)).round(4)  # coeff of variation
    stats["range"]     = num_df.max() - num_df.min()
    stats["zeros_pct"] = ((num_df == 0).sum() / len(num_df) * 100).round(2)

    # IQR outliers
    Q1, Q3  = num_df.quantile(.25), num_df.quantile(.75)
    IQR     = Q3 - Q1
    stats["outlier_iqr"]   = ((num_df < Q1 - 1.5*IQR) | (num_df > Q3 + 1.5*IQR)).sum()
    stats["outlier_iqr_pct"] = (stats["outlier_iqr"] / len(df) * 100).round(2)

    # Z-score outliers
    z_threshold = CFG["ZSCORE_THRESHOLD"]
    zscores = (num_df - num_df.mean()) / num_df.std()
    stats["outlier_zscore"] = (zscores.abs() > z_threshold).sum()

    # Normality tests
    sw_p, da_p, ad_sig = {}, {}, {}
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) < 8:
            sw_p[col] = da_p[col] = np.nan; ad_sig[col] = "—"; continue
        s_test = s.sample(min(5000, len(s)), random_state=42)
        try:
            sw_p[col] = round(shapiro(s_test)[1], 4)
        except Exception:
            sw_p[col] = np.nan
        try:
            da_p[col] = round(normaltest(s_test)[1], 4)
        except Exception:
            da_p[col] = np.nan
        try:
            res = anderson(s_test, dist="norm")
            # significance level index 2 = 5%
            ad_sig[col] = "Normal" if res.statistic < res.critical_values[2] else "Non-normal"
        except Exception:
            ad_sig[col] = "—"

    stats["shapiro_p"]  = pd.Series(sw_p)
    stats["dagostino_p"]= pd.Series(da_p)
    stats["anderson"]   = pd.Series(ad_sig)
    stats["is_normal"]  = stats["shapiro_p"].apply(
        lambda p: "✔ Normal" if (not np.isnan(p) and p > CFG["NORMALITY_ALPHA"]) else "✘ Non-normal"
    )

    _print_section("NUMERICAL FEATURE STATISTICS")
    print(stats.to_string())
    return stats


def plot_numerical_charts(df: pd.DataFrame) -> dict:
    """Return dict of chart lists: histograms, boxplots, violins, qq, kde, ecdf."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample   = _sample(df, CFG["SAMPLE_ROWS_PLOTS"])
    out = defaultdict(list)

    for col in num_cols:
        data = sample[col].dropna()
        if len(data) < 2:
            continue

        # ── histogram + KDE overlay ──
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=50, density=True, alpha=0.65, color="#3498db", edgecolor="white")
        try:
            if data.nunique() >= 3:
                kde = scipy_stats.gaussian_kde(data)
                xs  = np.linspace(data.min(), data.max(), 300)
                ax.plot(xs, kde(xs), color="#e74c3c", lw=2, label="KDE")
        except Exception:
            pass  # singular covariance / low-variance column — skip KDE overlay
        ax.set_title(f"Histogram + KDE: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col); ax.set_ylabel("Density"); ax.legend(fontsize=8)
        plt.tight_layout()
        out["histograms"].append((col, _fig_to_b64(fig)))

        # ── boxplot ──
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.boxplot(data, vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#aed6f1", color="#1a5276"),
                   medianprops=dict(color="#e74c3c", lw=2),
                   whiskerprops=dict(color="#1a5276"),
                   flierprops=dict(marker=".", color="#e74c3c", alpha=0.4))
        ax.set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold"); ax.set_xlabel(col)
        plt.tight_layout()
        out["boxplots"].append((col, _fig_to_b64(fig)))

        # ── violin (falls back to boxplot if KDE is singular) ──
        fig, ax = plt.subplots(figsize=(5, 4))
        try:
            clean_data = data.dropna()
            if clean_data.nunique() < 3:
                raise ValueError("Too few unique values for violin KDE")
            parts = ax.violinplot(clean_data, vert=True, showmedians=True)
            for pc in parts.get("bodies", []):
                pc.set_facecolor("#a9cce3"); pc.set_alpha(0.8)
            ax.set_title(f"Violin: {col}", fontsize=11, fontweight="bold")
        except Exception:
            ax.boxplot(data.dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#a9cce3", color="#1a5276"),
                       medianprops=dict(color="#e74c3c", lw=2))
            ax.set_title(f"Violin (fallback boxplot): {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel(col); ax.set_xticks([])
        plt.tight_layout()
        out["violins"].append((col, _fig_to_b64(fig)))

        # ── Q-Q plot ──
        fig, ax = plt.subplots(figsize=(5, 4))
        (osm, osr), (slope, intercept, _) = scipy_stats.probplot(data, dist="norm")
        ax.scatter(osm, osr, s=8, alpha=0.4, color="#2ecc71")
        ax.plot(osm, slope * np.array(osm) + intercept, color="#e74c3c", lw=2)
        ax.set_title(f"Q-Q Plot: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
        plt.tight_layout()
        out["qq"].append((col, _fig_to_b64(fig)))

        # ── ECDF ──
        fig, ax = plt.subplots(figsize=(5, 4))
        sorted_d = np.sort(data)
        cdf      = np.arange(1, len(sorted_d)+1) / len(sorted_d)
        ax.plot(sorted_d, cdf, color="#8e44ad", lw=2)
        ax.set_title(f"ECDF: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col); ax.set_ylabel("Cumulative Probability"); ax.set_ylim(0, 1)
        plt.tight_layout()
        out["ecdf"].append((col, _fig_to_b64(fig)))

    return dict(out)


# ═════════════════════════════════════════════════════════════════════════════
#  5.  CATEGORICAL  — deep analysis
# ═════════════════════════════════════════════════════════════════════════════
def categorical_analysis(df: pd.DataFrame) -> dict:
    log.info("Analysing categorical features …")
    cat_df  = df.select_dtypes(include=["object", "category", "bool"])
    results = {}

    _print_section("CATEGORICAL FEATURE ANALYSIS")
    for col in cat_df.columns:
        vc          = cat_df[col].value_counts(dropna=False)
        unique_cnt  = cat_df[col].nunique(dropna=True)
        entropy_val = scipy_stats.entropy(vc.values) if len(vc) > 1 else 0.0
        results[col] = dict(
            unique_count    = unique_cnt,
            top_n           = vc.head(CFG["TOP_CAT_N"]),
            imbalance_ratio = round(vc.iloc[0] / len(df), 4) if len(vc) else None,
            entropy         = round(entropy_val, 4),
            is_binary       = unique_cnt == 2,
            is_id_col       = unique_cnt / len(df) > CFG["HIGH_CARDINALITY_RATIO"],
        )
        print(f"\n  [{col}]  unique={unique_cnt}  entropy={entropy_val:.3f}  "
              f"top={vc.index[0] if len(vc) else 'N/A'} ({vc.iloc[0] if len(vc) else 0:,})")

    return results


def plot_cat_charts(df: pd.DataFrame) -> list:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    images   = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(CFG["TOP_CAT_N"])
        if vc.empty: continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        # Bar
        vc.plot(kind="barh", ax=axes[0], color="#2ecc71", edgecolor="white")
        axes[0].invert_yaxis()
        axes[0].set_title(f"Top Categories: {col}", fontweight="bold")
        axes[0].set_xlabel("Count")
        # Pie (top 8)
        top8 = vc.head(8)
        other = vc.iloc[8:].sum()
        if other > 0:
            top8 = pd.concat([top8, pd.Series({"Other": other})])
        axes[1].pie(top8.values, labels=[str(l) for l in top8.index],
                    autopct="%1.1f%%", startangle=90,
                    colors=sns.color_palette("pastel", len(top8)))
        axes[1].set_title(f"Distribution: {col}", fontweight="bold")
        plt.tight_layout()
        images.append((col, _fig_to_b64(fig)))
    return images


# ═════════════════════════════════════════════════════════════════════════════
#  6.  DATETIME ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def datetime_analysis(df: pd.DataFrame) -> dict:
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if not dt_cols:
        return {}

    log.info("Analysing %d datetime column(s) …", len(dt_cols))
    _print_section("DATETIME ANALYSIS")
    results = {}
    for col in dt_cols:
        s = df[col].dropna().sort_values()
        if s.empty: continue
        diffs = s.diff().dropna()
        results[col] = dict(
            min_date    = str(s.min()),
            max_date    = str(s.max()),
            date_range  = str(s.max() - s.min()),
            missing_cnt = df[col].isnull().sum(),
            median_gap  = str(diffs.median()),
            max_gap     = str(diffs.max()),
            n_unique    = s.nunique(),
            weekday_dist= s.dt.day_name().value_counts().to_dict(),
            hour_dist   = s.dt.hour.value_counts().to_dict() if hasattr(s.dt, "hour") else {},
        )
        print(f"\n  [{col}]  range: {results[col]['min_date']} → {results[col]['max_date']}")
        print(f"    Median gap: {results[col]['median_gap']}  Max gap: {results[col]['max_gap']}")
    return results


def plot_datetime_charts(df: pd.DataFrame, dt_results: dict) -> list:
    images = []
    for col, info in dt_results.items():
        s = df[col].dropna().sort_values()
        if s.empty: continue

        # Timeline count by month
        monthly = s.dt.to_period("M").value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(monthly.index.astype(str), monthly.values, color="#3498db", edgecolor="white")
        ax.set_title(f"Monthly Record Count: {col}", fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=60)
        plt.tight_layout()
        images.append((col + " — Monthly", _fig_to_b64(fig)))

        # Weekday distribution
        if info["weekday_dist"]:
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            wd = pd.Series(info["weekday_dist"]).reindex(order, fill_value=0)
            fig, ax = plt.subplots(figsize=(7, 4))
            wd.plot(kind="bar", ax=ax, color="#9b59b6", edgecolor="white")
            ax.set_title(f"Weekday Distribution: {col}", fontweight="bold")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            images.append((col + " — Weekday", _fig_to_b64(fig)))
    return images


# ═════════════════════════════════════════════════════════════════════════════
#  7.  TEXT COLUMN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_URL_RE   = re.compile(r"https?://\S+|www\.\S+")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")


def text_analysis(df: pd.DataFrame) -> dict:
    """Analyse string columns for text-specific metrics."""
    log.info("Analysing text columns …")
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        return {}

    _print_section("TEXT / STRING ANALYSIS")
    results = {}
    for col in obj_cols:
        s = df[col].dropna().astype(str)
        if s.empty: continue
        lengths    = s.str.len()
        word_cnts  = s.str.split().str.len()
        email_cnt  = s.str.contains(_EMAIL_RE).sum()
        url_cnt    = s.str.contains(_URL_RE).sum()
        phone_cnt  = s.str.contains(_PHONE_RE).sum()
        has_digits = s.str.contains(r"\d").sum()
        has_upper  = s.str.contains(r"[A-Z]").sum()

        results[col] = dict(
            avg_length  = round(lengths.mean(), 1),
            max_length  = int(lengths.max()),
            min_length  = int(lengths.min()),
            avg_words   = round(word_cnts.mean(), 1),
            email_count = int(email_cnt),
            url_count   = int(url_cnt),
            phone_count = int(phone_cnt),
            has_digits_pct = round(has_digits / len(s) * 100, 1),
            has_upper_pct  = round(has_upper  / len(s) * 100, 1),
        )
        print(f"\n  [{col}]  avg_len={results[col]['avg_length']}  "
              f"emails={email_cnt}  urls={url_cnt}  phones={phone_cnt}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  8.  OUTLIER DETECTION  (IQR · Z-score · Isolation Forest · LOF)
# ═════════════════════════════════════════════════════════════════════════════
def outlier_analysis(df: pd.DataFrame) -> dict:
    log.info("Running outlier detection …")
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num_df.empty:
        return {}

    results = {}

    # IQR
    Q1, Q3  = num_df.quantile(.25), num_df.quantile(.75)
    IQR     = Q3 - Q1
    iqr_mask = (num_df < Q1 - 1.5*IQR) | (num_df > Q3 + 1.5*IQR)
    results["iqr_outlier_counts"] = iqr_mask.sum().to_dict()

    # Z-score
    z       = ((num_df - num_df.mean()) / num_df.std()).abs()
    results["zscore_outlier_counts"] = (z > CFG["ZSCORE_THRESHOLD"]).sum().to_dict()

    # Isolation Forest & LOF (multivariate — needs sklearn)
    results["iso_forest_count"] = results["lof_count"] = None
    if SKLEARN_OK and len(num_df.columns) >= 2:
        clean = num_df.dropna()
        clean_sample = _sample(clean, CFG["SAMPLE_ROWS_PLOTS"])
        scaler = StandardScaler()
        X = scaler.fit_transform(clean_sample)

        try:
            iso = IsolationForest(contamination=CFG["ISOLATION_CONTAMINATION"], random_state=42, n_jobs=-1)
            iso_labels = iso.fit_predict(X)
            results["iso_forest_count"] = int((iso_labels == -1).sum())
        except Exception:
            pass

        try:
            lof = LocalOutlierFactor(n_neighbors=min(CFG["LOF_N_NEIGHBORS"], len(X)-1))
            lof_labels = lof.fit_predict(X)
            results["lof_count"] = int((lof_labels == -1).sum())
        except Exception:
            pass

    _print_section("OUTLIER ANALYSIS")
    print(f"  IQR outlier counts   : { {k: v for k,v in results['iqr_outlier_counts'].items() if v > 0} }")
    print(f"  Z-score outlier counts: { {k: v for k,v in results['zscore_outlier_counts'].items() if v > 0} }")
    if results["iso_forest_count"] is not None:
        print(f"  Isolation Forest (multivariate): {results['iso_forest_count']} outliers")
    if results["lof_count"] is not None:
        print(f"  LOF (multivariate)             : {results['lof_count']} outliers")
    return results


def plot_outlier_charts(df: pd.DataFrame, outlier_results: dict) -> list:
    """Scatter-matrix style outlier highlight for top outlier-heavy columns."""
    images = []
    iqr_counts = outlier_results.get("iqr_outlier_counts", {})
    top_cols = sorted(iqr_counts, key=iqr_counts.get, reverse=True)[:4]
    top_cols = [c for c in top_cols if iqr_counts[c] > 0]

    num_df = df.select_dtypes(include=[np.number])
    for col in top_cols:
        s = num_df[col].dropna()
        Q1, Q3 = s.quantile(.25), s.quantile(.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        is_out = (s < lower) | (s > upper)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(range(len(s[~is_out])), s[~is_out], s=5, alpha=0.3, color="#3498db", label="Normal")
        out_idx  = is_out[is_out].index
        ax.scatter([list(s.index).index(i) for i in out_idx if i in list(s.index)[:CFG["SAMPLE_ROWS_PLOTS"]]],
                   s[out_idx[:500]], s=12, alpha=0.7, color="#e74c3c", label="Outlier")
        ax.axhline(upper, ls="--", color="#e74c3c", lw=1, label=f"Upper {upper:.2f}")
        ax.axhline(lower, ls="--", color="#e67e22", lw=1, label=f"Lower {lower:.2f}")
        ax.set_title(f"Outlier Scatter: {col}  ({is_out.sum()} outliers)", fontweight="bold")
        ax.legend(fontsize=8); ax.set_xlabel("Row index"); ax.set_ylabel(col)
        plt.tight_layout()
        images.append((col, _fig_to_b64(fig)))

    return images


# ═════════════════════════════════════════════════════════════════════════════
#  9.  CORRELATION  (Pearson · Spearman · Kendall · Cramér's V · Pt-Biserial)
# ═════════════════════════════════════════════════════════════════════════════
def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct   = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct, correction=False)[0]
    n    = ct.sum().sum()
    r, k = ct.shape
    phi2 = max(0, chi2/n - (k-1)*(r-1)/(n-1))
    return math.sqrt(phi2 / min((k-1)/(n-1), (r-1)/(n-1))) if n > 1 else 0.0


def correlation_analysis(df: pd.DataFrame) -> dict:
    log.info("Computing correlation matrices …")
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(include=["object", "category"])
    results = {}

    # Pearson
    if num_df.shape[1] >= 2:
        results["pearson"]  = num_df.corr("pearson")
        results["spearman"] = num_df.corr("spearman")
        results["kendall"]  = num_df.corr("kendall")

        # High correlation pairs (Pearson)
        corr = results["pearson"]
        high_pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                v = corr.iloc[i,j]
                if abs(v) >= CFG["HIGH_CORR"]:
                    high_pairs.append((cols[i], cols[j], round(v,4)))
        results["high_corr_pairs"] = high_pairs
    else:
        results["pearson"] = results["spearman"] = results["kendall"] = pd.DataFrame()
        results["high_corr_pairs"] = []

    # Cramér's V (cat–cat)
    cat_cols = cat_df.columns.tolist()
    if len(cat_cols) >= 2:
        cv_mat = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
        for c1 in cat_cols:
            for c2 in cat_cols:
                try:
                    cv_mat.loc[c1,c2] = _cramers_v(df[c1].fillna("NA"), df[c2].fillna("NA"))
                except Exception:
                    cv_mat.loc[c1,c2] = np.nan
        results["cramers_v"] = cv_mat.astype(float)
    else:
        results["cramers_v"] = pd.DataFrame()

    # Point-biserial (binary cat × num)
    pb_results = []
    binary_cats = [c for c in cat_cols if df[c].nunique() == 2]
    for cat_c in binary_cats:
        enc = pd.factorize(df[cat_c])[0]
        for num_c in num_df.columns:
            mask = (~df[cat_c].isnull()) & (~df[num_c].isnull())
            if mask.sum() < 5: continue
            try:
                r, p = pointbiserialr(enc[mask], df[num_c][mask])
                pb_results.append(dict(cat_col=cat_c, num_col=num_c, r=round(r,4), p=round(p,4)))
            except Exception:
                pass
    results["point_biserial"] = pd.DataFrame(pb_results) if pb_results else pd.DataFrame()

    _print_section("CORRELATION ANALYSIS")
    hp = results["high_corr_pairs"]
    if hp:
        print(f"  ⚠  Highly correlated pairs (Pearson |r| ≥ {CFG['HIGH_CORR']}):")
        for a,b,v in hp: print(f"      {a}  ↔  {b}  :  {v}")
    else:
        print("  ✔ No highly correlated pairs found.")
    return results


def plot_corr_charts(corr_results: dict) -> dict:
    imgs = {}
    for key in ["pearson", "spearman", "kendall", "cramers_v"]:
        mat = corr_results.get(key, pd.DataFrame())
        if mat is None or mat.empty: continue
        n = mat.shape[0]
        sz = max(6, n * 0.55)
        fig, ax = plt.subplots(figsize=(sz, sz*0.85))
        mask = np.triu(np.ones_like(mat, dtype=bool)) if key != "cramers_v" else None
        cmap = "coolwarm" if key != "cramers_v" else "YlOrRd"
        sns.heatmap(mat.astype(float), annot=(n<=18), fmt=".2f", cmap=cmap,
                    mask=mask, ax=ax, center=0 if key!="cramers_v" else None,
                    vmin=-1 if key!="cramers_v" else 0, vmax=1,
                    linewidths=0.3, square=True, cbar_kws={"shrink":.7})
        title_map = dict(pearson="Pearson Correlation", spearman="Spearman Correlation",
                         kendall="Kendall Correlation", cramers_v="Cramér's V  (cat–cat)")
        ax.set_title(title_map[key], fontsize=13, fontweight="bold")
        plt.tight_layout()
        imgs[key] = _fig_to_b64(fig)
    return imgs


# ═════════════════════════════════════════════════════════════════════════════
#  10.  STATISTICAL TESTS  (ANOVA · Chi-squared)
# ═════════════════════════════════════════════════════════════════════════════
def statistical_tests(df: pd.DataFrame) -> dict:
    log.info("Running statistical tests …")
    results = dict(anova=[], chi2=[])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    # ANOVA: every num × every cat (low-cardinality)
    for cat_c in cat_cols:
        if df[cat_c].nunique() > 20: continue
        groups_dict = df.groupby(cat_c, observed=True)
        for num_c in num_cols:
            groups = [g[num_c].dropna().values for _, g in groups_dict if len(g[num_c].dropna()) >= 2]
            if len(groups) < 2: continue
            try:
                F, p = f_oneway(*groups)
                results["anova"].append(dict(
                    cat_col=cat_c, num_col=num_c,
                    F=round(F,4), p=round(p,6),
                    significant= p < CFG["ANOVA_ALPHA"],
                ))
            except Exception:
                pass

    # Chi-squared: every pair of low-cardinality cat columns
    for i, c1 in enumerate(cat_cols):
        if df[c1].nunique() > 30: continue
        for c2 in cat_cols[i+1:]:
            if df[c2].nunique() > 30: continue
            try:
                ct = pd.crosstab(df[c1].fillna("NA"), df[c2].fillna("NA"))
                chi2, p, dof, _ = chi2_contingency(ct)
                results["chi2"].append(dict(
                    col1=c1, col2=c2,
                    chi2=round(chi2,4), p=round(p,6), dof=dof,
                    significant= p < CFG["CHI2_ALPHA"],
                ))
            except Exception:
                pass

    _print_section("STATISTICAL TESTS")
    sig_anova = [r for r in results["anova"] if r["significant"]]
    sig_chi2  = [r for r in results["chi2"]  if r["significant"]]
    print(f"  ANOVA significant pairs  : {len(sig_anova)}")
    print(f"  Chi-squared significant  : {len(sig_chi2)}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  11.  PCA  &  t-SNE  (dimensionality overview)
# ═════════════════════════════════════════════════════════════════════════════
def pca_analysis(df: pd.DataFrame) -> tuple:
    """Returns (pca_img_b64, variance_df)."""
    if not SKLEARN_OK:
        return "", pd.DataFrame()
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num_df.shape[1] < 2:
        return "", pd.DataFrame()

    clean = num_df.dropna()
    if len(clean) < 5:
        return "", pd.DataFrame()

    scaler = StandardScaler()
    X = scaler.fit_transform(clean)
    n_components = min(len(clean.columns), 20, len(clean))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    var_df = pd.DataFrame({
        "PC":        [f"PC{i+1}" for i in range(n_components)],
        "var_explained": (explained * 100).round(3),
        "cum_var":       (cumulative * 100).round(3),
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(var_df["PC"], var_df["var_explained"], color="#3498db", edgecolor="white", label="Per PC")
    ax2 = ax.twinx()
    ax2.plot(var_df["PC"], var_df["cum_var"], color="#e74c3c", marker="o", ms=5, lw=2, label="Cumulative")
    ax2.axhline(90, ls="--", color="#e74c3c", lw=1, alpha=0.5, label="90% line")
    ax.set_title("PCA — Variance Explained", fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance (%)")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax.tick_params(axis="x", rotation=50)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig), var_df


def tsne_plot(df: pd.DataFrame, cat_col: str = None) -> str:
    """2-D t-SNE scatter, optionally coloured by cat_col."""
    if not SKLEARN_OK:
        return ""
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num_df.shape[1] < 2:
        return ""

    clean = num_df.dropna()
    if len(clean) < 10:
        return ""

    sample = _sample(clean, CFG["SAMPLE_ROWS_TSNE"])
    scaler = StandardScaler()
    X = scaler.fit_transform(sample)

    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1), n_iter=500)
        emb  = tsne.fit_transform(X)
    except Exception:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    if cat_col and cat_col in df.columns:
        labels = df.loc[sample.index, cat_col].fillna("NA").astype(str).values
        unique_labels = list(dict.fromkeys(labels))[:15]
        palette = sns.color_palette("tab20", len(unique_labels))
        color_map = {l: palette[i] for i, l in enumerate(unique_labels)}
        colors = [color_map.get(l, (0.5,0.5,0.5)) for l in labels]
        for lbl in unique_labels:
            mask = labels == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.5,
                       color=color_map[lbl], label=lbl)
        ax.legend(fontsize=7, markerscale=2, loc="best", ncol=2)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.4, color="#3498db")

    ax.set_title(f"t-SNE 2D Embedding  (n={len(sample):,})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dimension 1"); ax.set_ylabel("Dimension 2")
    plt.tight_layout()
    return _fig_to_b64(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  12.  PAIRPLOT  (top-N numerical columns by variance)
# ═════════════════════════════════════════════════════════════════════════════
def pairplot_chart(df: pd.DataFrame, cat_col: str = None) -> str:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return ""
    # Pick top columns by variance, limit to MAX_PAIRPLOT_COLS
    top_cols = num_df.var().sort_values(ascending=False).head(CFG["MAX_PAIRPLOT_COLS"]).index.tolist()
    sample   = _sample(df[top_cols + ([cat_col] if cat_col and cat_col in df.columns else [])], 3000)

    try:
        hue = cat_col if (cat_col and cat_col in sample.columns and sample[cat_col].nunique() <= 10) else None
        g = sns.pairplot(sample, vars=top_cols, hue=hue, plot_kws={"alpha":0.35,"s":12},
                         diag_kind="hist", height=2.2)
        g.fig.suptitle("Pairplot — Top Numerical Features", y=1.02, fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_b64(g.fig)
    except Exception:
        return ""


# ═════════════════════════════════════════════════════════════════════════════
#  13.  CLASS IMBALANCE  (auto-detect target)
# ═════════════════════════════════════════════════════════════════════════════
def detect_target(df: pd.DataFrame) -> str:
    keywords = {"target","label","y","class","output","result","churn","fraud",
                "default","survived","diagnosis","response","outcome","flag"}
    for col in df.columns:
        if col.lower().strip() in keywords:
            return col
    # Last col with low cardinality
    last = df.columns[-1]
    if df[last].nunique() <= 20:
        return last
    return ""


def class_imbalance_analysis(df: pd.DataFrame, target_col: str) -> dict:
    if not target_col or target_col not in df.columns:
        return {}
    log.info("Analysing class imbalance for target='%s' …", target_col)
    vc      = df[target_col].value_counts(dropna=False)
    n       = len(df)
    entropy = scipy_stats.entropy(vc.values)
    ratio   = vc.iloc[0] / vc.iloc[-1] if len(vc) > 1 else 1.0

    result = dict(
        target_col    = target_col,
        class_counts  = vc.to_dict(),
        class_pct     = (vc/n*100).round(2).to_dict(),
        majority_class= vc.index[0],
        minority_class= vc.index[-1],
        imbalance_ratio=round(ratio,2),
        entropy       = round(entropy,4),
        smote_needed  = ratio > 3,
    )

    _print_section("CLASS IMBALANCE ANALYSIS")
    print(f"  Target col      : {target_col}")
    print(f"  Class counts    :\n{vc.to_string()}")
    print(f"  Imbalance ratio : {ratio:.2f}  {'⚠ SMOTE recommended' if ratio>3 else '✔ Balanced'}")
    return result


def plot_class_imbalance(df: pd.DataFrame, target_col: str) -> str:
    if not target_col or target_col not in df.columns:
        return ""
    vc = df[target_col].value_counts(dropna=False).head(20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Bar
    colors = sns.color_palette("Set2", len(vc))
    vc.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white")
    axes[0].set_title(f"Class Distribution: {target_col}", fontweight="bold")
    axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    # Pie
    axes[1].pie(vc.values, labels=[str(l) for l in vc.index],
                autopct="%1.1f%%", colors=colors, startangle=90)
    axes[1].set_title(f"Class Proportions: {target_col}", fontweight="bold")
    plt.tight_layout()
    return _fig_to_b64(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  14.  FEATURE–TARGET RELATIONSHIPS  (num vs target, cat vs target)
# ═════════════════════════════════════════════════════════════════════════════
def feature_target_analysis(df: pd.DataFrame, target_col: str) -> list:
    if not target_col or target_col not in df.columns:
        return []
    log.info("Analysing feature–target relationships …")
    images = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    target_is_cat = df[target_col].nunique() <= 20

    if target_is_cat and num_cols:
        # KDE per class for each numerical feature
        for col in num_cols[:8]:
            fig, ax = plt.subplots(figsize=(7, 4))
            for cls in df[target_col].dropna().unique():
                subset = df[df[target_col] == cls][col].dropna()
                if len(subset) < 2: continue
                try:
                    if subset.nunique() >= 3:
                        subset.plot.kde(ax=ax, label=str(cls), alpha=0.7, lw=2)
                    else:
                        ax.hist(subset, density=True, alpha=0.5, label=str(cls), bins=10)
                except Exception:
                    ax.hist(subset, density=True, alpha=0.5, label=str(cls), bins=10)
            ax.set_title(f"{col}  by  {target_col}", fontweight="bold")
            ax.set_xlabel(col); ax.legend(fontsize=8)
            plt.tight_layout()
            images.append((f"{col} vs {target_col}", _fig_to_b64(fig)))

    elif not target_is_cat and num_cols:
        # Scatter plot for top correlated features
        sample = _sample(df, 5000)
        for col in num_cols[:6]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(sample[col], sample[target_col], s=8, alpha=0.3, color="#3498db")
            ax.set_title(f"{col}  vs  {target_col}", fontweight="bold")
            ax.set_xlabel(col); ax.set_ylabel(target_col)
            plt.tight_layout()
            images.append((f"{col} vs {target_col}", _fig_to_b64(fig)))

    return images


# ═════════════════════════════════════════════════════════════════════════════
#  15.  DUPLICATE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def duplicate_analysis(df: pd.DataFrame) -> dict:
    log.info("Checking duplicates …")
    full_dup  = df.duplicated().sum()
    col_dups  = {c: int(df[c].duplicated().sum()) for c in df.columns}
    info = dict(
        full_duplicate_count = int(full_dup),
        full_duplicate_pct   = round(full_dup / len(df) * 100, 3),
        per_column_duplicates= col_dups,
    )
    _print_section("DUPLICATE ANALYSIS")
    print(f"  Full duplicate rows : {full_dup:,}  ({info['full_duplicate_pct']} %)")
    return info


# ═════════════════════════════════════════════════════════════════════════════
#  16.  DATA QUALITY  (full suite)
# ═════════════════════════════════════════════════════════════════════════════
def data_quality_checks(df: pd.DataFrame) -> dict:
    log.info("Running data quality checks …")
    issues = dict(
        constant_cols       = [],
        near_constant_cols  = [],
        high_missing_cols   = [],
        high_cardinality    = [],
        highly_skewed       = [],
        potential_id_cols   = [],
        potential_leakage   = [],
        mixed_type_cols     = [],
        negative_where_positive_expected = [],
    )

    num_df = df.select_dtypes(include=[np.number])

    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)
        miss_pct = df[col].isnull().mean() * 100

        # Constant
        if n_unique <= 1:
            issues["constant_cols"].append(col)

        # Near-constant (>= 99% same value)
        elif df[col].value_counts(normalize=True, dropna=False).iloc[0] >= 0.99:
            issues["near_constant_cols"].append(col)

        # High missing
        if miss_pct >= CFG["HIGH_MISSING_PCT"]:
            issues["high_missing_cols"].append(col)

        # High cardinality (cat)
        if df[col].dtype == object and n_unique / len(df) > CFG["HIGH_CARDINALITY_RATIO"]:
            issues["high_cardinality"].append(col)

        # Potential ID
        if n_unique == len(df):
            issues["potential_id_cols"].append(col)

        # Mixed type detection
        if df[col].dtype == object:
            sample_vals = df[col].dropna().head(500)
            type_set = set(type(v).__name__ for v in sample_vals)
            if len(type_set) > 1:
                issues["mixed_type_cols"].append(col)

    # Highly skewed numerical
    for col in num_df.columns:
        sk = num_df[col].skew()
        if abs(sk) >= CFG["HIGH_SKEW"]:
            issues["highly_skewed"].append((col, round(sk, 3)))

    # Potential negative values in columns that should be positive
    positivity_keywords = ["age","price","cost","amount","count","qty","quantity","salary","revenue"]
    for col in num_df.columns:
        if any(kw in col.lower() for kw in positivity_keywords):
            if (num_df[col] < 0).sum() > 0:
                issues["negative_where_positive_expected"].append(col)

    _print_section("DATA QUALITY CHECKS")
    for k, v in issues.items():
        if v: print(f"  {k}: {v}")
    return issues


# ═════════════════════════════════════════════════════════════════════════════
#  17.  ML TASK DETECTION  +  ML READINESS SCORE
# ═════════════════════════════════════════════════════════════════════════════

def detect_ml_task(df: pd.DataFrame, target_col: str) -> dict:
    """
    Determine whether the target column implies:
      - Binary Classification
      - Multiclass Classification
      - Regression
      - Unknown (no target provided)

    Decision logic
    ──────────────
    1. No target column              → Unknown
    2. Target is non-numeric (object/category/bool)
                                     → Classification (binary if 2 unique, else multiclass)
    3. Target is numeric AND has ≤ 20 unique integer-like values
       AND those values look like discrete labels (all whole numbers)
                                     → Classification
    4. Everything else               → Regression
    """
    if not target_col or target_col not in df.columns:
        return dict(task="unknown", reason="No target column provided.", n_classes=None,
                    class_labels=None, target_dtype=None)

    s          = df[target_col].dropna()
    n_unique   = s.nunique()
    dtype_str  = str(s.dtype)
    target_dtype = dtype_str

    # Non-numeric → always classification
    if s.dtype == object or str(s.dtype) == "category" or s.dtype == bool:
        task = "binary_classification" if n_unique == 2 else "multiclass_classification"
        return dict(task=task,
                    reason=f"Target is non-numeric with {n_unique} unique value(s).",
                    n_classes=n_unique,
                    class_labels=sorted(s.unique().tolist()),
                    target_dtype=target_dtype)

    # Numeric — check if it looks like discrete class labels
    all_integers   = (s == s.round()).all()          # e.g. 0, 1, 2 — not 0.5, 1.7
    few_unique     = n_unique <= 20
    small_range    = (s.max() - s.min()) < 50        # classes rarely span more than 50

    if all_integers and few_unique and small_range:
        task = "binary_classification" if n_unique == 2 else "multiclass_classification"
        return dict(task=task,
                    reason=f"Target is integer-like with only {n_unique} unique value(s) — treated as labels.",
                    n_classes=n_unique,
                    class_labels=sorted(s.unique().tolist()),
                    target_dtype=target_dtype)

    # Continuous numeric → regression
    return dict(task="regression",
                reason=f"Target is continuous numeric ({n_unique} unique values, dtype={dtype_str}).",
                n_classes=None,
                class_labels=None,
                target_dtype=target_dtype)


def ml_readiness_score(
    df: pd.DataFrame,
    mv_df: pd.DataFrame,
    quality: dict,
    dup_info: dict,
    outlier_results: dict,
    target_col: str,
    task_info: dict,
) -> dict:
    """
    Score 0–100 with task-aware checks.

    Universal checks (apply to all tasks):
      • Missing values
      • Duplicate rows
      • Constant / near-constant columns
      • High-cardinality columns that need encoding
      • Feature-to-sample-size ratio (curse of dimensionality)
      • Multivariate outliers

    Classification-specific checks:
      • Class imbalance (ratio of majority to minority class)
      • Too few samples per class
      • Target leakage risk (feature perfectly correlated with target)

    Regression-specific checks:
      • Target skewness (predicting a skewed target is harder)
      • Target outliers (extreme values damage regression models)
      • Low variance in target (nothing to predict)
    """
    score      = 100
    deductions = []
    bonuses    = []
    task       = task_info.get("task", "unknown")

    def _deduct(pts, reason):
        nonlocal score
        score -= pts
        deductions.append((reason, -pts))

    def _bonus(pts, reason):
        nonlocal score
        score += pts
        bonuses.append((reason, +pts))

    # ── UNIVERSAL CHECKS ────────────────────────────────────────────────
    # 1. Missing values
    avg_miss = mv_df["missing_pct"].mean() if not mv_df.empty else 0
    cols_high_miss = len(quality.get("high_missing_cols", []))
    if avg_miss > 30:      _deduct(20, f"High avg missing across columns ({avg_miss:.1f}%)")
    elif avg_miss > 10:    _deduct(10, f"Moderate avg missing ({avg_miss:.1f}%)")
    if cols_high_miss > 0: _deduct(min(10, cols_high_miss * 3),
                                   f"{cols_high_miss} column(s) have ≥{CFG['HIGH_MISSING_PCT']:.0f}% missing")

    # 2. Duplicate rows
    dup_pct = dup_info.get("full_duplicate_pct", 0)
    if dup_pct > 10:   _deduct(10, f"High duplicate rows ({dup_pct}%)")
    elif dup_pct > 2:  _deduct(5,  f"Moderate duplicate rows ({dup_pct}%)")

    # 3. Constant / near-constant columns
    n_const = len(quality.get("constant_cols", []))
    n_near  = len(quality.get("near_constant_cols", []))
    if n_const: _deduct(min(10, n_const * 3), f"{n_const} constant column(s) — zero information")
    if n_near:  _deduct(min(5,  n_near  * 2), f"{n_near} near-constant column(s)")

    # 4. Skewed features (harder to model without transforms)
    n_skewed = len(quality.get("highly_skewed", []))
    if n_skewed > 5:   _deduct(8, f"{n_skewed} highly skewed features — transforms likely needed")
    elif n_skewed > 2: _deduct(4, f"{n_skewed} moderately skewed features")

    # 5. Feature-to-sample ratio (curse of dimensionality)
    n_features = len(df.select_dtypes(include=[np.number]).columns)
    n_samples  = len(df)
    ratio = n_features / max(n_samples, 1)
    if ratio > 0.5:    _deduct(15, f"Very high feature/sample ratio ({n_features} features, {n_samples} rows) — risk of overfitting")
    elif ratio > 0.1:  _deduct(7,  f"High feature/sample ratio ({n_features} features, {n_samples} rows)")

    # 6. Multivariate outliers
    iso = outlier_results.get("iso_forest_count")
    if iso and iso / max(n_samples, 1) > 0.10:
        _deduct(5, f"Many multivariate outliers detected ({iso}) by Isolation Forest")

    # 7. Bonus: large clean dataset
    if n_samples >= 10_000 and avg_miss < 5:
        _bonus(5, f"Large, mostly-complete dataset ({n_samples:,} rows)")

    # ── CLASSIFICATION-SPECIFIC CHECKS ──────────────────────────────────
    if task in ("binary_classification", "multiclass_classification") and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=True)

        # Class imbalance
        if len(vc) >= 2:
            imbalance_ratio = vc.iloc[0] / vc.iloc[-1]
            if imbalance_ratio > 20:   _deduct(15, f"Severe class imbalance (ratio={imbalance_ratio:.1f}x) — SMOTE/resampling critical")
            elif imbalance_ratio > 10: _deduct(10, f"High class imbalance (ratio={imbalance_ratio:.1f}x) — SMOTE recommended")
            elif imbalance_ratio > 3:  _deduct(5,  f"Moderate class imbalance (ratio={imbalance_ratio:.1f}x)")
            else:                      _bonus(3,   f"Well-balanced classes (ratio={imbalance_ratio:.1f}x)")

        # Samples per class (need at least ~50 per class for reliable training)
        min_class_count = int(vc.min())
        if min_class_count < 10:   _deduct(15, f"Minority class has only {min_class_count} sample(s) — far too few")
        elif min_class_count < 50: _deduct(8,  f"Minority class has only {min_class_count} samples — may cause poor generalisation")
        elif min_class_count >= 200: _bonus(3, f"Minority class well-represented ({min_class_count} samples)")

        # Too many classes for multiclass
        if task == "multiclass_classification" and task_info["n_classes"] > 50:
            _deduct(10, f"Very many classes ({task_info['n_classes']}) — consider hierarchical or label grouping")

    # ── REGRESSION-SPECIFIC CHECKS ───────────────────────────────────────
    if task == "regression" and target_col in df.columns:
        target_vals = df[target_col].dropna()

        # Target skewness — hard to predict a heavily skewed continuous target
        t_skew = abs(target_vals.skew())
        if t_skew > 3:    _deduct(10, f"Target '{target_col}' is severely skewed (skew={t_skew:.2f}) — log/Box-Cox transform recommended")
        elif t_skew > 1:  _deduct(5,  f"Target '{target_col}' is moderately skewed (skew={t_skew:.2f})")
        else:             _bonus(3,   f"Target '{target_col}' is approximately normal (skew={t_skew:.2f})")

        # Target outliers via IQR
        Q1, Q3   = target_vals.quantile(.25), target_vals.quantile(.75)
        IQR      = Q3 - Q1
        t_out    = ((target_vals < Q1 - 3*IQR) | (target_vals > Q3 + 3*IQR)).sum()
        t_out_pct= t_out / len(target_vals) * 100
        if t_out_pct > 5:  _deduct(8,  f"Target has {t_out} extreme outliers ({t_out_pct:.1f}%) — will distort regression")
        elif t_out_pct > 1:_deduct(4,  f"Target has {t_out} outliers ({t_out_pct:.1f}%)")

        # Near-zero variance in target — nothing to predict
        t_cv = target_vals.std() / abs(target_vals.mean()) if target_vals.mean() != 0 else 0
        if t_cv < 0.01:    _deduct(15, f"Target '{target_col}' has near-zero variance (CV={t_cv:.4f}) — nothing meaningful to predict")

    # ── FINAL SCORE ──────────────────────────────────────────────────────
    score = max(0, min(100, score))
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 45 else "F"

    # Task-specific recommended algorithms
    algo_map = {
        "binary_classification":    ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "SVM"],
        "multiclass_classification":["Random Forest", "XGBoost", "LightGBM", "Softmax Regression", "KNN"],
        "regression":               ["Linear Regression", "Ridge/Lasso", "Random Forest Regressor", "XGBoost Regressor", "SVR"],
        "unknown":                  ["— (no target column provided)"],
    }
    recommended_algos = algo_map.get(task, [])

    result = dict(
        score              = score,
        grade              = grade,
        task               = task,
        task_reason        = task_info.get("reason", ""),
        n_classes          = task_info.get("n_classes"),
        deductions         = deductions,
        bonuses            = bonuses,
        recommended_algos  = recommended_algos,
    )

    _print_section("ML TASK DETECTION  +  ML READINESS SCORE")
    task_label = task.replace("_", " ").title()
    print(f"  Detected Task   : {task_label}")
    print(f"  Reason          : {task_info.get('reason','—')}")
    if task_info.get("n_classes"):
        print(f"  Classes         : {task_info['n_classes']}  → {task_info.get('class_labels','')}")
    print(f"\n  ML Readiness    : {score}/100  (Grade: {grade})")
    print(f"\n  Deductions:")
    for reason, pts in deductions: print(f"    {pts:+d}  {reason}")
    if bonuses:
        print(f"\n  Bonuses:")
        for reason, pts in bonuses: print(f"    {pts:+d}  {reason}")
    print(f"\n  Recommended Algorithms:")
    for algo in recommended_algos: print(f"    • {algo}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  18.  FEATURE IMPORTANCE PROXY  &  MEMORY HINTS
# ═════════════════════════════════════════════════════════════════════════════
def feature_importance_proxy(df: pd.DataFrame) -> pd.Series:
    log.info("Computing feature importance proxy …")
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num_df.empty:
        return pd.Series(dtype=float)
    var    = num_df.var()
    ranked = (var / var.sum()).sort_values(ascending=False)
    _print_section("FEATURE IMPORTANCE PROXY  (Variance Ranking)")
    print(ranked.to_string())
    return ranked


def memory_optimization_hints(df: pd.DataFrame) -> list:
    hints = []
    for col in df.select_dtypes("int64").columns:
        mn, mx = int(df[col].min()), int(df[col].max())
        if   0 <= mn and mx <= 255:         hints.append((col, "uint8"))
        elif -128 <= mn and mx <= 127:      hints.append((col, "int8"))
        elif -32768 <= mn and mx <= 32767:  hints.append((col, "int16"))
        elif mn >= 0 and mx <= 65535:       hints.append((col, "uint16"))
    for col in df.select_dtypes("float64").columns:
        hints.append((col, "float32"))
    for col in df.select_dtypes("object").columns:
        if df[col].nunique() / len(df) < 0.50:
            hints.append((col, "category"))
    return hints


# ═════════════════════════════════════════════════════════════════════════════
#  19.  AUTOMATED INSIGHTS  (comprehensive)
# ═════════════════════════════════════════════════════════════════════════════
def generate_insights(
    df, mv_df, num_stats, corr_results, dup_info, quality,
    outlier_results, imbalance, ml_score, stat_tests,
) -> list:
    log.info("Generating automated insights …")
    insights = []

    # Missing
    for col, row in mv_df.iterrows():
        p = row["missing_pct"]
        if p >= 70: insights.append(f"🔴 Column '{col}' has {p:.0f}% missing — strongly consider dropping.")
        elif p >= 40: insights.append(f"🟠 Column '{col}' has {p:.0f}% missing — imputation needed.")
        elif p >= 10: insights.append(f"🟡 Column '{col}' has {p:.1f}% missing values.")

    # Skewness & normality
    if not num_stats.empty:
        for col, row in num_stats.iterrows():
            sk = row.get("skewness", 0)
            if abs(sk) >= 2:     insights.append(f"📊 '{col}' is severely skewed (skew={sk:.2f}) — log/Box-Cox transform recommended.")
            elif abs(sk) >= 1:   insights.append(f"📊 '{col}' is moderately skewed (skew={sk:.2f}).")
            if row.get("is_normal","") == "✘ Non-normal":
                insights.append(f"🔬 '{col}' failed normality tests — use non-parametric methods or transforms.")
            outs = row.get("outlier_iqr", 0)
            if outs > 0:
                pct_o = row.get("outlier_iqr_pct", 0)
                insights.append(f"🔍 '{col}' has {int(outs)} outliers ({pct_o:.1f}% of rows) by IQR.")

    # Correlations
    for a, b, v in corr_results.get("high_corr_pairs", []):
        insights.append(f"🔗 '{a}' ↔ '{b}' are highly correlated (Pearson r={v}) — risk of multicollinearity.")

    # Duplicates
    if dup_info["full_duplicate_count"] > 0:
        insights.append(f"♻ {dup_info['full_duplicate_count']:,} duplicate rows ({dup_info['full_duplicate_pct']}%) detected.")

    # Quality
    for col in quality.get("constant_cols",[]): insights.append(f"❌ '{col}' is constant — no predictive value.")
    for col in quality.get("near_constant_cols",[]): insights.append(f"⚠ '{col}' is near-constant (>99% same value).")
    for col in quality.get("potential_id_cols",[]): insights.append(f"🆔 '{col}' has all-unique values — likely an ID column, exclude from modelling.")
    for col in quality.get("negative_where_positive_expected",[]): insights.append(f"⚠ '{col}' contains negative values where positives are expected.")
    for col in quality.get("mixed_type_cols",[]): insights.append(f"⚠ '{col}' appears to have mixed data types.")

    # Imbalance
    if imbalance and imbalance.get("smote_needed"):
        insights.append(f"⚖ Target '{imbalance['target_col']}' is imbalanced (ratio={imbalance['imbalance_ratio']}) — consider SMOTE/oversampling.")

    # Multivariate outliers
    iso = outlier_results.get("iso_forest_count")
    lof = outlier_results.get("lof_count")
    if iso: insights.append(f"🌲 Isolation Forest detected {iso} multivariate outliers.")
    if lof: insights.append(f"📍 LOF detected {lof} local outliers.")

    # ANOVA
    sig = [r for r in stat_tests.get("anova",[]) if r["significant"]]
    if sig: insights.append(f"📐 {len(sig)} ANOVA tests significant — these features differ across classes.")

    # ML score
    if ml_score:
        g = ml_score["grade"]
        s = ml_score["score"]
        emoji = "🟢" if s>=75 else "🟡" if s>=60 else "🔴"
        insights.append(f"{emoji} ML Readiness Score: {s}/100 (Grade {g}).")

    _print_section("AUTOMATED INSIGHTS")
    for i, ins in enumerate(insights, 1): print(f"  {i:02d}. {ins}")
    return insights


# ═════════════════════════════════════════════════════════════════════════════
#  20.  CSV EXPORT OF ALL STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
def export_stats_csv(num_stats, mv_df, output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    if not num_stats.empty:
        num_stats.to_csv(f"{output_dir}/numerical_stats.csv")
    if not mv_df.empty:
        mv_df.to_csv(f"{output_dir}/missing_values.csv")
    log.info("Stats CSV files saved to '%s/'", output_dir)


# ═════════════════════════════════════════════════════════════════════════════
#  PRINT HELPER
# ═════════════════════════════════════════════════════════════════════════════
def _print_section(title: str):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")


# ═════════════════════════════════════════════════════════════════════════════
#  21.  HTML REPORT  (tabbed, dark-mode, professional)
# ═════════════════════════════════════════════════════════════════════════════
def generate_html_report(
    filepath, df, overview, mv_df, mv_charts,
    num_stats, num_charts,
    cat_results, cat_charts,
    dt_results, dt_charts,
    text_results,
    corr_results, corr_charts,
    outlier_results, outlier_charts,
    stat_tests,
    pca_img, pca_var_df,
    tsne_img,
    pairplot_img,
    imbalance_info, imbalance_img,
    feat_target_imgs,
    dup_info, quality,
    insights, ml_score,
    feat_importance,
    mem_hints,
    output_path,
):
    log.info("Building HTML report …")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── helpers ──
    def df2html(dfo, max_rows=500):
        if dfo is None or (hasattr(dfo,"empty") and dfo.empty):
            return "<p><em>No data.</em></p>"
        return dfo.head(max_rows).to_html(classes="tbl", border=0, table_id=f"t{id(dfo)}")

    def charts_row(items):
        """items = list of (title, b64) tuples."""
        if not items: return ""
        return '<div class="chart-row">' + "".join(_img_tag(b, t) for t, b in items if b) + "</div>"

    def kpi(val, label, color="#3498db"):
        return f'<div class="kpi" style="border-top:3px solid {color}"><div class="kv">{val}</div><div class="kl">{label}</div></div>'

    # ── KPI bar ──
    score_color = "#27ae60" if ml_score.get("score",0)>=75 else "#e67e22" if ml_score.get("score",0)>=60 else "#e74c3c"
    kpis = (
        kpi(f"{overview['rows']:,}",    "Rows",           "#3498db") +
        kpi(overview["columns"],        "Columns",         "#9b59b6") +
        kpi(overview["memory_mb"],      "Memory MB",       "#1abc9c") +
        kpi(len(overview["num_cols"]),  "Numeric Cols",    "#2980b9") +
        kpi(len(overview["cat_cols"]),  "Categorical",     "#e67e22") +
        kpi(len(overview["dt_cols"]),   "Datetime",        "#16a085") +
        kpi(dup_info["full_duplicate_count"],  "Duplicates","#e74c3c") +
        kpi(len(mv_df),                 "Cols w/ Missing", "#c0392b") +
        kpi(f"{ml_score.get('score',0)}/100  ({ml_score.get('grade','—')})",
            "ML Readiness",  score_color)
    )

    # ── dtype table ──
    dtype_rows = "".join(f"<tr><td>{c}</td><td>{t}</td></tr>" for c,t in overview["dtypes"].items())
    dtype_tbl  = f'<table class="tbl"><tr><th>Column</th><th>Type</th></tr>{dtype_rows}</table>'

    # ── quality table ──
    def _q(k, label):
        v = quality.get(k, [])
        vstr = ", ".join(str(x) for x in v) if v else "<span style='color:green'>✔ None</span>"
        return f"<tr><td>{label}</td><td>{vstr}</td></tr>"

    q_tbl = f"""<table class='tbl'>
        {_q('constant_cols','Constant columns')}
        {_q('near_constant_cols','Near-constant columns')}
        {_q('high_missing_cols','High-missing columns (≥{:.0f}%)'.format(CFG['HIGH_MISSING_PCT']))}
        {_q('high_cardinality','High-cardinality categorical')}
        {_q('potential_id_cols','Potential ID columns')}
        {_q('highly_skewed','Highly skewed columns')}
        {_q('mixed_type_cols','Mixed-type columns')}
        {_q('negative_where_positive_expected','Unexpected negatives')}
    </table>"""

    # ── insights list ──
    ins_html = "<ol class='ins-list'>" + "".join(f"<li>{i}</li>" for i in insights) + "</ol>" if insights else "<p>No insights.</p>"

    # ── memory hints ──
    hint_rows = "".join(f"<tr><td>{c}</td><td>Cast to <code>{t}</code></td></tr>" for c,t in mem_hints)
    hint_tbl  = (f"<table class='tbl'><tr><th>Column</th><th>Suggestion</th></tr>{hint_rows}</table>"
                 if hint_rows else "<p>No obvious optimisation hints.</p>")

    # ── corr tables ──
    def hc_tbl():
        rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{v}</td></tr>" for a,b,v in corr_results.get("high_corr_pairs",[]))
        return (f"<table class='tbl'><tr><th>Feature A</th><th>Feature B</th><th>Pearson r</th></tr>{rows}</table>"
                if rows else "<p style='color:green'>✔ No highly correlated pairs.</p>")

    def test_tbl(key, cols):
        rows = stat_tests.get(key, [])
        sig  = [r for r in rows if r.get("significant")]
        if not sig: return "<p style='color:green'>✔ No significant results.</p>"
        thead = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        tbody = "".join("<tr>" + "".join(f"<td>{r.get(c,'')}</td>" for c in cols) + "</tr>" for r in sig[:50])
        return f"<table class='tbl'>{thead}{tbody}</table>"

    # ── outlier summary ──
    iqr_d  = outlier_results.get("iqr_outlier_counts",{})
    z_d    = outlier_results.get("zscore_outlier_counts",{})
    out_rows = "".join(
        f"<tr><td>{c}</td><td>{iqr_d.get(c,0)}</td><td>{z_d.get(c,0)}</td></tr>"
        for c in set(list(iqr_d)+list(z_d)) if iqr_d.get(c,0)+z_d.get(c,0)>0
    )
    out_tbl = (f"<table class='tbl'><tr><th>Column</th><th>IQR Outliers</th><th>Z-score Outliers</th></tr>{out_rows}</table>"
               if out_rows else "<p style='color:green'>✔ No notable outliers.</p>")

    # ── feature importance ──
    fi_rows  = "".join(f"<tr><td>{c}</td><td>{v:.4f}</td></tr>" for c,v in feat_importance.items())
    fi_tbl   = (f"<table class='tbl'><tr><th>Feature</th><th>Variance Weight</th></tr>{fi_rows}</table>"
                if fi_rows else "<p>No numerical columns.</p>")

    # ── PCA table ──
    pca_tbl = df2html(pca_var_df) if not pca_var_df.empty else "<p>N/A</p>"

    # ── imbalance section ──
    if imbalance_info:
        imb_rows = "".join(
            f"<tr><td>{cls}</td><td>{cnt:,}</td><td>{imbalance_info['class_pct'].get(cls,0):.1f}%</td></tr>"
            for cls, cnt in imbalance_info["class_counts"].items()
        )
        imb_html = (f"<p><strong>Target:</strong> {imbalance_info['target_col']} &nbsp;|&nbsp; "
                    f"<strong>Ratio:</strong> {imbalance_info['imbalance_ratio']} &nbsp;|&nbsp; "
                    f"<strong>SMOTE needed:</strong> {'Yes ⚠' if imbalance_info['smote_needed'] else 'No ✔'}</p>"
                    f"<table class='tbl'><tr><th>Class</th><th>Count</th><th>%</th></tr>{imb_rows}</table>")
        if imbalance_img: imb_html += _img_tag(imbalance_img, "Class Distribution")
    else:
        imb_html = "<p>No target column identified.</p>"

    # ── text analysis table ──
    if text_results:
        t_rows = "".join(
            f"<tr><td>{c}</td><td>{v['avg_length']}</td><td>{v['avg_words']}</td>"
            f"<td>{v['email_count']}</td><td>{v['url_count']}</td><td>{v['phone_count']}</td>"
            f"<td>{v['has_digits_pct']}%</td></tr>"
            for c, v in text_results.items()
        )
        txt_tbl = (f"<table class='tbl'><tr><th>Column</th><th>Avg Len</th><th>Avg Words</th>"
                   f"<th>Emails</th><th>URLs</th><th>Phones</th><th>Has Digits%</th></tr>{t_rows}</table>")
    else:
        txt_tbl = "<p>No text columns.</p>"

    # ── datetime table ──
    if dt_results:
        dt_rows = "".join(
            f"<tr><td>{c}</td><td>{v['min_date']}</td><td>{v['max_date']}</td>"
            f"<td>{v['date_range']}</td><td>{v['n_unique']:,}</td><td>{v['missing_cnt']}</td></tr>"
            for c, v in dt_results.items()
        )
        dt_tbl = (f"<table class='tbl'><tr><th>Column</th><th>Min</th><th>Max</th>"
                  f"<th>Range</th><th>Unique</th><th>Missing</th></tr>{dt_rows}</table>")
    else:
        dt_tbl = "<p>No datetime columns detected.</p>"

    # ── pb correlation ──
    pb_df  = corr_results.get("point_biserial", pd.DataFrame())
    pb_html= df2html(pb_df) if not pb_df.empty else "<p>No binary categorical columns found.</p>"

    # ══════════════ CSS + JS ═════════════════════════════════════════════
    CSS = """
    <style>
    :root{--bg:#f0f2f8;--card:#fff;--text:#1a1a2e;--accent:#3d5af1;--accent2:#e94560;
          --border:#dee2ef;--th:#eef0fb;--green:#27ae60;--orange:#e67e22;--red:#e74c3c;
          --tab-active:#3d5af1;--tab-bg:#eef0fb;--shadow:0 4px 18px rgba(0,0,0,.09)}
    [data-theme=dark]{--bg:#12121f;--card:#1e1e30;--text:#e0e0f0;--border:#2e2e4a;
          --th:#252538;--tab-bg:#252538}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:14px;transition:background .3s,color .3s}
    a{color:var(--accent)}
    /* Header */
    header{background:linear-gradient(135deg,#1a1464,#3d5af1 60%,#e94560);color:#fff;padding:28px 40px;display:flex;align-items:center;justify-content:space-between}
    header h1{font-size:24px;font-weight:700;letter-spacing:.3px}
    header p{font-size:12px;opacity:.85;margin-top:4px}
    .dark-btn{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.4);color:#fff;padding:7px 16px;border-radius:20px;cursor:pointer;font-size:12px;transition:.2s}
    .dark-btn:hover{background:rgba(255,255,255,.3)}
    /* Container */
    .container{max-width:1280px;margin:24px auto;padding:0 20px}
    /* KPI */
    .kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:14px;margin-bottom:22px}
    .kpi{background:var(--card);border-radius:10px;padding:16px 14px;text-align:center;box-shadow:var(--shadow)}
    .kv{font-size:20px;font-weight:700;color:var(--accent)}
    .kl{font-size:11px;color:#888;margin-top:4px}
    /* Tabs */
    .tab-bar{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:0;border-bottom:2px solid var(--border)}
    .tab-btn{padding:10px 18px;border:none;background:var(--tab-bg);color:var(--text);cursor:pointer;border-radius:8px 8px 0 0;font-size:13px;font-weight:500;transition:.2s}
    .tab-btn:hover{background:var(--accent);color:#fff}
    .tab-btn.active{background:var(--tab-active);color:#fff}
    .tab-panel{display:none;background:var(--card);border-radius:0 8px 8px 8px;padding:24px 26px;box-shadow:var(--shadow)}
    .tab-panel.active{display:block}
    /* Section heading */
    h2{font-size:17px;color:var(--accent);font-weight:700;margin-bottom:14px;padding-bottom:6px;border-bottom:2px solid var(--border)}
    h3{font-size:14px;font-weight:600;color:var(--accent2);margin:16px 0 8px}
    /* Tables */
    .tbl{border-collapse:collapse;width:100%;font-size:12.5px}
    .tbl th,.tbl td{padding:7px 10px;border:1px solid var(--border);white-space:nowrap}
    .tbl th{background:var(--th);font-weight:600;position:sticky;top:0;z-index:1}
    .tbl tr:nth-child(even){background:var(--th)}
    .overflow{overflow-x:auto;max-height:420px;overflow-y:auto;border-radius:6px}
    /* Charts */
    .chart-row{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}
    figure.chart-wrap{display:inline-block;border:1px solid var(--border);border-radius:8px;background:var(--card);padding:10px;max-width:620px}
    figure.chart-wrap figcaption{font-size:11px;font-weight:600;color:#888;margin-bottom:6px;text-align:center}
    figure.chart-wrap img{max-width:100%;display:block;border-radius:4px}
    /* Insights */
    .ins-list{list-style:none;padding:0}
    .ins-list li{padding:9px 14px 9px 16px;border-left:4px solid var(--accent);background:var(--th);border-radius:0 6px 6px 0;margin-bottom:7px;line-height:1.5}
    /* Score gauge */
    .score-bar{height:18px;border-radius:9px;background:linear-gradient(90deg,#e74c3c,#e67e22,#f1c40f,#27ae60);width:100%;position:relative;margin:10px 0}
    .score-needle{position:absolute;top:-4px;width:4px;height:26px;background:#1a1a2e;border-radius:2px;transform:translateX(-50%)}
    /* Badge */
    .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700}
    .bg-green{background:#d5f5e3;color:#145a32}
    .bg-red{background:#fadbd8;color:#922b21}
    .bg-orange{background:#fdebd0;color:#784212}
    footer{text-align:center;padding:20px;color:#aaa;font-size:12px}
    </style>"""

    JS = """
    <script>
    // Tab switching
    function showTab(tabId, btn) {
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        btn.classList.add('active');
    }
    // Dark mode
    function toggleDark() {
        const b = document.body;
        const t = b.getAttribute('data-theme') === 'dark';
        b.setAttribute('data-theme', t ? 'light' : 'dark');
        document.querySelector('.dark-btn').textContent = t ? '🌙 Dark Mode' : '☀ Light Mode';
    }
    // DataTable search
    function filterTable(inputId, tableId) {
        const val = document.getElementById(inputId).value.toLowerCase();
        document.querySelectorAll('#' + tableId + ' tbody tr').forEach(r => {
            r.style.display = r.textContent.toLowerCase().includes(val) ? '' : 'none';
        });
    }
    </script>"""

    # ── score gauge ──
    s = ml_score.get("score", 0)
    grade = ml_score.get("grade", "—")
    needle_pct = s
    score_color_map = {range(75,101):"#27ae60", range(60,75):"#e67e22", range(0,60):"#e74c3c"}
    sc = "#e74c3c"
    for rng, col in score_color_map.items():
        if s in rng: sc = col; break
    ml_html = f"""
    <p style="font-size:28px;font-weight:700;color:{sc}">{s}/100 &nbsp; <span style="font-size:20px">Grade: {grade}</span></p>
    <div class='score-bar'><div class='score-needle' style='left:{needle_pct}%'></div></div>
    <ul style='margin-top:12px'>
    {"".join(f"<li style='margin:4px 0'><span class='badge {'bg-red' if pts<0 else 'bg-green'}'>{pts:+d}</span>  {r}</li>" for r, pts in ml_score.get("deductions",[]))}
    </ul>"""

    # ══════════════ TAB DEFINITIONS ════════════════════════════════════
    tabs = [
        ("tab-overview",  "📋 Overview"),
        ("tab-missing",   "❓ Missing"),
        ("tab-numerical", "🔢 Numerical"),
        ("tab-categorical","🏷 Categorical"),
        ("tab-datetime",  "📅 Datetime"),
        ("tab-text",      "📝 Text"),
        ("tab-outliers",  "🎯 Outliers"),
        ("tab-corr",      "🔗 Correlation"),
        ("tab-tests",     "🧪 Stat Tests"),
        ("tab-dimred",    "🌀 Dim Reduction"),
        ("tab-imbalance", "⚖ Imbalance"),
        ("tab-quality",   "✅ Quality"),
        ("tab-insights",  "💡 Insights"),
        ("tab-mlscore",   "🏆 ML Score"),
        ("tab-memory",    "💾 Memory"),
    ]

    tab_bar = '<div class="tab-bar">' + "".join(
        f'<button class="tab-btn{"  active" if i==0 else ""}" onclick="showTab(\'{tid}\', this)">{lbl}</button>'
        for i,(tid,lbl) in enumerate(tabs)
    ) + "</div>"

    # ── build each panel ──
    panels = {
        "tab-overview": f"""
            <h2>Dataset Overview</h2>
            <div class='kpi-grid'>{kpis}</div>
            <p style='font-size:11px;color:#888;margin-bottom:12px'>
              SHA-256 (16): <code>{overview['hash_sha256']}</code> &nbsp;|&nbsp; Profiled: {now}
            </p>
            <h3>Column Data Types</h3><div class='overflow'>{dtype_tbl}</div>
            <h3>First 5 Rows</h3><div class='overflow'>{df2html(overview['head'])}</div>
            <h3>Last 5 Rows</h3><div class='overflow'>{df2html(overview['tail'])}</div>
        """,

        "tab-missing": f"""
            <h2>Missing Value Analysis</h2>
            {"<p style='color:green;font-weight:600'>✔ No missing values found.</p>" if mv_df.empty
             else "<div class='overflow'>" + df2html(mv_df) + "</div>"}
            {charts_row([("Missing % per Column", mv_charts[0]),
                         ("Missing Pattern Heatmap", mv_charts[1]),
                         ("Nullity Dendrogram", mv_charts[2])])}
        """,

        "tab-numerical": f"""
            <h2>Numerical Feature Statistics</h2>
            <div class='overflow'>{df2html(num_stats)}</div>
            <h3>Histograms + KDE</h3>
            {charts_row([(c,b) for c,b in num_charts.get('histograms',[])])}
            <h3>Boxplots</h3>
            {charts_row([(c,b) for c,b in num_charts.get('boxplots',[])])}
            <h3>Violin Plots</h3>
            {charts_row([(c,b) for c,b in num_charts.get('violins',[])])}
            <h3>Q-Q Plots</h3>
            {charts_row([(c,b) for c,b in num_charts.get('qq',[])])}
            <h3>ECDF Plots</h3>
            {charts_row([(c,b) for c,b in num_charts.get('ecdf',[])])}
        """,

        "tab-categorical": f"""
            <h2>Categorical Feature Analysis</h2>
            {''.join(
                f"<h3>{col}</h3><table class='tbl'><tr><th>Category</th><th>Count</th></tr>" +
                "".join(f"<tr><td>{k}</td><td>{v:,}</td></tr>" for k,v in info['top_n'].items()) +
                f"</table><p style='font-size:11px;margin:4px 0 14px'>Unique: {info['unique_count']} | Entropy: {info['entropy']} | {'🆔 Likely ID' if info['is_id_col'] else ''} {'🔲 Binary' if info['is_binary'] else ''}</p>"
                for col, info in cat_results.items()
            )}
            <h3>Bar + Pie Charts</h3>
            {charts_row([(c,b) for c,b in cat_charts])}
        """,

        "tab-datetime": f"""
            <h2>Datetime Analysis</h2>
            {dt_tbl}
            {charts_row([(c,b) for c,b in dt_charts])}
        """,

        "tab-text": f"""
            <h2>Text / String Analysis</h2>
            {txt_tbl}
        """,

        "tab-outliers": f"""
            <h2>Outlier Analysis</h2>
            {out_tbl}
            <p style='margin-top:10px'>
              <strong>Isolation Forest (multivariate):</strong> {outlier_results.get('iso_forest_count','N/A')} &nbsp;|&nbsp;
              <strong>LOF:</strong> {outlier_results.get('lof_count','N/A')}
            </p>
            {charts_row([(c,b) for c,b in outlier_charts])}
        """,

        "tab-corr": f"""
            <h2>Correlation Analysis</h2>
            <h3>Highly Correlated Pairs (Pearson |r| ≥ {CFG['HIGH_CORR']})</h3>
            {hc_tbl()}
            <h3>Pearson Heatmap</h3>{_img_tag(corr_charts.get('pearson',''), 'Pearson')}
            <h3>Spearman Heatmap</h3>{_img_tag(corr_charts.get('spearman',''), 'Spearman')}
            <h3>Kendall Heatmap</h3>{_img_tag(corr_charts.get('kendall',''), 'Kendall')}
            <h3>Cramér's V  (Categorical–Categorical)</h3>{_img_tag(corr_charts.get('cramers_v',''), "Cramér's V")}
            <h3>Point-Biserial Correlations</h3>{pb_html}
        """,

        "tab-tests": f"""
            <h2>Statistical Tests</h2>
            <h3>ANOVA  (Significant only — α={CFG['ANOVA_ALPHA']})</h3>
            {test_tbl('anova', ['cat_col','num_col','F','p','significant'])}
            <h3>Chi-Squared  (Significant only — α={CFG['CHI2_ALPHA']})</h3>
            {test_tbl('chi2', ['col1','col2','chi2','p','dof','significant'])}
        """,

        "tab-dimred": f"""
            <h2>Dimensionality Reduction</h2>
            <h3>PCA — Variance Explained</h3>
            {_img_tag(pca_img, 'PCA Scree Plot')}
            <div class='overflow'>{pca_tbl}</div>
            <h3>Pairplot (top-{CFG['MAX_PAIRPLOT_COLS']} numerical features)</h3>
            {_img_tag(pairplot_img, 'Pairplot')}
            <h3>t-SNE 2D Embedding</h3>
            {_img_tag(tsne_img, 't-SNE')}
            <h3>Feature–Target Relationships</h3>
            {charts_row([(t,b) for t,b in feat_target_imgs])}
        """,

        "tab-imbalance": f"""
            <h2>Class Imbalance Analysis</h2>
            {imb_html}
        """,

        "tab-quality": f"""
            <h2>Data Quality Checks</h2>
            {q_tbl}
        """,

        "tab-insights": f"""
            <h2>Automated Insights</h2>
            {ins_html}
        """,

        "tab-mlscore": f"""
            <h2>ML Readiness Score</h2>
            {ml_html}
            <h3>Feature Importance Proxy (Variance Ranking)</h3>
            {fi_tbl}
        """,

        "tab-memory": f"""
            <h2>Memory Optimisation Suggestions</h2>
            {hint_tbl}
        """,
    }

    panels_html = "".join(
        f'<div id="{tid}" class="tab-panel{"  active" if i==0 else ""}">{panels.get(tid,"")}</div>'
        for i,(tid,_) in enumerate(tabs)
    )

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Data Profile Report — {Path(filepath).name}</title>
  {CSS}
</head>
<body>
<header>
  <div>
    <h1>📊 Ultimate Data Profile Report</h1>
    <p>Source: <strong>{Path(filepath).name}</strong> &nbsp;|&nbsp; Profiled: {now}</p>
  </div>
  <button class='dark-btn' onclick='toggleDark()'>🌙 Dark Mode</button>
</header>
<div class="container">
  {tab_bar}
  {panels_html}
</div>
<footer>Generated by Ultimate Data Profiler v2.0 &nbsp;|&nbsp; {now}</footer>
{JS}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("HTML report saved → %s", output_path)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    _print_section("ULTIMATE DATA PROFILER  v2.0")
    print("  The only profiling script a data scientist will ever need.")

    filepath = input("\n  Enter dataset path (CSV / Excel / JSON / Parquet / TSV): ")

    # Load
    df = load_dataset(filepath)

    # Overview
    overview = basic_overview(df)

    # Ask user for target column
    print(f"\n  Available columns: {df.columns.tolist()}")
    target_input = input("\n  Enter target column name (or press Enter to skip): ")

    if target_input and target_input in df.columns:
        target_col = target_input
        print(f"  🎯 Target column set to: '{target_col}'")
    elif target_input and target_input not in df.columns:
        print(f"  ⚠ Column '{target_input}' not found — skipping target analysis.")
        target_col = ""
    else:
        target_col = ""
        print("  ℹ No target column selected — skipping target-specific analysis.")

    # Missing
    mv_df    = missing_value_analysis(df)
    mv_charts= plot_missing_charts(df)

    # Numerical
    num_stats = numerical_analysis(df)
    num_charts= plot_numerical_charts(df)

    # Categorical
    cat_results = categorical_analysis(df)
    cat_charts  = plot_cat_charts(df)

    # Datetime
    dt_results = datetime_analysis(df)
    dt_charts  = plot_datetime_charts(df, dt_results)

    # Text
    text_results = text_analysis(df)

    # Outliers
    outlier_results = outlier_analysis(df)
    outlier_charts  = plot_outlier_charts(df, outlier_results)

    # Correlation
    corr_results = correlation_analysis(df)
    corr_charts  = plot_corr_charts(corr_results)

    # Statistical tests
    stat_tests = statistical_tests(df)

    # PCA + t-SNE + Pairplot
    pca_img, pca_var_df = _safe_call(pca_analysis, df) or ("", pd.DataFrame())
    tsne_img    = _safe_call(tsne_plot, df, target_col) or ""
    pairplot_img= _safe_call(pairplot_chart, df, target_col) or ""

    # Imbalance
    imbalance_info= class_imbalance_analysis(df, target_col)
    imbalance_img = plot_class_imbalance(df, target_col)

    # Feature–target
    feat_target_imgs = feature_target_analysis(df, target_col)

    # Duplicates
    dup_info = duplicate_analysis(df)

    # Data quality
    quality = data_quality_checks(df)

    # Feature importance
    feat_importance = feature_importance_proxy(df)

    # Memory hints
    mem_hints = memory_optimization_hints(df)

    # ML task detection + readiness
    task_info = detect_ml_task(df, target_col)
    ml_score  = ml_readiness_score(df, mv_df, quality, dup_info, outlier_results, target_col, task_info)

    # Insights
    insights = generate_insights(
        df, mv_df, num_stats, corr_results, dup_info, quality,
        outlier_results, imbalance_info, ml_score, stat_tests,
    )

    # Export stats CSVs
    export_stats_csv(num_stats, mv_df, CFG["STATS_CSV_DIR"])

    # HTML report
    output_html = CFG["REPORT_FILENAME"]
    generate_html_report(
        filepath      = filepath,
        df            = df,
        overview      = overview,
        mv_df         = mv_df,
        mv_charts     = mv_charts,
        num_stats     = num_stats,
        num_charts    = num_charts,
        cat_results   = cat_results,
        cat_charts    = cat_charts,
        dt_results    = dt_results,
        dt_charts     = dt_charts,
        text_results  = text_results,
        corr_results  = corr_results,
        corr_charts   = corr_charts,
        outlier_results=outlier_results,
        outlier_charts = outlier_charts,
        stat_tests    = stat_tests,
        pca_img       = pca_img,
        pca_var_df    = pca_var_df,
        tsne_img      = tsne_img,
        pairplot_img  = pairplot_img,
        imbalance_info= imbalance_info,
        imbalance_img = imbalance_img,
        feat_target_imgs=feat_target_imgs,
        dup_info      = dup_info,
        quality       = quality,
        insights      = insights,
        ml_score      = ml_score,
        feat_importance=feat_importance,
        mem_hints     = mem_hints,
        output_path   = output_html,
    )

    _print_section("✅  DATA PROFILING COMPLETED SUCCESSFULLY")
    print(f"  📄  Report   → {output_html}")
    print(f"  📁  CSV stats→ {CFG['STATS_CSV_DIR']}/")
    print()


if __name__ == "__main__":
    main()
