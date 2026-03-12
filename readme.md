<div align="center">

<img src="https://github.com/trx98/Predicting-Customer-Annual-Spending-Using-Behavioral-Metrics/blob/main/logo_new.png?raw=true" alt="KoshurAI Logo" width="180"/>

<br/>

# 📊 Ultimate Data Profiler

### *The only data profiling script a data scientist will ever need*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-orange?style=flat-square)](https://pep8.org/)
[![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen?style=flat-square)]()
[![Developed by KoshurAI](https://img.shields.io/badge/Developed%20by-KoshurAI-8a2be2?style=flat-square)]()

<br/>

**Drop in any dataset. Get a full statistical audit, rich visualisations, ML readiness score,
and a self-contained interactive HTML report — in a single command.**

<br/>

[Features](#-features) · [Quick Start](#-quick-start) · [Report Preview](#-report-preview) · [Installation](#-installation) · [Usage](#-usage) · [Configuration](#-configuration) · [Output](#-output-structure)

---

</div>

## ✨ Features

<table>
<tr>
<td width="50%">

**📋 Dataset Overview**
- Row / column counts, memory usage
- SHA-256 fingerprint for dataset versioning
- Data type breakdown across all column categories
- First & last row preview

**❓ Missing Value Analysis**
- Per-column count, percentage, and pattern
- Smart imputation strategy recommendation
- Bar chart, heatmap, and nullity dendrogram

**🔢 Numerical Profiling**
- 15+ statistics including IQR, CV, skewness, kurtosis
- Extended percentiles (1st → 99th)
- Normality tests: Shapiro-Wilk, D'Agostino-K², Anderson-Darling
- Outlier detection: IQR, Z-score, Isolation Forest, LOF

**🏷 Categorical Profiling**
- Frequency tables, Shannon entropy, imbalance ratio
- High-cardinality and ID column auto-detection
- Bar chart + pie chart per column

</td>
<td width="50%">

**🔗 Correlation Analysis**
- Pearson, Spearman, Kendall (numerical)
- Cramér's V (categorical ↔ categorical)
- Point-Biserial (binary ↔ numerical)

**🧪 Statistical Tests**
- ANOVA (numerical vs. categorical groups)
- Chi-squared (categorical independence)

**🌀 Dimensionality Reduction**
- PCA scree plot with cumulative variance
- t-SNE 2D embedding coloured by target
- Pairplot of top-N features

**🏆 ML Readiness Score**
- Auto-detects: Binary / Multiclass Classification or Regression
- Task-specific scoring with deductions & bonuses
- Recommended algorithms per task type

</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl missingno

# 2. Run the profiler
python data_profiler_v2.py
```

The script will prompt you for two things, then run everything automatically:

```
╔══════════════════════════════════════════════════════════╗
║         ULTIMATE DATA PROFILER  v2.0                    ║
╚══════════════════════════════════════════════════════════╝

  Enter dataset path (CSV / Excel / JSON / Parquet / TSV): titanic.csv

  Available columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Age', ...]
  Enter target column name (or press Enter to skip): Survived

  🎯 Target column set to: 'Survived'
  🔍 Detected Task: Binary Classification
  ...
  ✅  Data Profiling Completed Successfully
  📄  Report   → data_profile_report.html
  📁  CSV stats → profile_stats/
```

---

## 📸 Report Preview

The output is a **single self-contained HTML file** — no server, no internet, no dependencies. Open it in any browser.

| Tab | Contents |
|-----|----------|
| 📋 Overview | KPI bar, dtypes, sample rows, dataset fingerprint |
| ❓ Missing | Per-column stats, heatmap, dendrogram |
| 🔢 Numerical | Extended stats table, histogram+KDE, boxplot, violin, Q-Q, ECDF |
| 🏷 Categorical | Frequency tables, bar + pie charts |
| 📅 Datetime | Date range, gap analysis, monthly timeline, weekday distribution |
| 📝 Text | Length, word count, email/URL/phone pattern detection |
| 🎯 Outliers | IQR + Z-score table, Isolation Forest, LOF counts, scatter plots |
| 🔗 Correlation | Pearson, Spearman, Kendall heatmaps, Cramér's V, Point-Biserial |
| 🧪 Stat Tests | ANOVA and Chi-squared significant results |
| 🌀 Dim Reduction | PCA scree, t-SNE scatter, pairplot, feature-target plots |
| ⚖ Imbalance | Class distribution, ratio, SMOTE recommendation |
| ✅ Quality | All data quality flags in one place |
| 💡 Insights | Plain-English summary of every finding |
| 🏆 ML Score | Score gauge, grade, deductions, recommended algorithms |
| 💾 Memory | Dtype downcast suggestions per column |

> **Dark mode** is built in — toggle it with the button in the top-right corner.

---

## 📦 Installation

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 1.3 | Data loading and manipulation |
| `numpy` | ≥ 1.20 | Numerical operations |
| `matplotlib` | ≥ 3.4 | Chart rendering |
| `seaborn` | ≥ 0.11 | Statistical visualisations |
| `scipy` | ≥ 1.7 | Normality tests, KDE, statistical tests |
| `scikit-learn` | ≥ 0.24 | PCA, t-SNE, Isolation Forest, LOF |
| `openpyxl` | ≥ 3.0 | Excel file support (`.xlsx`) |
| `missingno` | ≥ 0.5 | Nullity dendrogram *(optional)* |

### Install All

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl missingno
```

### Minimal Install (no ML features)

```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl
```

> **Graceful degradation** — if `scikit-learn` or `missingno` are not installed, the corresponding sections (PCA, t-SNE, Isolation Forest, LOF, dendrogram) are silently skipped. Everything else still runs.

---

## 🖥 Usage

### Running the Script

```bash
python data_profiler_v2.py
```

That's it. The script will interactively ask you two questions and then run everything automatically:

**Step 1 — Enter your dataset path:**
```
Enter dataset path (CSV / Excel / JSON / Parquet / TSV): path/to/your/data.csv
```

**Step 2 — Enter your target column (or skip):**
```
Available columns: ['col1', 'col2', 'col3', ...]
Enter target column name (or press Enter to skip): col3
```

Pressing **Enter** without typing a column name skips target-specific analysis (ML readiness score, class imbalance, feature–target plots) and profiles the dataset as-is.

### Supported File Formats

| Format | Extension |
|--------|-----------|
| CSV (auto-detects encoding) |  |
| TSV |  |
| Excel | ,  |
| JSON |  |
| Parquet |  |
| Feather |  |
| Pickle |  |

---

## 🏆 ML Task Detection & Readiness Score

One of the key features is the automatic detection of the ML task type, followed by a task-aware quality score.

### How Task Detection Works

```
Target column provided?
│
├── No  ──────────────────────────────────────► Unknown (skip ML checks)
│
└── Yes
    │
    ├── dtype = object / category / bool ──────► Classification
    │     └── 2 unique values ────────────────► Binary Classification
    │     └── >2 unique values ───────────────► Multiclass Classification
    │
    └── dtype = numeric
          ├── ≤20 unique values
          │   AND all whole numbers
          │   AND range < 50 ──────────────────► Classification (treated as labels)
          │
          └── otherwise ──────────────────────► Regression
```

### Scoring Breakdown

The score starts at **100** and deductions are applied per finding. Bonuses can be earned for clean, well-structured data.

#### Universal Checks *(all task types)*

| Check | Max Deduction |
|-------|--------------|
| High average missing values (>30%) | −20 |
| Columns with ≥40% missing | −3 per column |
| High duplicate row rate (>10%) | −10 |
| Constant columns | −3 per column |
| Near-constant columns (>99% same value) | −2 per column |
| Highly skewed features (>5 columns) | −8 |
| High feature-to-sample ratio | −15 |
| Many multivariate outliers (>10%) | −5 |

#### Classification-Specific Checks

| Check | Max Deduction |
|-------|--------------|
| Severe class imbalance (>20× ratio) | −15 |
| High imbalance (>10× ratio) | −10 |
| Minority class < 10 samples | −15 |
| Too many classes (>50) for multiclass | −10 |

#### Regression-Specific Checks

| Check | Max Deduction |
|-------|--------------|
| Severely skewed target (skew > 3) | −10 |
| Extreme outliers in target (>5%) | −8 |
| Near-zero target variance | −15 |

#### Grade Scale

| Score | Grade | Interpretation |
|-------|-------|---------------|
| 90–100 | **A** | Dataset is clean and ready for modelling |
| 75–89 | **B** | Minor issues — address before training |
| 60–74 | **C** | Moderate issues — significant cleaning needed |
| 45–59 | **D** | Major issues — data may produce unreliable models |
| 0–44 | **F** | Critical problems — reassess data collection |

---

## ⚙️ Configuration

All thresholds are controlled by the `CFG` dictionary at the top of the script — no need to dig into the logic to tune behaviour.

```python
CFG = dict(
    # ── Flagging thresholds ──────────────────────────────────────
    HIGH_MISSING_PCT         = 40.0,   # % missing to flag a column as high-missing
    HIGH_CORR                = 0.85,   # Pearson |r| to flag as highly correlated
    HIGH_CARDINALITY_RATIO   = 0.50,   # unique/total ratio to flag as high-cardinality
    HIGH_SKEW                = 1.0,    # abs(skewness) threshold
    ZSCORE_THRESHOLD         = 3.0,    # Z-score cutoff for outlier detection

    # ── Performance (large datasets) ────────────────────────────
    SAMPLE_ROWS_PLOTS        = 50_000, # max rows used when rendering charts
    SAMPLE_ROWS_TSNE         = 5_000,  # max rows used for t-SNE embedding

    # ── Display limits ───────────────────────────────────────────
    TOP_CAT_N                = 20,     # top N categories shown per categorical column
    MAX_PAIRPLOT_COLS        = 6,      # max columns included in the pairplot

    # ── Statistical test significance ────────────────────────────
    NORMALITY_ALPHA          = 0.05,   # alpha for Shapiro-Wilk / D'Agostino / Anderson
    ANOVA_ALPHA              = 0.05,   # alpha for ANOVA significance flagging
    CHI2_ALPHA               = 0.05,   # alpha for Chi-squared significance flagging

    # ── Outlier detection ────────────────────────────────────────
    ISOLATION_CONTAMINATION  = 0.05,   # expected outlier proportion for Isolation Forest
    LOF_N_NEIGHBORS          = 20,     # number of neighbours for LOF

    # ── Output ───────────────────────────────────────────────────
    REPORT_FILENAME          = "data_profile_report.html",
    STATS_CSV_DIR            = "profile_stats",
)
```

---

## 📂 Output Structure

```
your_project/
│
├── data_profile_report.html      ← self-contained interactive HTML report
│                                    (open in any browser, no server needed)
│
└── profile_stats/
    ├── numerical_stats.csv       ← extended statistics for all numerical columns
    └── missing_values.csv        ← missing value summary with imputation recommendations
```

---

## 🔬 Analysis Modules Reference

<details>
<summary><strong>Missing Value Analysis</strong></summary>

Per-column report including:
- Missing count and percentage
- Imputation recommendation:
  - `Mean impute` — low-skew numerical columns
  - `Median impute` — high-skew numerical columns
  - `Mode impute` — categorical columns
  - `Forward-fill` — datetime columns
  - `DROP column` — columns with >70% missing

Visualisations:
- Bar chart coloured by severity (red > 40%, orange > 10%, blue < 10%)
- Heatmap of the missing pattern across a row sample
- Nullity dendrogram showing which columns tend to be missing together

</details>

<details>
<summary><strong>Outlier Detection</strong></summary>

Four independent methods applied:

| Method | Scope | How it works |
|--------|-------|-------------|
| IQR (1.5×) | Per-column | Flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR |
| Z-score (3σ) | Per-column | Flags values more than 3 standard deviations from the mean |
| Isolation Forest | Multivariate | Randomly isolates observations; anomalies are easier to isolate |
| LOF | Multivariate | Compares local density of a point to its neighbours |

</details>

<details>
<summary><strong>Correlation Analysis</strong></summary>

| Method | Columns | Captures |
|--------|---------|---------|
| Pearson | Numerical ↔ Numerical | Linear relationships |
| Spearman | Numerical ↔ Numerical | Monotonic / rank relationships |
| Kendall | Numerical ↔ Numerical | Rank concordance (robust to outliers) |
| Cramér's V | Categorical ↔ Categorical | Strength of association via Chi-squared |
| Point-Biserial | Binary ↔ Numerical | Correlation between a binary group and a continuous variable |

</details>

<details>
<summary><strong>Normality Tests</strong></summary>

Three tests are run per numerical column:

| Test | Best for | H₀ |
|------|---------|-----|
| Shapiro-Wilk | Small–medium samples (n < 5,000) | Data is normally distributed |
| D'Agostino-K² | Medium–large samples | Data is normally distributed |
| Anderson-Darling | Any sample size | Data follows a specified distribution |

If p-value > 0.05 (configurable), the column is marked **Normal**. Otherwise **Non-normal** — non-parametric methods or transformations are recommended.

</details>

<details>
<summary><strong>Text / String Analysis</strong></summary>

For every `object`-type column that appears to contain free text:

- Average, min, max character length
- Average word count
- Count of cells containing email addresses (regex-detected)
- Count of cells containing URLs
- Count of cells containing phone numbers
- Percentage of cells containing digits
- Percentage of cells containing uppercase characters

</details>

---

## 🛠 Performance

The profiler is designed to handle large datasets without running out of memory:

- All chart rendering is done on a **random sample** (default: 50,000 rows) — full dataset statistics are always computed on the complete data
- t-SNE uses a separate smaller sample (default: 5,000 rows) due to O(n²) complexity
- Pandas vectorised operations are used throughout — no Python loops over rows
- Type auto-correction runs before any analysis to avoid silent dtype issues

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

Built on top of the Python data science ecosystem:

[pandas](https://pandas.pydata.org/) · [NumPy](https://numpy.org/) · [Matplotlib](https://matplotlib.org/) · [seaborn](https://seaborn.pydata.org/) · [SciPy](https://scipy.org/) · [scikit-learn](https://scikit-learn.org/) · [missingno](https://github.com/ResidentMario/missingno)

---

<div align="center">

<img src="https://github.com/trx98/Predicting-Customer-Annual-Spending-Using-Behavioral-Metrics/blob/main/logo_new.png?raw=true" alt="KoshurAI" width="120"/>

<br/>

**Developed by [KoshurAI](https://github.com/trx98)**

*If this tool saved you time, consider giving it a ⭐ on GitHub.*

</div>
