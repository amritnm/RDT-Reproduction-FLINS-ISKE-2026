# RDT — FLINS/ISKE 2026 Reproduction Package

Code and (already-generated) results needed to reproduce the tables and figures in
*"Restricted Decision Trees: Hierarchical Feature Selection for Knowledge-Guided
Staged Decision Systems"* (17th FLINS & 21st ISKE Conference, 2026).

This is a trimmed-down copy of a larger research repository, containing only the
scripts, library code, and data required to regenerate the paper's results — not
the LaTeX source or unrelated exploratory notebooks.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`xgboost`/`catboost`/`gmpy2` are only needed for specific baselines (see below) and
are imported inside `try/except` blocks — the scripts degrade gracefully if any are
missing.

**Compiled acceleration modules**: `rdt_optimal_cython/`, `rdt_cython/`, and
`contree_lib/python/pycontree/` ship prebuilt Windows binaries (`.pyd`, built for
CPython 3.11; `pycontree` also ships a 3.13 build). If your Python version/platform
doesn't match, the RDT modules fall back to pure Python automatically (slower, but
numerically equivalent); ConTree has no fallback — rebuild from
[the upstream contree repo](https://github.com/AlgTUDelft/pyContree) or omit it
(`run_benchmark.py` skips it gracefully if unimportable).

## Directory map

| Path | Role |
|---|---|
| `rdt/`, `rdt_optimal_cython/`, `rdt_cython/` | Core RDT / ODT / two-pass-RDT implementations (pure Python + Cython acceleration) |
| `ensemble/` | RDT-Boost, RDT-AdaBoost, RDT-RandomForest |
| `osdt_lib/`, `contree_lib/` | Vendored optional baselines (OSDT, ConTree) — trimmed to the files actually imported |
| `revised_benchmark/` | Entry-point scripts that run the standalone-tree and gradient-boosting benchmarks |
| `kyc_illustration/` | Synthetic staged-cost KYC simulation (Tables 3 & 4) |
| `datasets/` | Dataset download scripts (zips are not vendored — re-download via these scripts) |
| `Final_Results/` | Raw `results_summary.csv` per dataset, as originally generated — lets you regenerate Tables 1 & 2 without rerunning the full benchmark |
| `analyse_boosting_mechanisms.py`, `boosting_mechanism_results.json` | Table 5 + Figure 2 source data |
| `extract_all_results.py`, `recompute_statistics.py`, `recompute_boosting_statistics.py`, `statistical_significance_test.py` | Aggregate `Final_Results/` into paper tables and run the Wilcoxon/Cohen's d significance tests |
| `generate_paper_figures.py` | Builds `fig1_scatter.pdf` and `fig2_convergence.pdf` |

## Reproducing from scratch

Run from the repo root unless noted otherwise.

1. **Datasets** — `python datasets/download_datasets.py` (core 12), then
   `python datasets/download_new_datasets.py` (Phishing Websites, Madelon, Musk v2,
   etc. needed for the boosting benchmark). Zips land in `datasets/`.
2. **Table 1 raw data** (standalone trees, 12 datasets) —
   ```bash
   cd revised_benchmark
   python run_benchmark.py --dataset "Adult Income"
   # ... repeat for each of the 12 datasets listed in the paper
   ```
   Writes `results_<Dataset>/results_summary.csv`; copy/merge into
   `Final_Results/Tree_Results/results_<Dataset>/` to match the aggregation
   scripts' expected layout (already populated with the results used in the paper).
3. **Table 2 raw data** (gradient boosting, 10 datasets) —
   ```bash
   cd revised_benchmark
   python run_boosting_benchmark_parallel.py --dataset "Adult Income"
   # ... repeat for each of the 10 boosting datasets
   ```
   Feeds `Final_Results/Ensemble_Results/`.
4. **Aggregate + significance tests** (from repo root):
   ```bash
   python extract_all_results.py
   python recompute_statistics.py
   python recompute_boosting_statistics.py
   python statistical_significance_test.py
   ```
5. **Table 5 + Figure 2 data** (mechanism analysis):
   ```bash
   python analyse_boosting_mechanisms.py
   ```
   Writes `boosting_mechanism_results.json` (already included with the paper's
   original run).
6. **Tables 3 & 4** (synthetic KYC staged-cost illustration):
   ```bash
   python kyc_illustration/simulate_kyc_cost.py
   ```
7. **Figures 1 & 2**:
   ```bash
   python generate_paper_figures.py
   ```

Since `Final_Results/` and `boosting_mechanism_results.json` already contain the
data used in the paper, steps 4–7 can be run immediately without repeating the
(multi-hour) benchmark sweeps in steps 1–3.

## Notes

- The standalone-tree and boosting benchmarks (`run_benchmark.py`,
  `run_boosting_benchmark_parallel.py`) are the slowest part of reproduction — see
  the paper's runtime analysis (Python implementation is 50–100x slower than
  sklearn's C++ backend; the Cython modules here recover 3–8x of that).
- All experiments use 5-fold stratified cross-validation with `seed=42`.