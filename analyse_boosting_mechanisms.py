#!/usr/bin/env python3
"""
analyse_boosting_mechanisms.py
=======================================================================
Measures three mechanisms explaining why feature-restriction performs well
in gradient boosting, comparing sklearn GB, CatBoost, and RDT-Boost:

  1. Reduced Greediness     – cumulative unique feature spread across iterations
  2. Enhanced Diversity     – average pairwise Jaccard similarity between trees
                              (lower Jaccard = more diverse ensemble)
  3. Implicit Regularisation– train vs test AUC learning curves; overfitting gap

Datasets : Adult Income, Bank Marketing, Spambase  (same 3 as Table 2)
Settings : depth=5, 100 estimators, 80/20 stratified split, seed=42

Usage
-----
  python analyse_boosting_mechanisms.py           # full run (sklearn + CatBoost + RDT)
  python analyse_boosting_mechanisms.py --quick   # 25 estimators (fast check)
  python analyse_boosting_mechanisms.py --dataset spambase
  python analyse_boosting_mechanisms.py --no-rdt  # skip RDT-Boost (slow)

Results printed as plain-text tables; raw data saved to boosting_mechanism_results.json.
"""

import argparse
import io
import json
import os
import sys
import tempfile
import time
import warnings
import zipfile

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── locate project root ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── sklearn ────────────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── CatBoost ───────────────────────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[WARN] catboost not installed; CatBoost columns will be skipped.")
    print("       Install with:  pip install catboost")

# ── RDT-Boost ──────────────────────────────────────────────────────────────────
try:
    from ensemble.gradient_boosting_rdt_fast import GradientBoostingRDTFast
    RDT_AVAILABLE = True
except ImportError:
    RDT_AVAILABLE = False
    print("[WARN] GradientBoostingRDTFast not available; RDT columns will be skipped.")

# ── paper values (Table 2) for self-validation ─────────────────────────────────
PAPER_AUC = {
    "adult":    {"sklearn": 0.927, "catboost": 0.921, "rdt": 0.918},
    "bank":     {"sklearn": 0.920, "catboost": 0.918, "rdt": 0.908},
    "spambase": {"sklearn": 0.986, "catboost": 0.986, "rdt": 0.976},
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (uses local ZIP files in datasets/ directory)
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS_DIR = os.path.join(ROOT, "datasets")

ZIP_MAP = {
    "adult":    "Adult Income.zip",
    "bank":     "Bank Marketing.zip",
    "spambase": "Spambase.zip",
    "madelon":  "Madelon.zip",
}


def _load_local_zip(zip_name: str):
    """Load a dataset from local ZIP (single CSV, last column = target)."""
    zip_path = os.path.join(DATASETS_DIR, zip_name)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV found in {zip_path}")
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
    target_col = df.columns[-1]
    y = LabelEncoder().fit_transform(df[target_col].astype(str))
    X = df.drop(columns=[target_col])
    for col in X.select_dtypes(["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.values.astype(np.float64), y.astype(np.float64), list(X.columns)


def load_dataset(name: str):
    key = name.lower().strip()
    aliases = {"adult income": "adult", "bank marketing": "bank", "spam": "spambase"}
    key = aliases.get(key, key)
    if key not in ZIP_MAP:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(ZIP_MAP.keys())}")
    return _load_local_zip(ZIP_MAP[key])


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_sklearn_feature_sets(model):
    """List of frozensets: features used by each tree in sklearn GB."""
    feature_sets = []
    for estimators in model.estimators_:
        tree = estimators[0].tree_
        feats = frozenset(int(f) for f in tree.feature if f >= 0)
        feature_sets.append(feats)
    return feature_sets


def get_catboost_feature_sets(model):
    """
    List of frozensets: features used by each tree in a CatBoost model.

    CatBoost uses Oblivious Decision Trees (ODT): every node at depth d uses
    the SAME feature (and threshold).  A depth-k tree therefore contributes
    at most k distinct features (fewer if the same feature is re-selected).

    Extraction uses CatBoost's JSON model export, which lists each tree's
    splits as an ordered list — one entry per depth level.
    """
    if not CATBOOST_AVAILABLE:
        return []
    # Export model to a temporary JSON file and parse it
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmpname = f.name
    try:
        model.save_model(tmpname, format="json")
        with open(tmpname, "r") as f:
            cbjson = json.load(f)
    finally:
        os.unlink(tmpname)

    feature_sets = []
    for tree in cbjson.get("oblivious_trees", []):
        feats = set()
        for split in tree.get("splits", []):
            # Numeric features → float_feature_index
            # Categorical features → cat_feature_index
            if "float_feature_index" in split:
                feats.add(int(split["float_feature_index"]))
            elif "cat_feature_index" in split:
                feats.add(int(split["cat_feature_index"]))
        feature_sets.append(frozenset(feats))
    return feature_sets


def get_rdt_feature_sets(model):
    """List of frozensets: features used by each tree in RDT-Boost."""
    feature_sets = []
    for tree in model.estimators_:
        if hasattr(tree, "depth_features_"):
            feats = frozenset(int(v) for v in tree.depth_features_.values() if v >= 0)
        else:
            feats = frozenset()
        feature_sets.append(feats)
    return feature_sets


# ═══════════════════════════════════════════════════════════════════════════════
# MECHANISM MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def cumulative_unique_features(feature_sets, n_total):
    """Returns list of (iteration, cumulative_count, fraction)."""
    seen = set()
    results = []
    for t, fs in enumerate(feature_sets, 1):
        seen |= fs
        results.append((t, len(seen), len(seen) / n_total))
    return results


def pairwise_jaccard_stats(feature_sets, max_pairs=500):
    """
    Mean and std of pairwise Jaccard similarity across all trees.
    Lower mean → more diverse ensemble.
    Samples at most `max_pairs` pairs to keep runtime tractable.
    """
    n = len(feature_sets)
    rng = np.random.RandomState(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(all_pairs) > max_pairs:
        chosen = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        all_pairs = [all_pairs[k] for k in chosen]
    scores = []
    for i, j in all_pairs:
        a, b = feature_sets[i], feature_sets[j]
        u = a | b
        if len(u) == 0:
            continue
        scores.append(len(a & b) / len(u))
    arr = np.array(scores)
    return float(arr.mean()), float(arr.std())


def staged_auc_sklearn(model, X, y):
    """Per-iteration AUC for sklearn GB via staged_predict_proba."""
    return [roc_auc_score(y, p[:, 1]) for p in model.staged_predict_proba(X)]


def staged_auc_catboost(model, X, y, n_estimators):
    """
    Per-iteration AUC for CatBoost.

    CatBoost does not expose staged_predict_proba, but supports
    predict_proba(X, ntree_start=0, ntree_end=t) to get predictions
    using only the first t trees.
    """
    aucs = []
    for t in range(1, n_estimators + 1):
        proba = model.predict_proba(X, ntree_start=0, ntree_end=t)
        aucs.append(roc_auc_score(y, proba[:, 1]))
    return aucs


def staged_auc_rdt(model, X, y):
    """Per-iteration AUC for RDT-Boost via staged_predict_proba."""
    return [roc_auc_score(y, p[:, 1]) for p in model.staged_predict_proba(X)]


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PER DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def _record(method_label, train_auc, test_auc, feature_sets, n_features):
    """Package mechanism results for one method into a dict."""
    cum = cumulative_unique_features(feature_sets, n_features)
    jacc_mean, jacc_std = pairwise_jaccard_stats(feature_sets)
    ck = {}
    for c in [10, 25, 50, 100]:
        if c <= len(cum):
            ck[c] = cum[c - 1][1]
    return {
        "final_auc":      test_auc[-1],
        "train_auc":      train_auc,
        "test_auc":       test_auc,
        "overfit_gap":    train_auc[-1] - test_auc[-1],
        "cum_feats_all":  cum,
        "cum_feats_at":   ck,
        "jaccard_mean":   jacc_mean,
        "jaccard_std":    jacc_std,
        "n_features_per_tree": (
            round(sum(len(fs) for fs in feature_sets) / len(feature_sets), 1)
            if feature_sets else None
        ),
    }


def analyse_dataset(name, n_estimators, depth=5, seed=42, run_rdt=True):
    print(f"\n{'='*70}")
    print(f"  Dataset: {name.upper()}   n_estimators={n_estimators}  depth={depth}")
    print(f"{'='*70}")

    X, y, feature_names = load_dataset(name)
    n_features = X.shape[1]
    print(f"  Shape: {X.shape}   features: {n_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    results = {}

    # ── 1. sklearn GradientBoostingClassifier ─────────────────────────────────
    print(f"\n  [sklearn] Training GradientBoostingClassifier ...")
    t0 = time.time()
    skl = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=depth,
        learning_rate=0.1, random_state=seed, subsample=1.0,
    )
    skl.fit(X_train, y_train)
    print(f"           Done in {time.time()-t0:.1f}s")

    skl_fs = get_sklearn_feature_sets(skl)
    skl_tr  = staged_auc_sklearn(skl, X_train, y_train)
    skl_te  = staged_auc_sklearn(skl, X_test,  y_test)
    results["sklearn"] = _record("sklearn", skl_tr, skl_te, skl_fs, n_features)

    # ── 2. CatBoost (ODT weak learners) ───────────────────────────────────────
    if CATBOOST_AVAILABLE:
        print(f"\n  [catboost] Training CatBoostClassifier (ODT) ...")
        t0 = time.time()
        cb = CatBoostClassifier(
            iterations=n_estimators,
            depth=depth,
            learning_rate=0.1,
            random_seed=seed,
            verbose=0,
            loss_function="Logloss",
            eval_metric="AUC",
        )
        cb.fit(X_train, y_train)
        print(f"            Done in {time.time()-t0:.1f}s")

        cb_fs = get_catboost_feature_sets(cb)
        print(f"            Computing CatBoost staged AUC ({n_estimators} iters) ...")
        cb_tr = staged_auc_catboost(cb, X_train, y_train, n_estimators)
        cb_te = staged_auc_catboost(cb, X_test,  y_test,  n_estimators)
        results["catboost"] = _record("catboost", cb_tr, cb_te, cb_fs, n_features)
    else:
        print("\n  [catboost] Skipped (not installed)")

    # ── 3. RDT-Boost ──────────────────────────────────────────────────────────
    if run_rdt and RDT_AVAILABLE:
        print(f"\n  [rdt] Training GradientBoostingRDTFast ...")
        print(f"        NOTE: slow (~{max(60, 36 * n_estimators)}s expected)")
        t0 = time.time()
        rdt = GradientBoostingRDTFast(
            n_estimators=n_estimators, max_depth=depth,
            learning_rate=0.1, random_state=seed, task="classification",
        )
        rdt.fit(X_train, y_train)
        print(f"        Done in {time.time()-t0:.1f}s")

        rdt_fs = get_rdt_feature_sets(rdt)
        rdt_tr = staged_auc_rdt(rdt, X_train, y_train)
        rdt_te = staged_auc_rdt(rdt, X_test,  y_test)
        results["rdt"] = _record("rdt", rdt_tr, rdt_te, rdt_fs, n_features)
    elif run_rdt:
        print("\n  [rdt] Skipped (GradientBoostingRDTFast not available)")

    return results, n_features


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

METHODS   = ["sklearn", "catboost", "rdt"]
METHOD_LB = {"sklearn": "sklearn", "catboost": "CatBoost", "rdt": "RDT-Boost"}
CHECKPOINTS = [10, 25, 50, 100]


def print_summary(all_results, n_estimators):
    ck = [c for c in CHECKPOINTS if c <= n_estimators]

    # ── Mechanism 1: Feature Spread ──────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("MECHANISM 1: Cumulative Unique Features Across Iterations")
    print("  (shows how fast each method exhausts its available feature vocabulary)")
    print("=" * 72)
    col_w = 8
    hdr = f"{'Dataset':<10} {'Method':<10} {'d':>4}"
    for c in ck:
        hdr += f"  {'@'+str(c):>{col_w}}"
    hdr += "  avg/tree"
    print(hdr)
    for ds, (res, nf) in all_results.items():
        for m in METHODS:
            if m not in res:
                continue
            r = res[m]
            cum = r["cum_feats_all"]
            avg = r["n_features_per_tree"]
            row = f"{ds:<10} {METHOD_LB[m]:<10} {nf:>4}"
            for c in ck:
                v = r["cum_feats_at"].get(c, "-")
                row += f"  {str(v):>{col_w}}"
            row += f"  {avg if avg else '-':>8}"
            print(row)
        print()

    # ── Mechanism 2: Pairwise Jaccard ────────────────────────────────────────
    print("=" * 72)
    print("MECHANISM 2: Pairwise Jaccard Similarity  (lower = more diverse ensemble)")
    print("  ODT/RDT constraint → smaller per-tree feature sets → lower Jaccard")
    print("=" * 72)
    print(f"{'Dataset':<10} {'sklearn':>10}  {'CatBoost':>10}  {'RDT-Boost':>10}  "
          f"{'CB vs SKL':>12}  {'RDT vs SKL':>12}")
    for ds, (res, nf) in all_results.items():
        skl_j = res["sklearn"]["jaccard_mean"] if "sklearn" in res else None
        cb_j  = res["catboost"]["jaccard_mean"] if "catboost" in res else None
        rdt_j = res["rdt"]["jaccard_mean"]      if "rdt" in res else None

        def _fmt_j(v):
            return f"{v:.4f}" if v is not None else "  n/a"
        def _fmt_delta(v, base):
            if v is None or base is None:
                return "  n/a"
            return f"{(v - base):+.4f}"

        print(f"{ds:<10} {_fmt_j(skl_j):>10}  {_fmt_j(cb_j):>10}  {_fmt_j(rdt_j):>10}  "
              f"{_fmt_delta(cb_j, skl_j):>12}  {_fmt_delta(rdt_j, skl_j):>12}")
    print()

    # ── Mechanism 3: Overfitting Gap ─────────────────────────────────────────
    print("=" * 72)
    print("MECHANISM 3: Overfitting Gap  (Train AUC − Test AUC)")
    print("  Smaller gap → implicit regularisation from capacity restriction")
    print("=" * 72)
    sel = [c for c in [1, 10, 25, 50, 100] if c <= n_estimators]
    hdr3 = f"{'Dataset':<10} {'Method':<10}"
    for c in sel:
        hdr3 += f"  {'gap@'+str(c):>9}"
    print(hdr3)
    for ds, (res, nf) in all_results.items():
        for m in METHODS:
            if m not in res:
                continue
            r = res[m]
            tr, te = r["train_auc"], r["test_auc"]
            row = f"{ds:<10} {METHOD_LB[m]:<10}"
            for c in sel:
                if c <= len(tr):
                    row += f"  {tr[c-1]-te[c-1]:>9.4f}"
                else:
                    row += f"  {'n/a':>9}"
            print(row)
        print()

    # ── Self-Validation ───────────────────────────────────────────────────────
    print("=" * 72)
    print("SELF-VALIDATION: Final AUC vs. Table 2 paper numbers (tol ±0.005)")
    print("=" * 72)
    print(f"{'Dataset':<10} {'Method':<10} {'Measured':>10}  {'Paper':>7}  {'Match':>8}")
    for ds, (res, nf) in all_results.items():
        key = ds.lower()
        for m in METHODS:
            if m not in res or key not in PAPER_AUC or m not in PAPER_AUC[key]:
                continue
            measured   = res[m]["final_auc"]
            paper_val  = PAPER_AUC[key][m]
            ok = abs(measured - paper_val) <= 0.005
            mark = "OK" if ok else "MISMATCH"
            print(f"{ds:<10} {METHOD_LB[m]:<10} {measured:>10.4f}  {paper_val:>7.3f}  {mark:>8}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyse boosting mechanisms: sklearn vs CatBoost vs RDT-Boost"
    )
    parser.add_argument("--quick",    action="store_true",
                        help="Use 25 estimators (fast directional check)")
    parser.add_argument("--no-rdt",   action="store_true",
                        help="Skip RDT-Boost (very slow; skip for sklearn+CatBoost only)")
    parser.add_argument("--dataset",  default="all",
                        help="adult | bank | spambase | all (default)")
    parser.add_argument("--depth",    type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    n_est = 25 if args.quick else 100
    run_rdt = not args.no_rdt

    if args.quick:
        print("[QUICK MODE] 25 estimators — AUC will NOT match Table 2 paper numbers.")
    if args.no_rdt:
        print("[--no-rdt] Skipping RDT-Boost; running sklearn + CatBoost only.")

    datasets = (["adult", "bank", "spambase", "madelon"]
                if args.dataset == "all" else [args.dataset])

    all_results = {}
    for ds in datasets:
        try:
            res, nf = analyse_dataset(ds, n_est, depth=args.depth,
                                      seed=args.seed, run_rdt=run_rdt)
            all_results[ds] = (res, nf)
        except Exception as exc:
            print(f"[ERROR] {ds}: {exc}")
            import traceback; traceback.print_exc()

    if all_results:
        print_summary(all_results, n_est)

        # ── JSON save ─────────────────────────────────────────────────────────
        out = {}
        for ds, (res, nf) in all_results.items():
            out[ds] = {}
            for m in METHODS:
                if m not in res:
                    continue
                r = res[m]
                out[ds][m] = {
                    "final_auc":          round(r["final_auc"], 4),
                    "jaccard_mean":       round(r["jaccard_mean"], 4),
                    "jaccard_std":        round(r["jaccard_std"], 4),
                    "overfit_gap":        round(r["overfit_gap"], 4),
                    "cum_feats_final":    r["cum_feats_all"][-1][1] if r["cum_feats_all"] else None,
                    "n_features_per_tree": r["n_features_per_tree"],
                    "train_auc_by_iter":  [round(v, 4) for v in r["train_auc"]],
                    "test_auc_by_iter":   [round(v, 4) for v in r["test_auc"]],
                }
        outfile = os.path.join(ROOT, "boosting_mechanism_results.json")
        with open(outfile, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nRaw results saved to: {outfile}")


if __name__ == "__main__":
    main()
