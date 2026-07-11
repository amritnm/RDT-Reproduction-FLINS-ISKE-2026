"""
Enhanced Boosting Benchmark with Depth Analysis and Feature Tracking.

This benchmark:
- Tests multiple depths (2, 3, 4, 5, 6)
- Tracks distinct features used at each stage
- Uses 5-fold cross-validation
- Saves all results to CSV
- Compares all boosting methods including the new fast RDT variant
"""

import sys
import os
import time
import warnings
import csv
from datetime import datetime
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Try to import optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not available")

from ensemble.adaboost_rdt import AdaBoostRDT
from ensemble.gradient_boosting_rdt import GradientBoostingRDT
from ensemble.gradient_boosting_rdt_fast import GradientBoostingRDTFast


def count_distinct_features(model, model_type, up_to_tree=None):
    """
    Count distinct features used across trees.
    
    Parameters
    ----------
    model : ensemble model
        The trained ensemble model
    model_type : str
        Type of model for handling different APIs
    up_to_tree : int, optional
        Only count features up to this tree index
    
    Returns
    -------
    int : Number of unique features used
    """
    features_set = set()
    
    if model_type in ['ada_rdt', 'gb_rdt', 'gb_rdt_fast']:
        # RDT models - use depth_features_
        estimators = model.estimators_[:up_to_tree] if up_to_tree else model.estimators_
        for tree in estimators:
            if hasattr(tree, 'depth_features_'):
                features_set.update(tree.depth_features_.values())
    
    elif model_type == 'gb_sklearn':
        # sklearn GradientBoosting
        estimators = model.estimators_
        if isinstance(estimators, np.ndarray) and estimators.ndim == 2:
            estimators = estimators.flatten()
        
        trees_to_check = estimators[:up_to_tree] if up_to_tree else estimators
        for tree in trees_to_check:
            if tree is not None and hasattr(tree, 'tree_'):
                feature_indices = tree.tree_.feature
                features_set.update(feature_indices[feature_indices >= 0])
    
    elif model_type == 'xgboost' and HAS_XGBOOST:
        # XGBoost - can't easily do staged feature counting, return final
        try:
            importance = model.get_booster().get_score(importance_type='weight')
            features_set = set(int(f[1:]) for f in importance.keys() if f.startswith('f'))
        except:
            return 0
    
    elif model_type == 'catboost' and HAS_CATBOOST:
        # CatBoost - can't easily do staged feature counting, return final
        try:
            importance = model.get_feature_importance()
            features_set = set(np.where(importance > 0)[0])
        except:
            return 0
    
    return len(features_set)


def get_staged_predictions(model, X, model_type):
    """Get predictions and probabilities at each stage."""
    staged_preds = []
    staged_probas = []
    
    if model_type in ['gb_rdt', 'gb_rdt_fast', 'gb_sklearn']:
        # Use staged_predict_proba
        if hasattr(model, 'staged_predict_proba'):
            for proba in model.staged_predict_proba(X):
                if proba.ndim > 1 and proba.shape[1] > 1:
                    pred = np.argmax(proba, axis=1)
                else:
                    pred = (proba[:, 1] > 0.5).astype(int) if proba.ndim > 1 else (proba > 0.5).astype(int)
                staged_preds.append(pred)
                staged_probas.append(proba)
    
    elif model_type == 'ada_rdt':
        # AdaBoost - use predict method iteratively
        if hasattr(model, 'staged_predict'):
            for pred in model.staged_predict(X):
                proba = np.zeros((len(pred), 2))
                proba[np.arange(len(pred)), pred.astype(int)] = 1.0
                staged_preds.append(pred)
                staged_probas.append(proba)
    
    elif model_type == 'xgboost' and HAS_XGBOOST:
        for i in range(1, model.n_estimators + 1):
            try:
                proba = model.predict_proba(X, iteration_range=(0, i))
            except TypeError:
                try:
                    proba = model.predict_proba(X, ntree_limit=i)
                except:
                    if i == model.n_estimators:
                        proba = model.predict_proba(X)
                    else:
                        continue
            
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            pred = np.argmax(proba, axis=1)
            staged_preds.append(pred)
            staged_probas.append(proba)
    
    elif model_type == 'catboost' and HAS_CATBOOST:
        for i in range(1, model.tree_count_ + 1):
            proba = model.predict_proba(X, ntree_end=i)
            pred = np.argmax(proba, axis=1)
            staged_preds.append(pred)
            staged_probas.append(proba)
    
    # Fallback
    if not staged_preds:
        final_pred = model.predict(X)
        final_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        if final_proba is not None:
            if final_proba.ndim == 1:
                final_proba = np.vstack([1 - final_proba, final_proba]).T
            return [final_pred], [final_proba]
        return [final_pred], [None]
    
    return staged_preds, staged_probas


def benchmark_single_config(model_name, model_class, params, X_train, X_val, y_train, y_val, total_features):
    """
    Benchmark a single model configuration.
    
    Returns list of result dictionaries (one per stage).
    """
    model_type = params.pop('_model_type', 'unknown')
    
    # Train model
    start_time = time.time()
    model = model_class(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Get staged predictions
    staged_preds, staged_probas = get_staged_predictions(model, X_val, model_type)
    
    results = []
    for stage_idx, (pred, proba) in enumerate(zip(staged_preds, staged_probas)):
        n_trees = stage_idx + 1
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, pred)
        
        if proba is not None and proba.shape[1] > 1:
            auc = roc_auc_score(y_val, proba[:, 1])
        else:
            auc = 0.0
        
        f1 = f1_score(y_val, pred, zero_division=0)
        
        # Count distinct features up to this stage
        distinct_features = count_distinct_features(model, model_type, up_to_tree=n_trees)
        
        results.append({
            'model_name': model_name,
            'n_trees': n_trees,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'train_time': train_time,
            'distinct_features': distinct_features,
            'total_features': total_features
        })
    
    return results


def run_depth_fold_experiment(X, y, depth, fold_idx, train_idx, val_idx, max_trees=100):
    """
    Run all models for a specific depth and fold.
    
    Returns list of result dictionaries.
    """
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    total_features = X.shape[1]
    n_samples = len(X_train)
    
    # Adaptive parameters
    learning_rate = 0.1
    subsample = 0.8 if n_samples > 10000 else 1.0
    min_samples_split = max(2, int(n_samples * 0.001))
    min_samples_leaf = max(1, int(n_samples * 0.0005))
    
    all_results = []
    
    # 1. AdaBoost RDT
    print(f"      -> AdaBoost RDT (depth={depth}, fold={fold_idx+1})...")
    try:
        results = benchmark_single_config(
            'AdaBoost-RDT',
            AdaBoostRDT,
            {
                'n_estimators': max_trees,
                'max_depth': depth,
                'learning_rate': 0.8,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'random_state': 42,
                '_model_type': 'ada_rdt'
            },
            X_train, X_val, y_train, y_val, total_features
        )
        for r in results:
            r['depth'] = depth
            r['fold'] = fold_idx + 1
        all_results.extend(results)
    except Exception as e:
        print(f"         ERROR: {e}")
    
    # 2. Gradient Boosting RDT-Optimal (Python)
    print(f"      -> GB RDT-Optimal (depth={depth}, fold={fold_idx+1})...")
    try:
        results = benchmark_single_config(
            'GB-RDT-Optimal',
            GradientBoostingRDT,
            {
                'n_estimators': max_trees,
                'max_depth': depth,
                'learning_rate': learning_rate,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'subsample': subsample,
                'random_state': 42,
                'verbose': 0,
                '_model_type': 'gb_rdt'
            },
            X_train, X_val, y_train, y_val, total_features
        )
        for r in results:
            r['depth'] = depth
            r['fold'] = fold_idx + 1
        all_results.extend(results)
    except Exception as e:
        print(f"         ERROR: {e}")
    
    # 3. Gradient Boosting RDT-Fast (Cython-optimized)
    print(f"      -> GB RDT-Fast (depth={depth}, fold={fold_idx+1})...")
    try:
        results = benchmark_single_config(
            'GB-RDT-Fast',
            GradientBoostingRDTFast,
            {
                'n_estimators': max_trees,
                'max_depth': depth,
                'learning_rate': learning_rate,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'subsample': subsample,
                'random_state': 42,
                'verbose': 0,
                '_model_type': 'gb_rdt_fast'
            },
            X_train, X_val, y_train, y_val, total_features
        )
        for r in results:
            r['depth'] = depth
            r['fold'] = fold_idx + 1
        all_results.extend(results)
    except Exception as e:
        print(f"         ERROR: {e}")
    
    # 4. sklearn Gradient Boosting
    print(f"      -> sklearn GB (depth={depth}, fold={fold_idx+1})...")
    try:
        results = benchmark_single_config(
            'GB-sklearn',
            GradientBoostingClassifier,
            {
                'n_estimators': max_trees,
                'max_depth': depth,
                'learning_rate': learning_rate,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'subsample': subsample,
                'random_state': 42,
                '_model_type': 'gb_sklearn'
            },
            X_train, X_val, y_train, y_val, total_features
        )
        for r in results:
            r['depth'] = depth
            r['fold'] = fold_idx + 1
        all_results.extend(results)
    except Exception as e:
        print(f"         ERROR: {e}")
    
    # 5. XGBoost (if available)
    if HAS_XGBOOST:
        print(f"      -> XGBoost (depth={depth}, fold={fold_idx+1})...")
        try:
            results = benchmark_single_config(
                'XGBoost',
                xgb.XGBClassifier,
                {
                    'n_estimators': max_trees,
                    'max_depth': depth,
                    'learning_rate': learning_rate,
                    'min_child_weight': min_samples_leaf,
                    'subsample': subsample,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    '_model_type': 'xgboost'
                },
                X_train, X_val, y_train, y_val, total_features
            )
            for r in results:
                r['depth'] = depth
                r['fold'] = fold_idx + 1
            all_results.extend(results)
        except Exception as e:
            print(f"         ERROR: {e}")
    
    # 6. CatBoost (if available)
    if HAS_CATBOOST:
        print(f"      -> CatBoost (depth={depth}, fold={fold_idx+1})...")
        try:
            results = benchmark_single_config(
                'CatBoost',
                cb.CatBoostClassifier,
                {
                    'iterations': max_trees,
                    'depth': depth,
                    'learning_rate': learning_rate,
                    'min_data_in_leaf': min_samples_leaf,
                    'subsample': subsample,
                    'random_state': 42,
                    'verbose': False,
                    '_model_type': 'catboost'
                },
                X_train, X_val, y_train, y_val, total_features
            )
            for r in results:
                r['depth'] = depth
                r['fold'] = fold_idx + 1
            all_results.extend(results)
        except Exception as e:
            print(f"         ERROR: {e}")
    
    return all_results


def run_full_benchmark(X, y, depths=[2, 3, 4, 5, 6], n_folds=5, max_trees=100):
    """
    Run complete benchmark across all depths and folds.
    
    Returns list of all results.
    """
    all_results = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    total_experiments = len(depths) * n_folds
    current_exp = 0
    
    for depth in depths:
        print(f"\n{'='*70}")
        print(f"DEPTH LEVEL: {depth}")
        print(f"{'='*70}")
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            current_exp += 1
            print(f"\n   Fold {fold_idx + 1}/{n_folds} (Experiment {current_exp}/{total_experiments})")
            
            fold_results = run_depth_fold_experiment(
                X, y, depth, fold_idx, train_idx, val_idx, max_trees
            )
            all_results.extend(fold_results)
    
    return all_results


def save_results_to_csv(results, filename='boosting_benchmark_results.csv'):
    """Save all results to CSV file."""
    if not results:
        print("No results to save!")
        return
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    fieldnames = [
        'timestamp', 'model_name', 'depth', 'fold', 'n_trees',
        'accuracy', 'auc', 'f1', 'train_time',
        'distinct_features', 'total_features', 'feature_usage_pct'
    ]
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'timestamp': timestamp,
                'model_name': result['model_name'],
                'depth': result['depth'],
                'fold': result['fold'],
                'n_trees': result['n_trees'],
                'accuracy': f"{result['accuracy']:.6f}",
                'auc': f"{result['auc']:.6f}",
                'f1': f"{result['f1']:.6f}",
                'train_time': f"{result['train_time']:.4f}",
                'distinct_features': result['distinct_features'],
                'total_features': result['total_features'],
                'feature_usage_pct': f"{100 * result['distinct_features'] / result['total_features']:.2f}"
            }
            writer.writerow(row)
    
    print(f"\n✓ Results saved to: {filepath}")
    return filepath


def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70 + "\n")
    
    # Group by model and depth
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        model = r['model_name']
        depth = r['depth']
        # Take final tree count only
        if r['n_trees'] >= 90:  # Near max_trees
            summary[model][depth].append(r)
    
    print("Final Performance (at max trees, averaged across folds):\n")
    print(f"{'Model':<20} {'Depth':<6} {'Accuracy':<10} {'AUC':<10} {'Features':<12} {'Time (s)':<10}")
    print("-" * 70)
    
    for model in sorted(summary.keys()):
        for depth in sorted(summary[model].keys()):
            fold_results = summary[model][depth]
            avg_acc = np.mean([r['accuracy'] for r in fold_results])
            avg_auc = np.mean([r['auc'] for r in fold_results])
            avg_feat = np.mean([r['distinct_features'] for r in fold_results])
            avg_time = np.mean([r['train_time'] for r in fold_results])
            
            print(f"{model:<20} {depth:<6} {avg_acc:<10.4f} {avg_auc:<10.4f} {avg_feat:<12.1f} {avg_time:<10.2f}")


def main():
    """Main execution."""
    print("="*70)
    print("ENHANCED BOOSTING BENCHMARK")
    print("With Depth Analysis and Feature Tracking")
    print("="*70 + "\n")
    
    # Configuration
    sample_size = 20000  # Manageable size for testing
    depths = [2, 3, 4, 5, 6]
    n_folds = 5
    max_trees = 100
    
    # Load data
    print("Loading Covertype dataset...")
    data = fetch_covtype()
    X, y = data.data, data.target
    y_binary = (y == 1).astype(int)
    
    if sample_size and sample_size < len(X):
        indices = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X = X[indices]
        y_binary = y_binary[indices]
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_binary)}\n")
    
    # Run benchmark
    results = run_full_benchmark(X, y_binary, depths, n_folds, max_trees)
    
    # Save to CSV
    csv_file = save_results_to_csv(results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*70)
    print("✓ BENCHMARK COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {csv_file}")
    print(f"Total rows: {len(results)}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
