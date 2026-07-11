"""
Parallel Boosting Benchmarking Script
Compares gradient boosting algorithms with parallel execution and comprehensive metrics

Usage:
    python run_boosting_benchmark_parallel.py --dataset "Adult Income" --depths 3 5
    python run_boosting_benchmark_parallel.py --dataset "HELOC" --sample_size 10000
    python run_boosting_benchmark_parallel.py --dataset "Bank Marketing" --n_estimators 100 --parallel 4

Algorithms Compared:
1. GradientBoostingRDTFast (Cython-based RDT)
2. GradientBoostingRDTFast (Cython-based Two-Pass RDT)  
3. sklearn GradientBoostingClassifier
4. XGBoost
5. CatBoost

Features:
- Parallel algorithm execution
- Feature counting (distinct features used)
- Command-line configurable sample size
- All RDT benchmark metrics
- Comprehensive output with visualizations
"""

import os
import sys
import argparse
import logging
import zipfile
import time
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ensemble'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rdt_optimal_cython'))

# Import boosting algorithms
from ensemble.gradient_boosting_rdt import GradientBoostingRDT
from ensemble.gradient_boosting_rdt_fast import GradientBoostingRDTFast

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available")

# Try to import CatBoost
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not available")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATHS = {
    'Adult Income': '../datasets/Adult Income.zip',
    'Bank Marketing': '../datasets/Bank Marketing.zip',
    'Cover Type': '../datasets/Cover Type.zip',
    'German Credit': '../datasets/German Credit.zip',
    'HELOC': '../datasets/HELOC.zip',
    'Breast Cancer': '../datasets/Breast Cancer.zip',
    'Dry Bean': '../datasets/Dry Bean.zip',
    'Ionosphere': '../datasets/Ionosphere.zip',
    'Occupancy Detection': '../datasets/Occupancy Detection.zip',
    'Pima Indians Diabetes': '../datasets/Pima Indians Diabetes.zip',
    'Spambase': '../datasets/Spambase.zip',
    # High-dimensional datasets added for boosting feature-reduction study
    'Phishing Websites': '../datasets/Phishing Websites.zip',   # 11,055 x 30 features
    'Madelon': '../datasets/Madelon.zip',                       # 2,000 x 500 features
    'Musk v2': '../datasets/Musk v2.zip',                       # 6,598 x 166 features
}


# =============================================================================
# RESOURCE DETECTION
# =============================================================================

def detect_optimal_parallel_jobs() -> int:
    """Auto-detect optimal number of parallel jobs."""
    try:
        mem = psutil.virtual_memory()
        available_ram_gb = mem.available / (1024**3)
        n_cpus = cpu_count()
        
        # Each boosting model needs ~4 GB RAM
        ram_limit = max(1, int(available_ram_gb / 4))
        cpu_limit = max(1, n_cpus - 1)
        optimal = min(ram_limit, cpu_limit, 6)
        
        print(f"\n💻 System Resources:")
        print(f"   Available RAM: {available_ram_gb:.1f} GB")
        print(f"   CPU cores: {n_cpus}")
        print(f"   ➜ Using: {optimal} parallel jobs\n")
        
        return optimal
    except:
        return 1


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset_name: str, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load dataset from zip file with optional sampling."""
    dataset_path = DATASET_PATHS[dataset_name]
    logging.info(f"Loading dataset: {dataset_name}")
    
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        csv_file = csv_files[0]
        zip_ref.extract(csv_file, 'temp_data')
    
    csv_path = f'temp_data/{csv_file}'
    df = pd.read_csv(csv_path)
    
    os.remove(csv_path)
    if os.path.exists('temp_data') and not os.listdir('temp_data'):
        os.rmdir('temp_data')
    
    X_df = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    
    # Identify categorical features
    categorical_indices = []
    for i, col in enumerate(X_df.columns):
        if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
            categorical_indices.append(i)
    
    # Simple label encoding for now (will apply onehot/target later)
    X = X_df.copy()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.values.astype(float)
    
    # Encode labels
    if y.dtype == object or y.dtype.kind == 'U':
        y = LabelEncoder().fit_transform(y)
    
    # Apply sampling if requested
    if sample_size is not None and sample_size < len(X):
        indices = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        logging.info(f"  Sampled to {sample_size} samples")
    
    logging.info(f"  Final: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    return X, y, categorical_indices


def apply_onehot_encoding(X: np.ndarray, categorical_indices: List[int]) -> np.ndarray:
    """Apply one-hot encoding to categorical features."""
    if not categorical_indices:
        return X
    
    numeric_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
    transformers = []
    
    if numeric_indices:
        transformers.append(('num', 'passthrough', numeric_indices))
    if categorical_indices:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_indices))
    
    ct = ColumnTransformer(transformers=transformers)
    X_encoded = ct.fit_transform(X)
    
    return X_encoded


def apply_target_encoding(X: np.ndarray, y: np.ndarray, categorical_indices: List[int], cv_folds: int = 5) -> np.ndarray:
    """Apply CV-aware target encoding to categorical features."""
    if not categorical_indices:
        return X.copy()
    
    X_encoded = X.copy()
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for cat_idx in categorical_indices:
        encoded_col = np.zeros(len(X))
        for train_idx, val_idx in kf.split(X, y):
            encoding_map = {}
            for category in np.unique(X[train_idx, cat_idx]):
                mask = X[train_idx, cat_idx] == category
                if mask.sum() > 0:
                    encoding_map[category] = y[train_idx[mask]].mean()
            global_mean = y[train_idx].mean()
            for i in val_idx:
                encoded_col[i] = encoding_map.get(X[i, cat_idx], global_mean)
        X_encoded[:, cat_idx] = encoded_col
    
    return X_encoded


# =============================================================================
# FEATURE COUNTING FUNCTIONS
# =============================================================================

def count_distinct_features(model, model_type: str) -> int:
    """
    Count distinct features used across all trees in the ensemble.
    
    Parameters
    ----------
    model : Trained ensemble model
    model_type : str
        Type: 'gb_rdt_fast', 'gb_sklearn', 'xgboost', 'catboost'
    
    Returns
    -------
    int : Number of unique features used
    """
    features_set = set()
    
    try:
        if model_type == 'gb_rdt_fast':
            # RDT models - use depth_features_
            for tree in model.estimators_:
                if hasattr(tree, 'depth_features_'):
                    features_set.update(tree.depth_features_.values())
        
        elif model_type == 'gb_sklearn':
            # sklearn GradientBoosting
            estimators = model.estimators_
            if isinstance(estimators, np.ndarray) and estimators.ndim == 2:
                estimators = estimators.flatten()
            
            for tree in estimators:
                if tree is not None and hasattr(tree, 'tree_'):
                    feature_indices = tree.tree_.feature
                    features_set.update(feature_indices[feature_indices >= 0])
        
        elif model_type == 'xgboost' and HAS_XGBOOST:
            # XGBoost
            importance = model.get_booster().get_score(importance_type='weight')
            features_set = set(int(f[1:]) for f in importance.keys() if f.startswith('f'))
        
        elif model_type == 'catboost' and HAS_CATBOOST:
            # CatBoost
            importance = model.get_feature_importance()
            features_set = set(np.where(importance > 0)[0])
        
        return len(features_set)
    except:
        return -1


# =============================================================================
# INTERPRETABILITY METRICS
# =============================================================================

def compute_interpretability_metrics(model, model_type: str, X: np.ndarray) -> Dict:
    """Compute interpretability metrics for ensemble."""
    try:
        distinct_features = count_distinct_features(model, model_type)
        total_features = X.shape[1]
        feature_usage_pct = (distinct_features / total_features * 100) if total_features > 0 else 0
        
        # Estimate average tree depth
        avg_depth = -1
        if model_type in ['gb_rdt_fast', 'gb_sklearn']:
            if hasattr(model, 'max_depth'):
                avg_depth = model.max_depth
        
        # Number of trees
        n_trees = 0
        if hasattr(model, 'n_estimators'):
            n_trees = model.n_estimators
        elif hasattr(model, 'estimators_'):
            n_trees = len(model.estimators_)
        
        return {
            'distinct_features': distinct_features,
            'total_features': total_features,
            'feature_usage_pct': feature_usage_pct,
            'avg_depth': avg_depth,
            'n_trees': n_trees
        }
    except Exception as e:
        return {
            'distinct_features': -1,
            'total_features': X.shape[1],
            'feature_usage_pct': 0,
            'avg_depth': -1,
            'n_trees': 0
        }


# =============================================================================
# PARALLEL WORKER FUNCTION
# =============================================================================

def benchmark_algorithm_worker(args):
    """
    Worker function to benchmark ONE boosting algorithm.
    
    Args:
        Tuple of (algo_name, X, y, depth, n_estimators, n_folds, progress_dict, algo_idx)
    
    Returns:
        Dictionary with results
    """
    algo_name, X, y, depth, n_estimators, n_folds, progress_dict, algo_idx = args
    
    start_time = time.time()
    
    try:
        # Update progress
        if progress_dict is not None:
            progress_dict[algo_idx] = f"⏳ {algo_name}"
        
        # Configure algorithm
        if algo_name == 'GB-RDT':
            # Original GradientBoostingRDT (Python implementation)
            model_class = GradientBoostingRDT
            params = {
                'n_estimators': n_estimators,
                'max_depth': depth,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }
            model_type = 'gb_rdt_fast'
        
        elif algo_name == 'GB-RDT-Fast':
            # GradientBoostingRDTFast (uses Cython OptimalRDTCython)
            model_class = GradientBoostingRDTFast
            params = {
                'n_estimators': n_estimators,
                'max_depth': depth,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }
            model_type = 'gb_rdt_fast'
        
        elif algo_name == 'GB-sklearn':
            model_class = GradientBoostingClassifier
            params = {
                'n_estimators': n_estimators,
                'max_depth': depth,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }
            model_type = 'gb_sklearn'
        
        elif algo_name == 'XGBoost' and HAS_XGBOOST:
            model_class = xgb.XGBClassifier
            params = {
                'n_estimators': n_estimators,
                'max_depth': depth,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            model_type = 'xgboost'
        
        elif algo_name == 'CatBoost' and HAS_CATBOOST:
            model_class = cb.CatBoostClassifier
            params = {
                'iterations': n_estimators,
                'depth': depth,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42,
                'verbose': False
            }
            model_type = 'catboost'
        
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        cv_auc_scores = []
        cv_f1_scores = []
        training_times = []
        prediction_times = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train
            model = model_class(**params)
            fold_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - fold_start
            
            # Predict
            pred_start = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - pred_start
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            cv_scores.append(accuracy)
            cv_f1_scores.append(f1)
            training_times.append(train_time)
            prediction_times.append(pred_time)
            
            # AUC for binary
            if len(np.unique(y)) == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        cv_auc_scores.append(roc_auc_score(y_test, y_proba))
                except:
                    cv_auc_scores.append(np.nan)
        
        # Train final model for interpretability metrics
        final_model = model_class(**params)
        final_model.fit(X, y)
        interp_metrics = compute_interpretability_metrics(final_model, model_type, X)
        
        # Update progress
        if progress_dict is not None:
            elapsed = time.time() - start_time
            progress_dict[algo_idx] = f"✓ {algo_name} ({elapsed/60:.1f}min)"
        
        return {
            'algorithm': algo_name,
            'depth': depth,
            'n_estimators': n_estimators,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'mean_auc': np.mean(cv_auc_scores) if cv_auc_scores else np.nan,
            'std_auc': np.std(cv_auc_scores) if cv_auc_scores else np.nan,
            'mean_f1': np.mean(cv_f1_scores),
            'std_f1': np.std(cv_f1_scores),
            'mean_train_time': np.mean(training_times),
            'std_train_time': np.std(training_times),
            'mean_pred_time': np.mean(prediction_times),
            'total_train_time': np.sum(training_times),
            'distinct_features': interp_metrics['distinct_features'],
            'total_features': interp_metrics['total_features'],
            'feature_usage_pct': interp_metrics['feature_usage_pct'],
            'avg_tree_depth': interp_metrics['avg_depth'],
            'n_trees': interp_metrics['n_trees'],
            'success': True
        }
        
    except Exception as e:
        if progress_dict is not None:
            progress_dict[algo_idx] = f"✗ {algo_name} (failed)"
        
        return {
            'algorithm': algo_name,
            'depth': depth,
            'n_estimators': n_estimators,
            'mean_accuracy': np.nan,
            'std_accuracy': np.nan,
            'mean_auc': np.nan,
            'std_auc': np.nan,
            'mean_f1': np.nan,
            'std_f1': np.nan,
            'mean_train_time': 0,
            'std_train_time': 0,
            'mean_pred_time': 0,
            'total_train_time': 0,
            'distinct_features': -1,
            'total_features': X.shape[1] if X is not None else 0,
            'feature_usage_pct': 0,
            'avg_tree_depth': -1,
            'n_trees': 0,
            'success': False,
            'error': str(e)
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_visualizations(df, dataset_name, output_dir):
    """Generate comprehensive visualization plots."""
    
    # 1. Performance vs Depth
    plt.figure(figsize=(12, 6))
    for algo in df['algorithm'].unique():
        df_algo = df[df['algorithm'] == algo]
        depths = df_algo['depth'].values
        means = df_algo['mean_accuracy'].values
        stds = df_algo['std_accuracy'].values
        plt.plot(depths, means, marker='o', label=algo, linewidth=2)
        plt.fill_between(depths, means-stds, means+stds, alpha=0.2)
    plt.xlabel('Tree Depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Performance vs Depth - {dataset_name}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/depth_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training Time Comparison
    plt.figure(figsize=(10, 6))
    df_time = df.groupby('algorithm')['mean_train_time'].mean().sort_values()
    df_time.plot(kind='barh', color='steelblue')
    plt.xlabel('Mean Training Time (seconds)', fontsize=12)
    plt.title(f'Training Time Comparison - {dataset_name}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Usage Comparison
    plt.figure(figsize=(10, 6))
    df_features = df.groupby('algorithm')[['distinct_features', 'total_features']].mean()
    df_features.plot(kind='bar', color=['steelblue', 'lightcoral'])
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title(f'Feature Usage - {dataset_name}', fontsize=14)
    plt.legend(['Distinct Features Used', 'Total Features'])
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Accuracy vs Training Time scatter
    plt.figure(figsize=(10, 6))
    for algo in df['algorithm'].unique():
        df_algo = df[df['algorithm'] == algo]
        plt.scatter(df_algo['mean_train_time'], df_algo['mean_accuracy'], 
                   label=algo, s=100, alpha=0.7)
    plt.xlabel('Mean Training Time (seconds)', fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize=12)
    plt.title(f'Accuracy vs Training Time - {dataset_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_time.png', dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Parallel Boosting Benchmarking')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_PATHS.keys()))
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size to use (default: full dataset)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 5, 7],
                       help='Tree depths to test (default: 3 5 7)')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of boosting iterations (default: 100)')
    parser.add_argument('--parallel', type=str, default='auto',
                       help='Number of parallel jobs (auto or integer, default: auto)')
    parser.add_argument('--encoding', type=str, default='label', 
                       choices=['label', 'onehot', 'target'],
                       help='Encoding type: label (default), onehot, or target')
    
    args = parser.parse_args()
    
    # Determine parallelism
    if args.parallel == 'auto':
        n_jobs = detect_optimal_parallel_jobs()
    else:
        n_jobs = min(int(args.parallel), 6)
        print(f"\n💻 Using {n_jobs} parallel jobs (user-specified)\n")
    
    # Setup
    dataset_name = args.dataset
    output_dir = f"results_boosting_{dataset_name.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/benchmark_log.txt'),
            logging.StreamHandler()
        ]
    )
    
    print("="*80)
    print(f"🚀 PARALLEL BOOSTING BENCHMARKING: {dataset_name}")
    print("="*80)
    print(f"Sample size: {args.sample_size if args.sample_size else 'Full dataset'}")
    print(f"Encoding: {args.encoding}")
    print(f"Parallelism: {n_jobs} jobs")
    print(f"Depths: {args.depths}")
    print(f"Estimators: {args.n_estimators}")
    print(f"CV Folds: {args.n_folds}")
    print("")
    
    # Load data
    X, y, categorical_indices = load_dataset(dataset_name, args.sample_size)
    
    # Apply encoding
    if args.encoding == 'onehot' and categorical_indices:
        print(f"Applying one-hot encoding to {len(categorical_indices)} categorical features...")
        X = apply_onehot_encoding(X, categorical_indices)
        print(f"Dataset shape after encoding: {X.shape}")
    elif args.encoding == 'target' and categorical_indices:
        print(f"Applying target encoding to {len(categorical_indices)} categorical features...")
        X = apply_target_encoding(X, y, categorical_indices)
        print(f"Dataset shape after encoding: {X.shape}")
    else:
        print(f"Using label encoding (default)")
    
    # Build algorithm list
    algorithms = ['GB-RDT', 'GB-RDT-Fast', 'GB-sklearn']
    if HAS_XGBOOST:
        algorithms.append('XGBoost')
    if HAS_CATBOOST:
        algorithms.append('CatBoost')
    
    print(f"Algorithms: {algorithms}\n")
    
    # Results storage
    all_results = []
    
    # Progress tracking
    with Manager() as manager:
        progress_dict = manager.dict()
        
        # For each depth
        for depth in args.depths:
            print(f"\n{'='*80}")
            print(f"DEPTH: {depth}")
            print(f"{'='*80}\n")
            
            # Prepare tasks
            tasks = []
            for idx, algo_name in enumerate(algorithms):
                tasks.append((
                    algo_name,
                    X,
                    y,
                    depth,
                    args.n_estimators,
                    args.n_folds,
                    progress_dict,
                    idx
                ))
            
            # Run in parallel
            if n_jobs > 1:
                print(f"Running {len(tasks)} algorithms in parallel...")
                with Pool(processes=min(n_jobs, len(tasks))) as pool:
                    results = pool.map(benchmark_algorithm_worker, tasks)
            else:
                print(f"Running {len(tasks)} algorithms sequentially...")
                results = [benchmark_algorithm_worker(task) for task in tasks]
            
            # Process results
            for result in results:
                if result['success']:
                    result['dataset'] = dataset_name
                    result['encoding'] = args.encoding
                    result['sample_size'] = args.sample_size if args.sample_size else len(X)
                    all_results.append(result)
                    print(f"  ✓ {result['algorithm']}: Acc={result['mean_accuracy']:.4f}, "
                          f"Features={result['distinct_features']}/{result['total_features']} "
                          f"({result['feature_usage_pct']:.1f}%), Time={result['mean_train_time']:.2f}s")
                else:
                    print(f"  ✗ {result['algorithm']}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Save results
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # Save comprehensive CSV
        df_results.to_csv(f'{output_dir}/results_summary.csv', index=False)
        
        # Save by depth
        for depth in args.depths:
            df_depth = df_results[df_results['depth'] == depth]
            if not df_depth.empty:
                df_depth.to_csv(f'{output_dir}/results_depth_{depth}.csv', index=False)
        
        # Generate visualizations
        try:
            generate_visualizations(df_results, dataset_name, output_dir)
            print("\n✓ Visualizations generated")
        except Exception as e:
            print(f"\n✗ Visualization error: {e}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        summary_cols = ['algorithm', 'depth', 'mean_accuracy', 'mean_f1', 'distinct_features', 
                       'feature_usage_pct', 'mean_train_time']
        print(df_results[summary_cols].to_string(index=False))
        
        # Save configuration
        config = {
            'dataset': dataset_name,
            'encoding': args.encoding,
            'sample_size': args.sample_size,
            'n_folds': args.n_folds,
            'depths': args.depths,
            'n_estimators': args.n_estimators,
            'timestamp': datetime.now().isoformat()
        }
        with open(f'{output_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}/")
    print("")


if __name__ == '__main__':
    main()
