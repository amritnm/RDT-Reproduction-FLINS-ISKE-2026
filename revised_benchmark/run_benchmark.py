"""
Comprehensive RDT Benchmarking Script
Addresses all Stanford reviewer concerns in single-phase execution

Usage:
    python run_benchmark.py --dataset "Adult Income"
    python run_benchmark.py --dataset "Bank Marketing"
    python run_benchmark.py --dataset "Cover Type"
    python run_benchmark.py --dataset "German Credit"
    python run_benchmark.py --dataset "HELOC"

Features:
- K-fold CV (5-fold outer, 3-fold inner for tuning)
- Hyperparameter tuning (depth-stratified GridSearchCV)
- Dual encoding (one-hot + CV-aware target encoding)
- Interpretability metrics
- OSDT/ConTree baselines (with fallback)
- ASCII-only logging (no Unicode errors)
- CSV + PNG outputs
"""

import os
import sys
import argparse
import logging
import zipfile
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Add paths for local implementations
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rdt'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rdt', 'optimal_restricted_tree'))

from oblivious_decision_tree import ObliviousDecisionTree
from restricted_decision_tree import RestrictedDecisionTree
from optimal_restricted_decision_tree import OptimalRestrictedDecisionTree

# Import OSDT wrapper
try:
    from osdt_wrapper import OSDTWrapper
    OSDT_AVAILABLE = True
except ImportError:
    OSDT_AVAILABLE = False
    logging.info("OSDT wrapper not available")

# Import ConTree (sklearn-compatible, no wrapper needed!)
try:
    from pycontree import ConTree
    CONTREE_AVAILABLE = True
except ImportError:
    CONTREE_AVAILABLE = False
    logging.info("ConTree not available")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATHS = {
    # Original datasets
    'Adult Income': '../datasets/Adult Income.zip',
    'Bank Marketing': '../datasets/Bank Marketing.zip',
    'Cover Type': '../datasets/Cover Type.zip',
    'German Credit': '../datasets/German Credit.zip',
    'HELOC': '../datasets/HELOC.zip',
    'Heart Disease': '../datasets/Heart Disease.zip',
    
    # New datasets
    'Skin Segmentation': '../datasets/Skin Segmentation.zip',
    'Magic Gamma Telescope': '../datasets/Magic Gamma Telescope.zip',
    'HTRU2': '../datasets/HTRU2.zip',
    'Occupancy Detection': '../datasets/Occupancy Detection.zip',
    'Page Blocks': '../datasets/Page Blocks.zip',
    'Spambase': '../datasets/Spambase.zip',
    'Satellite': '../datasets/Satellite.zip',
    'Wall-Following Robot': '../datasets/Wall-Following Robot.zip',
    'Segment': '../datasets/Segment.zip',
    'Steel Plates Faults': '../datasets/Steel Plates Faults.zip',
#    'QSAR Biodegradation': '../datasets/QSAR Biodegradation.zip',
    'Pima Indians Diabetes': '../datasets/Pima Indians Diabetes.zip',
    'Breast Cancer': '../datasets/Breast Cancer.zip',
    'Ionosphere': '../datasets/Ionosphere.zip',
    'Vehicle': '../datasets/Vehicle.zip',
    'Dry Bean': '../datasets/Dry Bean.zip',
}

ALGORITHMS = {
    'ODT': ObliviousDecisionTree,
    'RDT': RestrictedDecisionTree,
    'RDT-TwoPass': OptimalRestrictedDecisionTree,
    'sklearn-DT': DecisionTreeClassifier
}

# Add OSDT if wrapper is available
if OSDT_AVAILABLE:
    ALGORITHMS['OSDT'] = OSDTWrapper
    print("OSDT baseline added to benchmarking")

# Add ConTree if available (sklearn-compatible!)
if CONTREE_AVAILABLE:
    ALGORITHMS['ConTree'] = ConTree
    print("ConTree baseline added to benchmarking")


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Load dataset from zip file.
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        categorical_indices: Indices of categorical features
    """
    dataset_path = DATASET_PATHS[dataset_name]
    logging.info(f"Loading dataset: {dataset_name}")
    
    # Extract CSV from zip
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV found in {dataset_path}")
        csv_file = csv_files[0]
        zip_ref.extract(csv_file, 'temp_data')
    
    # Load data
    csv_path = f'temp_data/{csv_file}'
    df = pd.read_csv(csv_path)
    
    # Clean up temp file
    os.remove(csv_path)
    if os.path.exists('temp_data') and not os.listdir('temp_data'):
        os.rmdir('temp_data')
    
    # Separate features and target
    X_df = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    
    # Get feature names
    feature_names = X_df.columns.tolist()
    
    # Identify categorical features
    categorical_indices = []
    for i, col in enumerate(X_df.columns):
        if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
            categorical_indices.append(i)
    
    # Convert to numpy (with simple encoding for now)
    X = X_df.copy()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.values.astype(float)
    
    # Encode target if needed
    if y.dtype == object or y.dtype.kind == 'U':
        y = LabelEncoder().fit_transform(y)
    
    logging.info(f"  Samples: {len(X)}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    logging.info(f"  Categorical features: {len(categorical_indices)}")
    
    return X, y, feature_names, categorical_indices


def apply_onehot_encoding(X: np.ndarray, y: np.ndarray, 
                          categorical_indices: List[int]) -> Tuple[np.ndarray, List[str]]:
    """
    Apply one-hot encoding to categorical features.
    
    Returns:
        X_encoded: Encoded feature matrix
        new_feature_names: Updated feature names
    """
    if not categorical_indices:
        return X,  [f"feature_{i}" for i in range(X.shape[1])]
    
    # Identify numeric vs categorical columns
    numeric_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
    
    # Create column transformer
    transformers = []
    if numeric_indices:
        transformers.append(('num', 'passthrough', numeric_indices))
    if categorical_indices:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), 
                           categorical_indices))
    
    ct = ColumnTransformer(transformers=transformers)
    X_encoded = ct.fit_transform(X)
    
    # Generate feature names
    new_feature_names = []
    if numeric_indices:
        new_feature_names.extend([f"num_feature_{i}" for i in numeric_indices])
    if categorical_indices:
        cat_features = ct.named_transformers_['cat'].get_feature_names_out()
        new_feature_names.extend(cat_features.tolist())
    
    logging.info(f"  One-hot encoding: {X.shape[1]} -> {X_encoded.shape[1]} features")
    
    return X_encoded, new_feature_names


def apply_target_encoding_cv(X: np.ndarray, y: np.ndarray, 
                             categorical_indices: List[int], 
                             cv_folds: int = 5) -> np.ndarray:
    """
    Apply CV-aware target encoding to prevent leakage.
    
    For each fold:
      - Compute encoding on train portion
      - Apply to validation portion
    
    Returns:
        X_encoded: Encoded feature matrix
    """
    if not categorical_indices:
        return X.copy()
    
    X_encoded = X.copy()
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for cat_idx in categorical_indices:
        encoded_col = np.zeros(len(X))
        
        for train_idx, val_idx in kf.split(X, y):
            # Compute encoding on training fold
            encoding_map = {}
            for category in np.unique(X[train_idx, cat_idx]):
                mask = X[train_idx, cat_idx] == category
                if mask.sum() > 0:
                    encoding_map[category] = y[train_idx[mask]].mean()
            
            # Global mean for unseen categories
            global_mean = y[train_idx].mean()
            
            # Apply to validation fold
            for i in val_idx:
                cat = X[i, cat_idx]
                encoded_col[i] = encoding_map.get(cat, global_mean)
        
        X_encoded[:, cat_idx] = encoded_col
    
    logging.info(f"  Target encoding applied to {len(categorical_indices)} features")
    
    return X_encoded


# =============================================================================
# INTERPRETABILITY METRICS
# =============================================================================

def count_unique_features(model) -> int:
    """Count number of unique features used in tree."""
    try:
        if hasattr(model, 'tree_'):
            # sklearn tree
            tree = model.tree_
            features = tree.feature[tree.feature >= 0]
            return len(np.unique(features))
        elif hasattr(model, 'depth_features_'):
            # RDT/ODT - features stored by depth
            features = list(model.depth_features_.values())
            return len(set(features))
        elif hasattr(model, 'feature_indices_'):
            # Alternative attribute
            return len(set(model.feature_indices_))
        else:
            # Fallback: return max_depth as proxy (each depth uses 1 feature)
            return getattr(model, 'max_depth', -1)
    except Exception as e:
        logging.debug(f"count_unique_features failed: {e}")
        return -1


def compute_avg_path_length(model, X: np.ndarray) -> float:
    """Compute average path length from root to leaf."""
    try:
        if hasattr(model, 'decision_path'):
            # sklearn tree
            paths = model.decision_path(X)
            return float(paths.sum(axis=1).mean())
        elif hasattr(model, 'max_depth'):
            # For RDT/ODT, approximate as max_depth (complete binary tree)
            # Actual path length varies but this is reasonable approximation
            return float(model.max_depth)
        else:
            return -1.0
    except Exception as e:
        logging.debug(f"compute_avg_path_length failed: {e}")
        return -1.0


def get_tree_depth(model) -> int:
    """Get actual depth of tree."""
    try:
        if hasattr(model, 'tree_'):
            return int(model.tree_.max_depth)
        elif hasattr(model, 'max_depth'):
            return int(model.max_depth)
        elif hasattr(model, 'depth_features_'):
            # For RDT/ODT, infer from depth_features_ dict
            return max(model.depth_features_.keys()) + 1 if model.depth_features_ else -1
        else:
            return -1
    except Exception as e:
        logging.debug(f"get_tree_depth failed: {e}")
        return -1


def count_leaves(model) -> int:
    """Count number of leaf nodes."""
    try:
        if hasattr(model, 'tree_'):
            # sklearn tree
            tree = model.tree_
            return int((tree.feature == -2).sum())
        elif hasattr(model, 'max_depth'):
            # For complete binary tree: 2^depth leaves
            return int(2 ** model.max_depth)
        else:
            return -1
    except Exception as e:
        logging.debug(f"count_leaves failed: {e}")
        return -1


def compute_stability_across_folds(model_class, X: np.ndarray, y: np.ndarray, 
                                   params: dict, cv_folds: int = 5) -> float:
    """
    Compute structural stability across CV folds.
    
    Returns:
        Stability score between 0 and 1 (1 = identical structures)
    """
    try:
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        feature_lists = []
        
        for train_idx, _ in kf.split(X, y):
            model = model_class(**params)
            model.fit(X[train_idx], y[train_idx])
            
            if hasattr(model, 'depth_features_'):
                features = list(model.depth_features_.values())
            elif hasattr(model, 'tree_'):
                tree = model.tree_
                features = tree.feature[tree.feature >= 0].tolist()
            else:
                features = []
            
            feature_lists.append(set(features))
        
        # Compute Jaccard similarity between all pairs
        if not feature_lists or not feature_lists[0]:
            return 0.0
        
        similarities = []
        for i in range(len(feature_lists)):
            for j in range(i+1, len(feature_lists)):
                intersection = len(feature_lists[i] & feature_lists[j])
                union = len(feature_lists[i] | feature_lists[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 0.0
    except:
        return 0.0


def compute_interpretability_metrics(model, model_class, X: np.ndarray, y: np.ndarray, 
                                    params: dict) -> Dict[str, Any]:
    """Compute all interpretability metrics."""
    return {
        'features_used': count_unique_features(model),
        'avg_path_length': compute_avg_path_length(model, X),
        'actual_depth': get_tree_depth(model),
        'n_leaves': count_leaves(model),
        'stability': compute_stability_across_folds(model_class, X, y, params, cv_folds=5)
    }


# =============================================================================
# BENCHMARKING
# =============================================================================

def get_param_grid(algo_name: str, depth: int) -> Dict[str, List]:
    """Get depth-stratified parameter grid for algorithm."""
    base_grid = {
        'max_depth': [depth],
        'task': ['classification'],
        'criterion': ['gini']
    }
    
    if algo_name == 'sklearn-DT':
        return {
            'max_depth': [depth],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'ccp_alpha': [0.0, 0.01, 0.05]
        }
    elif algo_name == 'ConTree':
        # ConTree-specific parameters
        # Note: max_gap must be integer (C++ backend requirement)
        return {
            'max_depth': [depth],
            'max_gap': [0],  # Gap tolerance (0 = optimal, must be int)
            'max_gap_decay': [0.0, 0.1],  # Gap decay (this can be float)
            'time_limit': [300],  # 5 minutes per model
            'use_upper_bound': [True],
            'sort_gini': [True]
        }
    elif algo_name == 'OSDT':
        # OSDT-specific parameters (handled by wrapper)
        return {
            'max_depth': [depth],
            'timelimit': [60, 120],  # 1-2 minutes
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    else:
        # RDT/ODT variants
        base_grid.update({
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        })
        return base_grid


def benchmark_algorithm(algo_name: str, algo_class, X: np.ndarray, y: np.ndarray,
                       depth: int, encoding_type: str, n_folds: int = 5) -> Dict[str, Any]:
    """
    Benchmark algorithm with nested CV and hyperparameter tuning.
    
    Returns:
        Dictionary with performance metrics and best parameters
    """
    start_time = time.time()
    logging.info(f"    Benchmarking {algo_name} at depth {depth} ({encoding_type})")
    
    param_grid = get_param_grid(algo_name, depth)
    n_param_combinations = 1
    for v in param_grid.values():
        n_param_combinations *= len(v) if isinstance(v, list) else 1
    logging.info(f"      Hyperparameter combinations: {n_param_combinations}")
    
    # Check if sklearn-compatible (has get_params method)
    is_sklearn_compatible = algo_name == 'sklearn-DT'
    
    # Nested CV
    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
    
    cv_scores = []
    cv_auc_scores = []
    best_params_list = []
    training_times = []
    prediction_times = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        logging.info(f"        Fold {fold+1}/{n_folds}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        try:
            if is_sklearn_compatible:
                # Use GridSearchCV for sklearn
                grid = GridSearchCV(
                    algo_class(), 
                    param_grid, 
                    cv=inner_cv, 
                    scoring='accuracy',
                    n_jobs=1,
                    verbose=0
                )
                
                fold_start = time.time()
                grid.fit(X_train, y_train)
                train_time = time.time() - fold_start
                
                pred_start = time.time()
                y_pred = grid.predict(X_test)
                pred_time = time.time() - pred_start
                
                best_params = grid.best_params_
                
            else:
                # Manual hyperparameter tuning for RDT/ODT
                best_params = None
                best_val_score = -np.inf
                
                # Create param combinations
                param_combinations = [{}]
                for key, values in param_grid.items():
                    new_combinations = []
                    for combo in param_combinations:
                        for val in (values if isinstance(values, list) else [values]):
                            new_combo = combo.copy()
                            new_combo[key] = val
                            new_combinations.append(new_combo)
                    param_combinations = new_combinations
                
                # Inner CV for hyperparameter selection
                for params in param_combinations:
                    inner_scores = []
                    for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                        X_inner_train = X_train[inner_train_idx]
                        y_inner_train = y_train[inner_train_idx]
                        X_inner_val = X_train[inner_val_idx]
                        y_inner_val = y_train[inner_val_idx]
                        
                        model = algo_class(**params)
                        model.fit(X_inner_train, y_inner_train)
                        score = accuracy_score(y_inner_val, model.predict(X_inner_val))
                        inner_scores.append(score)
                    
                    mean_score = np.mean(inner_scores)
                    if mean_score > best_val_score:
                        best_val_score = mean_score
                        best_params = params
                
                # Train with best params on full training set
                fold_start = time.time()
                final_model = algo_class(**best_params)
                final_model.fit(X_train, y_train)
                train_time = time.time() - fold_start
                
                pred_start = time.time()
                y_pred = final_model.predict(X_test)
                pred_time = time.time() - pred_start
            
            training_times.append(train_time)
            prediction_times.append(pred_time)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            cv_scores.append(acc)
            
            # AUC (if binary)
            if len(np.unique(y)) == 2:
                try:
                    if is_sklearn_compatible:
                        y_pred_proba = grid.predict_proba(X_test)[:, 1]
                    else:
                        y_pred_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else None
                    if y_pred_proba is not None:
                        auc = roc_auc_score(y_test, y_pred_proba)
                        cv_auc_scores.append(auc)
                    else:
                        cv_auc_scores.append(np.nan)
                except:
                    cv_auc_scores.append(np.nan)
            
            best_params_list.append(best_params)
            
        except Exception as e:
            logging.warning(f"      Fold {fold+1} failed: {str(e)}")
            cv_scores.append(np.nan)
            cv_auc_scores.append(np.nan)
    
    # Aggregate results
    results = {
        'mean_accuracy': np.nanmean(cv_scores),
        'std_accuracy': np.nanstd(cv_scores),
        'mean_auc': np.nanmean(cv_auc_scores) if cv_auc_scores else np.nan,
        'std_auc': np.nanstd(cv_auc_scores) if cv_auc_scores else np.nan,
        'mean_train_time': np.mean(training_times) if training_times else 0,
        'mean_pred_time': np.mean(prediction_times) if prediction_times else 0,
        'cv_scores': cv_scores,
        'best_params': best_params_list[0] if best_params_list else {}
    }
    
    logging.info(f"      Accuracy: {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_depth_performance_plot(df: pd.DataFrame, dataset_name: str, output_dir: str):
    """Generate performance vs depth visualization."""
    plt.figure(figsize=(12, 6))
    
    for encoding in ['onehot', 'target']:
        df_enc = df[df['encoding'] == encoding]
        
        for algo in df_enc['algorithm'].unique():
            df_algo = df_enc[df_enc['algorithm'] == algo]
            
            depths = df_algo['depth'].values
            means = df_algo['mean_accuracy'].values
            stds = df_algo['std_accuracy'].values
            
            linestyle = '-' if encoding == 'onehot' else '--'
            label = f"{algo} ({encoding})"
            
            plt.plot(depths, means, marker='o', linestyle=linestyle, label=label)
            plt.fill_between(depths, means-stds, means+stds, alpha=0.2)
    
    plt.xlabel('Tree Depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Performance vs Depth - {dataset_name}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/depth_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("  Generated: depth_performance.png")


def generate_encoding_comparison_plot(df: pd.DataFrame, dataset_name: str, output_dir: str):
    """Generate encoding comparison visualization."""
    df_summary = df.groupby(['algorithm', 'encoding'])['mean_accuracy'].mean().reset_index()
    df_pivot = df_summary.pivot(index='algorithm', columns='encoding', values='mean_accuracy')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_pivot.plot(kind='bar', ax=ax)
    
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize=12)
    plt.title(f'Encoding Comparison - {dataset_name}', fontsize=14)
    plt.legend(title='Encoding', labels=['One-Hot', 'Target'])
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/encoding_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("  Generated: encoding_comparison.png")


def generate_interpretability_heatmap(df: pd.DataFrame, dataset_name: str, output_dir: str):
    """Generate interpretability metrics heatmap."""
    # Select interpretability columns
    interp_cols = ['features_used', 'avg_path_length', 'n_leaves', 'stability']
    df_interp = df[['algorithm', 'depth'] + interp_cols].copy()
    
    # Average across encodings
    df_avg = df_interp.groupby(['algorithm', 'depth']).mean().reset_index()
    
    # Create pivot for heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(interp_cols):
        df_pivot = df_avg.pivot(index='algorithm', columns='depth', values=col)
        sns.heatmap(df_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[idx])
        axes[idx].set_title(col.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_xlabel('Depth')
        axes[idx].set_ylabel('Algorithm')
    
    plt.suptitle(f'Interpretability Metrics - {dataset_name}', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/interpretability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("  Generated: interpretability_heatmap.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def setup_logging(output_dir: str):
    """Configure ASCII-only logging."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = f'{output_dir}/benchmark_log.txt'
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure new handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='ascii', errors='replace'),
            logging.StreamHandler()
        ]
    )


def save_results(df: pd.DataFrame, output_dir: str):
    """Save results to multiple CSV views."""
    # Main summary
    df.to_csv(f'{output_dir}/results_summary.csv', index=False)
    logging.info(f"  Saved: results_summary.csv")
    
    # By depth
    df_depth = df.groupby(['algorithm', 'depth', 'encoding']).agg({
        'mean_accuracy': 'mean',
        'std_accuracy': 'mean',
        'mean_train_time': 'mean'
    }).reset_index()
    df_depth.to_csv(f'{output_dir}/results_by_depth.csv', index=False)
    logging.info(f"  Saved: results_by_depth.csv")
    
    # By encoding
    df_encoding = df.groupby(['algorithm', 'encoding']).agg({
        'mean_accuracy': 'mean',
        'std_accuracy': 'mean'
    }).reset_index()
    df_encoding.to_csv(f'{output_dir}/results_by_encoding.csv', index=False)
    logging.info(f"  Saved: results_by_encoding.csv")
    
    # Hyperparameter tuning
    df_params = df[['algorithm', 'depth', 'encoding', 'best_params']].copy()
    df_params.to_csv(f'{output_dir}/hyperparameter_tuning.csv', index=False)
    logging.info(f"  Saved: hyperparameter_tuning.csv")
    
    # Interpretability metrics
    interp_cols = ['algorithm', 'depth', 'encoding', 'features_used', 
                   'avg_path_length', 'n_leaves', 'stability']
    df_interp = df[interp_cols].copy()
    df_interp.to_csv(f'{output_dir}/interpretability_metrics.csv', index=False)
    logging.info(f"  Saved: interpretability_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive RDT Benchmarking')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_PATHS.keys()),
                        help='Dataset to benchmark')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 5, 7, 9],
                        help='Tree depths to test (default: 3 5 7 9)')
    
    args = parser.parse_args()
    
    # Setup
    dataset_name = args.dataset
    output_dir = f"results_{dataset_name.replace(' ', '_')}"
    setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info(f"COMPREHENSIVE RDT BENCHMARKING: {dataset_name}")
    logging.info("="*80)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"CV folds: {args.n_folds}")
    logging.info(f"Depths: {args.depths}")
    logging.info("")
    
    # Load data
    X, y, feature_names, categorical_indices = load_dataset(dataset_name)
    
    # Check if dataset is binary (for OSDT compatibility)
    n_classes = len(np.unique(y))
    is_binary = (n_classes == 2)
    
    logging.info(f"Dataset characteristics:")
    logging.info(f"  Number of classes: {n_classes}")
    logging.info(f"  Binary classification: {is_binary}")
    
    # Create algorithm list for this specific dataset
    algorithms_to_use = {}
    for algo_name, algo_class in ALGORITHMS.items():
        if algo_name == 'OSDT':
            if is_binary:
                algorithms_to_use[algo_name] = algo_class
                logging.info(f"  OSDT included (binary dataset)")
            else:
                logging.info(f"  OSDT excluded (multi-class dataset - OSDT supports binary only)")
        else:
            algorithms_to_use[algo_name] = algo_class
    
    logging.info(f"\nAlgorithms to benchmark: {list(algorithms_to_use.keys())}")
    logging.info("")
    
    # Results storage
    all_results = []
    
    # Dual encoding execution
    for encoding_type in ['onehot', 'target']:
        logging.info("")
        logging.info(f"{'='*80}")
        logging.info(f"ENCODING: {encoding_type.upper()}")
        logging.info(f"{'='*80}")
        
        # Apply encoding
        if encoding_type == 'onehot':
            X_encoded, new_feature_names = apply_onehot_encoding(X, y, categorical_indices)
        else:
            X_encoded = apply_target_encoding_cv(X, y, categorical_indices, cv_folds=5)
            new_feature_names = feature_names
        
        # For each depth
        for depth in args.depths:
            logging.info(f"\n  Depth: {depth}")
            
            # For each algorithm (use filtered list based on dataset characteristics)
            for algo_name, algo_class in algorithms_to_use.items():
                try:
                    # Benchmark
                    results = benchmark_algorithm(
                        algo_name, algo_class, X_encoded, y, depth, encoding_type, args.n_folds
                    )
                    
                    # Train final model for interpretability metrics
                    final_model = algo_class(**results['best_params'])
                    final_model.fit(X_encoded, y)
                    
                    interp_metrics = compute_interpretability_metrics(
                        final_model, algo_class, X_encoded, y, results['best_params']
                    )
                    
                    # Store results
                    result_row = {
                        'dataset': dataset_name,
                        'encoding': encoding_type,
                        'algorithm': algo_name,
                        'depth': depth,
                        **results,
                        **interp_metrics
                    }
                    all_results.append(result_row)
                    
                except Exception as e:
                    logging.error(f"    ERROR with {algo_name}: {str(e)}")
    
    # Save results
    logging.info("")
    logging.info("="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)
    
    df = pd.DataFrame(all_results)
    save_results(df, output_dir)
    
    # Generate visualizations
    logging.info("")
    logging.info("="*80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("="*80)
    
    try:
        generate_depth_performance_plot(df, dataset_name, output_dir)
        generate_encoding_comparison_plot(df, dataset_name, output_dir)
        generate_interpretability_heatmap(df, dataset_name, output_dir)
    except Exception as e:
        logging.error(f"Visualization error: {str(e)}")
    
    # Summary
    logging.info("")
    logging.info("="*80)
    logging.info("BENCHMARK COMPLETE!")
    logging.info("="*80)
    logging.info(f"Results saved to: {output_dir}/")
    logging.info(f"Total experiments: {len(all_results)}")
    logging.info("")


if __name__ == '__main__':
    main()
