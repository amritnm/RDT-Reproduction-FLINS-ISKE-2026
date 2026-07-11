"""
Random Forest implementation using Cython-accelerated RDT as base learner.

Provides scikit-learn compatible interface with high performance.
"""

import numpy as np
from typing import Optional, Union
import sys
import os

# Add parent directory to path to import rdt_cython
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rdt_cython.cython_rdt import CythonRestrictedDecisionTree


class RandomForestRDT:
    """
    Random Forest using Restricted Decision Trees.
    
    Trains multiple RDTs on bootstrap samples and aggregates predictions
    via majority voting (classification) or averaging (regression).
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    
    task : {'classification', 'regression'}, default='classification'
        Type of task.
    
    criterion : str, optional
        Splitting criterion. If None, uses 'gini' for classification
        and 'mse' for regression.
    
    max_depth : int, default=5
        Maximum depth of each tree.
    
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
    
    max_features : {'sqrt', 'log2', None} or int, default='sqrt'
        Number of features to consider when looking for best split.
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - None: n_features (all features)
        - int: specific number of features
    
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate generalization score.
    
    random_state : int, optional
        Random seed for reproducibility.
    
    verbose : int, default=0
        Verbosity level (0 = silent, 1 = progress).
    
    Attributes
    ----------
    estimators_ : list
        The collection of fitted tree estimators.
    
    feature_importances_ : ndarray of shape (n_features,)
        Aggregated feature importance scores.
    
    oob_score_ : float
        Score of the training dataset obtained using out-of-bag estimate.
        Only available if oob_score=True.
    
    n_features_ : int
        Number of features when fit is performed.
    
    n_classes_ : int
        Number of classes (classification only).
    
    classes_ : ndarray
        The classes labels (classification only).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        task: str = 'classification',
        criterion: Optional[str] = None,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, None] = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.task = task.lower()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        
        # Will be set during fit
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self._oob_predictions = None
    
    def _get_max_features(self, n_features: int) -> int:
        """Calculate actual number of max_features."""
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRDT':
        """
        Build a forest of trees from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : RandomForestRDT
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        # Setup random state
        rng = np.random.RandomState(self.random_state)
        
        # Calculate max_features
        max_features = self._get_max_features(n_features)
        
        # Initialize OOB tracking
        if self.oob_score:
            if self.task == 'classification':
                self._oob_predictions = np.zeros((n_samples, self.n_classes_))
            else:
                self._oob_predictions = np.zeros(n_samples)
            oob_counts = np.zeros(n_samples, dtype=int)
        
        # Build trees
        self.estimators_ = []
        self._feature_subsets = []  # Track which features each tree uses
        
        for i in range(self.n_estimators):
            if self.verbose > 0:
                print(f"Training tree {i+1}/{self.n_estimators}...")
            
            # Bootstrap sampling
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
                
                # Track OOB samples
                if self.oob_score:
                    oob_mask = np.ones(n_samples, dtype=bool)
                    oob_mask[indices] = False
                    oob_indices = np.where(oob_mask)[0]
            else:
                X_bootstrap = X
                y_bootstrap = y
                oob_indices = []
            
            # Feature subsampling (Option 1: Per-Tree Feature Subset)
            # Randomly select max_features for this tree
            if max_features < n_features:
                feature_indices = rng.choice(n_features, size=max_features, replace=False)
                feature_indices = np.sort(feature_indices)  # Keep sorted for consistency
                X_bootstrap_subset = X_bootstrap[:, feature_indices]
                self._feature_subsets.append(feature_indices)
            else:
                feature_indices = np.arange(n_features)
                X_bootstrap_subset = X_bootstrap
                self._feature_subsets.append(feature_indices)
            
            # Train tree on feature subset
            tree = CythonRestrictedDecisionTree(
                task=self.task,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_bootstrap_subset, y_bootstrap)
            self.estimators_.append(tree)
            
            # Update OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                # Use same feature subset for OOB predictions
                X_oob_subset = X[oob_indices][:, feature_indices]
                oob_pred = tree.predict(X_oob_subset)
                if self.task == 'classification':
                    for idx, pred in zip(oob_indices, oob_pred):
                        pred_class = int(pred)
                        self._oob_predictions[idx, pred_class] += 1
                        oob_counts[idx] += 1
                else:
                    for idx, pred in zip(oob_indices, oob_pred):
                        self._oob_predictions[idx] += pred
                        oob_counts[idx] += 1
        
        # Calculate OOB score
        if self.oob_score:
            valid_oob = oob_counts > 0
            if np.any(valid_oob):
                if self.task == 'classification':
                    oob_pred = np.argmax(self._oob_predictions[valid_oob], axis=1)
                    self.oob_score_ = np.mean(oob_pred == y[valid_oob])
                else:
                    oob_pred = self._oob_predictions[valid_oob] / oob_counts[valid_oob]
                    self.oob_score_ = 1.0 - np.mean((oob_pred - y[valid_oob])**2) / np.var(y[valid_oob])
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        return self
    
    def _calculate_feature_importances(self):
        """Calculate feature importances based on tree depth usage."""
        importance = np.zeros(self.n_features_)
        
        for i, tree in enumerate(self.estimators_):
            if hasattr(tree, 'depth_features_'):
                # Map feature indices back to original feature space
                feature_subset = self._feature_subsets[i]
                for depth, subset_feature_idx in tree.depth_features_.items():
                    # Convert from subset index to original feature index
                    if 0 <= subset_feature_idx < len(feature_subset):
                        original_feature_idx = feature_subset[subset_feature_idx]
                        if 0 <= original_feature_idx < self.n_features_:
                            # Weight by inverse depth (shallower = more important)
                            weight = 1.0 / (depth + 1.0)
                            importance[original_feature_idx] += weight
        
        # Normalize
        if importance.sum() > 0:
            importance /= importance.sum()
        
        self.feature_importances_ = importance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or regression values for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self.estimators_:
            raise ValueError("Forest not fitted yet.")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        all_predictions = np.zeros((n_samples, self.n_estimators))
        for i, tree in enumerate(self.estimators_):
            # Use the same feature subset this tree was trained on
            feature_subset = self._feature_subsets[i]
            X_subset = X[:, feature_subset]
            all_predictions[:, i] = tree.predict(X_subset)
        
        # Aggregate predictions
        if self.task == 'classification':
            # Majority voting
            predictions = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                unique, counts = np.unique(all_predictions[i, :], return_counts=True)
                predictions[i] = int(unique[np.argmax(counts)])
            return predictions
        else:
            # Average
            return np.mean(all_predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Only available for classification tasks.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.estimators_:
            raise ValueError("Forest not fitted yet.")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        proba = np.zeros((n_samples, self.n_classes_))
        
        for i, tree in enumerate(self.estimators_):
            # Use the same feature subset this tree was trained on
            feature_subset = self._feature_subsets[i]
            X_subset = X[:, feature_subset]
            predictions = tree.predict(X_subset)
            for j, pred in enumerate(predictions):
                pred_class = int(pred)
                if 0 <= pred_class < self.n_classes_:
                    proba[j, pred_class] += 1
        
        # Normalize to get probabilities
        proba /= self.n_estimators
        
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the score on the given data.
        
        For classification: accuracy
        For regression: R^2 score
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels or values.
        
        Returns
        -------
        score : float
            Score value.
        """
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(y_pred == y)
        else:
            # R^2 score
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
