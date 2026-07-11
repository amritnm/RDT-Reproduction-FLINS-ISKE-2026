"""
AdaBoost implementation using Cython-accelerated RDT as base learner.

Provides scikit-learn compatible interface with high performance.
"""

import numpy as np
from typing import Optional
import sys
import os

# Add parent directory to path to import rdt_cython
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rdt_cython.cython_rdt import CythonRestrictedDecisionTree


class AdaBoostRDT:
    """
    AdaBoost (Adaptive Boosting) using Restricted Decision Trees.
    
    Trains trees sequentially, weighting samples based on previous errors.
    Each tree focuses on samples that previous trees misclassified.
    
    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting stages.
    
    learning_rate : float, default=1.0
        Shrinkage parameter. Lower values require more estimators
        but can improve generalization.
    
    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME'
        Boosting algorithm:
        - 'SAMME': Discrete AdaBoost (multi-class)
        - 'SAMME.R': Real AdaBoost (uses probabilities, binary only)
    
    max_depth : int, default=3
        Maximum depth of each tree (stumps or shallow trees work best).
    
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
    
    random_state : int, optional
        Random seed for reproducibility.
    
    verbose : int, default=0
        Verbosity level (0 = silent, 1 = progress).
    
    Attributes
    ----------
    estimators_ : list
        The collection of fitted tree estimators.
    
    estimator_weights_ : ndarray of shape (n_estimators,)
        Weights for each estimator in the boosted ensemble.
    
    estimator_errors_ : ndarray of shape (n_estimators,)
        Classification error for each estimator.
    
    feature_importances_ : ndarray of shape (n_features,)
        Aggregated feature importance scores.
    
    n_features_ : int
        Number of features when fit is performed.
    
    n_classes_ : int
        Number of classes.
    
    classes_ : ndarray
        The classes labels.
    
    Notes
    -----
    AdaBoost is only for classification tasks. For regression, use
    GradientBoostingRDT instead.
    """
    
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: str = 'SAMME',
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm.upper()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        
        if self.algorithm not in ['SAMME', 'SAMME.R']:
            raise ValueError("algorithm must be 'SAMME' or 'SAMME.R'")
        
        # Will be set during fit
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(n_estimators)
        self.estimator_errors_ = np.zeros(n_estimators)
        self.feature_importances_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostRDT':
        """
        Build a boosted ensemble from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values (class labels).
        
        Returns
        -------
        self : AdaBoostRDT
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Get class information
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        # Setup random state
        rng = np.random.RandomState(self.random_state)
        
        # Boosting loop
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
        for i in range(self.n_estimators):
            if self.verbose > 0:
                print(f"Training estimator {i+1}/{self.n_estimators}...")
            
            # Sample with replacement according to weights
            sample_indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )
            
            # Train tree on weighted sample
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]
            
            tree = CythonRestrictedDecisionTree(
                task='classification',
                criterion='gini',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sampled, y_sampled)
            
            # Get predictions on full training set
            y_pred = tree.predict(X)
            
            # Calculate weighted error
            incorrect = (y_pred != y).astype(float)
            error = np.sum(sample_weights * incorrect)
            
            # Handle edge cases
            if error <= 0:
                # Perfect classifier - use it and stop
                self.estimators_.append(tree)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(0.0)
                break
            
            if error >= 1 - 1.0 / self.n_classes_:
                # Worse than random - stop boosting
                if self.verbose > 0:
                    print(f"Stopping early at iteration {i+1}: error too high ({error:.4f})")
                break
            
            # Calculate estimator weight (SAMME algorithm)
            estimator_weight = self.learning_rate * (
                np.log((1.0 - error) / error) + 
                np.log(self.n_classes_ - 1)
            )
            
            # Update sample weights
            sample_weights *= np.exp(estimator_weight * incorrect)
            sample_weights /= np.sum(sample_weights)  # Normalize
            
            # Store estimator
            self.estimators_.append(tree)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_errors_.append(error)
        
        # Convert to arrays
        self.estimator_weights_ = np.array(self.estimator_weights_)
        self.estimator_errors_ = np.array(self.estimator_errors_)
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        return self
    
    def _calculate_feature_importances(self):
        """Calculate weighted feature importances."""
        importance = np.zeros(self.n_features_)
        
        for tree, weight in zip(self.estimators_, self.estimator_weights_):
            if hasattr(tree, 'depth_features_'):
                for depth, feature_idx in tree.depth_features_.items():
                    if 0 <= feature_idx < self.n_features_:
                        # Weight by estimator weight and inverse depth
                        tree_weight = weight / (depth + 1.0)
                        importance[feature_idx] += tree_weight
        
        # Normalize
        if importance.sum() > 0:
            importance /= importance.sum()
        
        self.feature_importances_ = importance
    
    def _predict_staged(self, X: np.ndarray, n_estimators: int) -> np.ndarray:
        """Predict using first n_estimators."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Accumulate weighted predictions
        predictions = np.zeros((n_samples, self.n_classes_))
        
        for tree, weight in zip(
            self.estimators_[:n_estimators],
            self.estimator_weights_[:n_estimators]
        ):
            tree_pred = tree.predict(X)
            for i, pred in enumerate(tree_pred):
                pred_class = int(pred)
                if 0 <= pred_class < self.n_classes_:
                    predictions[i, pred_class] += weight
        
        # Return class with highest weighted vote
        return np.argmax(predictions, axis=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if not self.estimators_:
            raise ValueError("Estimator not fitted yet.")
        
        return self._predict_staged(X, len(self.estimators_))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if not self.estimators_:
            raise ValueError("Estimator not fitted yet.")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Accumulate weighted predictions
        predictions = np.zeros((n_samples, self.n_classes_))
        
        for tree, weight in zip(self.estimators_, self.estimator_weights_):
            tree_pred = tree.predict(X)
            for i, pred in enumerate(tree_pred):
                pred_class = int(pred)
                if 0 <= pred_class < self.n_classes_:
                    predictions[i, pred_class] += weight
        
        # Normalize to get probabilities
        predictions = np.exp(predictions)
        predictions /= predictions.sum(axis=1, keepdims=True)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        scores : ndarray of shape (n_samples, n_classes)
            Decision function values.
        """
        if not self.estimators_:
            raise ValueError("Estimator not fitted yet.")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Accumulate weighted predictions
        scores = np.zeros((n_samples, self.n_classes_))
        
        for tree, weight in zip(self.estimators_, self.estimator_weights_):
            tree_pred = tree.predict(X)
            for i, pred in enumerate(tree_pred):
                pred_class = int(pred)
                if 0 <= pred_class < self.n_classes_:
                    scores[i, pred_class] += weight
        
        return scores
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels.
        
        Returns
        -------
        score : float
            Mean accuracy.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def staged_score(self, X: np.ndarray, y: np.ndarray):
        """
        Return staged scores for X, y.
        
        Yields accuracy after each boosting iteration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels.
        
        Yields
        ------
        score : float
            Accuracy after each iteration.
        """
        for i in range(1, len(self.estimators_) + 1):
            y_pred = self._predict_staged(X, i)
            yield np.mean(y_pred == y)
