"""
Gradient Boosting implementation using Cython-accelerated RDT as base learner.

Provides scikit-learn compatible interface with high performance.
"""

import numpy as np
from typing import Optional, Union
import sys
import os

# Add parent directory to path to import rdt_optimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import OptimalRestrictedDecisionTree for globally optimal feature selection
from rdt.optimal_restricted_tree.optimal_restricted_decision_tree import OptimalRestrictedDecisionTree


class GradientBoostingRDT:
    """
    Gradient Boosting using Restricted Decision Trees.
    
    Trains trees sequentially to fit the negative gradient (residuals)
    of the loss function. Supports both classification and regression.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    
    learning_rate : float, default=0.1
        Shrinkage parameter. Lower values require more estimators
        but can improve generalization.
    
    task : {'classification', 'regression'}, default='classification'
        Type of task.
    
    loss : str, optional
        Loss function to optimize:
        - Classification: 'deviance' (logistic), 'exponential'
        - Regression: 'ls' (least squares), 'lad' (least absolute deviation)
        If None, uses 'deviance' for classification and 'ls' for regression.
    
    max_depth : int, default=3
        Maximum depth of each tree (shallow trees typically work best).
    
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
    
    subsample : float, default=1.0
        Fraction of samples to use for fitting each tree.
        If < 1.0, implements stochastic gradient boosting.
    
    validation_fraction : float, default=0.1
        Fraction of training data to use for early stopping validation.
        Only used if n_iter_no_change is set.
    
    n_iter_no_change : int, optional
        Number of iterations with no improvement to wait before stopping.
        If None, no early stopping is performed.
    
    tol : float, default=1e-4
        Tolerance for early stopping. Training stops if validation score
        doesn't improve by at least tol for n_iter_no_change iterations.
    
    random_state : int, optional
        Random seed for reproducibility.
    
    verbose : int, default=0
        Verbosity level (0 = silent, 1 = progress).
    
    Attributes
    ----------
    estimators_ : list
        The collection of fitted tree estimators.
    
    train_score_ : ndarray of shape (n_estimators,)
        Training score at each iteration.
    
    init_prediction_ : float or ndarray
        The initial prediction (baseline).
    
    feature_importances_ : ndarray of shape (n_features,)
        Aggregated feature importance scores.
    
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
        learning_rate: float = 0.1,
        task: str = 'classification',
        loss: Optional[str] = None,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 1e-4,
        use_ordered_boosting: bool = True,
        categorical_features: Optional[list] = None,
        prior_weight: float = 1.0,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.task = task.lower()
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.use_ordered_boosting = use_ordered_boosting
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.prior_weight = prior_weight
        self.random_state = random_state
        self.verbose = verbose
        
        if self.task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")
        
        # Set default loss
        if self.loss is None:
            self.loss = 'deviance' if self.task == 'classification' else 'ls'
        
        # Validate loss for task
        if self.task == 'classification':
            if self.loss not in ['deviance', 'exponential']:
                raise ValueError("For classification, loss must be 'deviance' or 'exponential'")
        else:
            if self.loss not in ['ls', 'lad']:
                raise ValueError("For regression, loss must be 'ls' or 'lad'")
        
        # Will be set during fit
        self.estimators_ = []
        self.train_score_ = []
        self.init_prediction_ = None
        self.feature_importances_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.global_prior_ = None
        self.cat_encodings_ = {}
    
    def _compute_init_prediction(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """Compute initial prediction (baseline)."""
        if self.task == 'classification':
            # For binary classification with deviance loss, use log-odds
            if self.loss == 'deviance':
                # Count positive class (assuming binary 0/1)
                pos_ratio = np.mean(y)
                pos_ratio = np.clip(pos_ratio, 1e-7, 1 - 1e-7)
                return np.log(pos_ratio / (1 - pos_ratio))
            else:
                # For exponential loss, start with 0
                return 0.0
        else:
            # For regression, use mean
            return np.mean(y)
    
    def _compute_residuals(
        self,
        y: np.ndarray,
        predictions: np.ndarray
    ) -> np.ndarray:
        """Compute negative gradient (residuals) for current predictions."""
        if self.task == 'regression':
            if self.loss == 'ls':
                # Least squares: residual = y - y_pred
                return y - predictions
            else:  # lad
                # Least absolute deviation: sign of residual
                return np.sign(y - predictions)
        else:  # classification
            if self.loss == 'deviance':
                # Logistic loss gradient
                # For binary classification: y - sigmoid(f)
                # Clip predictions to prevent overflow in exp
                predictions_clipped = np.clip(predictions, -50, 50)
                prob = 1.0 / (1.0 + np.exp(-predictions_clipped))
                # Clip probabilities to prevent numerical issues
                prob = np.clip(prob, 1e-7, 1 - 1e-7)
                return y - prob
            else:  # exponential
                # Exponential loss gradient
                # For binary classification (y in {-1, 1}): y * exp(-y * f)
                y_signed = 2 * y - 1  # Convert from {0,1} to {-1,1}
                # Clip to prevent overflow
                exp_term = np.exp(-np.clip(y_signed * predictions, -50, 50))
                return y_signed * exp_term
    
    def _update_predictions(
        self,
        predictions: np.ndarray,
        tree_predictions: np.ndarray
    ) -> np.ndarray:
        """Update predictions with new tree predictions."""
        return predictions + self.learning_rate * tree_predictions
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRDT':
        """
        Build a gradient boosting ensemble from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : GradientBoostingRDT
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            if self.n_classes_ != 2:
                raise ValueError("Currently only binary classification is supported")
        
        # Setup random state
        rng = np.random.RandomState(self.random_state)
        
        # Split for validation if early stopping is enabled
        if self.n_iter_no_change is not None:
            n_val = int(n_samples * self.validation_fraction)
            indices = rng.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        n_train = len(X_train)
        
        # Initialize predictions with baseline
        self.init_prediction_ = self._compute_init_prediction(y_train)
        predictions = np.full(n_train, self.init_prediction_, dtype=np.float64)
        
        if X_val is not None:
            val_predictions = np.full(len(X_val), self.init_prediction_, dtype=np.float64)
            best_val_score = -np.inf
            no_improvement_count = 0
        
        # Boosting loop
        self.estimators_ = []
        self.train_score_ = []
        
        for i in range(self.n_estimators):
            # Compute residuals (negative gradient)
            residuals = self._compute_residuals(y_train, predictions)
            
            # Subsample if requested
            if self.subsample < 1.0:
                sample_size = int(n_train * self.subsample)
                sample_indices = rng.choice(n_train, size=sample_size, replace=False)
                X_sample = X_train[sample_indices]
                residuals_sample = residuals[sample_indices]
            else:
                X_sample = X_train
                residuals_sample = residuals
            
            # Fit tree to residuals using Optimal RDT (globally optimal feature selection)
            tree = OptimalRestrictedDecisionTree(
                task='regression',  # Always fit to residuals
                criterion='mse',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sample, residuals_sample)
            
            # Update predictions
            tree_preds = tree.predict(X_train)
            predictions = self._update_predictions(predictions, tree_preds)
            
            # Store estimator
            self.estimators_.append(tree)
            
            # Compute training score
            if self.task == 'classification':
                train_score = np.mean((predictions > 0) == y_train)
            else:
                # R^2 score
                ss_res = np.mean((y_train - predictions) ** 2)
                ss_tot = np.var(y_train)
                train_score = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            
            self.train_score_.append(train_score)
            
            if self.verbose > 0:
                print(f"Iteration {i+1}/{self.n_estimators}, Train Score: {train_score:.4f}")
            
            # Early stopping check
            if X_val is not None:
                # Update validation predictions
                val_tree_preds = tree.predict(X_val)
                val_predictions = self._update_predictions(val_predictions, val_tree_preds)
                
                # Compute validation score
                if self.task == 'classification':
                    val_score = np.mean((val_predictions > 0) == y_val)
                else:
                    ss_res = np.mean((y_val - val_predictions) ** 2)
                    ss_tot = np.var(y_val)
                    val_score = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                
                if self.verbose > 0:
                    print(f"  Val Score: {val_score:.4f}")
                
                # Check for improvement
                if val_score > best_val_score + self.tol:
                    best_val_score = val_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= self.n_iter_no_change:
                        if self.verbose > 0:
                            print(f"Early stopping at iteration {i+1}")
                        break
        
        # Convert to array
        self.train_score_ = np.array(self.train_score_)
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        return self
    
    def _calculate_feature_importances(self):
        """Calculate weighted feature importances."""
        importance = np.zeros(self.n_features_)
        
        for tree in self.estimators_:
            if hasattr(tree, 'depth_features_'):
                for depth, feature_idx in tree.depth_features_.items():
                    if 0 <= feature_idx < self.n_features_:
                        # Weight by inverse depth (shallower = more important)
                        weight = 1.0 / (depth + 1.0)
                        importance[feature_idx] += weight
        
        # Normalize
        if importance.sum() > 0:
            importance /= importance.sum()
        
        self.feature_importances_ = importance
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict raw values (before applying link function)."""
        if not self.estimators_:
            raise ValueError("Estimator not fitted yet.")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        predictions = np.full(n_samples, self.init_prediction_, dtype=np.float64)
        
        # Add contributions from all trees
        for tree in self.estimators_:
            tree_preds = tree.predict(X)
            predictions += self.learning_rate * tree_preds
        
        return predictions
    
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
        raw_predictions = self._predict_raw(X)
        
        if self.task == 'classification':
            # Convert to class labels
            return (raw_predictions > 0).astype(int)
        else:
            return raw_predictions
    
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
        
        raw_predictions = self._predict_raw(X)
        
        # Apply sigmoid to get probabilities
        prob_positive = 1.0 / (1.0 + np.exp(-raw_predictions))
        prob_negative = 1.0 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Decision function values.
        """
        return self._predict_raw(X)
    
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
    
    def staged_predict(self, X: np.ndarray):
        """
        Predict at each stage for X.
        
        Yields predictions after each boosting iteration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Yields
        ------
        y : ndarray of shape (n_samples,)
            Predictions after each iteration.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        predictions = np.full(n_samples, self.init_prediction_, dtype=np.float64)
        
        for tree in self.estimators_:
            tree_preds = tree.predict(X)
            predictions += self.learning_rate * tree_preds
            
            if self.task == 'classification':
                yield (predictions > 0).astype(int)
            else:
                yield predictions.copy()
    
    def staged_predict_proba(self, X: np.ndarray):
        """
        Predict class probabilities at each stage for X.
        
        Only available for classification tasks.
        
        Yields probabilities after each boosting iteration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Yields
        ------
        p : ndarray of shape (n_samples, n_classes)
            Class probabilities after each iteration.
        """
        if self.task != 'classification':
            raise ValueError("staged_predict_proba only available for classification")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        predictions = np.full(n_samples, self.init_prediction_, dtype=np.float64)
        
        for tree in self.estimators_:
            tree_preds = tree.predict(X)
            predictions += self.learning_rate * tree_preds
            
            # Apply sigmoid to get probabilities
            prob_positive = 1.0 / (1.0 + np.exp(-np.clip(predictions, -50, 50)))
            prob_negative = 1.0 - prob_positive
            
            yield np.column_stack([prob_negative, prob_positive])
