"""
sklearn-compatible wrapper for OSDT (Optimal Sparse Decision Trees)

This wrapper adapts the OSDT implementation from https://github.com/xiyanghu/OSDT
to make it compatible with sklearn's API for use in GridSearchCV and cross-validation.
"""

import sys
import os
import numpy as np
import warnings

# Add OSDT to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'osdt_lib', 'src'))

try:
    from osdt import OSDT as OSDT_Original
    OSDT_AVAILABLE = True
except ImportError as e:
    print(f"OSDT import failed: {e}")
    OSDT_AVAILABLE = False


class OSDTWrapper:
    """
    sklearn-compatible wrapper for OSDT.
    
    OSDT is designed for binary classification and uses optimization to find
    the optimal sparse decision tree.
    
    Parameters
    ----------
    max_depth : int, optional (default=3)
        Maximum depth of the tree. Maps to OSDT's MAXDEPTH parameter.
    task : str, optional (default='classification')
        Task type (included for API compatibility, not used by OSDT).
    criterion : str, optional (default='gini')
        Splitting criterion (included for API compatibility, not used by OSDT).
    min_samples_split : int, optional (default=2)
        Minimum samples required to split (maps to OSDT's lamb parameter).
    min_samples_leaf : int, optional (default=1)
        Minimum samples required at leaf (maps to OSDT's lamb parameter).
    lamb : float, optional (default=None)
        OSDT's regularization parameter. If None, computed from min_samples_split.
    timelimit : int, optional (default=60)
        Time limit for OSDT optimization in seconds.
    
    Notes
    -----
    - OSDT only supports binary classification
    - OSDT may be slow on larger datasets due to optimization
    - The wrapper converts multi-class problems to binary (one-vs-rest)
    """
    
    def __init__(self, max_depth=3, task='classification', criterion='gini',
                 min_samples_split=2, min_samples_leaf=1, lamb=None,
                 timelimit=60):
        self.max_depth = max_depth
        self.task = task
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.lamb = lamb
        self.timelimit = timelimit
        self.model_ = None
        self.classes_ = None
        self.is_binary_ = False
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn API)."""
        return {
            'max_depth': self.max_depth,
            'task': self.task,
            'criterion': self.criterion,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'lamb': self.lamb,
            'timelimit': self.timelimit
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator (sklearn API)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def fit(self, X, y):
        """
        Fit the OSDT model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.array(X)
        y = np.array(y)
        
        # Store classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Check if binary
        if n_classes != 2:
            warnings.warn(
                f"OSDT only supports binary classification. "
                f"Found {n_classes} classes. Converting to binary (majority class vs rest).",
                UserWarning
            )
            # Convert to binary: majority class = 1, others = 0
            majority_class = np.argmax(np.bincount(y.astype(int)))
            y_binary = (y == majority_class).astype(int)
            self.is_binary_ = False
            self.majority_class_ = majority_class
        else:
            # Ensure binary labels are 0 and 1
            y_binary = np.where(y == self.classes_[0], 0, 1)
            self.is_binary_ = True
        
        # Map parameters to OSDT
        if self.lamb is None:
            # Estimate lamb from min_samples_split
            # OSDT's lamb is roughly: min_samples_split / (2 * n_samples)
            lamb_value = max(0.001, self.min_samples_split / (2.0 * len(y)))
        else:
            lamb_value = self.lamb
        
        # Suppress OSDT's verbose output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Create and fit OSDT model
            self.model_ = OSDT_Original(
                lamb=lamb_value,
                MAXDEPTH=self.max_depth,
                timelimit=self.timelimit,
                init_cart=True,  # Use CART initialization
                prior_metric='curiosity'
            )
            
            try:
                self.model_.fit(X, y_binary)
            except Exception as e:
                # If OSDT fails, fall back to simple majority classifier
                warnings.warn(f"OSDT fitting failed: {e}. Using majority classifier.", UserWarning)
                self.model_ = None
                self.majority_pred_ = int(y_binary.sum() / len(y_binary) >= 0.5)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        X = np.array(X)
        
        if self.model_ is None:
            # Fallback: majority classifier
            return np.full(len(X), self.classes_[self.majority_pred_])
        
        # Get OSDT prediction (returns 0 or 1)
        y_pred_binary = self.model_.predict(X)
        
        # Convert back to original labels
        if self.is_binary_:
            # Map 0 -> class 0, 1 -> class 1
            y_pred = np.where(y_pred_binary == 0, self.classes_[0], self.classes_[1])
        else:
            # Map based on majority class conversion
            y_pred = np.where(y_pred_binary == 1, self.majority_class_, 
                            # For non-majority, pick first non-majority class
                            next(c for c in self.classes_ if c != self.majority_class_))
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Note: OSDT doesn't provide probabilistic predictions,
        so this returns hard 0/1 probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities (hard 0/1).
        """
        y_pred = self.predict(X)
        n_samples = len(X)
        n_classes = len(self.classes_)
        
        proba = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(y_pred):
            class_idx = np.where(self.classes_ == pred)[0][0]
            proba[i, class_idx] = 1.0
        
        return proba


def get_osdt_wrapper():
    """
    Factory function to get OSDT wrapper if available.
    
    Returns
    -------
    OSDTWrapper or None
        Returns OSDTWrapper class if OSDT is available, None otherwise.
    """
    if OSDT_AVAILABLE:
        return OSDTWrapper
    else:
        return None
