"""
Cython-accelerated Optimal Restricted Decision Tree.

This implementation maintains the same optimal two-pass algorithm as OptimalRestrictedDecisionTree
but uses optimized NumPy operations and optional Cython acceleration for performance-critical sections.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
from collections import Counter

# Try to import Cython-accelerated functions
try:
    from .src._optimal_tree_cython import (
        calculate_gini_fast,
        calculate_entropy_fast,
        calculate_mse_fast,
        find_best_threshold_vectorized
    )
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


class Node:
    """Lightweight node for Optimal RDT."""
    __slots__ = ['indices', 'left', 'right', 'split_info', 'value', 'class_counts']
    
    def __init__(
        self,
        indices: List[int],
        split_info: Optional[Dict[str, Any]] = None,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
        value: Optional[Union[int, float]] = None,
        class_counts: Optional[Dict[int, int]] = None
    ):
        self.indices = indices
        self.left = left
        self.right = right
        self.split_info = split_info
        self.value = value
        self.class_counts = class_counts
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class OptimalRDTCython:
    """
    Cython-accelerated Optimal Restricted Decision Tree.
    
    Uses the same optimal two-pass algorithm as OptimalRestrictedDecisionTree but with
    performance optimizations using vectorized NumPy operations and optional Cython.
    
    Parameters
    ----------
    task : str, default='classification'
        'classification' or 'regression'
    criterion : str, optional
        Splitting criterion. If None, defaults to 'gini' for classification or 'mse' for regression.
    max_depth : int, default=5
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node
    """
    
    def __init__(
        self,
        task: str = 'classification',
        criterion: Optional[str] = None,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        self.task = task.lower()
        if self.task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")
        
        if criterion is None:
            self.criterion = 'gini' if self.task == 'classification' else 'mse'
        else:
            self.criterion = criterion.lower()
        
        # Validate criterion
        if self.task == 'classification' and self.criterion not in ['gini', 'entropy']:
            raise ValueError("For classification, criterion must be 'gini' or 'entropy'")
        if self.task == 'regression' and self.criterion not in ['mse', 'mae']:
            raise ValueError("For regression, criterion must be 'mse' or 'mae'")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
        # Set during fit
        self.root = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.depth_features_ = {}  # Critical for feature tracking
        
        # Cache for performance
        self._X = None
        self._y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimalRDTCython':
        """Build the optimal restricted decision tree."""
        X = np.asarray(X, dtype=np.float64, order='C')
        y = np.asarray(y, dtype=np.float64)
        
        self.n_features_ = X.shape[1]
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        # Cache data for fast access
        self._X = X
        self._y = y
        
        indices = np.arange(len(X), dtype=np.int64)
        self.root = self._build_tree_optimal(indices)
        
        # Clear cache to save memory
        self._X = None
        self._y = None
        
        return self
    
    def _build_tree_optimal(self, root_indices: np.ndarray) -> Node:
        """Build tree using optimal two-pass algorithm with vectorized operations."""
        root = Node(indices=root_indices.tolist())
        current_level = [root]
        
        for depth in range(self.max_depth):
            if not current_level:
                break
            
            # Filter nodes that should be split
            nodes_to_split = []
            for node in current_level:
                if self._should_split(node, depth):
                    nodes_to_split.append(node)
                else:
                    self._make_leaf(node)
            
            if not nodes_to_split:
                break
            
            # OPTIMAL: Find best feature for this depth across ALL nodes
            best_feature, node_thresholds = self._find_optimal_feature_for_depth(nodes_to_split)
            
            if best_feature is None:
                for node in nodes_to_split:
                    self._make_leaf(node)
                break
            
            # Store feature for this depth
            self.depth_features_[depth] = best_feature
            
            # Split all nodes using their optimal thresholds
            next_level = []
            for node in nodes_to_split:
                threshold = node_thresholds.get(id(node))
                if threshold is None:
                    self._make_leaf(node)
                    continue
                
                left_indices, right_indices = self._split_indices(
                    node.indices, best_feature, threshold
                )
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    self._make_leaf(node)
                    continue
                
                # Create children
                node.split_info = {'feature': best_feature, 'threshold': threshold}
                node.left = Node(indices=left_indices)
                node.right = Node(indices=right_indices)
                next_level.extend([node.left, node.right])
            
            current_level = next_level
        
        # Make remaining nodes leaves
        for node in current_level:
            if node.value is None:
                self._make_leaf(node)
        
        return root
    
    def _should_split(self, node: Node, depth: int) -> bool:
        """Check if node should be split."""
        if depth >= self.max_depth:
            return False
        if len(node.indices) < self.min_samples_split:
            return False
        y_node = self._y[node.indices]
        if len(np.unique(y_node)) == 1:
            return False
        return True
    
    def _make_leaf(self, node: Node) -> None:
        """Convert node to leaf."""
        y_node = self._y[node.indices]
        if self.task == 'classification':
            counter = Counter(y_node)
            node.value = counter.most_common(1)[0][0]
            node.class_counts = dict(counter)
        else:
            node.value = np.mean(y_node)
    
    def _find_optimal_feature_for_depth(
        self,
        nodes: List[Node]
    ) -> Tuple[Optional[int], Dict[int, float]]:
        """
        Find optimal feature for all nodes at this depth (two-pass algorithm).
        
        Returns
        -------
        best_feature : int or None
        node_thresholds : dict mapping node_id to threshold
        """
        best_feature = None
        best_total_improvement = -np.inf
        best_node_thresholds = {}
        
        # Try each feature
        for feature_idx in range(self.n_features_):
            total_improvement = 0.0
            node_thresholds = {}
            valid_split_found = False
            
            # For each node, find its best threshold for this feature
            for node in nodes:
                threshold, improvement = self._find_best_threshold_for_node_feature(
                    node.indices, feature_idx
                )
                
                if threshold is not None:
                    total_improvement += improvement
                    node_thresholds[id(node)] = threshold
                    valid_split_found = True
            
            # Check if this feature is best so far
            if valid_split_found and total_improvement > best_total_improvement:
                best_total_improvement = total_improvement
                best_feature = feature_idx
                best_node_thresholds = node_thresholds
        
        return best_feature, best_node_thresholds
    
    def _find_best_threshold_for_node_feature(
        self,
        indices: List[int],
        feature_idx: int
    ) -> Tuple[Optional[float], float]:
        """
        Find best threshold for a specific feature at a specific node.
        Uses vectorized operations for speed.
        """
        if len(indices) < self.min_samples_split:
            return None, 0.0
        
        indices_arr = np.array(indices, dtype=np.int64)
        feature_values = self._X[indices_arr, feature_idx]
        y_subset = self._y[indices_arr]
        
        # Get unique values efficiently
        unique_vals = np.unique(feature_values)
        if len(unique_vals) < 2:
            return None, 0.0
        
        # Compute candidate thresholds
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        
        best_improvement = -np.inf
        best_threshold = None
        
        # Vectorized threshold search
        for threshold in thresholds:
            left_mask = feature_values <= threshold
            
            n_left = np.sum(left_mask)
            n_right = len(feature_values) - n_left
            
            # Check minimum samples constraint
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            
            y_left = y_subset[left_mask]
            y_right = y_subset[~left_mask]
            
            # Calculate improvement
            improvement = self._calculate_improvement(y_subset, y_left, y_right)
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_threshold = threshold
        
        if best_threshold is None:
            return None, 0.0
        
        return best_threshold, best_improvement
    
    def _split_indices(
        self,
        indices: List[int],
        feature_idx: int,
        threshold: float
    ) -> Tuple[List[int], List[int]]:
        """Split indices based on feature threshold."""
        indices_arr = np.array(indices, dtype=np.int64)
        feature_values = self._X[indices_arr, feature_idx]
        left_mask = feature_values <= threshold
        
        left_indices = indices_arr[left_mask].tolist()
        right_indices = indices_arr[~left_mask].tolist()
        
        return left_indices, right_indices
    
    def _calculate_criterion(self, y: np.ndarray) -> float:
        """Calculate splitting criterion (uses Cython if available)."""
        if len(y) == 0:
            return 0.0
        
        if self.criterion == 'gini':
            if HAS_CYTHON:
                return calculate_gini_fast(y)
            else:
                _, counts = np.unique(y, return_counts=True)
                probs = counts / len(y)
                return 1.0 - np.sum(probs ** 2)
        
        elif self.criterion == 'entropy':
            if HAS_CYTHON:
                return calculate_entropy_fast(y)
            else:
                _, counts = np.unique(y, return_counts=True)
                probs = counts / len(y)
                return -np.sum(probs * np.log2(probs + 1e-10))
        
        elif self.criterion == 'mse':
            if HAS_CYTHON:
                return calculate_mse_fast(y)
            else:
                return np.var(y)
        
        elif self.criterion == 'mae':
            median = np.median(y)
            return np.mean(np.abs(y - median))
    
    def _calculate_improvement(
        self,
        y_parent: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> float:
        """Calculate improvement from split."""
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = len(y_parent)
        
        if n_total == 0:
            return 0.0
        
        parent_criterion = self._calculate_criterion(y_parent)
        left_criterion = self._calculate_criterion(y_left)
        right_criterion = self._calculate_criterion(y_right)
        
        weighted_child = (n_left / n_total) * left_criterion + (n_right / n_total) * right_criterion
        improvement = parent_criterion - weighted_child
        
        return improvement
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X."""
        if self.root is None:
            raise ValueError("Tree not fitted yet. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> Union[int, float]:
        """Predict value for a single sample."""
        if node.is_leaf():
            return node.value
        
        feature_idx = node.split_info['feature']
        threshold = node.split_info['threshold']
        
        if x[feature_idx] <= threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if self.root is None:
            raise ValueError("Tree not fitted yet. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        probas = np.array([self._predict_proba_sample(x, self.root) for x in X])
        return probas
    
    def _predict_proba_sample(self, x: np.ndarray, node: Node) -> np.ndarray:
        """Get probability distribution for a single sample."""
        if node.is_leaf():
            probs = np.zeros(self.n_classes_)
            if node.class_counts is not None:
                total = sum(node.class_counts.values())
                for cls, count in node.class_counts.items():
                    cls_idx = np.where(self.classes_ == cls)[0][0]
                    probs[cls_idx] = count / total
            return probs
        
        feature_idx = node.split_info['feature']
        threshold = node.split_info['threshold']
        
        if x[feature_idx] <= threshold:
            return self._predict_proba_sample(x, node.left)
        else:
            return self._predict_proba_sample(x, node.right)
