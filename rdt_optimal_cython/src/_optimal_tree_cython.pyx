# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-accelerated functions for Optimal Restricted Decision Tree.

This module contains performance-critical functions implemented in Cython
with C type declarations for maximum speed.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log2, sqrt
from libc.stdlib cimport malloc, free

# Initialize NumPy C API
cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double calculate_gini_fast(cnp.ndarray[cnp.float64_t, ndim=1] y):
    """
    Calculate Gini impurity with C-speed performance.
    
    Parameters
    ----------
    y : np.ndarray[float64]
        Target values
    
    Returns
    -------
    double : Gini impurity
    """
    cdef int n = y.shape[0]
    if n == 0:
        return 0.0
    
    cdef int i
    cdef double gini = 1.0
    cdef double count
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unique_vals
    cdef cnp.ndarray[cnp.int64_t, ndim=1] counts
    
    # Get unique values and counts
    unique_vals, counts = np.unique(y, return_counts=True)
    
    # Calculate Gini
    for i in range(len(unique_vals)):
        count = <double>counts[i] / <double>n
        gini -= count * count
    
    return gini


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double calculate_entropy_fast(cnp.ndarray[cnp.float64_t, ndim=1] y):
    """
    Calculate entropy with C-speed performance.
    
    Parameters
    ----------
    y : np.ndarray[float64]
        Target values
    
    Returns
    -------
    double : Entropy
    """
    cdef int n = y.shape[0]
    if n == 0:
        return 0.0
    
    cdef int i
    cdef double entropy = 0.0
    cdef double prob
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unique_vals
    cdef cnp.ndarray[cnp.int64_t, ndim=1] counts
    
    # Get unique values and counts
    unique_vals, counts = np.unique(y, return_counts=True)
    
    # Calculate entropy
    for i in range(len(unique_vals)):
        prob = <double>counts[i] / <double>n
        if prob > 0.0:
            entropy -= prob * log2(prob)
    
    return entropy


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double calculate_mse_fast(cnp.ndarray[cnp.float64_t, ndim=1] y):
    """
    Calculate mean squared error (variance) with C-speed performance.
    
    Parameters
    ----------
    y : np.ndarray[float64]
        Target values
    
    Returns
    -------
    double : MSE (variance)
    """
    cdef int n = y.shape[0]
    if n == 0:
        return 0.0
    
    cdef int i
    cdef double mean = 0.0
    cdef double variance = 0.0
    cdef double diff
    
    # Calculate mean
    for i in range(n):
        mean += y[i]
    mean /= <double>n
    
    # Calculate variance
    for i in range(n):
        diff = y[i] - mean
        variance += diff * diff
    
    variance /= <double>n
    
    return variance


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple find_best_threshold_vectorized(
    cnp.ndarray[cnp.float64_t, ndim=1] feature_values,
    cnp.ndarray[cnp.float64_t, ndim=1] y_values,
    int min_samples_leaf,
    str criterion
):
    """
    Find best threshold for splitting using vectorized operations.
    
    Parameters
    ----------
    feature_values : np.ndarray[float64]
        Feature values for samples
    y_values : np.ndarray[float64]
        Target values
    min_samples_leaf : int
        Minimum samples required in each leaf
    criterion : str
        Splitting criterion ('gini', 'entropy', 'mse')
    
    Returns
    -------
    tuple : (best_threshold, best_improvement)
    """
    cdef int n = feature_values.shape[0]
    if n < 2 * min_samples_leaf:
        return None, 0.0
    
    # Get sorted indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sorted_idx = np.argsort(feature_values)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sorted_features = feature_values[sorted_idx]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sorted_y = y_values[sorted_idx]
    
    cdef double best_threshold = 0.0
    cdef double best_improvement = -1e10
    cdef double threshold, improvement
    cdef int i, n_left, n_right
    cdef bint threshold_found = False
    
    # Try each potential split point
    for i in range(min_samples_leaf, n - min_samples_leaf):
        # Skip if values are the same
        if sorted_features[i] == sorted_features[i-1]:
            continue
        
        threshold = (sorted_features[i-1] + sorted_features[i]) / 2.0
        n_left = i
        n_right = n - i
        
        # Calculate improvement
        improvement = _calculate_split_improvement(
            sorted_y, n_left, n_right, criterion
        )
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_threshold = threshold
            threshold_found = True
    
    if not threshold_found:
        return None, 0.0
    
    return best_threshold, best_improvement


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _calculate_split_improvement(
    cnp.ndarray[cnp.float64_t, ndim=1] sorted_y,
    int n_left,
    int n_right,
    str criterion
):
    """
    Calculate improvement from a split (internal function).
    
    This function calculates how much the split improves the criterion.
    """
    cdef int n_total = n_left + n_right
    cdef double parent_criterion, left_criterion, right_criterion
    cdef double improvement
    
    # Calculate parent criterion
    if criterion == 'gini':
        parent_criterion = calculate_gini_fast(sorted_y)
    elif criterion == 'entropy':
        parent_criterion = calculate_entropy_fast(sorted_y)
    else:  # mse
        parent_criterion = calculate_mse_fast(sorted_y)
    
    # Calculate left child criterion
    if criterion == 'gini':
        left_criterion = calculate_gini_fast(sorted_y[:n_left])
    elif criterion == 'entropy':
        left_criterion = calculate_entropy_fast(sorted_y[:n_left])
    else:  # mse
        left_criterion = calculate_mse_fast(sorted_y[:n_left])
    
    # Calculate right child criterion
    if criterion == 'gini':
        right_criterion = calculate_gini_fast(sorted_y[n_left:])
    elif criterion == 'entropy':
        right_criterion = calculate_entropy_fast(sorted_y[n_left:])
    else:  # mse
        right_criterion = calculate_mse_fast(sorted_y[n_left:])
    
    # Calculate weighted improvement
    improvement = parent_criterion - (
        (<double>n_left / <double>n_total) * left_criterion +
        (<double>n_right / <double>n_total) * right_criterion
    )
    
    return improvement


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.int64_t, ndim=1] split_indices_fast(
    cnp.ndarray[cnp.int64_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=2] X,
    int feature_idx,
    double threshold
):
    """
    Split indices based on feature threshold (returns left indices, mask for efficiency).
    
    Parameters
    ----------
    indices : np.ndarray[int64]
        Indices to split
    X : np.ndarray[float64, ndim=2]
        Feature matrix
    feature_idx : int
        Feature index to split on
    threshold : double
        Threshold value
    
    Returns
    -------
    np.ndarray[int64] : Left indices (values <= threshold)
    """
    cdef int n = indices.shape[0]
    cdef int i, count = 0
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_indices
    cdef list left_list = []
    
    # Count and collect left indices
    for i in range(n):
        if X[indices[i], feature_idx] <= threshold:
            left_list.append(indices[i])
    
    left_indices = np.array(left_list, dtype=np.int64)
    return left_indices
