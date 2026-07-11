# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False

"""
Complete optimized Cython implementation of RDT/ODT.
Single-file implementation for easier compilation and maximum performance.
"""

cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport log2, fabs
cimport numpy as cnp
import numpy as np

cnp.import_array()


# ============================================================================
# CRITERION CALCULATIONS (Inline for maximum performance)
# ============================================================================

cdef inline double calculate_gini(cnp.int64_t* y_int, long n_samples, long n_classes) nogil:
    """Ultra-fast Gini calculation."""
    if n_samples == 0:
        return 0.0
    
    cdef long* counts = <long*>malloc(n_classes * sizeof(long))
    memset(counts, 0, n_classes * sizeof(long))
    
    cdef long i
    for i in range(n_samples):
        counts[y_int[i]] += 1
    
    cdef double gini = 1.0
    cdef double p
    cdef double n_double = <double>n_samples
    
    for i in range(n_classes):
        if counts[i] > 0:
            p = <double>counts[i] / n_double
            gini -= p * p
    
    free(counts)
    return gini


cdef inline double calculate_mse(double* y_vals, long n_samples) nogil:
    """Ultra-fast MSE calculation."""
    if n_samples == 0:
        return 0.0
    
    cdef double mean = 0.0
    cdef long i
    for i in range(n_samples):
        mean += y_vals[i]
    mean /= <double>n_samples
    
    cdef double mse = 0.0
    cdef double diff
    for i in range(n_samples):
        diff = y_vals[i] - mean
        mse += diff * diff
    
    return mse / <double>n_samples


# ============================================================================
# TREE BUILDER (Core implementation)
# ============================================================================

cdef class FastTreeBuilder:
    """
    High-performance tree builder with pre-sorted indices.
    Optimized for both RDT and ODT.
    """
    
    cdef double[:, ::1] X  # C-contiguous for cache efficiency
    cdef double[::1] y
    cdef cnp.int64_t[:, ::1] sorted_indices  # Pre-sorted indices per feature (C-contiguous)
    cdef long n_samples
    cdef long n_features
    cdef long n_classes
    cdef bint is_classification
    cdef str criterion_name
    cdef long min_samples_split
    cdef long min_samples_leaf
    cdef long max_depth
    
    # Buffers for efficiency
    cdef cnp.int64_t* y_int_buffer
    cdef double* y_double_buffer
    cdef cnp.int64_t* left_buffer
    cdef cnp.int64_t* right_buffer
    
    def __cinit__(
        self,
        double[:, ::1] X,
        double[::1] y,
        long n_classes,
        bint is_classification,
        str criterion,
        long min_samples_split,
        long min_samples_leaf,
        long max_depth
    ):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_classes = n_classes
        self.is_classification = is_classification
        self.criterion_name = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        
        # Pre-sort all features
        self.sorted_indices = self._presort_features()
        
        # Allocate buffers
        self.y_int_buffer = <cnp.int64_t*>malloc(self.n_samples * sizeof(cnp.int64_t))
        self.y_double_buffer = <double*>malloc(self.n_samples * sizeof(double))
        self.left_buffer = <cnp.int64_t*>malloc(self.n_samples * sizeof(cnp.int64_t))
        self.right_buffer = <cnp.int64_t*>malloc(self.n_samples * sizeof(cnp.int64_t))
    
    def __dealloc__(self):
        free(self.y_int_buffer)
        free(self.y_double_buffer)
        free(self.left_buffer)
        free(self.right_buffer)
    
    def _presort_features(self):
        """Pre-sort all features once."""
        cdef cnp.ndarray[cnp.int64_t, ndim=2] sorted_idx = np.zeros(
            (self.n_features, self.n_samples), dtype=np.int64
        )
        
        X_np = np.asarray(self.X)
        for i in range(self.n_features):
            sorted_idx[i, :] = np.argsort(X_np[:, i])
        
        return sorted_idx
    
    cdef double calculate_criterion_at_indices(
        self,
        cnp.int64_t* indices,
        long n_indices
    ) nogil:
        """Calculate criterion for given indices."""
        if self.is_classification:
            # Gather y values
            for i in range(n_indices):
                self.y_int_buffer[i] = <long>self.y[indices[i]]
            return calculate_gini(self.y_int_buffer, n_indices, self.n_classes)
        else:
            # Gather y values
            for i in range(n_indices):
                self.y_double_buffer[i] = self.y[indices[i]]
            return calculate_mse(self.y_double_buffer, n_indices)
    
    cdef tuple find_best_split_for_feature(
        self,
        long feature_idx,
        cnp.int64_t* node_indices,
        long n_node_samples
    ):
        """
        Find best threshold for a specific feature at a node.
        Returns (threshold, improvement) or (0, -1) if no valid split.
        """
        if n_node_samples < self.min_samples_split:
            return (0.0, -1.0)
        
        # Create membership mask
        cdef bint* in_node = <bint*>malloc(self.n_samples * sizeof(bint))
        memset(in_node, 0, self.n_samples * sizeof(bint))
        
        cdef long i
        cdef cnp.int64_t idx
        for i in range(n_node_samples):
            in_node[node_indices[i]] = True
        
        # Filter sorted indices to this node
        cdef cnp.int64_t* sorted_node_idx = <cnp.int64_t*>malloc(n_node_samples * sizeof(cnp.int64_t))
        cdef long count = 0
        for i in range(self.n_samples):
            idx = self.sorted_indices[feature_idx, i]
            if in_node[idx]:
                sorted_node_idx[count] = idx
                count += 1
        
        free(in_node)
        
        if count < 2:
            free(sorted_node_idx)
            return (0.0, -1.0)
        
        # Calculate parent criterion
        cdef double parent_criterion = self.calculate_criterion_at_indices(
            node_indices, n_node_samples
        )
        
        # Try all thresholds
        cdef double best_threshold = 0.0
        cdef double best_improvement = -1.0
        cdef double current_val, next_val, threshold
        cdef long n_left, n_right
        cdef double weighted_criterion, improvement
        cdef double left_crit, right_crit
        
        for i in range(count - 1):
            current_val = self.X[sorted_node_idx[i], feature_idx]
            next_val = self.X[sorted_node_idx[i + 1], feature_idx]
            
            if current_val >= next_val - 1e-10:
                continue
            
            n_left = i + 1
            n_right = count - n_left
            
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            
            threshold = (current_val + next_val) / 2.0
            
            # Calculate weighted criterion
            left_crit = self.calculate_criterion_at_indices(
                sorted_node_idx, n_left
            )
            right_crit = self.calculate_criterion_at_indices(
                &sorted_node_idx[n_left], n_right
            )
            
            weighted_criterion = (
                (<double>n_left / <double>count) * left_crit +
                (<double>n_right / <double>count) * right_crit
            )
            
            improvement = parent_criterion - weighted_criterion
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_threshold = threshold
        
        free(sorted_node_idx)
        return (best_threshold, best_improvement)
    
    def find_best_feature_for_depth(
        self,
        list nodes_at_depth
    ):
        """Find best feature for RDT (same feature at depth)."""
        cdef long best_feature = -1
        cdef double best_total_improvement = -1.0
        cdef long feature_idx
        cdef double total_improvement, threshold, improvement
        cdef cnp.int64_t[::1] node_idx_view
        cdef long n_samples
        
        for feature_idx in range(self.n_features):
            total_improvement = 0.0
            
            for node_info in nodes_at_depth:
                node_indices = np.asarray(node_info[0], dtype=np.int64)
                node_idx_view = node_indices
                n_samples = len(node_indices)
                
                threshold, improvement = self.find_best_split_for_feature(
                    feature_idx, &node_idx_view[0], n_samples
                )
                
                if improvement > 0:
                    total_improvement += improvement
            
            if total_improvement > best_total_improvement:
                best_total_improvement = total_improvement
                best_feature = feature_idx
        
        return best_feature
    
    def find_best_threshold(
        self,
        long feature_idx,
        cnp.ndarray[cnp.int64_t, ndim=1] node_indices
    ):
        """Find best threshold for specific feature at node."""
        cdef cnp.int64_t[::1] node_idx_view = node_indices
        cdef long n_samples = len(node_indices)
        cdef double threshold, improvement
        
        threshold, improvement = self.find_best_split_for_feature(
            feature_idx, &node_idx_view[0], n_samples
        )
        
        if improvement < 0:
            return None
        return threshold
    
    def split_node(
        self,
        long feature_idx,
        double threshold,
        cnp.ndarray[cnp.int64_t, ndim=1] node_indices
    ):
        """Split node indices based on feature and threshold."""
        cdef cnp.int64_t[::1] node_idx_view = node_indices
        cdef long n_samples = len(node_indices)
        cdef long n_left = 0, n_right = 0
        cdef long i
        cdef cnp.int64_t idx
        
        with nogil:
            for i in range(n_samples):
                idx = node_idx_view[i]
                if self.X[idx, feature_idx] <= threshold:
                    self.left_buffer[n_left] = idx
                    n_left += 1
                else:
                    self.right_buffer[n_right] = idx
                    n_right += 1
        
        # Convert to numpy arrays
        left_indices = np.zeros(n_left, dtype=np.int64)
        right_indices = np.zeros(n_right, dtype=np.int64)
        
        cdef cnp.int64_t[::1] left_view = left_indices
        cdef cnp.int64_t[::1] right_view = right_indices
        
        with nogil:
            memcpy(&left_view[0], self.left_buffer, n_left * sizeof(cnp.int64_t))
            memcpy(&right_view[0], self.right_buffer, n_right * sizeof(cnp.int64_t))
        
        return left_indices, right_indices
    
    def get_leaf_value(self, cnp.ndarray[cnp.int64_t, ndim=1] node_indices):
        """Get prediction value for leaf node."""
        cdef cnp.int64_t[::1] idx_view = node_indices
        cdef long n_samples = len(node_indices)
        cdef long* counts
        cdef long i, cls
        cdef long max_count = 0
        cdef long best_class = 0
        cdef double mean = 0.0
        
        if self.is_classification:
            # Most common class
            counts = <long*>malloc(self.n_classes * sizeof(long))
            memset(counts, 0, self.n_classes * sizeof(long))
            
            with nogil:
                for i in range(n_samples):
                    cls = <long>self.y[idx_view[i]]
                    counts[cls] += 1
                    if counts[cls] > max_count:
                        max_count = counts[cls]
                        best_class = cls
            
            free(counts)
            return best_class
        else:
            # Mean value
            with nogil:
                for i in range(n_samples):
                    mean += self.y[idx_view[i]]
                mean /= <double>n_samples
            
            return mean


def build_tree_cython(X, y, n_classes, is_classification, criterion, 
                     min_samples_split, min_samples_leaf, max_depth):
    """Factory function to create FastTreeBuilder."""
    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    
    return FastTreeBuilder(
        X_c, y_c, n_classes, is_classification,
        criterion, min_samples_split, min_samples_leaf, max_depth
    )
