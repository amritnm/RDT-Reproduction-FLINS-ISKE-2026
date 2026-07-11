# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

"""
Ultra-fast criterion calculations for decision trees.
Optimized for maximum performance with nogil support.
"""

cimport cython
from libc.math cimport log2, fabs, sqrt
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memset
cimport numpy as cnp
import numpy as np

cnp.import_array()


cdef inline double fast_gini_impurity(
    long* class_counts,
    long n_classes,
    long n_samples
) nogil:
    """
    Fast Gini impurity: 1 - sum(p_i^2)
    Optimized with minimal operations.
    """
    if n_samples == 0:
        return 0.0
    
    cdef double gini = 1.0
    cdef double p
    cdef long i
    cdef double n_samples_double = <double>n_samples
    
    for i in range(n_classes):
        if class_counts[i] > 0:
            p = <double>class_counts[i] / n_samples_double
            gini -= p * p
    
    return gini


cdef inline double fast_entropy(
    long* class_counts,
    long n_classes,
    long n_samples
) nogil:
    """
    Fast entropy: -sum(p_i * log2(p_i))
    """
    if n_samples == 0:
        return 0.0
    
    cdef double entropy = 0.0
    cdef double p
    cdef long i
    cdef double n_samples_double = <double>n_samples
    
    for i in range(n_classes):
        if class_counts[i] > 0:
            p = <double>class_counts[i] / n_samples_double
            entropy -= p * log2(p)
    
    return entropy


cdef inline double fast_variance(
    double* values,
    long n_samples
) nogil:
    """
    Fast variance calculation using Welford's online algorithm.
    More numerically stable than naive approach.
    """
    if n_samples == 0:
        return 0.0
    
    cdef double mean = 0.0
    cdef double m2 = 0.0
    cdef double delta
    cdef long i
    
    for i in range(n_samples):
        delta = values[i] - mean
        mean += delta / <double>(i + 1)
        m2 += delta * (values[i] - mean)
    
    if n_samples < 2:
        return 0.0
    
    return m2 / <double>n_samples


cdef inline double fast_mae_with_median(
    double* values,
    long n_samples,
    double median
) nogil:
    """
    Fast MAE given pre-computed median.
    """
    if n_samples == 0:
        return 0.0
    
    cdef double mae = 0.0
    cdef long i
    
    for i in range(n_samples):
        mae += fabs(values[i] - median)
    
    return mae / <double>n_samples


cdef class CriterionBase:
    """Base class for criteria with optimized calculations."""
    
    def __cinit__(self, double[:] y, long n_classes, bint is_classification):
        self.y = y
        self.n_samples = y.shape[0]
        self.n_classes = n_classes
        self.is_classification = is_classification
        
        # Allocate reusable buffers
        if is_classification:
            self.class_counts = <long*>malloc(n_classes * sizeof(long))
        else:
            self.value_buffer = <double*>malloc(self.n_samples * sizeof(double))
            self.class_counts = NULL
    
    def __dealloc__(self):
        if self.class_counts != NULL:
            free(self.class_counts)
        if self.value_buffer != NULL:
            free(self.value_buffer)
    
    cdef inline void count_classes(
        self,
        long* indices,
        long n_indices
    ) nogil:
        """Count classes for given indices."""
        cdef long i, idx, cls
        
        # Reset counts
        memset(self.class_counts, 0, self.n_classes * sizeof(long))
        
        # Count
        for i in range(n_indices):
            idx = indices[i]
            cls = <long>self.y[idx]
            self.class_counts[cls] += 1
    
    cdef inline void gather_values(
        self,
        long* indices,
        long n_indices
    ) nogil:
        """Gather values for regression."""
        cdef long i, idx
        
        for i in range(n_indices):
            idx = indices[i]
            self.value_buffer[i] = self.y[idx]
    
    cdef double calculate(
        self,
        long* indices,
        long n_indices
    ) nogil:
        """Calculate criterion - to be overridden."""
        return 0.0
    
    cdef double calculate_weighted(
        self,
        long* left_indices,
        long n_left,
        long* right_indices,
        long n_right
    ) nogil:
        """Calculate weighted criterion for split."""
        cdef long n_total = n_left + n_right
        if n_total == 0:
            return 0.0
        
        cdef double left_criterion = self.calculate(left_indices, n_left)
        cdef double right_criterion = self.calculate(right_indices, n_right)
        
        cdef double weighted = (
            (<double>n_left / <double>n_total) * left_criterion +
            (<double>n_right / <double>n_total) * right_criterion
        )
        
        return weighted


cdef class GiniCriterion(CriterionBase):
    """Optimized Gini impurity criterion."""
    
    cdef double calculate(
        self,
        long* indices,
        long n_indices
    ) nogil:
        if n_indices == 0:
            return 0.0
        
        self.count_classes(indices, n_indices)
        return fast_gini_impurity(self.class_counts, self.n_classes, n_indices)


cdef class EntropyCriterion(CriterionBase):
    """Optimized Entropy criterion."""
    
    cdef double calculate(
        self,
        long* indices,
        long n_indices
    ) nogil:
        if n_indices == 0:
            return 0.0
        
        self.count_classes(indices, n_indices)
        return fast_entropy(self.class_counts, self.n_classes, n_indices)


cdef class MSECriterion(CriterionBase):
    """Optimized MSE criterion."""
    
    cdef double calculate(
        self,
        long* indices,
        long n_indices
    ) nogil:
        if n_indices == 0:
            return 0.0
        
        self.gather_values(indices, n_indices)
        return fast_variance(self.value_buffer, n_indices)


# Comparison function for qsort
cdef int compare_doubles(const void* a, const void* b) noexcept nogil:
    cdef double da = (<double*>a)[0]
    cdef double db = (<double*>b)[0]
    if da < db:
        return -1
    elif da > db:
        return 1
    else:
        return 0


cdef class MAECriterion(CriterionBase):
    """Optimized MAE criterion."""
    
    cdef double calculate(
        self,
        long* indices,
        long n_indices
    ) nogil:
        if n_indices == 0:
            return 0.0
        
        self.gather_values(indices, n_indices)
        
        # Find median using qsort
        qsort(self.value_buffer, n_indices, sizeof(double), compare_doubles)
        
        cdef double median
        if n_indices % 2 == 1:
            median = self.value_buffer[n_indices // 2]
        else:
            median = (self.value_buffer[n_indices // 2 - 1] + 
                     self.value_buffer[n_indices // 2]) / 2.0
        
        return fast_mae_with_median(self.value_buffer, n_indices, median)


def create_criterion(str criterion_name, double[:] y, long n_classes, bint is_classification):
    """Factory function to create criterion object."""
    if criterion_name == 'gini':
        return GiniCriterion(y, n_classes, is_classification)
    elif criterion_name == 'entropy':
        return EntropyCriterion(y, n_classes, is_classification)
    elif criterion_name == 'mse':
        return MSECriterion(y, n_classes, is_classification)
    elif criterion_name == 'mae':
        return MAECriterion(y, n_classes, is_classification)
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")
