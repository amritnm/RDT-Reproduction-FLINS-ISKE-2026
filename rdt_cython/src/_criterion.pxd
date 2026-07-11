# cython: language_level=3

"""Header file for criterion module."""

cdef class CriterionBase:
    cdef double[:] y
    cdef long n_samples
    cdef long n_classes
    cdef bint is_classification
    cdef long* class_counts
    cdef double* value_buffer
    
    cdef inline void count_classes(self, long* indices, long n_indices) nogil
    cdef inline void gather_values(self, long* indices, long n_indices) nogil
    cdef double calculate(self, long* indices, long n_indices) nogil
    cdef double calculate_weighted(self, long* left_indices, long n_left, 
                                   long* right_indices, long n_right) nogil


cdef class GiniCriterion(CriterionBase):
    cdef double calculate(self, long* indices, long n_indices) nogil


cdef class EntropyCriterion(CriterionBase):
    cdef double calculate(self, long* indices, long n_indices) nogil


cdef class MSECriterion(CriterionBase):
    cdef double calculate(self, long* indices, long n_indices) nogil


cdef class MAECriterion(CriterionBase):
    cdef double calculate(self, long* indices, long n_indices) nogil


cdef inline double fast_gini_impurity(long* class_counts, long n_classes, long n_samples) nogil
cdef inline double fast_entropy(long* class_counts, long n_classes, long n_samples) nogil
cdef inline double fast_variance(double* values, long n_samples) nogil
cdef inline double fast_mae_with_median(double* values, long n_samples, double median) nogil
