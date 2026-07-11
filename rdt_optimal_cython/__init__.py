"""
Cython-accelerated Optimal Restricted Decision Tree implementation.

This package provides a high-performance version of the OptimalRestrictedDecisionTree
that uses Cython for speed-critical operations while maintaining the same optimal
two-pass feature selection algorithm.
"""

from .optimal_rdt_cython import OptimalRDTCython

__all__ = ['OptimalRDTCython']
