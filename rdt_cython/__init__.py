"""
Ultra-fast Cython implementation of Restricted and Oblivious Decision Trees.

This package provides production-grade implementations optimized for:
- Large datasets (millions of samples)
- High-dimensional data (hundreds of features)
- Ensemble methods (Random Forests, Gradient Boosting)

Performance: 100-500x faster than pure Python implementations.
"""

from .cython_rdt import CythonRestrictedDecisionTree

__version__ = '1.0.0'
__all__ = ['CythonRestrictedDecisionTree']
