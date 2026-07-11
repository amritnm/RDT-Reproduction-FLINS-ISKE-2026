"""
Cython-accelerated source modules for Optimal RDT.
"""

# Try to import compiled Cython modules
try:
    from ._optimal_tree_cython import (
        calculate_gini_fast,
        calculate_entropy_fast,
        calculate_mse_fast,
        find_best_threshold_vectorized,
        split_indices_fast
    )
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    # Fallback implementations will be used from the main module

__all__ = [
    'calculate_gini_fast',
    'calculate_entropy_fast',
    'calculate_mse_fast',
    'find_best_threshold_vectorized',
    'split_indices_fast',
    'HAS_CYTHON'
]
