"""
Ensemble methods for Restricted Decision Trees.

High-performance Random Forest, AdaBoost, and Gradient Boosting implementations
using Cython-accelerated RDT as base learners.
"""

from .random_forest_rdt import RandomForestRDT
from .adaboost_rdt import AdaBoostRDT
from .gradient_boosting_rdt import GradientBoostingRDT
from .gradient_boosting_rdt_fast import GradientBoostingRDTFast

__all__ = [
    'RandomForestRDT',
    'AdaBoostRDT',
    'GradientBoostingRDT',
    'GradientBoostingRDTFast',
]

__version__ = '1.0.0'
