# ccagg/__init__.py

"""
ccagg
============
Public API for the multi-view CCA aggregation pipeline.

Usage
-----
from ccagg import (
    train_test_split_by_group,
    permute_score_significance_multidim,
    permute_rgcca_significance_multidim,
    plot_permutation_summary,
)
"""

from . import cca_extended, model, permutation, robust_inference
from . import stats, utils, validation, visualisation

__version__ = "0.1.0"

__all__ = [
    "cca_extended",
    "model",
    "permutation",
    "robust_inference",
    "stats",
    "utils",
    "validation",
    "visualisation",
]