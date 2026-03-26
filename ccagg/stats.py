# ccagg/stats.py

"""
Statistical summaries and model-evaluation helpers for ccagg.

This module contains:
- cross-validated parameter evaluation
- latent-space \(R^2\) metrics
- cross-loading calculations

Notes
-----
`evaluate_param` is kept here for convenience, but conceptually it is closer
to model selection / validation than to a pure statistical primitive.
"""


from __future__ import annotations

import warnings
from typing import Iterable, List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from .utils import get_fold_data, make_model_init_kwargs



def evaluate_param(
    model_class,
    params,
    view_metadata,
    combined_group,
    cv=10,
    random_state=42,
    epochs=100,
    tol=1e-6,
    model_kwargs=None,
    design=None,
    response_view=0,
    latent_dimensions=1,
):
    rng = np.random.default_rng(random_state)
    n_samples = view_metadata[0]["data"].shape[0]
    n_views = len(view_metadata)
    
    split_iter, cv_eff = _make_cv_splitter(
        n_samples=n_samples,
        groups=combined_group,
        cv=cv,
        random_state=random_state,
    )
    
    if split_iter is None or cv_eff < 2:
        return np.nan, {"fold_scores": [], "params": params, "cv_eff": 0}
    
    effective_design = (
        np.asarray(design)
        if design is not None
        else (np.ones((n_views, n_views)) - np.eye(n_views))
    )
    
    fold_scores = []
    
    for train_idx, test_idx in split_iter:
        try:
            train_views, test_views = get_fold_data(
                train_idx,
                test_idx,
                view_metadata,
            )
            
            init_kwargs = make_model_init_kwargs(
                model_class=model_class,
                params=params,
                model_kwargs=model_kwargs,
                design=design,
                random_state=int(rng.integers(0, 2**32 - 1)),
                latent_dimensions=latent_dimensions,
                epochs=epochs,
                tol=tol,
            )
            
            model = model_class(**init_kwargs)
            model.fit(train_views)
            
            corrs = model.pairwise_correlations(test_views)
            
            vals = []
            for v in range(n_views):
                if v == response_view:
                    continue
                if effective_design[response_view, v] != 0:
                    vals.extend(corrs[response_view, v, :].tolist())
            
            score = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
            if np.isfinite(score):
                fold_scores.append(score)
        
        except Exception as e:
            warnings.warn(f"Fold failed: {e}. Skipping.")
            continue
    
    if len(fold_scores) == 0:
        return np.nan, {
            "fold_scores": [],
            "params": params,
            "cv_eff": cv_eff,
        }
    
    return float(np.mean(fold_scores)), {
        "fold_scores": fold_scores,
        "params": params,
        "cv_eff": cv_eff,
    }


def _make_cv_splitter(
    n_samples,
    groups=None,
    cv=10,
    random_state=42,
):
    x_dummy = np.zeros((n_samples, 1))

    # Case 1: cv is already a splitter object
    if hasattr(cv, "split"):
        try:
            if groups is not None:
                split_iter = list(cv.split(x_dummy, groups=groups))
            else:
                split_iter = list(cv.split(x_dummy))
        except TypeError:
            split_iter = list(cv.split(x_dummy))

        cv_eff = len(split_iter)
        if cv_eff < 2:
            return None, 0
        return split_iter, cv_eff

    # Case 2: cv is an integer
    if not isinstance(cv, int) or cv < 2:
        return None, 0

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        cv_eff = min(cv, len(unique_groups))

        if cv_eff < 2:
            return None, 0

        splitter = GroupKFold(n_splits=cv_eff)
        split_iter = list(splitter.split(x_dummy, groups=groups))
        return split_iter, cv_eff

    cv_eff = min(cv, n_samples)
    if cv_eff < 2:
        return None, 0

    splitter = KFold(
        n_splits=cv_eff,
        shuffle=True,
        random_state=random_state,
    )
    split_iter = list(splitter.split(x_dummy))
    return split_iter, cv_eff

def compute_r2(model, views, response_view=0):
    """
    Compute R^2 by regressing the response‐view scores on all predictor‐view scores.

    Parameters
    ----------
    model : fitted CCA‐type model exposing .transform(views) -> list of arrays
    views : list of arrays, one per view
    response_view : int, index of the response view

    Returns
    -------
    float  the coefficient of determination of the multiple‐linear regression
           y = response_scores[:,0] ~ X where X is horizontally stacked predictor_scores
    """
    scores = model.transform(views)
    # ensure we have a list of views; some implementations return an array by mistake
    if not isinstance(scores, (list, tuple)):
        raise ValueError("model.transform must return a list (one array per view).")

    K = scores[0].shape[1]  # number of components
    # build y
    y = scores[response_view]
    # flatten to 1d if K==1
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    # gather all other views
    X_list = []
    for v_idx, arr in enumerate(scores):
        if v_idx == response_view:
            continue
        if arr.ndim == 1:
            # single‐dim array
            X_list.append(arr.reshape(-1, 1))
        elif arr.ndim == 2:
            X_list.append(arr)
        else:
            raise ValueError(f"Unexpected score array shape {arr.shape} for view {v_idx}")

    if not X_list:
        warnings.warn("No predictor views found; returning NaN for R2")
        return np.nan

    # horizontally stack
    X = np.hstack(X_list)
    # X must be 2D
    if X.ndim != 2:
        raise ValueError(f"Predictor matrix X has bad ndim={X.ndim}")

    # Check variability
    if np.allclose(X.std(axis=0), 0) or np.allclose(np.std(y), 0):
        return np.nan

    # fit & score
    lr = LinearRegression()
    lr.fit(X, y)
    return lr.score(X, y)


def compute_r2_per_dimension(model, views, response=0):
    scores = model.transform(views) 
    r2 = []

    for i in range(model.latent_dimensions):
        scores_dim = [arr[:, i].reshape(-1, 1) for arr in scores] 
        X = np.hstack([scores_dim[j] for j in range(len(scores)) if j != response])  
        y = scores_dim[response]  
        r2_dim = LinearRegression().fit(X, y).score(X, y)
        r2.append(r2_dim)

    return r2


def compute_test_AVE(model, test_views, response=0):
    scores = model.transform(test_views)
    X = np.hstack([scores[j] for j in range(len(scores)) if j != response])
    y = scores[response]
    return LinearRegression().fit(X, y).score(X, y)


def calculate_cross_loadings(
    raw_views_list: Iterable[np.ndarray],
    scores_list: Iterable[np.ndarray],
    target_view_idx: int,
    predictor_view_indices: Optional[List[int]] = None,
) -> List[Optional[np.ndarray]]:
    """
    Calculates cross-loadings: correlations between raw variables of
    predictor views and the scores (canonical variates) of a target view.

    Parameters
    ----------
    raw_views_list : Iterable[np.ndarray]
        A list/tuple of NumPy arrays, where each array X_j (samples x features_j)
        represents the original raw data for each view.
    scores_list : Iterable[np.ndarray]
        A list/tuple of NumPy arrays, where each array Z_i (samples x latent_dims_i)
        represents the canonical variates (scores) for each view.
        Typically the output of model.transform(raw_views_list).
    target_view_idx : int
        The index of the view in scores_list whose scores (Z_target)
        will be used for correlation.
    predictor_view_indices : Optional[List[int]], optional
        A list of integer indices specifying which views in raw_views_list
        should be used as predictor views (their raw data X_j will be
        correlated with Z_target). If None, all views other than
        target_view_idx will be considered predictor views.

    Returns
    -------
    List[Optional[np.ndarray]]
        A list of NumPy arrays. Each array corresponds to a predictor view
        (in the order of predictor_view_indices if provided, or natural order
        excluding target_view_idx).
        The shape of each output array is (n_features_predictor_j, n_latent_dims_target).
        Contains None for a predictor index if it's the target_view_idx or if
        data is unsuitable for correlation.
    """
    num_raw_views = len(raw_views_list)
    num_score_views = len(scores_list)

    if num_raw_views != num_score_views:
        raise ValueError(
            f"Mismatch in number of raw views ({num_raw_views}) and score views ({num_score_views})."
        )
    if not (0 <= target_view_idx < num_score_views):
        raise ValueError(
            f"target_view_idx {target_view_idx} is out of bounds for {num_score_views} views."
        )

    target_scores = scores_list[target_view_idx] # Z_target

    if target_scores is None or not isinstance(target_scores, np.ndarray) or target_scores.ndim != 2 or target_scores.shape[0] < 2:
        warnings.warn(
            f"Target view {target_view_idx} scores are unsuitable for correlation. "
            f"Shape: {getattr(target_scores, 'shape', 'N/A')}. Returning None for cross-loadings."
        )
        num_predictors_expected = len(predictor_view_indices) if predictor_view_indices is not None else num_raw_views -1
        return [None] * max(0, num_predictors_expected) # Ensure non-negative count

    if predictor_view_indices is None:
        predictor_view_indices = [
            i for i in range(num_raw_views) if i != target_view_idx
        ]
    
    if not predictor_view_indices: # No predictors specified or left
        return []

    all_cross_loadings = []
    for pred_idx in predictor_view_indices:
        if not (0 <= pred_idx < num_raw_views):
            warnings.warn(f"Predictor view index {pred_idx} is out of bounds. Skipping.")
            all_cross_loadings.append(None)
            continue
        
        # Cross-loadings are not typically defined for a view with its own scores
        # in this context (that would be regular loadings).
        # However, if the user explicitly asks for it, we could allow it,
        # but your request implies X_j vs Z_response where j != response.
        if pred_idx == target_view_idx:
            warnings.warn(
                f"Predictor view index {pred_idx} is the same as target_view_idx. "
                "Cross-loadings with self are typically regular loadings. Appending None."
            )
            all_cross_loadings.append(None)
            continue

        predictor_raw_data = raw_views_list[pred_idx] # X_j

        if predictor_raw_data is None or not isinstance(predictor_raw_data, np.ndarray) or \
           predictor_raw_data.ndim != 2 or predictor_raw_data.shape[0] != target_scores.shape[0] or \
           predictor_raw_data.shape[0] < 2:
            warnings.warn(
                f"Predictor view {pred_idx} data is unsuitable for correlation or mismatched with target scores. "
                f"Shape: {getattr(predictor_raw_data, 'shape', 'N/A')}. Appending None."
            )
            all_cross_loadings.append(None)
            continue

        n_samples, n_features_predictor = predictor_raw_data.shape
        _, n_latent_dims_target = target_scores.shape
        
        # Initialize cross-loading matrix for this predictor view
        current_cross_loadings = np.full(
            (n_features_predictor, n_latent_dims_target), np.nan
        )

        for i in range(n_features_predictor): # For each feature in X_j
            for k in range(n_latent_dims_target): # For each component in Z_target
                x_feature_col = predictor_raw_data[:, i]
                z_score_col = target_scores[:, k]

                # Handle NaNs for this pair of columns before correlation
                valid_mask = ~np.isnan(x_feature_col) & ~np.isnan(z_score_col)
                x_valid = x_feature_col[valid_mask]
                z_valid = z_score_col[valid_mask]

                if len(x_valid) >= 2 and np.std(x_valid) > 1e-9 and np.std(z_valid) > 1e-9:
                    # Check for non-zero standard deviation to avoid runtime warning with np.corrcoef
                    current_cross_loadings[i, k] = np.corrcoef(x_valid, z_valid)[0, 1]
                # else, it remains NaN if not enough data or zero variance

        all_cross_loadings.append(current_cross_loadings)
        
    return all_cross_loadings


