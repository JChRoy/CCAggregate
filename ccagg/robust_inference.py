# ccagg/robust_inference.py


"""
Multi-split discovery/inference evaluation and permutation testing for
multi-view latent-dimension models.

This module implements outer-split workflows that separate feature discovery
from held-out inference. For each outer split, features are selected on a
discovery set, reduced one-dimensional models are refit on the selected
features, and association strength is evaluated on an independent inference
set.

Three complementary workflows are provided:

* ``_run_multisplit_full_pipeline`` — observed-data evaluation only
  (held-out test correlations across repeated outer discovery/inference
  splits).
* ``multisplit_discovery_inference_test`` — split-wise locked-model
  permutation inference on held-out inference sets, returning per-split
  effect estimates, null reference intervals, and combined p-values across
  splits.
* ``omnibus_full_pipeline_permutation_test`` — full end-to-end omnibus
  permutation inference in which the entire discovery/tuning/selection/
  reduced-refit/held-out-evaluation pipeline is rerun under permutation to
  obtain a global null distribution.

Together, these workflows support complementary goals: assessing the
stability of held-out effects across repeated discovery/inference splits,
testing split-wise significance with locked-model inference, and testing
overall end-to-end pipeline significance with a global omnibus permutation
test.
"""

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .model import parallel_best_param_ensemble_multidim
from .stats import evaluate_param
from .utils import (get_fold_data, 
                    safe_float, 
                    make_model_init_kwargs, 
                    block_permutation_index, 
                    subset_view_metadata, 
                    extract_fold_scores, 
                    select_best_idx_from_cv)

#%%

###################################### 
# Helper functions

##################################
# 1. Metadata and data subsetting


def apply_feature_masks(view_metadata, feature_masks):
    return [
        {**meta, "data": meta["data"][:, mask]}
        for meta, mask in zip(view_metadata, feature_masks)
    ]



##########################################
# 2. Array formatting and result stacking

def _to_1d_float_array(x):
    if x is None:
        return np.array([], dtype=float)
    return np.asarray(x, dtype=float).ravel()


def _stack_padded_1d(arrays, fill_value=np.nan):
    arrays = [_to_1d_float_array(a) for a in arrays]
    max_len = max((a.size for a in arrays), default=0)

    out = np.full((len(arrays), max_len), fill_value, dtype=float)
    for i, arr in enumerate(arrays):
        if arr.size > 0:
            out[i, : arr.size] = arr
    return out


def _stack_padded_null(null_list):
    """
    Stack split-wise null arrays of shape (n_dims, n_permutations) into a
    padded cube of shape (n_splits, max_dims, max_permutations).
    """
    n_splits = len(null_list)
    max_dims = max((arr.shape[0] for arr in null_list), default=0)
    max_perms = max((arr.shape[1] for arr in null_list), default=0)

    out = np.full((n_splits, max_dims, max_perms), np.nan, dtype=float)

    for i, arr in enumerate(null_list):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            continue
        out[i, : arr.shape[0], : arr.shape[1]] = arr

    return out



def _aggregate_over_splits(x, stat):
    """
    Aggregate columns of a 2D array across splits, ignoring NaNs.

    Parameters
    ----------
    x : ndarray
        If 2D, shape is assumed to be (n_splits, n_columns).
        If 1D, a scalar aggregate is returned.
    stat : {'mean', 'median', 'fraction_positive'}
        Aggregate statistic.

    Returns
    -------
    float or ndarray
        Column-wise aggregate(s).
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        columns = [x]
        return_scalar = True
    elif x.ndim == 2:
        columns = [x[:, j] for j in range(x.shape[1])]
        return_scalar = False
    else:
        raise ValueError("`x` must be 1D or 2D.")

    out = []
    for col in columns:
        valid = col[np.isfinite(col)]
        if valid.size == 0:
            out.append(np.nan)
            continue

        if stat == "mean":
            out.append(float(np.mean(valid)))
        elif stat == "median":
            out.append(float(np.median(valid)))
        elif stat == "fraction_positive":
            out.append(float(np.mean(valid > 0)))
        else:
            raise ValueError(
                "`aggregate_stat` must be one of "
                "{'mean', 'median', 'fraction_positive'}."
            )

    if return_scalar:
        return out[0]

    return np.asarray(out, dtype=float)



###################################
# 3. Correlation helper

def mean_response_vs_predictors_corr(model, views, response_view=0):
    corrs = model.pairwise_correlations(views)
    other_view_indices = [j for j in range(len(views)) if j != response_view]

    if not other_view_indices:
        return np.nan

    vals = [corrs[response_view, j, 0] for j in other_view_indices]
    return float(np.nanmean(vals))


##########################################
# 4. Tuning helpers


def tune_reduced_1d_model(
    model_class,
    reduced_discovery_metadata,
    discovery_group,
    sample_param_func,
    n_iter=50,
    cv=10,
    random_state=0,
    epochs=200,
    model_kwargs=None,
    design=None,
    response_view=0,
    selection_rule="max_mean",
    param_complexity_fn=None,
):
    param_list = sample_param_func(
        n_iter=n_iter,
        seed=random_state,
        n_views=len(reduced_discovery_metadata),
        latent_dimensions=1,
    )
    
    if param_list is None or len(param_list) == 0:
        return {}, []
    
    cv_records = []
    
    for params_idx, params in enumerate(param_list):
        mean_cv, cv_aux = evaluate_param(
            model_class=model_class,
            params=params,
            view_metadata=reduced_discovery_metadata,
            combined_group=discovery_group,
            cv=cv,
            random_state=random_state + params_idx,
            epochs=epochs,
            tol=params.get("tol", 1e-5) if isinstance(params, dict) else 1e-5,
            model_kwargs=model_kwargs,
            design=design,
            response_view=response_view,
            latent_dimensions=1,
        )
        
        mean_cv = safe_float(mean_cv)
        fold_scores = extract_fold_scores(cv_aux)
        
        if fold_scores is not None:
            valid = fold_scores[np.isfinite(fold_scores)]
            if valid.size > 0:
                if valid.size >= 2:
                    sd_cv = float(np.std(valid, ddof=1))
                    se_cv = float(sd_cv / np.sqrt(valid.size))
                else:
                    sd_cv = np.nan
                    se_cv = np.nan
                n_cv = int(valid.size)
            else:
                sd_cv = np.nan
                se_cv = np.nan
                n_cv = 0
        else:
            sd_cv = np.nan
            se_cv = np.nan
            n_cv = 0
        
        cv_records.append(
            {
                "mean_cv": mean_cv,
                "sd_cv": sd_cv,
                "se_cv": se_cv,
                "n_cv": n_cv,
                "params": params,
            }
        )
    
    best_idx = select_best_idx_from_cv(
        cv_records=cv_records,
        param_list=param_list,
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
    )
    
    return param_list[best_idx], cv_records

############################################
# 5. P-value combination and other summary 

def _combine_pvalues_cauchy(pvalues, weights=None, clip=1e-15):
    """
    Cauchy combination test. Useful when p-values may be dependent.

    Parameters
    ----------
    pvalues : array-like
        Vector of p-values.
    weights : array-like or None
        Optional non-negative weights. If None, uniform weights are used.
    clip : float
        Numerical clipping away from 0 and 1.

    Returns
    -------
    p_combined : float
    """
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p) & (p > 0) & (p <= 1)
    p = p[valid]

    if p.size == 0:
        return np.nan

    p = np.clip(p, clip, 1 - clip)

    if weights is None:
        w = np.full(p.size, 1.0 / p.size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)[valid]
        w = np.clip(w, 0, np.inf)
        if np.sum(w) == 0:
            return np.nan
        w = w / np.sum(w)

    stat = np.sum(w * np.tan((0.5 - p) * np.pi))
    p_combined = 0.5 - np.arctan(stat) / np.pi
    return float(np.clip(p_combined, 0.0, 1.0))


def _combine_pvalues_min_bonferroni(pvalues):
    """
    Conservative p-value combination valid under arbitrary dependence.
    """
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p) & (p > 0) & (p <= 1)
    p = p[valid]

    if p.size == 0:
        return np.nan

    return float(min(1.0, p.size * np.min(p)))


def combine_pvalues_multisplit(pvalues, method="cauchy", weights=None):
    if method == "cauchy":
        return _combine_pvalues_cauchy(pvalues, weights=weights)
    if method == "min_bonferroni":
        return _combine_pvalues_min_bonferroni(pvalues)

    raise ValueError("method must be 'cauchy' or 'min_bonferroni'.")
    

def _empirical_p_value_greater(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]

    if not np.isfinite(observed) or null_values.size == 0:
        return np.nan

    count = np.sum(null_values >= observed)
    return float((count + 1) / (null_values.size + 1))


def pval_and_ci(obs_val, null_dist):
    valid_null = np.asarray(null_dist, dtype=float)
    valid_null = valid_null[np.isfinite(valid_null)]

    if valid_null.size == 0:
        return np.nan, np.array([np.nan, np.nan])

    p_value = (1 + np.sum(valid_null >= obs_val)) / (valid_null.size + 1)
    ci_95 = np.percentile(valid_null, [2.5, 97.5])
    return float(p_value), ci_95

def _summarize_columns(mat):
    """
    Column-wise summaries ignoring NaN.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        raise ValueError("mat must be 2D.")

    n_cols = mat.shape[1]

    n_valid = np.zeros(n_cols, dtype=int)
    mean = np.full(n_cols, np.nan, dtype=float)
    median = np.full(n_cols, np.nan, dtype=float)
    sd = np.full(n_cols, np.nan, dtype=float)
    q025 = np.full(n_cols, np.nan, dtype=float)
    q250 = np.full(n_cols, np.nan, dtype=float)
    q750 = np.full(n_cols, np.nan, dtype=float)
    q975 = np.full(n_cols, np.nan, dtype=float)

    for j in range(n_cols):
        vals = mat[:, j]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            continue

        n_valid[j] = vals.size
        mean[j] = float(np.mean(vals))
        median[j] = float(np.median(vals))

        if vals.size >= 2:
            sd[j] = float(np.std(vals, ddof=1))

        q025[j], q250[j], q750[j], q975[j] = np.percentile(
            vals,
            [2.5, 25, 75, 97.5],
        )

    return {
        "n_valid": n_valid,
        "mean": mean,
        "median": median,
        "sd": sd,
        "q025": q025,
        "q250": q250,
        "q750": q750,
        "q975": q975,
    }


def _induced_block_permutation_index(blocks, keys):
    """
    Create a within-block permutation index induced by shared random keys.

    Parameters
    ----------
    blocks : array-like
        Block labels for the current subset of samples.
    keys : array-like
        One random key per sample in the same subset. Samples are permuted
        within block by sorting these keys.

    Returns
    -------
    ndarray
        Permutation index for the subset.
    """
    blocks = np.asarray(blocks)
    keys = np.asarray(keys, dtype=float)

    if blocks.shape[0] != keys.shape[0]:
        raise ValueError("`blocks` and `keys` must have the same length.")

    perm_idx = np.empty(blocks.shape[0], dtype=int)

    for block in np.unique(blocks):
        idx = np.flatnonzero(blocks == block)
        order = np.argsort(keys[idx], kind="mergesort")
        perm_idx[idx] = idx[order]

    return perm_idx


def _permute_response_view_metadata_global(
    view_metadata,
    response_view=0,
    permutation_blocks=None,
    seed=None,
):
    """
    Permute only the response view rows, leaving all predictor views unchanged.

    Parameters
    ----------
    view_metadata : list of dict
        Each element must contain a 'data' key and may contain 'confounds'.
    response_view : int
        Index of the response / psychopathology view.
    permutation_blocks : array-like or None
        Exchangeability blocks for row-wise permutation. If None, permutation
        is global.
    seed : int or None
        Random seed.

    Returns
    -------
    permuted_view_metadata : list of dict
    """
    rng = np.random.default_rng(seed)
    n_samples = view_metadata[response_view]["data"].shape[0]

    if permutation_blocks is None:
        perm = rng.permutation(n_samples)
    else:
        perm = block_permutation_index(permutation_blocks, rng)

    out = []
    for v, meta in enumerate(view_metadata):
        if v == response_view:
            out.append(
                {
                    **meta,
                    "data": meta["data"][perm],
                    "confounds": (
                        meta["confounds"][perm]
                        if meta.get("confounds", None) is not None
                        else None
                    ),
                }
            )
        else:
            out.append({**meta})

    return out


def _sign_from_value(x):
     x = safe_float(x)
     if not np.isfinite(x) or x == 0:
         return 1.0
     return 1.0 if x > 0 else -1.0
 
 
def _discovery_alignment_sign(model, discovery_views, response_view=0):
     """
     Orient a locked 1D model using only the discovery set.
 
     The sign is chosen so that the mean response-vs-predictors correlation on
     the discovery data is non-negative. This avoids using held-out inference
     data to choose the orientation while making held-out effects comparable
     across outer splits.
 
     Returns
     -------
     sign : float
         Either  1.0 or -1.0.
     discovery_corr : float
         The pre-alignment discovery-set correlation used to choose the sign.
     """
     discovery_corr = mean_response_vs_predictors_corr(
         model, discovery_views, response_view=response_view
     )
     return _sign_from_value(discovery_corr), discovery_corr


#%%

###################################### 
# Discovery and Inference estimation 

def _run_one_outer_split_full_pipeline(
    split_idx,
    split_seed,
    split_random_state,
    model_class,
    view_metadata,
    combined_group,
    train_test_split_by_group,
    sample_param_func,
    response_view=0,
    inference_size=0.3,
    latent_dimensions=6,
    discovery_n_splits=100,
    discovery_n_iter=50,
    reduced_cv=10,
    selection_threshold=0.9,
    model_kwargs=None,
    design=None,
    epochs=200,
    selection_rule="max_mean",
    param_complexity_fn=None,
    n_jobs_discovery=1,
):
    """
    Run one outer discovery/inference split of the full pipeline.

    This function performs:
    1. A group-aware outer split into discovery and inference sets.
    2. Feature-selection discovery on the discovery set across multiple latent
       dimensions.
    3. Thresholding of average feature-selection frequencies for each latent
       dimension.
    4. Refitting of a reduced one-dimensional model on the selected features.
    5. Evaluation of held-out inference correlation for each latent dimension.

    No inference-stage permutations are performed.

    Parameters
    ----------
    split_idx : int
        Index of the outer split.
    split_seed : int
        Random seed used for the outer discovery/inference split.
    split_random_state : int
        Random state used for tuning and final model fitting within this split.
    model_class : type
        Model class to instantiate and fit.
    view_metadata : sequence
        Metadata describing each view, including feature matrices and any
        associated annotations needed downstream.
    combined_group : array-like
        Group labels used for group-aware splitting.
    train_test_split_by_group : callable
        Function that splits indices into discovery and inference sets while
        respecting group structure.
    sample_param_func : callable
        Function used to sample candidate hyperparameters.
    response_view : int, default=0
        Index of the response view.
    inference_size : float, default=0.3
        Proportion of samples allocated to the inference set.
    latent_dimensions : int, default=6
        Maximum number of latent dimensions considered during discovery.
    discovery_n_splits : int, default=100
        Number of resampling splits used in the discovery stage.
    discovery_n_iter : int, default=50
        Number of hyperparameter samples evaluated during discovery and reduced
        model tuning.
    reduced_cv : int, default=10
        Number of cross-validation folds used for reduced-model tuning.
    selection_threshold : float, default=0.9
        Threshold applied to average selection frequencies to determine selected
        features.
    model_kwargs : dict or None, default=None
        Additional keyword arguments passed to the model constructor.
    design : any, default=None
        Optional design object passed through to model initialization/tuning.
    epochs : int, default=200
        Number of training epochs for supported models.
    selection_rule : str, default="max_mean"
        Rule used to choose the best hyperparameter configuration.
    param_complexity_fn : callable or None, default=None
        Optional function used to score or penalize hyperparameter complexity.
    n_jobs_discovery : int, default=1
        Number of parallel jobs used within the discovery stage.

    Returns
    -------
    dict
        Dictionary containing:
        - "split_idx": int
        - "split_seed": int
        - "split_random_state": int
        - "discovery_idx": ndarray
        - "inference_idx": ndarray
        - "obs_test_corr": ndarray
            Held-out inference correlations, one per latent dimension.
        - "discovery_results": dict
            Full discovery-stage output for this split.
    """
    combined_group = np.asarray(combined_group)

    discovery_idx, inference_idx = train_test_split_by_group(
        combined_group,
        test_size=inference_size,
        seed=int(split_seed),
    )

    discovery_metadata = subset_view_metadata(view_metadata, discovery_idx)
    discovery_group = combined_group[discovery_idx]

    discovery_results = parallel_best_param_ensemble_multidim(
        model_class=model_class,
        combined_group=discovery_group,
        view_metadata=discovery_metadata,
        train_test_split_by_group=train_test_split_by_group,
        sample_param_func=sample_param_func,
        n_splits=discovery_n_splits,
        n_iter=discovery_n_iter,
        cv=reduced_cv,
        epochs=epochs,
        n_jobs=n_jobs_discovery,
        model_kwargs=model_kwargs,
        design=design,
        run_final_ensemble=False,
        response_view=response_view,
        latent_dimensions=latent_dimensions,
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
    )

    avg_sel_freq = discovery_results.get("average_selection_frequency", None)
    if avg_sel_freq is None or len(avg_sel_freq) == 0:
        return {
            "split_idx": int(split_idx),
            "split_seed": int(split_seed),
            "split_random_state": int(split_random_state),
            "discovery_idx": discovery_idx,
            "inference_idx": inference_idx,
            "obs_test_corr": np.array([], dtype=float),
            "discovery_results": discovery_results,
        }

    k_max = int(avg_sel_freq[0].shape[1])
    obs_test_corr = np.full(k_max, np.nan, dtype=float)

    for k in range(k_max):
        feature_masks_k = [
            np.asarray(freq_matrix[:, k] >= selection_threshold, dtype=bool)
            for freq_matrix in avg_sel_freq
        ]

        if any(mask.sum() == 0 for mask in feature_masks_k):
            continue

        masked_full_metadata = apply_feature_masks(
            view_metadata,
            feature_masks_k,
        )

        disc_views_k, inf_views_k = get_fold_data(
            discovery_idx,
            inference_idx,
            masked_full_metadata,
        )

        reduced_discovery_metadata = subset_view_metadata(
            masked_full_metadata,
            discovery_idx,
        )

        best_params_k, _ = tune_reduced_1d_model(
            model_class=model_class,
            reduced_discovery_metadata=reduced_discovery_metadata,
            discovery_group=discovery_group,
            sample_param_func=sample_param_func,
            n_iter=discovery_n_iter,
            cv=reduced_cv,
            random_state=int(split_random_state) + 1000 + k,
            epochs=epochs,
            model_kwargs=model_kwargs,
            design=design,
            response_view=response_view,
            selection_rule=selection_rule,
            param_complexity_fn=param_complexity_fn,
        )

        tol_k = (best_params_k.get("tol", 1e-5) if isinstance(best_params_k, dict) else 1e-5)

        model = model_class(
            **make_model_init_kwargs(
                model_class=model_class,
                params=best_params_k,
                model_kwargs=model_kwargs,
                design=design,
                random_state=int(split_random_state) + 2000 + k,
                latent_dimensions=1,
                epochs=epochs,
                tol=tol_k,
            )
        )
        model.fit(disc_views_k)

        align_sign_k, _ = _discovery_alignment_sign(
            model,
            disc_views_k,
            response_view=response_view,
        )

        obs_test_corr[k] = align_sign_k * mean_response_vs_predictors_corr(
            model,
            inf_views_k,
            response_view=response_view,
        )

    return {
        "split_idx": int(split_idx),
        "split_seed": int(split_seed),
        "split_random_state": int(split_random_state),
        "discovery_idx": discovery_idx,
        "inference_idx": inference_idx,
        "obs_test_corr": obs_test_corr,
        "discovery_results": discovery_results,
    }


def _run_multisplit_full_pipeline(
    model_class,
    view_metadata,
    combined_group,
    train_test_split_by_group,
    sample_param_func,
    outer_split_seeds,
    outer_split_random_states,
    response_view=0,
    inference_size=0.3,
    latent_dimensions=6,
    discovery_n_splits=100,
    discovery_n_iter=50,
    reduced_cv=10,
    selection_threshold=0.9,
    model_kwargs=None,
    design=None,
    epochs=200,
    selection_rule="max_mean",
    param_complexity_fn=None,
    n_jobs_discovery=1,
    n_jobs_splits=1,
    show_progress=False,
    progress_desc="Outer splits",
):
    """
    Run the full discovery/inference pipeline across multiple outer splits.

    For each outer split, the function runs the complete observed-data pipeline
    by calling `_run_one_outer_split_full_pipeline`, then collects the held-out
    inference correlations across splits into a padded matrix.

    Parameters
    ----------
    model_class : type
        Model class to instantiate and fit.
    view_metadata : sequence
        Metadata describing each view, including feature matrices and any
        associated annotations needed downstream.
    combined_group : array-like
        Group labels used for group-aware splitting.
    train_test_split_by_group : callable
        Function that splits indices into discovery and inference sets while
        respecting group structure.
    sample_param_func : callable
        Function used to sample candidate hyperparameters.
    outer_split_seeds : array-like
        Seeds used to generate the outer discovery/inference splits.
    outer_split_random_states : array-like
        Random states used for tuning and fitting within each outer split.
        Must have the same length as `outer_split_seeds`.
    response_view : int, default=0
        Index of the response view.
    inference_size : float, default=0.3
        Proportion of samples allocated to the inference set in each outer
        split.
    latent_dimensions : int, default=6
        Maximum number of latent dimensions considered during discovery.
    discovery_n_splits : int, default=100
        Number of resampling splits used in the discovery stage.
    discovery_n_iter : int, default=50
        Number of hyperparameter samples evaluated during discovery and reduced
        model tuning.
    reduced_cv : int, default=10
        Number of cross-validation folds used for reduced-model tuning.
    selection_threshold : float, default=0.9
        Threshold applied to average selection frequencies to determine selected
        features.
    model_kwargs : dict or None, default=None
        Additional keyword arguments passed to the model constructor.
    design : any, default=None
        Optional design object passed through to model initialization/tuning.
    epochs : int, default=200
        Number of training epochs for supported models.
    selection_rule : str, default="max_mean"
        Rule used to choose the best hyperparameter configuration.
    param_complexity_fn : callable or None, default=None
        Optional function used to score or penalize hyperparameter complexity.
    n_jobs_discovery : int, default=1
        Number of parallel jobs used within each discovery stage.
    n_jobs_splits : int, default=1
        Number of parallel jobs used across outer splits.
    show_progress : bool, default=False
        Whether to display a progress bar when running splits sequentially.
    progress_desc : str, default="Outer splits"
        Description shown in the progress bar.

    Returns
    -------
    dict
        Dictionary containing:
        - "split_results": list of dict
            Per-split outputs from `_run_one_outer_split_full_pipeline`.
        - "obs_test_corr_by_split": ndarray
            Padded matrix of held-out inference correlations with one row per
            outer split.
        - "outer_split_seeds": ndarray
            Seeds used for the outer splits.
        - "outer_split_random_states": ndarray
            Random states used within the outer splits.

    Raises
    ------
    ValueError
        If `outer_split_seeds` and `outer_split_random_states` do not have the
        same length.
    """
    outer_split_seeds = np.asarray(outer_split_seeds).ravel()
    outer_split_random_states = np.asarray(outer_split_random_states).ravel()

    if outer_split_seeds.size != outer_split_random_states.size:
        raise ValueError(
            "outer_split_seeds and outer_split_random_states must have the "
            "same length."
        )

    worker_kwargs = dict(
        model_class=model_class,
        view_metadata=view_metadata,
        combined_group=combined_group,
        train_test_split_by_group=train_test_split_by_group,
        sample_param_func=sample_param_func,
        response_view=response_view,
        inference_size=inference_size,
        latent_dimensions=latent_dimensions,
        discovery_n_splits=discovery_n_splits,
        discovery_n_iter=discovery_n_iter,
        reduced_cv=reduced_cv,
        selection_threshold=selection_threshold,
        model_kwargs=model_kwargs,
        design=design,
        epochs=epochs,
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
        n_jobs_discovery=n_jobs_discovery,
    )

    if n_jobs_splits == 1:
        split_results = []
        iterator = zip(
            range(outer_split_seeds.size),
            outer_split_seeds,
            outer_split_random_states,
        )
        iterator = tqdm(
            iterator,
            total=outer_split_seeds.size,
            disable=not show_progress,
            desc=progress_desc,
        )

        for split_idx, split_seed, split_random_state in iterator:
            split_results.append(
                _run_one_outer_split_full_pipeline(
                    split_idx=int(split_idx),
                    split_seed=int(split_seed),
                    split_random_state=int(split_random_state),
                    **worker_kwargs,
                )
            )
    else:
        split_results = Parallel(n_jobs=n_jobs_splits, backend="loky")(
            delayed(_run_one_outer_split_full_pipeline)(
                split_idx=int(split_idx),
                split_seed=int(split_seed),
                split_random_state=int(split_random_state),
                **worker_kwargs,
            )
            for split_idx, split_seed, split_random_state in zip(
                range(outer_split_seeds.size),
                outer_split_seeds,
                outer_split_random_states,
            )
        )

    obs_matrix = _stack_padded_1d([res["obs_test_corr"] for res in split_results])

    return {
        "split_results": split_results,
        "obs_test_corr_by_split": obs_matrix,
        "outer_split_seeds": outer_split_seeds.astype(np.uint32),
        "outer_split_random_states": outer_split_random_states.astype(np.uint32),
    }





#%%

######################################
# Multi-split permutation inference

def discovery_inference_permutation_test(
    model_class,
    view_metadata,
    combined_group,
    train_test_split_by_group,
    sample_param_func,
    response_view=0,
    inference_size=0.3,
    discovery_split_seed=123,
    discovery_n_splits=100,
    discovery_n_iter=50,
    reduced_cv=10,
    n_permutations=1000,
    selection_threshold=0.9,
    model_kwargs=None,
    design=None,
    epochs=200,
    random_state=0,
    selection_rule="max_mean",
    param_complexity_fn=None,
    permutation_blocks=None,
    n_jobs_discovery=1,
    latent_dimensions=1,
    global_permutation_keys=None,
):
    """
    Single discovery/inference locked-model permutation test.

    If `global_permutation_keys` is provided, null permutations in the
    inference set are induced from those shared keys. This is useful for
    multisplit inference, where dependence across overlapping splits should
    be reflected in the aggregate null distribution.

    Parameters
    ----------
    latent_dimensions : int, default=1
        Maximum number of latent dimensions considered during discovery.
    global_permutation_keys : ndarray or None, default=None
        Optional array of shape (n_permutations, n_samples). If provided,
        these shared keys are used to induce within-block permutations on the
        inference set. If None, split-local permutations are generated using
        `random_state`.

    Returns
    -------
    dict
        Dictionary with observed held-out correlations, permutation p-values,
        permutation reference intervals, null samples, and discovery output.
    """
    combined_group = np.asarray(combined_group)

    if permutation_blocks is None:
        permutation_blocks = combined_group
    else:
        permutation_blocks = np.asarray(permutation_blocks)
        if permutation_blocks.shape[0] != combined_group.shape[0]:
            raise ValueError(
                "`permutation_blocks` must have the same length as "
                "`combined_group`."
            )

    if global_permutation_keys is not None:
        global_permutation_keys = np.asarray(
            global_permutation_keys,
            dtype=float,
        )
        if global_permutation_keys.ndim != 2:
            raise ValueError("`global_permutation_keys` must be a 2D array.")
        if global_permutation_keys.shape[1] != combined_group.shape[0]:
            raise ValueError(
                "`global_permutation_keys` must have shape "
                "(n_permutations, n_samples)."
            )
        n_permutations_eff = int(global_permutation_keys.shape[0])
        rng = None
    else:
        n_permutations_eff = int(n_permutations)
        rng = np.random.default_rng(random_state)

    discovery_idx, inference_idx = train_test_split_by_group(
        combined_group,
        test_size=inference_size,
        seed=int(discovery_split_seed),
    )

    discovery_metadata = subset_view_metadata(view_metadata, discovery_idx)
    discovery_group = combined_group[discovery_idx]

    discovery_results = parallel_best_param_ensemble_multidim(
        model_class=model_class,
        combined_group=discovery_group,
        view_metadata=discovery_metadata,
        train_test_split_by_group=train_test_split_by_group,
        sample_param_func=sample_param_func,
        n_splits=discovery_n_splits,
        n_iter=discovery_n_iter,
        cv=reduced_cv,
        epochs=epochs,
        n_jobs=n_jobs_discovery,
        model_kwargs=model_kwargs,
        design=design,
        run_final_ensemble=False,
        response_view=response_view,
        latent_dimensions=latent_dimensions,
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
    )

    avg_sel_freq = discovery_results.get("average_selection_frequency", None)
    if avg_sel_freq is None or len(avg_sel_freq) == 0:
        return {
            "discovery_idx": discovery_idx,
            "inference_idx": inference_idx,
            "obs_test_corr": np.array([], dtype=float),
            "p_test_corr": np.array([], dtype=float),
            "ci_test_corr": np.empty((0, 2), dtype=float),
            "null_test_corr": np.empty((0, n_permutations_eff), dtype=float),
            "discovery_results": discovery_results,
        }

    k_max = int(avg_sel_freq[0].shape[1])

    obs_test_corr = np.full(k_max, np.nan, dtype=float)
    p_test_corr = np.full(k_max, np.nan, dtype=float)
    ci_test_corr = np.full((k_max, 2), np.nan, dtype=float)
    null_test_corr = np.full(
        (k_max, n_permutations_eff),
        np.nan,
        dtype=float,
    )

    permutation_blocks_inf = permutation_blocks[inference_idx]

    for k in range(k_max):
        feature_masks_k = [
            np.asarray(freq_matrix[:, k] >= selection_threshold, dtype=bool)
            for freq_matrix in avg_sel_freq
        ]

        if any(mask.sum() == 0 for mask in feature_masks_k):
            continue

        masked_full_metadata = apply_feature_masks(
            view_metadata,
            feature_masks_k,
        )

        disc_views_k, inf_views_k = get_fold_data(
            discovery_idx,
            inference_idx,
            masked_full_metadata,
        )

        reduced_discovery_metadata = subset_view_metadata(
            masked_full_metadata,
            discovery_idx,
        )

        best_params_k, _ = tune_reduced_1d_model(
            model_class=model_class,
            reduced_discovery_metadata=reduced_discovery_metadata,
            discovery_group=discovery_group,
            sample_param_func=sample_param_func,
            n_iter=discovery_n_iter,
            cv=reduced_cv,
            random_state=int(random_state) + 1000 + k,
            epochs=epochs,
            model_kwargs=model_kwargs,
            design=design,
            response_view=response_view,
            selection_rule=selection_rule,
            param_complexity_fn=param_complexity_fn,
        )

        tol_k = (
            best_params_k.get("tol", 1e-5)
            if isinstance(best_params_k, dict)
            else 1e-5
        )

        model = model_class(
            **make_model_init_kwargs(
                model_class=model_class,
                params=best_params_k,
                model_kwargs=model_kwargs,
                design=design,
                random_state=int(random_state) + 2000 + k,
                latent_dimensions=1,
                epochs=epochs,
                tol=tol_k,
            )
        )
        model.fit(disc_views_k)

        align_sign_k, _ = _discovery_alignment_sign(
            model,
            disc_views_k,
            response_view=response_view,
        )

        obs_test_corr[k] = align_sign_k * mean_response_vs_predictors_corr(
            model,
            inf_views_k,
            response_view=response_view,
        )

        for b in range(n_permutations_eff):
            if global_permutation_keys is None:
                perm_idx = block_permutation_index(
                    permutation_blocks_inf,
                    rng,
                )
            else:
                perm_idx = _induced_block_permutation_index(
                    permutation_blocks_inf,
                    global_permutation_keys[b, inference_idx],
                )

            perm_views = [
                x if v_idx != response_view else x[perm_idx, :]
                for v_idx, x in enumerate(inf_views_k)
            ]

            null_test_corr[k, b] = align_sign_k * mean_response_vs_predictors_corr(
                model,
                perm_views,
                response_view=response_view,
            )

        if np.isfinite(obs_test_corr[k]) and np.any(np.isfinite(null_test_corr[k])):
            p_test_corr[k], ci_test_corr[k] = pval_and_ci(
                obs_test_corr[k],
                null_test_corr[k],
            )

    return {
        "discovery_idx": discovery_idx,
        "inference_idx": inference_idx,
        "obs_test_corr": obs_test_corr,
        "p_test_corr": p_test_corr,
        "ci_test_corr": ci_test_corr,
        "null_test_corr": null_test_corr,
        "discovery_results": discovery_results,
    }

def _run_one_multisplit_rep(
    rep_idx,
    split_seed,
    split_random_state,
    model_class,
    view_metadata,
    combined_group,
    train_test_split_by_group,
    sample_param_func,
    response_view,
    inference_size,
    discovery_n_splits,
    discovery_n_iter,
    reduced_cv,
    n_permutations,
    selection_threshold,
    model_kwargs,
    design,
    epochs,
    selection_rule,
    param_complexity_fn,
    permutation_blocks,
    n_jobs_discovery,
    latent_dimensions,
    global_permutation_keys,
):
    out = discovery_inference_permutation_test(
        model_class=model_class,
        view_metadata=view_metadata,
        combined_group=combined_group,
        train_test_split_by_group=train_test_split_by_group,
        sample_param_func=sample_param_func,
        response_view=response_view,
        inference_size=inference_size,
        discovery_split_seed=int(split_seed),
        discovery_n_splits=discovery_n_splits,
        discovery_n_iter=discovery_n_iter,
        reduced_cv=reduced_cv,
        n_permutations=n_permutations,
        selection_threshold=selection_threshold,
        model_kwargs=model_kwargs,
        design=design,
        epochs=epochs,
        random_state=int(split_random_state),
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
        permutation_blocks=permutation_blocks,
        n_jobs_discovery=n_jobs_discovery,
        latent_dimensions=latent_dimensions,
        global_permutation_keys=global_permutation_keys,
    )

    return {
        "rep_idx": int(rep_idx),
        "split_seed": int(split_seed),
        "split_random_state": int(split_random_state),
        "obs_test_corr": _to_1d_float_array(out.get("obs_test_corr")),
        "p_test_corr": _to_1d_float_array(out.get("p_test_corr")),
        "null_ref_interval_95": np.asarray(
            out.get("ci_test_corr", np.empty((0, 2))),
            dtype=float,
        ),
        "raw": out,
    }



def multisplit_discovery_inference_test(
    model_class,
    view_metadata,
    combined_group,
    train_test_split_by_group,
    sample_param_func,
    response_view=0,
    inference_size=0.3,
    discovery_split_seeds=None,
    n_repeats=None,
    discovery_n_splits=100,
    discovery_n_iter=50,
    reduced_cv=10,
    n_permutations=1000,
    selection_threshold=0.9,
    model_kwargs=None,
    design=None,
    epochs=200,
    random_state=0,
    selection_rule="max_mean",
    param_complexity_fn=None,
    permutation_blocks=None,
    n_jobs_discovery=1,
    n_jobs_splits=1,
    combine_p_method="cauchy",
    show_progress=True,
    latent_dimensions=1,
    aggregate_stat="median",
):
    """
    Repeated discovery/inference locked-model permutation inference.

    This wraps `discovery_inference_permutation_test` across many outer splits
    and aggregates split-wise effect estimates and p-values.

    Two multisplit inference summaries are returned:

    1. Combined split-wise p-values
       (`combined_p_test_corr`, `combined_p_test_corr_min_bonferroni`)
    2. A direct permutation test for an aggregate effect across splits
       (`aggregate_p_test_corr`), where the aggregate statistic is controlled
       by `aggregate_stat`

    Parameters
    ----------
    discovery_split_seeds : array-like or None, default=None
        Seeds controlling the outer discovery/inference split. If None,
        `n_repeats` random seeds are generated.
    n_repeats : int or None, default=None
        Number of outer repeats if `discovery_split_seeds` is None.
    selection_rule : {'max_mean', 'one_se'}, default='max_mean'
        Strategy for selecting hyperparameters from CV results.
        - 'max_mean':
            Choose the configuration with the highest mean CV score.
        - 'one_se':
            Choose the simplest configuration whose mean CV score is within
            one standard error of the best mean CV score. Simplicity is
            determined by `param_complexity_fn` when provided; otherwise the
            first qualifying candidate is used.
    param_complexity_fn : callable or None, default=None
        Optional function that maps a hyperparameter dictionary to a numeric
        complexity score, where smaller values indicate simpler models. This
        is only used when `selection_rule='one_se'`.
    combine_p_method : {'cauchy', 'min_bonferroni'}, default='cauchy'
        Method for combining split-wise p-values per latent dimension.
    latent_dimensions : int, default=1
        Maximum number of latent dimensions considered during discovery.
    aggregate_stat : {'mean', 'median', 'fraction_positive'}, default='median'
        Aggregate statistic used for direct multisplit permutation inference.

    Returns
    -------
    dict
        Contains split-wise raw outputs, descriptive summaries, split-wise
        combined p-values, and a direct permutation-calibrated aggregate test
        across splits.
    """
    combined_group = np.asarray(combined_group)

    if discovery_split_seeds is None:
        if n_repeats is None:
            raise ValueError(
                "Provide either `discovery_split_seeds` or `n_repeats`."
            )
        rng = np.random.default_rng(random_state)
        discovery_split_seeds = rng.integers(
            0,
            2**32 - 1,
            size=int(n_repeats),
            dtype=np.uint32,
        )
    else:
        discovery_split_seeds = np.asarray(discovery_split_seeds).ravel()
        if discovery_split_seeds.size == 0:
            raise ValueError("`discovery_split_seeds` must not be empty.")
        n_repeats = int(discovery_split_seeds.size)

    rng = np.random.default_rng(random_state + 1)
    split_random_states = rng.integers(
        0,
        2**32 - 1,
        size=int(n_repeats),
        dtype=np.uint32,
    )

    if permutation_blocks is None:
        permutation_blocks = combined_group
    else:
        permutation_blocks = np.asarray(permutation_blocks)
        if permutation_blocks.shape[0] != combined_group.shape[0]:
            raise ValueError(
                "`permutation_blocks` must have the same length as "
                "`combined_group`."
            )

    # Shared keys synchronize permutation replicate b across all outer splits.
    rng_perm = np.random.default_rng(random_state + 2)
    global_permutation_keys = rng_perm.random(
        size=(int(n_permutations), combined_group.shape[0]),
    )

    worker_kwargs = dict(
        model_class=model_class,
        view_metadata=view_metadata,
        combined_group=combined_group,
        train_test_split_by_group=train_test_split_by_group,
        sample_param_func=sample_param_func,
        response_view=response_view,
        inference_size=inference_size,
        discovery_n_splits=discovery_n_splits,
        discovery_n_iter=discovery_n_iter,
        reduced_cv=reduced_cv,
        n_permutations=n_permutations,
        selection_threshold=selection_threshold,
        model_kwargs=model_kwargs,
        design=design,
        epochs=epochs,
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
        permutation_blocks=permutation_blocks,
        n_jobs_discovery=n_jobs_discovery,
        latent_dimensions=latent_dimensions,
        global_permutation_keys=global_permutation_keys,
    )

    if n_jobs_splits == 1:
        split_results = []
        iterator = zip(
            range(n_repeats),
            discovery_split_seeds,
            split_random_states,
        )
        iterator = tqdm(
            iterator,
            total=n_repeats,
            disable=not show_progress,
            desc="Multi-split locked-model inference",
        )

        for rep_idx, split_seed, split_random_state in iterator:
            split_results.append(
                _run_one_multisplit_rep(
                    rep_idx=int(rep_idx),
                    split_seed=int(split_seed),
                    split_random_state=int(split_random_state),
                    **worker_kwargs,
                )
            )
    else:
        split_results = Parallel(n_jobs=n_jobs_splits, backend="loky")(
            delayed(_run_one_multisplit_rep)(
                rep_idx=int(rep_idx),
                split_seed=int(split_seed),
                split_random_state=int(split_random_state),
                **worker_kwargs,
            )
            for rep_idx, split_seed, split_random_state in zip(
                range(n_repeats),
                discovery_split_seeds,
                split_random_states,
            )
        )

    obs_matrix = _stack_padded_1d(
        [res["obs_test_corr"] for res in split_results]
    )
    p_matrix = _stack_padded_1d(
        [res["p_test_corr"] for res in split_results]
    )
    null_cube = _stack_padded_null(
        [
            np.asarray(
                res["raw"].get("null_test_corr", np.empty((0, 0))),
                dtype=float,
            )
            for res in split_results
        ]
    )

    effect_summary = _summarize_columns(obs_matrix)
    p_summary = _summarize_columns(p_matrix)

    n_dims = obs_matrix.shape[1]

    p_combined = np.full(n_dims, np.nan, dtype=float)
    p_combined_min_bonf = np.full(n_dims, np.nan, dtype=float)

    for k in range(n_dims):
        pvals_k = p_matrix[:, k]
        p_combined[k] = combine_pvalues_multisplit(
            pvals_k,
            method=combine_p_method,
        )
        p_combined_min_bonf[k] = combine_pvalues_multisplit(
            pvals_k,
            method="min_bonferroni",
        )

    aggregate_obs_test_corr = _aggregate_over_splits(
        obs_matrix,
        aggregate_stat,
    )
    
    n_perms_eff = null_cube.shape[2] if null_cube.ndim == 3 else 0
    aggregate_null_test_corr = np.full(
         (n_dims, n_perms_eff),
         np.nan,
         dtype=float,
     )
     
    aggregate_p_test_corr = np.full(n_dims, np.nan, dtype=float)
    aggregate_ci_test_corr = np.full((n_dims, 2), np.nan, dtype=float)
 
    for k in range(n_dims):
        if null_cube.ndim != 3 or k >= null_cube.shape[1]:
            continue
 
        null_mat_k = null_cube[:, k, :]
        agg_null_k = _aggregate_over_splits(null_mat_k, aggregate_stat)
        aggregate_null_test_corr[k, : agg_null_k.size] = agg_null_k
 
        if np.isfinite(aggregate_obs_test_corr[k]):
            aggregate_p_test_corr[k], aggregate_ci_test_corr[k] = pval_and_ci(
                aggregate_obs_test_corr[k],
                agg_null_k,
             )
 
    return {
        "split_results": split_results,
        "obs_test_corr_by_split": obs_matrix,
        "p_test_corr_by_split": p_matrix,
        "null_test_corr_by_split": null_cube,
        "effect_summary": effect_summary,
        "p_summary": p_summary,
        "combined_p_test_corr": p_combined,
        "combined_p_test_corr_min_bonferroni": p_combined_min_bonf,
        "aggregate_stat": aggregate_stat,
        "aggregate_obs_test_corr": aggregate_obs_test_corr,
        "aggregate_null_test_corr": aggregate_null_test_corr,
        "aggregate_p_test_corr": aggregate_p_test_corr,
        "aggregate_ci_test_corr": aggregate_ci_test_corr,
        "discovery_split_seeds": np.asarray(
             discovery_split_seeds, dtype=np.uint32
         ),
        "split_random_states": split_random_states.astype(np.uint32),
     }
    





#%%

## Omnibus test


def _compute_omnibus_stat(
    obs_test_corr_by_split,
    use_absolute_corr=True,
):
    """
    One global omnibus statistic for the full pipeline.

    Default statistic:
        T = mean over outer splits of the sum over dimensions of held-out
            absolute correlations

    Missing / non-retained dimensions contribute 0.

    Parameters
    ----------
    obs_test_corr_by_split : ndarray, shape (n_outer_splits, n_dims)
        Held-out correlations from the full pipeline.
    use_absolute_corr : bool, default True
        If True, use absolute correlations to respect sign indeterminacy.

    Returns
    -------
    stat : float
        Scalar omnibus statistic.
    aux : dict
        Helpful summaries.
    """
    mat = np.asarray(obs_test_corr_by_split, dtype=float)
    if mat.ndim != 2:
        raise ValueError("obs_test_corr_by_split must be 2D.")

    # Missing / non-retained dimensions contribute zero omnibus signal.
    mat0 = np.where(np.isfinite(mat), mat, 0.0)

    if use_absolute_corr:
        contrib = np.abs(mat0)
    else:
        contrib = np.clip(mat0, 0.0, None)

    split_scores = np.sum(contrib, axis=1)
    stat = float(np.mean(split_scores)) if split_scores.size > 0 else np.nan

    return stat, {
        "split_scores": split_scores,
        "per_dim_mean": np.mean(contrib, axis=0) if mat.shape[1] > 0 else np.array([]),
        "per_dim_median": (np.median(contrib, axis=0) if mat.shape[1] > 0 else np.array([])),
        "n_outer_splits": int(mat.shape[0]),
        "n_dims": int(mat.shape[1]),
        "use_absolute_corr": bool(use_absolute_corr),
    }


def _run_one_omnibus_permutation(
    perm_seed,
    model_class,
    view_metadata,
    combined_group,
    train_test_split_by_group,
    sample_param_func,
    outer_split_seeds,
    outer_split_random_states,
    response_view=0,
    permutation_blocks=None,
    inference_size=0.3,
    latent_dimensions=6,
    discovery_n_splits=100,
    discovery_n_iter=50,
    reduced_cv=10,
    selection_threshold=0.9,
    model_kwargs=None,
    design=None,
    epochs=200,
    selection_rule="max_mean",
    param_complexity_fn=None,
    n_jobs_discovery=1,
    n_jobs_splits=1,
    use_absolute_corr=True,
):
    """
    One full-pipeline permutation replicate.
    """
    permuted_view_metadata = _permute_response_view_metadata_global(
        view_metadata=view_metadata,
        response_view=response_view,
        permutation_blocks=permutation_blocks,
        seed=int(perm_seed),
    )

    perm_pipeline = _run_multisplit_full_pipeline(
        model_class=model_class,
        view_metadata=permuted_view_metadata,
        combined_group=combined_group,
        train_test_split_by_group=train_test_split_by_group,
        sample_param_func=sample_param_func,
        outer_split_seeds=outer_split_seeds,
        outer_split_random_states=outer_split_random_states,
        response_view=response_view,
        inference_size=inference_size,
        latent_dimensions=latent_dimensions,
        discovery_n_splits=discovery_n_splits,
        discovery_n_iter=discovery_n_iter,
        reduced_cv=reduced_cv,
        selection_threshold=selection_threshold,
        model_kwargs=model_kwargs,
        design=design,
        epochs=epochs,
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
        n_jobs_discovery=n_jobs_discovery,
        n_jobs_splits=n_jobs_splits,
        show_progress=False,
    )

    stat, stat_aux = _compute_omnibus_stat(
        perm_pipeline["obs_test_corr_by_split"],
        use_absolute_corr=use_absolute_corr,
    )

    return {
        "perm_seed": int(perm_seed),
        "global_stat": stat,
        "per_dim_mean": stat_aux["per_dim_mean"],
        "split_scores": stat_aux["split_scores"],
    }


def omnibus_full_pipeline_permutation_test(
    model_class,
    view_metadata,
    combined_group,
    train_test_split_by_group,
    sample_param_func,
    response_view=0,
    permutation_blocks=None,
    inference_size=0.3,
    latent_dimensions=6,
    outer_split_seeds=None,
    n_outer_splits=None,
    discovery_n_splits=100,
    discovery_n_iter=50,
    reduced_cv=10,
    selection_threshold=0.9,
    model_kwargs=None,
    design=None,
    epochs=200,
    selection_rule="max_mean",
    param_complexity_fn=None,
    n_perm=100,
    random_state=0,
    use_absolute_corr=True,
    n_jobs_discovery=1,
    n_jobs_splits=1,
    n_jobs_perm=1,
    show_progress=True,
):
    """
    Full end-to-end omnibus permutation test.

    In each permutation replicate, rows of the response view are permuted
    within exchangeability blocks and the entire discovery/tuning/selection/
    reduced-refit/held-out-evaluation workflow is rerun.

    Default omnibus statistic
    -------------------------
    Mean across outer splits of the sum of absolute held-out latent
    correlations across retained dimensions, with missing dimensions
    contributing zero.

    Parameters
    ----------
    outer_split_seeds : array-like or None
        Seeds controlling the outer discovery/inference splits. If None,
        `n_outer_splits` seeds are generated.
    n_outer_splits : int or None
        Number of outer splits if `outer_split_seeds` is None.
    permutation_blocks : array-like or None
        Exchangeability blocks for global permutation of the response view.
        If None, `combined_group` is used.
    n_perm : int, default 100
        Number of full-pipeline permutation replicates. Because each replicate
        reruns the entire workflow, values such as 100-200 are often a
        realistic compromise.
    use_absolute_corr : bool, default True
        Use absolute held-out correlations in the omnibus statistic to respect
        sign indeterminacy.

    Returns
    -------
    results : dict
        Observed omnibus statistic, null distribution, empirical p-value, and
        full observed pipeline outputs.
    """
    if outer_split_seeds is None:
        if n_outer_splits is None:
            raise ValueError(
                "Provide either `outer_split_seeds` or `n_outer_splits`."
            )
        rng = np.random.default_rng(random_state)
        outer_split_seeds = rng.integers(
            0,
            2**32 - 1,
            size=int(n_outer_splits),
            dtype=np.uint32,
        )
    else:
        outer_split_seeds = np.asarray(outer_split_seeds).ravel()
        if outer_split_seeds.size == 0:
            raise ValueError("`outer_split_seeds` must not be empty.")
        n_outer_splits = int(outer_split_seeds.size)

    rng = np.random.default_rng(random_state + 1)
    outer_split_random_states = rng.integers(
        0,
        2**32 - 1,
        size=int(n_outer_splits),
        dtype=np.uint32,
    )

    if permutation_blocks is None:
        permutation_blocks = np.asarray(combined_group)
    else:
        permutation_blocks = np.asarray(permutation_blocks)

    # --------------------------------------------------------------
    # 1. Observed full pipeline
    # --------------------------------------------------------------
    observed_pipeline = _run_multisplit_full_pipeline(
        model_class=model_class,
        view_metadata=view_metadata,
        combined_group=combined_group,
        train_test_split_by_group=train_test_split_by_group,
        sample_param_func=sample_param_func,
        outer_split_seeds=outer_split_seeds,
        outer_split_random_states=outer_split_random_states,
        response_view=response_view,
        inference_size=inference_size,
        latent_dimensions=latent_dimensions,
        discovery_n_splits=discovery_n_splits,
        discovery_n_iter=discovery_n_iter,
        reduced_cv=reduced_cv,
        selection_threshold=selection_threshold,
        model_kwargs=model_kwargs,
        design=design,
        epochs=epochs,
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
        n_jobs_discovery=n_jobs_discovery,
        n_jobs_splits=n_jobs_splits,
        show_progress=show_progress,
        progress_desc="Observed outer splits",
    )

    observed_stat, observed_stat_aux = _compute_omnibus_stat(
        observed_pipeline["obs_test_corr_by_split"],
        use_absolute_corr=use_absolute_corr,
    )

    # --------------------------------------------------------------
    # 2. Full-pipeline permutations
    # --------------------------------------------------------------
    rng = np.random.default_rng(random_state + 2)
    perm_seeds = rng.integers(
        0,
        2**32 - 1,
        size=int(n_perm),
        dtype=np.uint32,
    )

    worker_kwargs = dict(
        model_class=model_class,
        view_metadata=view_metadata,
        combined_group=combined_group,
        train_test_split_by_group=train_test_split_by_group,
        sample_param_func=sample_param_func,
        outer_split_seeds=outer_split_seeds,
        outer_split_random_states=outer_split_random_states,
        response_view=response_view,
        permutation_blocks=permutation_blocks,
        inference_size=inference_size,
        latent_dimensions=latent_dimensions,
        discovery_n_splits=discovery_n_splits,
        discovery_n_iter=discovery_n_iter,
        reduced_cv=reduced_cv,
        selection_threshold=selection_threshold,
        model_kwargs=model_kwargs,
        design=design,
        epochs=epochs,
        selection_rule=selection_rule,
        param_complexity_fn=param_complexity_fn,
        n_jobs_discovery=n_jobs_discovery,
        n_jobs_splits=n_jobs_splits,
        use_absolute_corr=use_absolute_corr,
    )

    if n_jobs_perm == 1:
        perm_results = []
        iterator = tqdm(
            perm_seeds,
            disable=not show_progress,
            desc="Full-pipeline permutations",
        )

        for perm_seed in iterator:
            out = _run_one_omnibus_permutation(
                perm_seed=int(perm_seed),
                **worker_kwargs,
            )
            perm_results.append(out)

            iterator.set_postfix(
                obs=f"{observed_stat:.4f}" if np.isfinite(observed_stat) else "nan",
                last_perm=(
                    f"{out['global_stat']:.4f}"
                    if np.isfinite(out["global_stat"])
                    else "nan"
                ),
            )
    else:
        # To avoid oversubscription, it is usually best to keep
        # n_jobs_splits=1 and n_jobs_discovery=1 when n_jobs_perm > 1.
        perm_results = Parallel(n_jobs=n_jobs_perm, backend="loky")(
            delayed(_run_one_omnibus_permutation)(
                perm_seed=int(perm_seed),
                **worker_kwargs,
            )
            for perm_seed in perm_seeds
        )

    null_stats = np.asarray(
        [res["global_stat"] for res in perm_results],
        dtype=float,
    )
    null_per_dim_mean = _stack_padded_1d(
        [res["per_dim_mean"] for res in perm_results]
    )

    p_value = _empirical_p_value_greater(observed_stat, null_stats)

    valid_null = null_stats[np.isfinite(null_stats)]
    null_ref_interval_95 = (
        np.percentile(valid_null, [2.5, 97.5])
        if valid_null.size > 0
        else np.array([np.nan, np.nan])
    )

    return {
        "observed_global_stat": observed_stat,
        "observed_split_scores": observed_stat_aux["split_scores"],
        "observed_per_dim_mean": observed_stat_aux["per_dim_mean"],
        "observed_per_dim_median": observed_stat_aux["per_dim_median"],
        "observed_test_corr_by_split": observed_pipeline["obs_test_corr_by_split"],
        "observed_pipeline": observed_pipeline,
        "null_global_stats": null_stats,
        "null_per_dim_mean": null_per_dim_mean,
        "null_ref_interval_95": null_ref_interval_95,
        "p_value": p_value,
        "n_perm": int(n_perm),
        "n_outer_splits": int(n_outer_splits),
        "use_absolute_corr": bool(use_absolute_corr),
        "perm_results": perm_results,
        "outer_split_seeds": np.asarray(outer_split_seeds, dtype=np.uint32),
        "outer_split_random_states": np.asarray(
            outer_split_random_states,
            dtype=np.uint32,
        ),
    }
