# ccagg/model.py


"""
Parallel hyperparameter search with ensemble evaluation for multi-view CCA.

Pipeline
--------
1. ``parallel_best_param_ensemble_multidim`` — repeated train/test splits,
   hyperparameter search via inner cross-validation, best-model selection,
   and optional bootstrap ensemble evaluation.
2. ``run_rcca_stability_selection_multidim`` — feature stability selection
   via repeated rCCA fits across splits.

Supporting functions
--------------------
train_test_split_by_group
    Group-stratified train/test splitting.
evaluate_param2
    Inner cross-validation scoring for a single hyperparameter set.
compute_r2_per_dimension
    Coefficient of determination between latent scores across dimensions.
calculate_cross_loadings
    Cross-loadings between raw features and latent scores.
evaluate_bootstrap_cca_performance
    Diagnostic bootstrap ensemble evaluation (performance estimation only).
"""

# ── Standard library ──────────────────────────────────────────────────────────
import time
import warnings
from copy import deepcopy

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.utils import resample

# ── Internal ──────────────────────────────────────────────────────────────────
from .utils import train_test_split_by_group, get_fold_data, subset_view_metadata
from .stats import evaluate_param, compute_r2_per_dimension, compute_test_AVE, calculate_cross_loadings


#%%
# =============================================================================
# MAIN PIPELINE
# =============================================================================


def parallel_best_param_ensemble_multidim(
    model_class,
    combined_group,
    view_metadata,
    train_test_split_by_group,
    sample_param_func,
    n_splits=10,
    n_iter=50,
    cv=10,
    epochs=200,
    n_jobs=12,
    model_kwargs=None,
    design=None,
    run_final_ensemble=True,
    n_ensemble_runs=20,
    response_view=0,
    latent_dimensions=1,
    n_jobs_inner_ensemble=1,
    selection_abs_weight_tol=1e-6,
    n_features_per_view=None,
    split_seeds=None,
    selection_rule="max_mean",
    param_complexity_fn=None,
):
    """
    Repeated train/test resampling for multiview CCA with nested
    hyperparameter tuning.

    Key behavior:
    - Hyperparameters are selected using inner CV only.
    - Outer test scores are computed only for evaluation / diagnostics.
    - If selection_rule == "one_se" and fold-level CV scores are available
      from evaluate_param2(...), the one-standard-error rule is used.
    - If fold-level CV scores are unavailable, selection falls back to the
      maximum mean inner-CV score.

    Parameters
    ----------
    selection_rule : {"max_mean", "one_se"}
        Rule used to choose the best hyperparameter setting from inner CV.
    param_complexity_fn : callable or None
        Optional function mapping a parameter dict to a scalar complexity
        score. Smaller values are treated as "simpler" when using the
        one-standard-error rule.
    """

    views = [meta["data"] for meta in view_metadata]
    n_views = len(views)

    if n_features_per_view is None:
        n_features_per_view = [v.shape[1] for v in views]

    if split_seeds is None:
        rng = np.random.default_rng(20)
        split_seeds = rng.integers(0, 2**32 - 1, size=n_splits, dtype=np.uint32)
    else:
        split_seeds = np.asarray(split_seeds, dtype=np.uint32)
        n_splits = len(split_seeds)

    def _extract_fold_scores(cv_aux):
        """
        Try to recover fold-level CV scores from the second output returned
        by evaluate_param. If unavailable, return None.
        """
        if cv_aux is None:
            return None

        if isinstance(cv_aux, dict):
            for key in ("fold_scores", "scores", "cv_scores"):
                if key in cv_aux:
                    try:
                        arr = np.asarray(cv_aux[key], dtype=float).ravel()
                        return arr
                    except Exception:
                        return None

        if isinstance(cv_aux, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(cv_aux, dtype=float).ravel()
                if arr.size > 0:
                    return arr
            except Exception:
                return None

        return None

    def _safe_float(x, default=np.nan):
       
        try:
            val = float(x)
            return val if np.isfinite(val) else default
        except Exception:
            return default

    def _select_best_idx_from_cv(cv_records, param_list):
        
        """
        Choose the best hyperparameter index using inner-CV summaries only.
        """
        
        means_0 = [rec["mean_cv"] for rec in cv_records]
        means = np.asarray(means_0, dtype=float)

        if means.size == 0 or np.all(np.isnan(means)):
            return 0

        best_mean_idx = int(np.nanargmax(means))

        if selection_rule != "one_se":
            return best_mean_idx

        ses_0 = [rec["se_cv"] for rec in cv_records]
        ses = np.asarray(ses_0, dtype=float)
        best_se = ses[best_mean_idx]

        if not np.isfinite(best_se):
            return best_mean_idx

        cutoff = means[best_mean_idx] - best_se
        candidate_idxs = np.flatnonzero(
            np.isfinite(means) & (means >= cutoff)
        )

        if candidate_idxs.size == 0:
            return best_mean_idx

        if param_complexity_fn is not None:
            complexities = []
            for idx in candidate_idxs:
                try:
                    complexities.append(param_complexity_fn(param_list[idx]))
                except Exception:
                    complexities.append(np.nan)

            complexities = np.asarray(complexities, dtype=float)
            if np.any(np.isfinite(complexities)):
                local_best = int(np.nanargmin(complexities))
                return int(candidate_idxs[local_best])

        # Fallback: first acceptable candidate.
        return int(candidate_idxs[0])

    def _compute_outer_test_score(model, eval_views):
        
        """
        Diagnostic outer-test score across design-weighted pairs for the
        first latent dimension.
        """
        
        corrs = model.pairwise_correlations(eval_views)

        effective_design = (
            design
            if design is not None
            else (np.ones((n_views, n_views)) - np.eye(n_views))
        )

        pair_scores = []
        for i in range(n_views):
            for j in range(i + 1, n_views):
                if effective_design[i, j] != 0:
                    pair_scores.append(corrs[i, j, 0])

        if not pair_scores:
            return np.nan

        return float(np.nanmean(pair_scores))

    def run_split(split_idx):
        try:
            param_search_selection_counts = [np.zeros((n_features_per_view[v], latent_dimensions), dtype=int) for v in range(n_views)]

            split_cv_scores = []
            split_cv_records = []
            split_outer_test_scores_per_iter = []

            seed = int(split_seeds[split_idx])

            # ----------------------------------------------------------
            # Outer split
            # ----------------------------------------------------------
            train_idx, test_idx = train_test_split_by_group(combined_group, test_size=0.3, seed=seed)

            train_views, test_views = get_fold_data(train_idx, test_idx, view_metadata)

            train_metadata = [
                {
                    **meta,
                    "data": meta["data"][train_idx],
                    "confounds": (
                        meta["confounds"][train_idx]
                        if meta["confounds"] is not None
                        else None
                    ),
                }
                for meta in view_metadata
            ]
            train_group = combined_group[train_idx]

            # ----------------------------------------------------------
            # Sample candidate hyperparameters
            # ----------------------------------------------------------
            param_list = sample_param_func(
                n_iter=n_iter,
                seed=seed,
                n_views=n_views,
                latent_dimensions=latent_dimensions,
            )

            # ----------------------------------------------------------
            # Inner-CV search
            # ----------------------------------------------------------
            for params_idx, params in enumerate(param_list):
                ave_cv_score, cv_aux = evaluate_param(
                    model_class=model_class,
                    params=params,
                    view_metadata=train_metadata,
                    combined_group=train_group,
                    cv=cv,
                    random_state=seed + params_idx,
                    epochs=epochs,
                    tol=params.get("tol", 1e-5) if params is not None else 1e-5,
                    model_kwargs=model_kwargs,
                    design=design,
                    response_view=response_view,
                    latent_dimensions=latent_dimensions,
                )

                mean_cv = _safe_float(ave_cv_score)
                fold_scores = _extract_fold_scores(cv_aux)

                if fold_scores is not None:
                    valid_fold_scores = fold_scores[
                        np.isfinite(fold_scores)
                    ]
                    if valid_fold_scores.size > 0:
                        sd_cv = float(np.nanstd(valid_fold_scores, ddof=1))
                        se_cv = float(
                            sd_cv / np.sqrt(valid_fold_scores.size)
                        )
                        n_cv = int(valid_fold_scores.size)
                    else:
                        sd_cv = np.nan
                        se_cv = np.nan
                        n_cv = 0
                else:
                    sd_cv = np.nan
                    se_cv = np.nan
                    n_cv = 0

                split_cv_scores.append(mean_cv)
                split_cv_records.append(
                    {
                        "mean_cv": mean_cv,
                        "sd_cv": sd_cv,
                        "se_cv": se_cv,
                        "n_cv": n_cv,
                        "params": params,
                    }
                )

                # Outer-test score for this candidate
                current_model_init_kwargs = {
                    **(params or {}),
                    **(model_kwargs or {}),
                }
                if design is not None:
                    current_model_init_kwargs["design"] = design
                if model_class.__name__ != "GRCCA2":
                    current_model_init_kwargs["early_stopping"] = True
                    current_model_init_kwargs["epochs"] = epochs
                    current_model_init_kwargs["tol"] = 1e-5

                temp_model = model_class(**current_model_init_kwargs)
                temp_model.fit(train_views)

                diag_outer_score = _compute_outer_test_score(temp_model, test_views)
                split_outer_test_scores_per_iter.append(diag_outer_score)


                # Track feature usage for diagnostics
                weights_attr = getattr(temp_model, "weights_", None)
                if weights_attr is not None:
                    for v_idx_loop, w_iter in enumerate(weights_attr):
                        if w_iter is None:
                            continue
                        mask = (
                            np.abs(w_iter) > selection_abs_weight_tol
                        ).astype(int)
                        param_search_selection_counts[v_idx_loop] += mask


            # Select best params using inner CV only
            best_idx = _select_best_idx_from_cv(split_cv_records, param_list)
            best_params_for_this_split = param_list[best_idx]


            # Fit best model on full outer-train partition
            best_model_per_split = model_class(
                **(best_params_for_this_split or {}),
                **(model_kwargs or {}),
                **({"design": design} if design is not None else {}),
                **(
                    {
                        "early_stopping": True,
                        "epochs": best_params_for_this_split.get(
                            "epochs", epochs
                        ),
                        "tol": best_params_for_this_split.get("tol", 1e-5),
                    }
                    if model_class.__name__ != "GRCCA2"
                    else {}
                ),
            )
            best_model_per_split.fit(train_views)

            corr_mat_tr = best_model_per_split.pairwise_correlations(train_views)
            pred_idx = [ v for v in range(corr_mat_tr.shape[0]) if v != response_view ]   
            train_corrs_all_dims = corr_mat_tr[response_view, pred_idx, :]
            corr_mat_te = best_model_per_split.pairwise_correlations(test_views)
            test_corrs_all_dims = corr_mat_te[response_view, pred_idx, :]

            train_r2_per_dim = compute_r2_per_dimension(best_model_per_split, train_views, response_view)
            test_r2_per_dim = compute_r2_per_dimension(best_model_per_split, test_views, response_view)

            train_loadings = ( best_model_per_split.loadings_(train_views) if hasattr(best_model_per_split, "loadings_") else [])
            test_loadings = ( best_model_per_split.loadings_(test_views) if hasattr(best_model_per_split, "loadings_") else [])
             
            train_scores_list = best_model_per_split.transform(train_views)
            test_scores_list = best_model_per_split.transform(test_views)

            train_cross_loadings_val = calculate_cross_loadings(
                raw_views_list=train_views,
                scores_list=train_scores_list,
                target_view_idx=response_view
            )
            test_cross_loadings_val = calculate_cross_loadings(
                raw_views_list=test_views,
                scores_list=test_scores_list,
                target_view_idx=response_view
            )

            param_search_selection_freq_this_split = [ param_search_selection_counts[v] / float(n_iter) for v in range(n_views)]                


            selection_freq_sm = []
            weights_attr = getattr(best_model_per_split, "weights_", None)
            if weights_attr is not None:
                for v_idx, w in enumerate(weights_attr):
                    if w is not None:
                        freq_matrix = (np.abs(w) > selection_abs_weight_tol).astype(int)
                        selection_freq_sm.append(freq_matrix)
                    else:
                        selection_freq_sm.append(
                            np.zeros((n_features_per_view[v_idx], latent_dimensions)))
            else:
                selection_freq_sm = [np.zeros((nf, latent_dimensions)) for nf in n_features_per_view]

            bootstrap_output = None
            if run_final_ensemble:
                bootstrap_output = evaluate_bootstrap_cca_performance(
                    train_views=train_views,
                    eval_views=test_views,
                    model_class=model_class,
                    param_set=deepcopy(best_params_for_this_split),
                    random_state=seed,
                    n_runs=n_ensemble_runs,
                    model_kwargs=model_kwargs,
                    design=design,
                    response_view=response_view,
                    n_jobs_inner=n_jobs_inner_ensemble,
                )

            return {
                "best_params": best_params_for_this_split,
                "best_param_index": best_idx,
                "cv_scores_all_params": split_cv_scores,
                "cv_records_all_params": split_cv_records,
                "outer_test_scores_all_params": split_outer_test_scores_per_iter,
                "train_r2_per_dim": train_r2_per_dim,
                "test_r2_per_dim": test_r2_per_dim,
                "train_corrs_all_dims": train_corrs_all_dims,
                "test_corrs_all_dims": test_corrs_all_dims,
                "train_scores": train_scores_list,
                "test_scores": test_scores_list,
                "train_loadings_single_best_model": train_loadings,
                "test_loadings_single_best_model": test_loadings,
                "train_cross_loadings_single_best_model": train_cross_loadings_val,
                "test_cross_loadings_single_best_model": test_cross_loadings_val,
                "param_search_selection_freq_this_split": param_search_selection_freq_this_split,
                "selection_freq_single_best_model": selection_freq_sm,
                "weights": getattr(best_model_per_split, "weights_", None),
                "bootstrap_results": bootstrap_output,
                "split_seed": seed,
            }

        except Exception as e:
            warnings.warn(f"Split {split_idx} failed with error: {e}")
            return None

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------
    results_list = []
    batch_size = 10
    pause_secs = 2.0

    for batch_start in range(0, n_splits, batch_size):
        batch_end = min(n_splits, batch_start + batch_size)
        print(f"Processing splits {batch_start}–{batch_end - 1} ...")
        batch_idxs = list(range(batch_start, batch_end))

        batch_res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(run_split)(i) for i in batch_idxs)
            
        results_list.extend([r for r in batch_res if r is not None])

        if batch_end < n_splits:
            print(f"Pausing {pause_secs} s before next batch ...")
            time.sleep(pause_secs)

    if not results_list:
        return {}


    # Aggregate average selection frequency across splits
    average_selection_freq = []
    for view_idx in range(n_views):
        freq_matrices_for_view = []
        for res in results_list:
            sel_freq = res.get("selection_freq_single_best_model")
            if sel_freq is not None and view_idx < len(sel_freq):
                freq_matrices_for_view.append(sel_freq[view_idx])

        if freq_matrices_for_view:
            shapes = {m.shape for m in freq_matrices_for_view}
            if len(shapes) > 1:
                warnings.warn(
                    f"Inconsistent shapes for selection frequency in view "
                    f"{view_idx}. Skipping averaging."
                )
                max_shape = max(shapes, key=lambda s: s[0] * s[1])
                average_selection_freq.append(np.zeros(max_shape))
            else:
                stacked_matrices = np.stack(freq_matrices_for_view, axis=0)
                mean_freq_matrix = np.mean(stacked_matrices, axis=0)
                average_selection_freq.append(mean_freq_matrix)
        else:
            average_selection_freq.append(np.array([]))


    # Align and average weights across splits
    weight_lists_per_split = [
        res.get("weights")
        for res in results_list
        if res.get("weights") is not None
    ]

    avg_weights = None
    average_scores = None

    if weight_lists_per_split:
        S = len(weight_lists_per_split)
        V = len(weight_lists_per_split[0])

        W0_ref = weight_lists_per_split[0][0]
        p0, K = W0_ref.shape

        aligned_weights = [[None] * V for _ in range(S)]

        for s in range(S):
            Ws = weight_lists_per_split[s]
            W0 = Ws[0]

            corr_block = np.corrcoef(W0_ref.T, W0.T)[:K, K:]
            corr_block = np.nan_to_num(np.abs(corr_block), nan=0.0, posinf=0.0, neginf=0.0)

            rows, cols = linear_sum_assignment(-corr_block)

            for v in range(V):
                aligned_weights[s][v] = Ws[v][:, cols].copy()

            for j in range(K):
                r = np.corrcoef(
                    W0_ref[:, j],
                    aligned_weights[s][0][:, j],
                )[0, 1]
                if not np.isfinite(r):
                    r = 1.0
                sign = 1 if r >= 0 else -1
                for v in range(V):
                    aligned_weights[s][v][:, j] *= sign

        avg_weights = []
        for v in range(V):
            stack = np.stack(
                [aligned_weights[s][v] for s in range(S)],
                axis=0,
            )
            avg_weights.append(stack.mean(axis=0))

        average_scores = [views[v] @ avg_weights[v] for v in range(V)]

    final_results = {
        
        ## Detailed outputs:
        "split_seeds": [res.get("split_seed") for res in results_list],
        "selection_freq_per_split": [res.get("param_search_selection_freq_this_split") for res in results_list],
        "selection_freq_best_model": [res.get("selection_freq_single_best_model") for res in results_list],
        "boostrp_evaluations_per_split": [res.get("bootstrap_results") for res in results_list],
        "train_r2_per_dim": [res.get("train_r2_per_dim") for res in results_list],
        "test_r2_per_dim": [res.get("test_r2_per_dim") for res in results_list],
        "train_loadings": [res.get("train_loadings_single_best_model") for res in results_list],
        "test_loadings": [res.get("test_loadings_single_best_model") for res in results_list],
        "train_cross_loadings": [res.get("train_cross_loadings_single_best_model") for res in results_list],
        "test_cross_loadings": [res.get("test_cross_loadings_single_best_model") for res in results_list],
        "best_params_per_split": [res.get("best_params") for res in results_list],
        "best_param_index_per_split": [res.get("best_param_index") for res in results_list],
        "cv_scores_all_params_per_split": [res.get("cv_scores_all_params") for res in results_list],
        "cv_records_all_params_per_split": [res.get("cv_records_all_params") for res in results_list],
        "test_scores_all_params_per_split": [res.get("outer_test_scores_all_params") for res in results_list],
        "outer_test_scores_all_params_per_split": [res.get("outer_test_scores_all_params") for res in results_list],

        ## aggregated variables:
        "average_selection_frequency": average_selection_freq,
        "average_scores": average_scores,
        "average_weights": avg_weights,
        "train_corrs_all_dims": [res.get("train_corrs_all_dims") for res in results_list],
        "test_corrs_all_dims": [res.get("test_corrs_all_dims") for res in results_list],
        "train_scores": [res.get("train_scores") for res in results_list],
        "test_scores": [res.get("test_scores") for res in results_list],

    }

    return final_results


def parallel_stability_selection_multidim(
    model_class,
    combined_group,
    view_metadata,
    train_test_split_by_group,
    sample_param_func,
    n_splits=50,
    n_iter=50,
    cv=10,
    epochs=200,
    n_jobs=1,
    model_kwargs=None,
    design=None,
    response_view=0,
    latent_dimensions=1,
    selection_abs_weight_tol=1e-6,
    split_seeds=None,
    selection_rule="max_mean",
    param_complexity_fn=None,
    test_size=0.3,
):
    """
    Repeated train/test stability-selection: 
    - only passes kwargs actually accepted by the model
    - dynamic inner CV
    - simpler outputs focused on discovery/stability
    """
    
    views = [meta["data"] for meta in view_metadata]
    n_views = len(views)
    n_features_per_view = [v.shape[1] for v in views]
    
    if split_seeds is None:
        rng = np.random.default_rng(20)
        split_seeds = rng.integers(
            0,
            2**32 - 1,
            size=n_splits,
            dtype=np.uint32,
        )
    else:
        split_seeds = np.asarray(split_seeds, dtype=np.uint32)
        n_splits = int(len(split_seeds))
    
    effective_design = (
        np.asarray(design)
        if design is not None
        else (np.ones((n_views, n_views)) - np.eye(n_views))
    )
    
    def run_split(split_idx):
        seed = int(split_seeds[split_idx])
        
        try:
            train_idx, test_idx = train_test_split_by_group(
                combined_group,
                test_size=test_size,
                seed=seed,
            )
            
            train_views, test_views = get_fold_data(
                train_idx,
                test_idx,
                view_metadata,
            )
            train_metadata = subset_view_metadata(view_metadata, train_idx)
            train_group = np.asarray(combined_group)[train_idx]
            
            param_list = sample_param_func(
                n_iter=n_iter,
                seed=seed,
                n_views=n_views,
                latent_dimensions=latent_dimensions,
            )
            
            if param_list is None or len(param_list) == 0:
                raise RuntimeError("sample_param_func returned no parameters.")
            
            cv_records = []
            
            for params_idx, params in enumerate(param_list):
                mean_cv, cv_aux = evaluate_param(
                    model_class=model_class,
                    params=params,
                    view_metadata=train_metadata,
                    combined_group=train_group,
                    cv=cv,
                    random_state=seed + params_idx,
                    epochs=epochs,
                    tol=(
                        params.get("tol", 1e-5)
                        if isinstance(params, dict)
                        else 1e-5
                    ),
                    model_kwargs=model_kwargs,
                    design=design,
                    response_view=response_view,
                    latent_dimensions=latent_dimensions,
                )
                
                mean_cv = safe_float(mean_cv)
                fold_scores = extract_fold_scores(cv_aux)
                
                if fold_scores is not None:
                    valid = fold_scores[np.isfinite(fold_scores)]
                    if valid.size >= 2:
                        sd_cv = float(np.std(valid, ddof=1))
                        se_cv = float(sd_cv / np.sqrt(valid.size))
                        n_cv = int(valid.size)
                    elif valid.size == 1:
                        sd_cv = np.nan
                        se_cv = np.nan
                        n_cv = 1
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
            best_params = param_list[best_idx]
            
            init_kwargs = make_model_init_kwargs(
                model_class=model_class,
                params=best_params,
                model_kwargs=model_kwargs,
                design=design,
                random_state=seed,
                latent_dimensions=latent_dimensions,
                epochs=epochs,
                tol=(
                    best_params.get("tol", 1e-5)
                    if isinstance(best_params, dict)
                    else 1e-5
                ),
            )
            
            model = model_class(**init_kwargs)
            model.fit(train_views)
            
            corr_mat_tr = model.pairwise_correlations(train_views)
            corr_mat_te = model.pairwise_correlations(test_views)
            
            pred_idx = [
                v
                for v in range(n_views)
                if v != response_view and effective_design[response_view, v] != 0
            ]
            
            if len(pred_idx) > 0:
                train_corrs_all_dims = corr_mat_tr[response_view, pred_idx, :]
                test_corrs_all_dims = corr_mat_te[response_view, pred_idx, :]
            else:
                train_corrs_all_dims = np.full(
                    (0, latent_dimensions),
                    np.nan,
                    dtype=float,
                )
                test_corrs_all_dims = np.full(
                    (0, latent_dimensions),
                    np.nan,
                    dtype=float,
                )
            
            selection_mask = [
                np.zeros(
                    (n_features_per_view[v], latent_dimensions),
                    dtype=int,
                )
                for v in range(n_views)
            ]
            
            weights_attr = getattr(model, "weights_", None)
            if weights_attr is not None:
                for v_idx, w in enumerate(weights_attr):
                    w = _coerce_weight_matrix(w)
                    if w is None:
                        continue
                    
                    kk = min(latent_dimensions, w.shape[1])
                    selection_mask[v_idx][:, :kk] = (
                        np.abs(w[:, :kk]) > selection_abs_weight_tol
                    ).astype(int)
            
            return {
                "best_params": best_params,
                "best_param_index": int(best_idx),
                "cv_records_all_params": cv_records,
                "train_corrs_all_dims": train_corrs_all_dims,
                "test_corrs_all_dims": test_corrs_all_dims,
                "selection_mask": selection_mask,
                "split_seed": seed,
            }
        
        except Exception as e:
            warnings.warn(f"Split {split_idx} failed with error: {e}")
            return None
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_split)(i) for i in range(n_splits)
    )
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        return {}
    
    average_selection_frequency = []
    for v_idx in range(n_views):
        mats = [res["selection_mask"][v_idx] for res in results]
        average_selection_frequency.append(np.mean(np.stack(mats, axis=0), axis=0))
    
    return {
        "best_params_per_split": [r["best_params"] for r in results],
        "best_param_index_per_split": [r["best_param_index"] for r in results],
        "cv_records_all_params_per_split": [
            r["cv_records_all_params"] for r in results
        ],
        "train_corrs_all_dims": [r["train_corrs_all_dims"] for r in results],
        "test_corrs_all_dims": [r["test_corrs_all_dims"] for r in results],
        "average_selection_frequency": average_selection_frequency,
        "split_seeds": [r["split_seed"] for r in results],
    }


#%%
# =============================================================================
# DIAGNOSTIC BOOTSTRAP ENSEMBLE EVALUATION
# =============================================================================


def evaluate_bootstrap_cca_performance(
    train_views,
    eval_views,
    model_class,
    param_set,
    n_runs=20,
    random_state=42,
    model_kwargs=None,
    response_view=0,
    design=None,
    n_jobs_inner=1,
):
    rng = np.random.default_rng(random_state)
    current_model_kwargs = model_kwargs or {}
    n_train_samples = train_views[0].shape[0]

    def _single_member(seed):
        train_views_resampled = [resample(view, replace=True, n_samples=n_train_samples, random_state=seed) for view in train_views]

        model_epochs = param_set.get("epochs", current_model_kwargs.get("epochs", 100))
        if not (isinstance(model_epochs, (int, float)) and model_epochs > 0):
            model_epochs = 100
        model_tol = param_set.get("tol", current_model_kwargs.get("tol", 1e-3))
        if not (isinstance(model_tol, (int, float)) and model_tol > 0):
            model_tol = 1e-3

        member_model = model_class(
            **(param_set or {}),
            **(current_model_kwargs or {}),
            **(
                {
                    "early_stopping": True,
                    "epochs": model_epochs,
                    "tol": model_tol,
                }
                if model_class.__name__ != "GRCCA2"
                else {}
            ),
            **({"design": design} if design is not None else {}),
        )

        try:
            member_model.fit(train_views_resampled)
        except Exception as e:
            warnings.warn(f"Ensemble member (seed {seed}) failed to fit: {e}")
            return np.nan, np.nan

        member_eval_ave = np.nan
        member_eval_mean_correlation = np.nan

        try:
            if not any(v.shape[0] == 0 for v in eval_views):
                member_eval_ave = compute_test_AVE(member_model, eval_views, response=response_view)
                avg_pairwise_corr_result = member_model.score(eval_views)

                if isinstance(avg_pairwise_corr_result, (list, tuple)):
                    member_eval_mean_correlation = avg_pairwise_corr_result[0]
                elif isinstance(avg_pairwise_corr_result, (float, np.floating)):
                    member_eval_mean_correlation = avg_pairwise_corr_result
            else:
                warnings.warn(f"Ensemble member (seed {seed}): eval_views empty.")
        except Exception as e:
            warnings.warn(f"Ensemble member (seed {seed}) failed during evaluation: {e}")

        return member_eval_ave, member_eval_mean_correlation

    seeds = rng.integers(0, 2**32 - 1, size=n_runs)
    member_results = Parallel(n_jobs=n_jobs_inner)(
        delayed(_single_member)(seed) for seed in seeds
    )

    valid_aves = [r[0] for r in member_results if r is not None and not np.isnan(r[0])]
    valid_corrs = [ r[1] for r in member_results if r is not None and not np.isnan(r[1])]

    if not valid_aves:
        warnings.warn("No valid ensemble members for final metrics.")
        return {
            "ensemble_avg_eval_ave": np.nan,
            "ensemble_avg_eval_correlation": np.nan,
            "n_valid_ensemble_members": 0,
            "n_total_ensemble_runs": n_runs,
            "ensemble_random_state": random_state,
        }

    return {
        "ensemble_avg_eval_ave": np.nanmean(valid_aves),
        "ensemble_avg_eval_correlation": np.nanmean(valid_corrs),
        "n_valid_ensemble_members": len(valid_aves),
        "n_total_ensemble_runs": n_runs,
        "ensemble_random_state": random_state,
    }




#%%

def run_rcca_stability_selection_multidim(
    results_full,
    view_metadata,
    rCCA_class,
    train_test_split_by_group,
    combined_group,
    threshold=0.5,
    c_grid=None,
    n_jobs=1,
    response_view=0,
    fixed_c=None,
):
    """
    Stage 2 of the regularised CCA (rCCA) stability-selection pipeline.

    Following Stage 1 (bootstrap-based stability selection), this function:

      1. Tunes the regularisation parameter *c* using cross-validated
         pairwise canonical correlations on the first latent dimension,
         which typically captures the dominant mode of cross-view
         covariation.

      2. For each latent dimension *k* identified in Stage 1, fits a
         final 1-D rCCA model restricted to the stable feature set
         (i.e. features whose selection frequency exceeded *threshold*
         across bootstrap resamples). Model performance is evaluated on
         held-out data across repeated random splits.

    Parameters
    ----------
    results_full : dict
        Output of Stage 1. Must contain:
          - ``"split_seeds"``  : list of int — RNG seeds for reproducible
                                 train/test splits.
          - ``"average_selection_frequency"`` : list of ndarray, one per
            view, each of shape (n_features_v, K) — bootstrap selection
            frequencies per feature per latent dimension.
    view_metadata : list of dict
        One entry per data view. Each dict must contain:
          - ``"data"``      : ndarray, shape (n_subjects, n_features).
          - ``"confounds"`` : ndarray, shape (n_subjects, n_confounds),
                              optional. Confounds are regressed from both
                              train and test partitions using train-fit
                              OLS coefficients.
          - ``"type"``      : str, ``"continuous"`` or ``"copula"``.
          - ``"name"``      : str, human-readable label for logging.
    rCCA_class : class
        Instantiable rCCA model class (e.g. ``GCCA``). Must implement
        ``.fit(views)``, ``.transform(views)``,
        ``.pairwise_correlations(views)``, and optionally
        ``.loadings_(views)``.
    train_test_split_by_group : callable
        ``fn(groups, test_size, seed) → (train_idx, test_idx)``.
        Splits subject indices while respecting group structure (e.g.
        site or family) to prevent data leakage.
    combined_group : array-like, shape (n_subjects,)
        Group labels used by *train_test_split_by_group* (e.g. scanner
        site IDs).
    threshold : float, optional
        Stability-selection inclusion threshold. Features with a
        bootstrap selection frequency > *threshold* are retained.
        Default: 0.5 (i.e. selected in the majority of resamples).
    c_grid : array-like of float, optional
        Grid of regularisation values to search during *c*-tuning.
        Defaults to ``np.logspace(-6, 0, 10)``.
    n_jobs : int, optional
        Number of parallel workers (passed to ``joblib.Parallel``).
        Default: 1.
    response_view : int, optional
        Index of the *response* view (e.g. the clinical view). Canonical
        correlations and R² are computed between this view and all
        remaining views. Default: 0.
    fixed_c : float or None, optional
        If the model has no *c* parameter, or if all cross-validation
        folds return NaN, this value is used as a fallback. Default: None.

    Returns
    -------
    dict with keys:

        ``"optimal_c_found"`` : float or None
            Regularisation value selected by cross-validation.
        ``"average_selection_frequency"`` : list of ndarray
            Passed through from Stage 1 (shape: n_features_v × K per view).
        ``"split_seeds"`` : list of int
            RNG seeds used for train/test splits.
        ``"train_corrs"`` : ndarray, shape (n_splits, K)
            Mean train canonical correlation (response vs. all other views)
            per split and latent dimension.
        ``"test_corrs"`` : ndarray, shape (n_splits, K)
            Mean test canonical correlation per split and latent dimension.
        ``"train_r2"`` : ndarray, shape (n_splits, K)
            Train R² of response-view reconstruction per split and dimension.
        ``"test_r2"`` : ndarray, shape (n_splits, K)
            Test R² per split and latent dimension.
        ``"weights"`` : nested list, shape (n_splits, K, V)
            Canonical weights. Entry [s, k, v] is an ndarray for view v
            in split s and dimension k, or None if that view was absent.
        ``"train_scores"`` : nested list, shape (n_splits, K, V)
            Canonical variates (scores) on the training partition.
        ``"test_scores"`` : nested list, shape (n_splits, K, V)
            Canonical variates on the test partition.
        ``"train_loadings"`` : nested list, shape (n_splits, K, V)
            Pearson correlations between each feature and its own
            canonical variate (train partition).
        ``"test_loadings"`` : nested list, shape (n_splits, K, V)
            Loadings on the test partition.
        ``"train_cross_loadings"`` : nested list, shape (n_splits, K, V)
            Pearson correlations between each feature and the
            *response-view* canonical variate (train partition).
        ``"test_cross_loadings"`` : nested list, shape (n_splits, K, V)
            Cross-loadings on the test partition.
    """


    #  Unpack Stage 1 outputs                                             
    views = [meta["data"] for meta in view_metadata]
    split_seeds = results_full["split_seeds"]
    n_splits = len(split_seeds)

    # avg_sel[v] has shape (n_features_v, K):
    # entry [f, k] is the fraction of bootstrap resamples in which feature f was selected for latent dimension k
    avg_sel = results_full["average_selection_frequency"]
    V = len(avg_sel)          # number of views
    K = avg_sel[0].shape[1]  # number of latent dimensions

    print(
        f"\n{'='*60}\n"
        f"  Stage 2 — rCCA on Stability-Selected Features\n"
        f"  Views: {V}  |  Latent dimensions: {K}  |  "
        f"Stability threshold: {threshold}\n"
        f"{'='*60}"
    )


    def _build_filtered_metadata(selected_indices_per_view):
        """
        Construct a new view_metadata list whose ``"data"`` arrays are
        restricted to the stability-selected features for a given latent
        dimension. All other metadata fields (confounds, type, name) are
        inherited unchanged, ensuring that confound regression and copula
        transforms are applied correctly within each fold.

        Parameters
        ----------
        selected_indices_per_view : list of ndarray
            ``selected_indices_per_view[v]`` contains the column indices
            of features selected for view *v*.

        Returns
        -------
        list of dict
        """
        filtered_metadata = []
        for v_idx, meta in enumerate(view_metadata):
            X_full = meta["data"]
            raw_indices = selected_indices_per_view[v_idx]

            # Guard against out-of-bounds indices from Stage 1
            valid_indices = raw_indices[raw_indices < X_full.shape[1]]
            n_dropped = len(raw_indices) - len(valid_indices)
            if n_dropped > 0:
                warnings.warn(
                    f"View '{meta.get('name', v_idx)}': {n_dropped} "
                    f"selected feature indices exceed the view's column "
                    f"count and were discarded."
                )

            new_meta = {
                "data": X_full[:, valid_indices],
                "name": meta.get("name", f"view_{v_idx}"),
                "type": meta.get("type", "continuous"),
            }
            if "confounds" in meta:
                new_meta["confounds"] = meta["confounds"]

            filtered_metadata.append(new_meta)
        return filtered_metadata

    def _partition_and_drop_empty(tr_arrays, te_arrays, meta_list,
                                  split_idx, phase_label):
        """
        After a train/test split, remove views that have zero retained
        features in either partition (this can occur when the stable
        feature set for a view is empty, or when constant-variance
        features are removed during standardisation).

        Returns
        -------
        tr_kept, te_kept : lists of ndarray
        kept_original_indices : list of int
            Original view indices (into the V-length view list) for the
            retained views. Used to map model outputs back to their
            correct position in the full V-view result arrays.
        """
        tr_kept, te_kept, kept_original_indices = [], [], []
        for v_idx, (tr_v, te_v) in enumerate(zip(tr_arrays, te_arrays)):
            if tr_v.shape[1] > 0 and te_v.shape[1] > 0:
                tr_kept.append(tr_v)
                te_kept.append(te_v)
                kept_original_indices.append(v_idx)
            else:
                warnings.warn(
                    f"[{phase_label}] Split {split_idx}, "
                    f"View '{meta_list[v_idx].get('name', v_idx)}': "
                    f"0 features after train/test split — view excluded "
                    f"from model for this split."
                )
        return tr_kept, te_kept, kept_original_indices

    
    ############################################################################################################
    #  Step 1 — Regularisation parameter (c) selection via cross-validated canonical correlation on Dimension 1   
    
    # We tune *c* exclusively on the first latent dimension  to capture the strongest and most reproducible mode of covariation.
    # Using only one dime nsion at this stage avoids overfitting *c* to the idiosyncrasies of weaker dimensions.
    
    print(
        f"\n{'─'*60}\n"
        f"  Step 1 of 2 — Regularisation tuning (Dimension 1)\n"
        f"{'─'*60}"
    )

    stable_features_dim0 = [np.where(avg_sel[v][:, 0] > threshold)[0] for v in range(V)]
    filtered_metadata_dim0 = _build_filtered_metadata(stable_features_dim0)
    n_selected_dim0 = [m["data"].shape[1] for m in filtered_metadata_dim0]

    print(f"  Stable features (Dim 1): {n_selected_dim0} per view")

    if any(n == 0 for n in n_selected_dim0):
        warnings.warn(
            "At least one view has zero stable features for Dimension 1. "
            "Regularisation tuning may be unreliable."
        )

    # Check whether the model class accepts a regularisation parameter.
    can_tune_c = "c" in rCCA_class.__init__.__code__.co_varnames

    if can_tune_c:
        if c_grid is None:
            c_grid = np.logspace(-6, 0, 10)

        print(
            f"\n  Searching {len(c_grid)} regularisation values "
            f"over {n_splits} splits (n_jobs={n_jobs}) …\n"
        )

        test_correlations_per_c = []

        for c_value in c_grid:
            print(f"    c = {c_value:.2e}", end="  →  ", flush=True)

            def _evaluate_c_on_split(split_idx, _c=c_value):
                """
                Fit a 1-D rCCA model with regularisation *_c* on the
                training partition of split *split_idx* and return the
                mean pairwise canonical correlation between the response
                view and all other views on the held-out test partition.
                """
                seed = split_seeds[split_idx]
                train_idx, test_idx = train_test_split_by_group(
                    combined_group, test_size=0.3, seed=seed
                )

                # Confound regression and standardisation are performed
                # inside get_fold_data using train-set statistics
                # only, preventing test-set leakage.
                tr_all, te_all = get_fold_data(train_idx, test_idx, filtered_metadata_dim0)

                tr_kept, te_kept, kept_indices = _partition_and_drop_empty(
                    tr_all, te_all, filtered_metadata_dim0,
                    split_idx, "c-tuning"
                )

                if len(tr_kept) < 2:
                    warnings.warn(
                        f"[c-tuning] Split {split_idx}: fewer than 2 views "
                        f"retained — returning NaN."
                    )
                    return np.nan

                try:
                    response_idx_local = kept_indices.index(response_view)
                except ValueError:
                    warnings.warn(
                        f"[c-tuning] Split {split_idx}: response view "
                        f"({response_view}) has no stable features in this "
                        f"split — returning NaN."
                    )
                    return np.nan

                try:
                    model = rCCA_class(latent_dimensions=1, c=_c).fit(tr_kept)
                    test_corr_matrix = model.pairwise_correlations(te_kept)

                    # Average over all pairwise correlations that involve
                    # the response view.
                    pairwise_corrs = [ test_corr_matrix[response_idx_local, j, 0] for j in range(len(te_kept)) if j != response_idx_local]
        
                    return ( np.nanmean(pairwise_corrs) if pairwise_corrs else np.nan)

                except Exception as exc:
                    warnings.warn(
                        f"[c-tuning] Split {split_idx}: model fit failed "
                        f"({exc}) — returning NaN."
                    )
                    return np.nan

            split_corrs = Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_c_on_split)(i) for i in range(n_splits)
            )
            mean_test_corr = np.nanmean(split_corrs)
            test_correlations_per_c.append(mean_test_corr)
            print(f"mean test correlation = {mean_test_corr:.4f}")

        if all(np.isnan(test_correlations_per_c)):
            warnings.warn(
                "All regularisation-tuning folds returned NaN. "
                "Cannot select an optimal 'c'."
            )
            optimal_c = fixed_c
            if fixed_c is not None:
                print(f"\n  ⚠  Falling back to fixed_c = {fixed_c:.2e}")
            else:
                print("\n  ⚠  Proceeding with model-default regularisation.")
        else:
            best_c_idx = int(np.nanargmax(test_correlations_per_c))
            optimal_c = c_grid[best_c_idx]
            print(
                f"\n  ✓  Selected regularisation: c = {optimal_c:.2e}  "
                f"(mean test r = {test_correlations_per_c[best_c_idx]:.4f})"
            )

    else:
        # Model has no regularisation parameter (e.g. standard CCA).
        optimal_c = fixed_c
        msg = (
            f"fixed_c = {fixed_c:.2e}"
            if fixed_c is not None
            else "no regularisation parameter"
        )
        print(f"rCCA class has no 'c' parameter — {msg}.")

    ############################################################################################################
    #  Step 2 — Final 1D models for each latent dimension              
    
    # For each dimension k, features are restricted to the stable set identified by Stage 1 for that dimension independently. 
    # A 1D  model is then fitted across repeated train/test splits to quantify the variance and reliability of canonical weights, loadings, cross-loadings, and explained variance.

    print(
        f"\n{'─'*60}\n"
        f"  Step 2 of 2 — Final model fitting across {K} dimension(s)\n"
        f"{'─'*60}"
    )

    all_dim_results_list = []  # One entry per dimension k.

    for k in range(K):
        print(f"\n  Dimension {k + 1} / {K}")

        # Stable feature set for dimension k.
        stable_features_dim_k = [ np.where(avg_sel[v][:, k] > threshold)[0] for v in range(V) ]
        # Keep only views that actually have features
        kept_view_indices_dim_k = [ v for v in range(V) if len(stable_features_dim_k[v]) > 0]

        if len(kept_view_indices_dim_k) < 2:
            warnings.warn(
                f"[Dim {k + 1}] Fewer than 2 views with stable features. "
                f"Skipping this dimension."
            )

            all_dim_results_list.append({
                "train_corrs": np.full(n_splits, np.nan),
                "test_corrs":  np.full(n_splits, np.nan),
                "train_r2": np.full(n_splits, np.nan),
                "test_r2": np.full(n_splits, np.nan),
                "stable_feature_indices": [np.array([]) for _ in range(V)],
                "weights_per_split_and_view": [[None] * V for _ in range(n_splits)], 
                "train_scores_per_split_and_view": [[None] * V for _ in range(n_splits)],
                "test_scores_per_split_and_view": [[None] * V for _ in range(n_splits)],
                "train_loadings_per_split_and_view": [[None] * V for _ in range(n_splits)],
                "test_loadings_per_split_and_view": [[None] * V for _ in range(n_splits)],
                "train_cross_loadings_per_split_and_view": [[None] * V for _ in range(n_splits)],
                "test_cross_loadings_per_split_and_view": [[None] * V for _ in range(n_splits)]
            })

            continue

        # Build filtered metadata WITHOUT zero-feature views
        filtered_metadata_k = []
        for v in kept_view_indices_dim_k:
            meta = view_metadata[v]
            selected_idx = stable_features_dim_k[v]

            new_meta = {
                "data": meta["data"][:, selected_idx],
                "name": meta.get("name", f"view_{v}"),
                "type": meta.get("type", "continuous"),
            }

            if "confounds" in meta:
                new_meta["confounds"] = meta["confounds"]

            filtered_metadata_k.append(new_meta)

        print("Stable features (retained views): " +", ".join(
           f"{view_metadata[v]['name']}="
           f"{len(stable_features_dim_k[v])}"
                for v in kept_view_indices_dim_k
            )
        )
        n_selected_k = [m["data"].shape[1] for m in filtered_metadata_k]
        view_names = [m.get("name", f"view_{i}") for i, m in enumerate(filtered_metadata_k)]

        print("Stable features: " + ", ".join(f"{name}={n}" for name, n in zip(view_names, n_selected_k)))
        stable_feature_indices_dim_k = [stable_features_dim_k[v] if v in kept_view_indices_dim_k else np.array([], dtype=int) for v in range(V)]
            


        #  Inner function: fit model and compute metrics for one split   
        def _run_single_split(split_idx, _k=k, _meta_k=filtered_metadata_k):
            
            """
            For a single train/test split:

            1. Apply confound regression and standardisation (train stats).
            2. Exclude views with zero stable features.
            3. Fit a 1-D rCCA model on the training partition.
            4. Compute canonical correlations, R², weights, loadings, and
               cross-loadings on both train and test partitions.
            5. Map all outputs back to a V-length list so that absent
               views are represented as None, preserving alignment with
               the original view ordering for downstream aggregation.
            """
            
            seed = split_seeds[split_idx]
            train_idx, test_idx = train_test_split_by_group(combined_group, test_size=0.3, seed=seed)

            # Confound regression and standardisation are performed inside get_fold_data using train-set statistics only, preventing test-set leakage.
            tr_all, te_all = get_fold_data(train_idx, test_idx, _meta_k)
            tr_kept, te_kept, kept_loc_indices = _partition_and_drop_empty(tr_all, te_all, _meta_k, split_idx, f"Dim {_k + 1}")
            kept_indices = [kept_view_indices_dim_k[i] for i in kept_loc_indices]

            _null_result = {
                "ctr":  np.nan,
                "cte":  np.nan,
                "r2tr": np.array([np.nan]),
                "r2te": np.array([np.nan]),
                "w":    [None] * V,
                "str":  [None] * V,
                "ste":  [None] * V,
                "ltr":  [None] * V,
                "lte":  [None] * V,
                "xtr":  [None] * V,
                "xte":  [None] * V,
            }

            if len(tr_kept) < 2:
                warnings.warn(
                    f"[Dim {_k + 1}] Split {split_idx}: fewer than 2 views "
                    f"retained — split skipped."
                )
                return {**_null_result,
                        "error": "Insufficient views for model fit"}

            try:
                response_idx_local = kept_indices.index(response_view)
            except ValueError:
                warnings.warn(
                    f"[Dim {_k + 1}] Split {split_idx}: response view "
                    f"({response_view}) absent after feature filtering — "
                    f"split skipped."
                )
                return {**_null_result,
                        "error": "Response view absent after feature filtering"}

            model_params = {"latent_dimensions": 1}
            if optimal_c is not None:
                model_params["c"] = optimal_c

            try:
                
                model = rCCA_class(**model_params).fit(tr_kept)

                train_corr_matrix = model.pairwise_correlations(tr_kept)
                test_corr_matrix = model.pairwise_correlations(te_kept)

                train_corrs = [train_corr_matrix[response_idx_local, j, 0] for j in range(len(tr_kept)) if j != response_idx_local]
                test_corrs = [test_corr_matrix[response_idx_local, j, 0] for j in range(len(te_kept)) if j != response_idx_local]

                ctr_mean = np.nanmean(train_corrs) if train_corrs else np.nan
                cte_mean = np.nanmean(test_corrs) if test_corrs else np.nan

                r2_train = compute_r2_per_dimension(model, tr_kept, response_idx_local)[0]
                r2_test = compute_r2_per_dimension(model, te_kept, response_idx_local)[0]

                canonical_weights = (model.weights_ if hasattr(model, "weights_") else [])
                scores_train = model.transform(tr_kept)
                scores_test = model.transform(te_kept)
                
                loadings_train = (model.loadings_(tr_kept) if hasattr(model, "loadings_") else [])
                loadings_test = (model.loadings_(te_kept) if hasattr(model, "loadings_") else [])

                cross_loadings_train = calculate_cross_loadings(tr_kept, scores_train, response_idx_local)
                cross_loadings_test = calculate_cross_loadings(te_kept, scores_test, response_idx_local)

                full_w = [None] * V
                full_str = [None] * V
                full_ste = [None] * V
                full_ltr = [None] * V
                full_lte = [None] * V
                full_xtr = [None] * V
                full_xte = [None] * V

                for local_idx, original_idx in enumerate(kept_indices):
                    if local_idx < len(canonical_weights) and canonical_weights[local_idx] is not None:
                        full_w[original_idx] = canonical_weights[local_idx]
                    if local_idx < len(scores_train) and scores_train[local_idx] is not None:
                        full_str[original_idx] = scores_train[local_idx]
                    if local_idx < len(scores_test) and scores_test[local_idx] is not None:
                        full_ste[original_idx] = scores_test[local_idx]
                    if local_idx < len(loadings_train) and loadings_train[local_idx] is not None:
                        full_ltr[original_idx] = loadings_train[local_idx]
                    if local_idx < len(loadings_test) and loadings_test[local_idx] is not None:
                        full_lte[original_idx] = loadings_test[local_idx]
                    if local_idx < len(cross_loadings_train) and cross_loadings_train[local_idx] is not None:
                        full_xtr[original_idx] = cross_loadings_train[local_idx]
                    if local_idx < len(cross_loadings_test) and cross_loadings_test[local_idx] is not None:
                        full_xte[original_idx] = cross_loadings_test[local_idx]

                return {
                    "ctr":  ctr_mean,
                    "cte":  cte_mean,
                    "r2tr": r2_train,
                    "r2te": r2_test,
                    "w":    full_w,
                    "str":  full_str,
                    "ste":  full_ste,
                    "ltr":  full_ltr,
                    "lte":  full_lte,
                    "xtr":  full_xtr,
                    "xte":  full_xte,
                }

            except Exception as exc:
                warnings.warn(
                    f"[Dim {_k + 1}] Split {split_idx}: model fit or metric "
                    f"computation failed ({exc})."
                )
                import traceback
                traceback.print_exc()
                return {**_null_result, "error": str(exc)}

        # Run all splits in parallel for dimension k.
        split_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_split)(i) for i in range(n_splits)
        )

        #  Aggregate per-split outputs into per-dimension summary arrays   #
        ctrs_dim_k = np.array([r["ctr"] for r in split_results])
        ctes_dim_k = np.array([r["cte"] for r in split_results])
        r2trs_dim_k = np.array([r["r2tr"] for r in split_results])
        r2tes_dim_k = np.array([r["r2te"] for r in split_results])

        all_dim_results_list.append({
            # Scalar metrics across splits — shape (n_splits,)
            "train_corrs": ctrs_dim_k,
            "test_corrs":  ctes_dim_k,
            "train_r2": r2trs_dim_k,
            "test_r2": r2tes_dim_k,
            "stable_feature_indices": stable_feature_indices_dim_k,

            # Per-split, per-view objects — shape (n_splits, V).
            # Entry [s][v] is an ndarray or None.
            "weights_per_split_and_view": [r["w"] for r in split_results],
            "train_scores_per_split_and_view": [r["str"] for r in split_results],
            "test_scores_per_split_and_view": [r["ste"] for r in split_results],
            "train_loadings_per_split_and_view": [r["ltr"] for r in split_results],
            "test_loadings_per_split_and_view": [r["lte"] for r in split_results],
            "train_cross_loadings_per_split_and_view": [r["xtr"] for r in split_results],
            "test_cross_loadings_per_split_and_view": [r["xte"] for r in split_results],
        })

        # Print per-dimension summary.
        valid_test_corrs = ctrs_dim_k[~np.isnan(ctes_dim_k)]
        valid_test_r2 = r2tes_dim_k[~np.isnan(r2tes_dim_k).any(axis=-1)] \
            if r2tes_dim_k.ndim > 1 \
            else r2tes_dim_k[~np.isnan(r2tes_dim_k)]
        print(
            f"  Results (Dim {k + 1}): "
            f"mean test r = {np.nanmean(ctes_dim_k):.4f} "
            f"± {np.nanstd(ctes_dim_k):.4f}  |  "
            f"mean test R² = {np.nanmean(r2tes_dim_k):.4f} "
            f"± {np.nanstd(r2tes_dim_k):.4f}  "
            f"({(~np.isnan(ctes_dim_k)).sum()}/{n_splits} valid splits)"
        )


    #  Step 3 — Assemble final output dictionary                        

    if not all_dim_results_list:
        warnings.warn(
            "No dimensions were successfully processed after filtering. "
            "Returning empty results."
        )
        return {
            "optimal_c_found": optimal_c,
            "average_selection_frequency": avg_sel,
            "split_seeds": split_seeds,
            "stable_feature_indices": [[np.array([]) for _ in range(V)] for _ in range(K)],
            "train_corrs": np.full((n_splits, K), np.nan),
            "test_corrs": np.full((n_splits, K), np.nan),
            "train_r2": np.full((n_splits, K), np.nan),
            "test_r2": np.full((n_splits, K), np.nan),
            "weights": [[None] * V for _ in range(n_splits)],
            "train_scores": [[None] * V for _ in range(n_splits)],
            "test_scores":  [[None] * V for _ in range(n_splits)],
            "train_loadings": [[None] * V for _ in range(n_splits)],
            "test_loadings":  [[None] * V for _ in range(n_splits)],
            "train_cross_loadings": [[None] * V for _ in range(n_splits)],
            "test_cross_loadings":  [[None] * V for _ in range(n_splits)],
        }

    # Stack scalar metrics to (n_splits, K).
    final_train_corrs = np.stack([d["train_corrs"] for d in all_dim_results_list], axis=1)
    final_test_corrs = np.stack([d["test_corrs"] for d in all_dim_results_list], axis=1)
    final_train_r2 = np.stack([np.squeeze(d["train_r2"]) for d in all_dim_results_list], axis=1)
    final_test_r2 = np.stack([np.squeeze(d["test_r2"]) for d in all_dim_results_list], axis=1)

    final_stable_feature_indices = [ d["stable_feature_indices"] for d in all_dim_results_list]
       
    final_weights = [[None] * K for _ in range(n_splits)]
    final_train_scores = [[None] * K for _ in range(n_splits)]
    final_test_scores = [[None] * K for _ in range(n_splits)]
    final_train_loadings = [[None] * K for _ in range(n_splits)]
    final_test_loadings = [[None] * K for _ in range(n_splits)]
    final_train_cross_loadings = [[None] * K for _ in range(n_splits)]
    final_test_cross_loadings = [[None] * K for _ in range(n_splits)]

    for s_idx in range(n_splits):
        for k_idx, dim_k_data in enumerate(all_dim_results_list):
            final_weights[s_idx][k_idx] = dim_k_data["weights_per_split_and_view"][s_idx]
            final_train_scores[s_idx][k_idx] = dim_k_data["train_scores_per_split_and_view"][s_idx]
            final_test_scores[s_idx][k_idx] = dim_k_data["test_scores_per_split_and_view"][s_idx]
            final_train_loadings[s_idx][k_idx] = dim_k_data["train_loadings_per_split_and_view"][s_idx]
            final_test_loadings[s_idx][k_idx] = dim_k_data["test_loadings_per_split_and_view"][s_idx]
            final_train_cross_loadings[s_idx][k_idx] = dim_k_data["train_cross_loadings_per_split_and_view"][s_idx]
            final_test_cross_loadings[s_idx][k_idx] = dim_k_data["test_cross_loadings_per_split_and_view"][s_idx]

    print(f"\n{'='*60}\n  Stage 2 complete.\n{'='*60}\n")

    return {
        "optimal_c_found": optimal_c,
        # list of (n_features_v, K)
        "average_selection_frequency": avg_sel,
        "split_seeds": split_seeds,
        "stable_feature_indices": final_stable_feature_indices,

        "train_corrs": final_train_corrs,  # (n_splits, K)
        "test_corrs": final_test_corrs,   # (n_splits, K)
        "train_r2": final_train_r2,     # (n_splits, K)
        "test_r2": final_test_r2,      # (n_splits, K)

        # Nested lists of shape (n_splits, K), where each leaf is a
        # V-length list of ndarrays or None values.
        "weights": final_weights,
        "train_scores": final_train_scores,
        "test_scores": final_test_scores,
        "train_loadings": final_train_loadings,
        "test_loadings": final_test_loadings,
        "train_cross_loadings": final_train_cross_loadings,
        "test_cross_loadings": final_test_cross_loadings,
    }