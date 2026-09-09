# ccagg/validation.py

import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .model import parallel_stability_selection_multidim
from .robust_inference import (
    combine_pvalues_multisplit,
    pval_and_ci,
    tune_reduced_1d_model,
)
from .utils import (
    block_permutation_index,
    get_fold_data,
    make_model_init_kwargs,
    subset_view_metadata,
)

#%% Helper functions

def subset_design_matrix(design, kept_indices):
    """
    Restrict a (n_views, n_views) design matrix to a subset of views.
 
    Parameters
    ----------
    design : array-like of shape (n_views, n_views), or None
        Connectivity matrix over the full set of views. None means
        fully connected, in which case None is returned so the estimator
        falls back to its own default.
    kept_indices : sequence of int
        Global view indices that survived filtering, in the order the
        views are now passed to the estimator.
 
    Returns
    -------
    ndarray of shape (len(kept_indices), len(kept_indices)), or None
    """
    if design is None:
        return None
    d = np.asarray(design, dtype=float)
    idx = np.asarray(list(kept_indices), dtype=int)
    return d[np.ix_(idx, idx)]
 
 
def _drop_empty_views_after_split(train_views, test_views, kept_global_views,
                                  min_features=1, min_samples=2):
    """
    Remove views that lost all usable features during preprocessing.
 
    Stability selection can leave a view with no retained features, and
    confound regression can zero out a view entirely within a fold. Fitting
    with such a view raises deep inside the estimator, so drop it here and
    report which global view indices remain.
 
    Returns
    -------
    train_kept, test_kept : list of ndarray
    kept_global : list of int
        Global view indices still present, in the same order.
    """
    train_kept, test_kept, kept_global = [], [], []
 
    for tr, te, g in zip(train_views, test_views, kept_global_views):
        tr = np.asarray(tr, dtype=float)
        te = np.asarray(te, dtype=float)
 
        if tr.ndim != 2 or te.ndim != 2:
            continue
        if tr.shape[1] < min_features or te.shape[1] < min_features:
            continue
        if tr.shape[0] < min_samples or te.shape[0] < min_samples:
            continue
        # a view with no variance carries no information and breaks scaling
        if not np.any(np.nanstd(tr, axis=0) > 1e-12):
            continue
 
        train_kept.append(tr)
        test_kept.append(te)
        kept_global.append(g)
 
    return train_kept, test_kept, kept_global

 
def _first_dim_score(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(float).ravel()
    if x.ndim == 2:
        if x.shape[1] < 1:
            return np.array([], dtype=float)
        return x[:, 0].astype(float).ravel()
    raise ValueError("Score array must be 1D or 2D.")


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size != y.size or x.size < 2:
        return np.nan
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])



def mean_abs_response_corr_from_scores(scores, response_view=0):
    """
    Mean absolute correlation between the response-view latent score and each
    predictor-view latent score, using the first latent dimension only.
    """
    y = _first_dim_score(scores[response_view])

    vals = []
    for j in range(len(scores)):
        if j == response_view:
            continue
        xj = _first_dim_score(scores[j])
        r = _safe_corr(y, xj)
        if np.isfinite(r):
            vals.append(abs(r))

    return float(np.nanmean(vals)) if len(vals) > 0 else np.nan


def fit_latent_response_regression(train_scores, response_view=0):
    """
    Fit y_response ~ scores_other_views on training latent scores.
    """
    pred_views = [j for j in range(len(train_scores)) if j != response_view]
    if len(pred_views) == 0:
        return None, None

    y_train = _first_dim_score(train_scores[response_view])
    X_train = np.column_stack(
        [_first_dim_score(train_scores[j]) for j in pred_views]
    )

    if (
        X_train.ndim != 2
        or X_train.shape[0] != y_train.shape[0]
        or X_train.shape[1] == 0
    ):
        return None, None

    reg = LinearRegression().fit(X_train, y_train)
    return reg, pred_views


def score_latent_response_regression(reg, scores, pred_views, response_view=0):
    """
    Evaluate held-out R^2 for the response latent score.
    """
    if reg is None or pred_views is None or len(pred_views) == 0:
        return np.nan

    y = _first_dim_score(scores[response_view])
    X = np.column_stack([_first_dim_score(scores[j]) for j in pred_views])

    if (
        X.ndim != 2
        or X.shape[0] != y.shape[0]
        or X.shape[1] == 0
    ):
        return np.nan

    try:
        return float(reg.score(X, y))
    except Exception:
        return np.nan
    





#%%

##############################################
# Leave one out cross validation accross sites

def leave_one_site_out(
    selector_class,
    estimator_class,
    view_metadata,
    site_labels,
    train_test_split_by_group,
    sample_param_func_selector,
    sample_param_func_estimator,
    discovery_groups=None,
    response_view=0,
    latent_dimensions=3,
    discovery_n_splits=50,
    discovery_n_iter=100,
    discovery_cv=5,
    reduced_cv=5,
    selection_threshold=0.9,
    selector_model_kwargs=None,
    estimator_model_kwargs=None,
    selector_design=None,
    estimator_design=None,
    epochs_selector=200,
    epochs_estimator=200,
    random_state=0,
    selection_rule="max_mean",
    param_complexity_fn_selector=None,
    param_complexity_fn_estimator=None,
    selection_abs_weight_tol=1e-6,
    permutation_blocks=None,
    n_permutations=0,
    combine_p_method="cauchy",
    n_jobs_discovery=1,
    n_jobs_sites=1,
    min_test_n=30,
    skip_small_sites=True,
    show_progress=True,
):
    """
    LOSO validation for a two-stage pipeline:
      - selector_class: e.g. ElasticMCCA
      - estimator_class: e.g. GCCA

    This version uses score-based held-out metrics:
      1) mean absolute response-vs-predictor score correlation
      2) held-out latent-space R^2 for response score prediction

    Important: Use metadata without site dummies in confounds.
    """

    site_labels = np.asarray(site_labels)
    n_samples = view_metadata[0]["data"].shape[0]

    if site_labels.shape[0] != n_samples:
        raise ValueError("site_labels length does not match number of samples.")

    if discovery_groups is None:
        discovery_groups = site_labels
    discovery_groups = np.asarray(discovery_groups)

    if discovery_groups.shape[0] != n_samples:
        raise ValueError(
            "discovery_groups length does not match number of samples."
        )

    if permutation_blocks is not None:
        permutation_blocks = np.asarray(permutation_blocks)
        if permutation_blocks.shape[0] != n_samples:
            raise ValueError(
                "permutation_blocks length does not match number of samples."
            )

    held_out_sites = np.unique(site_labels)
    V = len(view_metadata)

    def run_one_site(site_idx, held_out_site):
        test_idx = np.flatnonzero(site_labels == held_out_site)
        train_idx = np.flatnonzero(site_labels != held_out_site)

        if skip_small_sites and test_idx.size < min_test_n:
            return {
                "site": held_out_site,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "skipped": True,
                "skip_reason": f"n_test < {min_test_n}",
                "obs_test_corr": np.full(latent_dimensions, np.nan),
                "obs_test_r2": np.full(latent_dimensions, np.nan),
                "p_test_corr": np.full(latent_dimensions, np.nan),
                "p_test_r2": np.full(latent_dimensions, np.nan),
                "ci_test_corr": np.full((latent_dimensions, 2), np.nan),
                "ci_test_r2": np.full((latent_dimensions, 2), np.nan),
                "null_test_corr": np.full((latent_dimensions, n_permutations), np.nan),
                "null_test_r2": np.full((latent_dimensions, n_permutations), np.nan),
                "stable_feature_indices": [[np.array([], dtype=int) for _ in range(V)] for _ in range(latent_dimensions)],
                "discovery_results": {},
            }

        train_metadata = subset_view_metadata(view_metadata, train_idx)
        train_groups = discovery_groups[train_idx]

        discovery_results = parallel_stability_selection_multidim(
            model_class=selector_class,
            combined_group=train_groups,
            view_metadata=train_metadata,
            train_test_split_by_group=train_test_split_by_group,
            sample_param_func=sample_param_func_selector,
            n_splits=discovery_n_splits,
            n_iter=discovery_n_iter,
            cv=discovery_cv,
            epochs=epochs_selector,
            n_jobs=n_jobs_discovery,
            model_kwargs=selector_model_kwargs,
            design=selector_design,
            response_view=response_view,
            latent_dimensions=latent_dimensions,
            selection_abs_weight_tol=selection_abs_weight_tol,
            selection_rule=selection_rule,
            param_complexity_fn=param_complexity_fn_selector,
            test_size=0.3,
        )

        if not discovery_results:
            return {
                "site": held_out_site,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "skipped": False,
                "skip_reason": "discovery_results empty",
                "obs_test_corr": np.full(latent_dimensions, np.nan),
                "obs_test_r2": np.full(latent_dimensions, np.nan),
                "p_test_corr": np.full(latent_dimensions, np.nan),
                "p_test_r2": np.full(latent_dimensions, np.nan),
                "ci_test_corr": np.full((latent_dimensions, 2), np.nan),
                "ci_test_r2": np.full((latent_dimensions, 2), np.nan),
                "null_test_corr": np.full((latent_dimensions, n_permutations), np.nan),
                "null_test_r2": np.full((latent_dimensions, n_permutations), np.nan),
                "stable_feature_indices": [[np.array([], dtype=int) for _ in range(V)] for _ in range(latent_dimensions)],
                "discovery_results": discovery_results,
            }

        avg_sel = discovery_results["average_selection_frequency"]
        K = avg_sel[0].shape[1]

        obs_test_corr = np.full(latent_dimensions, np.nan)
        obs_test_r2 = np.full(latent_dimensions, np.nan)
        p_test_corr = np.full(latent_dimensions, np.nan)
        p_test_r2 = np.full(latent_dimensions, np.nan)
        ci_test_corr = np.full((latent_dimensions, 2), np.nan)
        ci_test_r2 = np.full((latent_dimensions, 2), np.nan)
        null_test_corr = np.full((latent_dimensions, n_permutations), np.nan)
        null_test_r2 = np.full((latent_dimensions, n_permutations), np.nan)

        stable_feature_indices = [[np.array([], dtype=int) for _ in range(V)] for _ in range(latent_dimensions)]
        
        for k in range(min(latent_dimensions, K)):
            feature_masks_k = []
            for v in range(V):
                mask = np.asarray(avg_sel[v][:, k] >= selection_threshold, dtype=bool)
                feature_masks_k.append(mask)
                stable_feature_indices[k][v] = np.flatnonzero(mask)

            kept_global_views = [v for v in range(V) if np.any(feature_masks_k[v])]

            if len(kept_global_views) < 2:
                continue

            if response_view not in kept_global_views:
                continue

            filtered_full_metadata = []
            for v in kept_global_views:
                meta = view_metadata[v]
                filtered_full_metadata.append(
                    {
                        **meta,
                        "data": meta["data"][:, feature_masks_k[v]],
                    }
                )

            local_response_view = kept_global_views.index(response_view)

            estimator_design_k = subset_design_matrix(estimator_design, kept_global_views)
            filtered_train_metadata = subset_view_metadata(filtered_full_metadata, train_idx)

            try:
                param_grid_est = sample_param_func_estimator(
                    n_iter=None,
                    seed=None,
                    n_views=len(filtered_full_metadata),
                    latent_dimensions=1,
                )
                n_iter_est = len(param_grid_est)
            except Exception:
                n_iter_est = 50

            best_params_k, _ = tune_reduced_1d_model(
                model_class=estimator_class,
                reduced_discovery_metadata=filtered_train_metadata,
                discovery_group=train_groups,
                sample_param_func=sample_param_func_estimator,
                n_iter=n_iter_est,
                cv=reduced_cv,
                random_state=random_state + 10000 * site_idx + k,
                epochs=epochs_estimator,
                model_kwargs=estimator_model_kwargs,
                design=estimator_design_k,
                response_view=local_response_view,
                selection_rule=selection_rule,
                param_complexity_fn=param_complexity_fn_estimator,
            )

            try:
                train_views_k, test_views_k = get_fold_data(
                    train_idx,
                    test_idx,
                    filtered_full_metadata,
                )
            except Exception as e:
                warnings.warn(
                    f"[Site {held_out_site}] Dim {k + 1}: "
                    f"preprocessing failed ({e})."
                )
                continue

            train_views_k, test_views_k, kept_after_split = (
                _drop_empty_views_after_split(
                    train_views_k,
                    test_views_k,
                    kept_global_views,
                )
            )

            if len(train_views_k) < 2:
                continue

            if response_view not in kept_after_split:
                continue

            local_response_after_split = kept_after_split.index(response_view)
            estimator_design_k_split = subset_design_matrix(estimator_design, kept_after_split)
            
            best_tol = (
                best_params_k.get("tol", 1e-5)
                if isinstance(best_params_k, dict)
                else 1e-5
            )

            try:
                est_init_kwargs = make_model_init_kwargs(
                    model_class=estimator_class,
                    params=best_params_k,
                    model_kwargs=estimator_model_kwargs,
                    design=estimator_design_k_split,
                    random_state=random_state + 20000 * site_idx + k,
                    latent_dimensions=1,
                    epochs=epochs_estimator,
                    tol=best_tol,
                )

                estimator = estimator_class(**est_init_kwargs)
                estimator.fit(train_views_k)

                train_scores = estimator.transform(train_views_k)
                test_scores = estimator.transform(test_views_k)

                obs_test_corr[k] = mean_abs_response_corr_from_scores(test_scores, response_view=local_response_after_split)

                reg, pred_views = fit_latent_response_regression(train_scores, response_view=local_response_after_split)

                obs_test_r2[k] = score_latent_response_regression(
                    reg,
                    test_scores,
                    pred_views,
                    response_view=local_response_after_split,
                )

            except Exception as e:
                warnings.warn(
                    f"[Site {held_out_site}] Dim {k + 1}: "
                    f"fit/eval failed ({e})."
                )
                continue

            if n_permutations > 0:
                rng_local = np.random.default_rng(random_state + 30000 * site_idx + k)

                blocks_test = (permutation_blocks[test_idx] if permutation_blocks is not None else None)

                for b in range(n_permutations):
                    if blocks_test is None:
                        perm_idx = rng_local.permutation(test_idx.size)
                    else:
                        perm_idx = block_permutation_index(blocks_test, rng_local)

                    perm_test_views = [x if i != local_response_after_split else x[perm_idx, :] for i, x in enumerate(test_views_k)]
                    try:
                        perm_test_scores = estimator.transform(perm_test_views)

                        null_test_corr[k, b] = mean_abs_response_corr_from_scores(
                            perm_test_scores,
                            response_view=local_response_after_split,
                        )

                        null_test_r2[k, b] = score_latent_response_regression(
                            reg,
                            perm_test_scores,
                            pred_views,
                            response_view=local_response_after_split,
                        )
                    except Exception:
                        null_test_corr[k, b] = np.nan
                        null_test_r2[k, b] = np.nan

                p_test_corr[k], ci_test_corr[k] = pval_and_ci(obs_test_corr[k], null_test_corr[k])
                p_test_r2[k], ci_test_r2[k] = pval_and_ci(obs_test_r2[k], null_test_r2[k])
                    

        return {
            "site": held_out_site,
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "skipped": False,
            "skip_reason": None,
            "obs_test_corr": obs_test_corr,
            "obs_test_r2": obs_test_r2,
            "p_test_corr": p_test_corr,
            "p_test_r2": p_test_r2,
            "ci_test_corr": ci_test_corr,
            "ci_test_r2": ci_test_r2,
            "null_test_corr": null_test_corr,
            "null_test_r2": null_test_r2,
            "stable_feature_indices": stable_feature_indices,
            "discovery_results": discovery_results,
        }

    site_tasks = list(enumerate(held_out_sites))

    if n_jobs_sites == 1:
        site_results = []
        iterator = tqdm(
            site_tasks,
            total=len(site_tasks),
            disable=not show_progress,
            desc="Leave-one-site-out",
        )
        for site_idx, held_out_site in iterator:
            site_results.append(run_one_site(site_idx, held_out_site))
    else:
        site_results = Parallel(n_jobs=n_jobs_sites, backend="loky")(
            delayed(run_one_site)(site_idx, held_out_site)
            for site_idx, held_out_site in site_tasks
        )

    n_sites = len(site_results)

    obs_test_corr_by_site = np.full((n_sites, latent_dimensions), np.nan)
    obs_test_r2_by_site = np.full((n_sites, latent_dimensions), np.nan)
    p_test_corr_by_site = np.full((n_sites, latent_dimensions), np.nan)
    p_test_r2_by_site = np.full((n_sites, latent_dimensions), np.nan)
    ci_test_corr_by_site = np.full((n_sites, latent_dimensions, 2), np.nan)
    ci_test_r2_by_site = np.full((n_sites, latent_dimensions, 2), np.nan)

    site_order = []
    skipped_sites = []
    site_n_test = []

    for i, res in enumerate(site_results):
        site_order.append(res["site"])
        site_n_test.append(res["n_test"])

        if res.get("skipped", False):
            skipped_sites.append((res["site"], res.get("skip_reason")))

        corr = np.asarray(res["obs_test_corr"], dtype=float).ravel()
        r2 = np.asarray(res["obs_test_r2"], dtype=float).ravel()
        p_corr = np.asarray(res["p_test_corr"], dtype=float).ravel()
        p_r2 = np.asarray(res["p_test_r2"], dtype=float).ravel()
        ci_corr = np.asarray(res["ci_test_corr"], dtype=float)
        ci_r2 = np.asarray(res["ci_test_r2"], dtype=float)

        kk = min(latent_dimensions, corr.size)
        obs_test_corr_by_site[i, :kk] = corr[:kk]

        kk = min(latent_dimensions, r2.size)
        obs_test_r2_by_site[i, :kk] = r2[:kk]

        kk = min(latent_dimensions, p_corr.size)
        p_test_corr_by_site[i, :kk] = p_corr[:kk]

        kk = min(latent_dimensions, p_r2.size)
        p_test_r2_by_site[i, :kk] = p_r2[:kk]

        if ci_corr.ndim == 2:
            kk = min(latent_dimensions, ci_corr.shape[0])
            ci_test_corr_by_site[i, :kk, :] = ci_corr[:kk, :]

        if ci_r2.ndim == 2:
            kk = min(latent_dimensions, ci_r2.shape[0])
            ci_test_r2_by_site[i, :kk, :] = ci_r2[:kk, :]

    combined_p_test_corr = np.full(latent_dimensions, np.nan)
    combined_p_test_corr_min_bonf = np.full(latent_dimensions, np.nan)
    combined_p_test_r2 = np.full(latent_dimensions, np.nan)
    combined_p_test_r2_min_bonf = np.full(latent_dimensions, np.nan)

    if n_permutations > 0:
        for k in range(latent_dimensions):
            combined_p_test_corr[k] = combine_pvalues_multisplit(p_test_corr_by_site[:, k], method=combine_p_method)
            combined_p_test_corr_min_bonf[k] = combine_pvalues_multisplit(p_test_corr_by_site[:, k], method="min_bonferroni")
            combined_p_test_r2[k] = combine_pvalues_multisplit(p_test_r2_by_site[:, k], method=combine_p_method)
            combined_p_test_r2_min_bonf[k] = combine_pvalues_multisplit(p_test_r2_by_site[:, k], method="min_bonferroni")

    return {
        "site_order": np.asarray(site_order, dtype=object),
        "site_n_test": np.asarray(site_n_test, dtype=int),
        "skipped_sites": skipped_sites,
        "site_results": site_results,
        "obs_test_corr_by_site": obs_test_corr_by_site,
        "obs_test_r2_by_site": obs_test_r2_by_site,
        "p_test_corr_by_site": p_test_corr_by_site,
        "p_test_r2_by_site": p_test_r2_by_site,
        "ci_test_corr_by_site": ci_test_corr_by_site,
        "ci_test_r2_by_site": ci_test_r2_by_site,
        "combined_p_test_corr": combined_p_test_corr,
        "combined_p_test_corr_method": combine_p_method,
        "combined_p_test_corr_min_bonferroni": combined_p_test_corr_min_bonf,
        "combined_p_test_r2": combined_p_test_r2,
        "combined_p_test_r2_method": combine_p_method,
        "combined_p_test_r2_min_bonferroni": combined_p_test_r2_min_bonf,
        "selection_threshold": float(selection_threshold),
        "latent_dimensions": int(latent_dimensions),
        "n_sites": int(n_sites),
        "n_permutations": int(n_permutations),
        "min_test_n": int(min_test_n),
    }
