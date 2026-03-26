# ccagg/permutation.py


"""
Permutation-based significance testing for multi-view CCA latent dimensions
conditional on the discovery reduced model.
 

Two complementary strategies are implemented:

* ``permute_score_significance_multidim``  — score-based (no model refitting).
* ``permute_rgcca_significance_multidim``  — model-refitting (rGCCA / rCCA).

These procedures provide the primary inferential results by testing held-out
association strength conditional on the reduced model defined in discovery.
For robustness-focused or full-pipeline permutation sensitivity analyses that
rerun larger parts of the workflow under permutation, see ``robust.py``.

"""

import warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from .utils import train_test_split_by_group, get_fold_data
from .stats import compute_r2

#%%

# =============================================================================
# PERMUTATION TESTING — SCORE-BASED (no model refitting)
# =============================================================================

def permute_score_significance_multidim(
    original_results: dict,
    response_view: int = 0,
    n_permutations: int = 500,
    random_state: int | None = None,
    n_jobs: int = 1,
) -> dict:
    """
    Assess the statistical significance of multi-view CCA latent dimensions
    by permuting response-view scores from an already-fitted model.

    Rather than re-fitting the model, this function uses the latent scores
    stored in ``original_results`` and breaks the association between the
    response view and all predictor views by randomly shuffling the response-
    view scores. This is the computationally cheap permutation approach and
    is appropriate when scores have already been obtained from a stable model.

    Strategy
    --------
    For each cross-validation split *s* and latent dimension *k*:

    1. Retrieve the fitted latent scores for the response view
       (shape: ``[N_samples, 1]``) and all valid predictor views.
    2. For each of *B* permutations, shuffle the response-view scores and
       recompute:

       * **Canonical correlation** – mean Pearson *r* between the permuted
         response-view scores and each predictor-view's scores.
       * **R²** – variance in the permuted response-view scores explained by
         a multiple linear regression on all predictor-view scores.

    3. Derive an empirical *p*-value and a 95 % null-distribution CI for each
       observed metric.

    Parameters
    ----------
    original_results : dict
        Output of the stability / cross-validation routine.  Must contain:

        * ``"train_scores"`` / ``"test_scores"`` –
          nested structure ``[S][K][V]`` of score arrays (``[N, 1]``) or
          ``None`` for views excluded by feature selection.
        * ``"train_corrs"`` / ``"test_corrs"`` –
          ``(S, K)`` arrays of observed mean canonical correlations.
        * ``"train_r2"`` / ``"test_r2"`` –
          ``(S, K)`` arrays of observed R² values.

    response_view : int, default 0
        Index of the view whose scores are permuted (typically the clinical /
        phenotypic view).
    n_permutations : int, default 500
        Number of permutations *B*.
    random_state : int or None, default None
        Seed for the master random-number generator.  Per-split and per-
        permutation seeds are derived deterministically from this seed,
        ensuring exact reproducibility even under parallel execution.
    n_jobs : int, default 1
        Number of parallel workers (passed to ``joblib.Parallel``).

    Returns
    -------
    dict
        Keys follow the pattern ``{obs|p|ci}_{corr|r2}_{train|test}``:

        * ``obs_*``  – ``(S, K)`` observed metric arrays.
        * ``p_*``    – ``(S, K)`` empirical *p*-values.
        * ``ci_*``   – ``(S, K, 2)`` 2.5th / 97.5th percentile CIs of the
          null distribution.

    Notes
    -----
    * Views with ``None`` scores (excluded by prior feature selection) are
      silently skipped; the corresponding cells are filled with ``np.nan``.
    * Each permutation uses an independent seed derived from the split seed,
      so results are fully reproducible regardless of ``n_jobs``.
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # 1.  Unpack observed scores and metrics
    # ------------------------------------------------------------------
    train_scores = original_results["train_scores"]   # [S][K][V]
    test_scores  = original_results["test_scores"]    # [S][K][V]

    obs_corr_train = np.array(original_results["train_corrs"])  # (S, K)
    obs_corr_test  = np.array(original_results["test_corrs"])   # (S, K)
    obs_r2_train   = np.array(original_results["train_r2"])     # (S, K)
    obs_r2_test    = np.array(original_results["test_r2"])      # (S, K)

    S, K = obs_corr_test.shape          # splits × latent dimensions
    V    = len(train_scores[0][0])      # total number of views (incl. removed)

    # Pre-draw one seed per split so parallel workers are fully independent
    split_seeds = rng.integers(low=0, high=2**32 - 1, size=S)

    # ------------------------------------------------------------------
    # 2.  Per-split worker
    # ------------------------------------------------------------------
    def _process_split(s: int, split_seed: int):
        """
        Build the null distribution for all K dimensions in split *s*.

        Returns a 12-tuple of per-split arrays consumed by the caller.
        """
        local_rng    = np.random.default_rng(split_seed)
        perm_seeds   = local_rng.integers(0, 2**32 - 1, size=n_permutations)

        # Null containers: (B, K), initialised to NaN
        null_corr_train = np.full((n_permutations, K), np.nan)
        null_corr_test  = np.full((n_permutations, K), np.nan)
        null_r2_train   = np.full((n_permutations, K), np.nan)
        null_r2_test    = np.full((n_permutations, K), np.nan)

        for k in range(K):
            # ------------------------------------------------------
            # 2a.  Validate response-view scores for (s, k)
            # ------------------------------------------------------
            resp_scores_train = train_scores[s][k][response_view]  # (N_tr, 1)
            resp_scores_test  = test_scores[s][k][response_view]   # (N_te, 1)

            if resp_scores_train is None or resp_scores_test is None:
                warnings.warn(
                    f"Split {s}, Dim {k + 1}: response view {response_view} "
                    f"has no scores (excluded by feature selection). "
                    f"Null distribution for this cell will be NaN."
                )
                continue   # null arrays already NaN

            # ------------------------------------------------------
            # 2b.  Collect valid predictor-view scores for (s, k)
            # ------------------------------------------------------
            pred_scores_train = [
                train_scores[s][k][j]
                for j in range(V)
                if j != response_view and train_scores[s][k][j] is not None
            ]
            pred_scores_test = [
                test_scores[s][k][j]
                for j in range(V)
                if j != response_view and test_scores[s][k][j] is not None
            ]

            if not pred_scores_train or not pred_scores_test:
                warnings.warn(
                    f"Split {s}, Dim {k + 1}: no valid predictor views after "
                    f"filtering. Null distribution for this cell will be NaN."
                )
                continue

            N_train = resp_scores_train.shape[0]
            N_test  = resp_scores_test.shape[0]

            # Concatenate predictor scores for R² regression: (N, n_pred_views)
            X_pred_train = np.hstack(
                [arr[:, 0].reshape(-1, 1) for arr in pred_scores_train]
            )
            X_pred_test = np.hstack(
                [arr[:, 0].reshape(-1, 1) for arr in pred_scores_test]
            )

            # ------------------------------------------------------
            # 2c.  Permutation loop for dimension k
            # ------------------------------------------------------
            for b, pseed in enumerate(perm_seeds):
                perm_rng = np.random.default_rng(pseed)

                # Shuffle response-view scores independently for train/test
                perm_resp_train = resp_scores_train[perm_rng.permutation(N_train), :]
                perm_resp_test  = resp_scores_test[perm_rng.permutation(N_test),  :]

                # ---- Canonical correlation (mean over predictor views) ----
                null_corr_train[b, k] = np.nanmean([
                    np.corrcoef(perm_resp_train[:, 0], arr[:, 0])[0, 1]
                    for arr in pred_scores_train
                ])
                null_corr_test[b, k] = np.nanmean([
                    np.corrcoef(perm_resp_test[:, 0], arr[:, 0])[0, 1]
                    for arr in pred_scores_test
                ])

                # ---- R²: response ~ all predictor views (OLS) ------------
                y_train = perm_resp_train[:, 0]
                if (
                    np.all(np.std(X_pred_train, axis=0) > 1e-9)
                    and np.std(y_train) > 1e-9
                ):
                    null_r2_train[b, k] = (
                        LinearRegression().fit(X_pred_train, y_train)
                        .score(X_pred_train, y_train)
                    )

                y_test = perm_resp_test[:, 0]
                if (
                    np.all(np.std(X_pred_test, axis=0) > 1e-9)
                    and np.std(y_test) > 1e-9
                ):
                    null_r2_test[b, k] = (
                        LinearRegression().fit(X_pred_test, y_test)
                        .score(X_pred_test, y_test)
                    )

        # ------------------------------------------------------
        # 2d.  Empirical p-values and 95 % CIs for split s
        # ------------------------------------------------------
        def _pval_and_ci(obs_val: float, null_dist: np.ndarray):
            """
            Compute an empirical *p*-value and 95 % CI from a null distribution.

            The p-value uses the +1 correction in both numerator and denominator
            (Phipson & Smyth 2010) to avoid *p* = 0 and to remain valid when
            the observed statistic is included in the null.
            """
            if np.isnan(obs_val):
                return np.nan, np.array([np.nan, np.nan])

            valid_null = null_dist[~np.isnan(null_dist)]
            if valid_null.size == 0:
                return np.nan, np.array([np.nan, np.nan])

            p_value = (1 + np.sum(valid_null >= obs_val)) / (valid_null.size + 1)
            ci_95   = np.percentile(valid_null, [2.5, 97.5])
            return p_value, ci_95

        # Observed scalars for this split (shape K)
        obs_corr_tr_s = obs_corr_train[s]
        obs_corr_te_s = obs_corr_test[s]
        obs_r2_tr_s   = obs_r2_train[s]
        obs_r2_te_s   = obs_r2_test[s]

        p_corr_tr  = np.array([_pval_and_ci(obs_corr_tr_s[k], null_corr_train[:, k])[0] for k in range(K)])
        p_corr_te  = np.array([_pval_and_ci(obs_corr_te_s[k], null_corr_test[:, k])[0]  for k in range(K)])
        ci_corr_tr = np.stack([_pval_and_ci(obs_corr_tr_s[k], null_corr_train[:, k])[1] for k in range(K)])
        ci_corr_te = np.stack([_pval_and_ci(obs_corr_te_s[k], null_corr_test[:, k])[1]  for k in range(K)])

        p_r2_tr  = np.array([_pval_and_ci(obs_r2_tr_s[k], null_r2_train[:, k])[0] for k in range(K)])
        p_r2_te  = np.array([_pval_and_ci(obs_r2_te_s[k], null_r2_test[:, k])[0]  for k in range(K)])
        ci_r2_tr = np.stack([_pval_and_ci(obs_r2_tr_s[k], null_r2_train[:, k])[1] for k in range(K)])
        ci_r2_te = np.stack([_pval_and_ci(obs_r2_te_s[k], null_r2_test[:, k])[1]  for k in range(K)])

        return (
            obs_corr_tr_s, p_corr_tr, ci_corr_tr,
            obs_corr_te_s, p_corr_te, ci_corr_te,
            obs_r2_tr_s,   p_r2_tr,   ci_r2_tr,
            obs_r2_te_s,   p_r2_te,   ci_r2_te,
        )

    # ------------------------------------------------------------------
    # 3.  Parallel execution over splits
    # ------------------------------------------------------------------
    split_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_split)(s, split_seeds[s]) for s in range(S)
    )

    # ------------------------------------------------------------------
    # 4.  Graceful handling of failed splits (replace with NaN arrays)
    # ------------------------------------------------------------------
    _nan_K   = lambda: np.full(K, np.nan)         # noqa: E731
    _nan_K2  = lambda: np.full((K, 2), np.nan)    # noqa: E731

    padded = []
    for result in split_results:
        if result is None:
            padded.append((
                _nan_K(), _nan_K(), _nan_K2(),
                _nan_K(), _nan_K(), _nan_K2(),
                _nan_K(), _nan_K(), _nan_K2(),
                _nan_K(), _nan_K(), _nan_K2(),
            ))
        else:
            padded.append(result)

    # Transpose list-of-tuples → tuple-of-lists, then stack over splits
    cols = list(zip(*padded))

    return {
        "obs_corr_train": np.stack(cols[0],  axis=0),          # (S, K)
        "p_corr_train":   np.stack(cols[1],  axis=0),          # (S, K)
        "ci_corr_train":  np.stack(cols[2],  axis=0),          # (S, K, 2)
        "obs_corr_test":  np.stack(cols[3],  axis=0),
        "p_corr_test":    np.stack(cols[4],  axis=0),
        "ci_corr_test":   np.stack(cols[5],  axis=0),
        "obs_r2_train":   np.stack(cols[6],  axis=0),
        "p_r2_train":     np.stack(cols[7],  axis=0),
        "ci_r2_train":    np.stack(cols[8],  axis=0),
        "obs_r2_test":    np.stack(cols[9],  axis=0),
        "p_r2_test":      np.stack(cols[10], axis=0),
        "ci_r2_test":     np.stack(cols[11], axis=0),
    }

#%%

# =============================================================================
# PERMUTATION TESTING — MODEL-REFITTING (rGCCA / rCCA)
# =============================================================================


def permute_rgcca_significance_multidim(
    view_metadata: list[dict],
    original_stability_results: dict,
    rCCA_class,
    combined_group: np.ndarray,
    response_view: int = 0,
    n_permutations: int = 500,
    random_state: int | None = None,
    n_jobs: int = 1,
    test_size: float = 0.3,
    selection_threshold: float = 0.5,
) -> dict:
    """
    Assess the statistical significance of rGCCA / rCCA latent dimensions by
    re-fitting the model from scratch on permuted data.

    Unlike the score-based approach, this function re-fits the CCA model on
    each permuted dataset, which is the correct procedure when the latent
    scores were derived from a model that performed feature selection (via
    stability selection).  The computational cost is therefore substantially
    higher than ``permute_score_significance_multidim``.

    Strategy
    --------
    For each latent dimension *k*:

    1. Retain only the features selected with frequency > ``selection_threshold``
       across the stability-selection runs (dimension-specific feature sets).
    2. For each of *S* cross-validation splits and *B* permutations:

       a. Randomly permute the rows of the response view (``response_view``)
          in the full dataset, preserving all other views intact.
       b. Re-split the permuted data using the **same** split seed as the
          original analysis to ensure train / test partitions are identical.
       c. Pre-process each partition independently (confound regression,
          z-scoring) to prevent data leakage.
       d. Fit a one-dimensional rGCCA / rCCA model on the permuted training
          data and evaluate it on the permuted test data.
       e. Record the mean canonical correlation (response view vs. all
          predictor views) and the R² of regressing the response-view scores
          on all predictor-view scores.

    3. Aggregate the *B* null values per split and compute an empirical
       *p*-value (Phipson & Smyth 2010) and a 95 % CI.

    Parameters
    ----------
    view_metadata : list of dict
        One dict per view with keys:

        * ``"data"``      – raw feature matrix ``(N, F_v)``.
        * ``"confounds"`` – confound matrix ``(N, C)`` or ``None``.
        * ``"type"``      – ``"continuous"`` (used by the preprocessor).
        * ``"name"``      – human-readable view label.

    original_stability_results : dict
        Output of the stability-selection / cross-validation routine.  Must
        contain:

        * ``"split_seeds"``                – 1-D array of length *S*.
        * ``"train_corrs"`` / ``"test_corrs"``   – ``(S, K)`` arrays.
        * ``"train_r2"`` / ``"test_r2"``         – ``(S, K)`` arrays.
        * ``"average_selection_frequency"``       – list of *V* arrays,
          each ``(F_v, K)``, giving the selection frequency of every feature
          in every latent dimension.
        * ``"optimal_c_found"``           – regularisation parameter (or
          ``None``).

    rCCA_class : class
        Un-instantiated rGCCA / rCCA model class.  Must support
        ``.fit(views)``, ``.pairwise_correlations(views)``, and
        expose a ``weights_`` attribute.
    combined_group : np.ndarray, shape (N,)
        Group / site labels used to stratify the train / test split so that
        all samples from the same group remain in the same partition.
    response_view : int, default 0
        Index of the view to permute.
    n_permutations : int, default 500
        Number of permutations *B* per split.
    random_state : int or None, default None
        Master seed for reproducibility.
    n_jobs : int, default 1
        Number of parallel workers.
    test_size : float, default 0.3
        Fraction of groups held out for testing in each split.
    selection_threshold : float, default 0.5
        Minimum stability-selection frequency for a feature to be retained.

    Returns
    -------
    dict
        Same structure as ``permute_score_significance_multidim``:
        ``{obs|p|ci}_{corr|r2}_{train|test}``, all ``(S, K[, 2])``.

    Notes
    -----
    * The response view is permuted **before** the train / test split.
      Because the same split seed is used as in the original analysis, the
      permuted train and test partitions contain the same *subjects* as their
      original counterparts, ensuring comparability.
    * Fitting is performed with ``latent_dimensions=1`` regardless of *K*,
      because each dimension's feature set was selected independently during
      stability selection.
    * Cells for dimensions with no selected features are filled with ``np.nan``.
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # 1.  Unpack metadata from the stability-selection results
    # ------------------------------------------------------------------
    S = len(original_stability_results["split_seeds"])
    optimal_c = original_stability_results["optimal_c_found"]
    # list[V] of (F_v, K)
    avg_sel_freq = original_stability_results["average_selection_frequency"]

    # (S, K) or (S,)
    obs_corr_train = original_stability_results["train_corrs"]
    obs_corr_test = original_stability_results["test_corrs"]
    obs_r2_train = original_stability_results["train_r2"]
    obs_r2_test = original_stability_results["test_r2"]

    # Guarantee (S, K) shape even when K = 1 causes a squeeze to (S,)
    def _ensure_2d(arr):
        return arr[:, np.newaxis] if arr.ndim == 1 else arr

    obs_corr_train = _ensure_2d(obs_corr_train)
    obs_corr_test = _ensure_2d(obs_corr_test)
    obs_r2_train = _ensure_2d(obs_r2_train)
    obs_r2_test = _ensure_2d(obs_r2_test)

    K = obs_r2_train.shape[1]
    V = len(view_metadata)

    # Pre-allocate result arrays; NaN is the default for skipped cells
    p_corr_train = np.full((S, K), np.nan)
    p_corr_test = np.full((S, K), np.nan)
    p_r2_train = np.full((S, K), np.nan)
    p_r2_test = np.full((S, K), np.nan)
    ci_corr_train = np.full((S, K, 2), np.nan)
    ci_corr_test = np.full((S, K, 2), np.nan)
    ci_r2_train = np.full((S, K, 2), np.nan)
    ci_r2_test = np.full((S, K, 2), np.nan)

    # ------------------------------------------------------------------
    # 2.  Helper: empirical p-value and 95 % CI
    # ------------------------------------------------------------------
    def _pval_and_ci(obs_val: float, null_dist: np.ndarray, label: str = ""):
        """
        Return ``(p_value, [ci_lower, ci_upper])`` from a 1-D null distribution.

        Uses the +1 correction (Phipson & Smyth 2010) so that *p* is never
        exactly zero and remains a valid upper bound.
        """
        valid_null = null_dist[~np.isnan(null_dist)]
        if valid_null.size == 0:
            if label:
                warnings.warn(f"Empty null distribution for {label}; p and CI set to NaN.")
            return np.nan, np.array([np.nan, np.nan])

        # p_value = (1 + np.sum(null_dist >= obs_val)) / (null_dist.size + 1)
        p_value = (1 + np.sum(valid_null >= obs_val)) / (valid_null.size + 1)
        ci_95 = np.percentile(valid_null, [2.5, 97.5])
        return p_value, ci_95

    # ------------------------------------------------------------------
    # 3.  Outer loop: one pass per latent dimension
    # ------------------------------------------------------------------
    for k in range(K):
        print(f"\n--- Permutation test: Latent Dimension {k + 1} / {K} ---")

        # --------------------------------------------------------------
        # 3a.  Feature selection for dimension k
        # --------------------------------------------------------------
        # Each element of avg_sel_freq is an (F_v, K) matrix; column k
        # gives the selection frequency of every feature for dimension k.
        feature_masks_k = [
            freq_matrix[:, k] > selection_threshold
            for freq_matrix in avg_sel_freq
        ]
        views_k = [
            meta["data"][:, mask]
            for meta, mask in zip(view_metadata, feature_masks_k)
        ]

        if any(v.shape[1] == 0 for v in views_k):
            warnings.warn(
                f"Dimension {k + 1}: no features survive the selection threshold "
                f"({selection_threshold}) in at least one view. "
                f"All cells for this dimension will be NaN."
            )
            continue   # NaN placeholders already in place

        # Build filtered view_metadata (data replaced with selected features)
        view_metadata_k = [
            {**meta, "data": views_k[v_idx]}
            for v_idx, meta in enumerate(view_metadata)
        ]

        # --------------------------------------------------------------
        # 3b.  Build task list: (perm_seed, split_idx) for all S × B jobs
        # --------------------------------------------------------------
        task_seeds = rng.integers(0, 2**32 - 1, size=S * n_permutations)
        task_splits = np.repeat(np.arange(S), n_permutations)
        tasks = list(zip(task_seeds.tolist(), task_splits.tolist()))

        # --------------------------------------------------------------
        # 3c.  Single-permutation worker (closure over k and metadata)
        # --------------------------------------------------------------

        def _run_one_permutation(perm_seed: int, split_idx: int):
            """
            Fit a 1-D CCA model on permuted data for one (split, permutation)
            pair and return canonical-correlation and R² statistics.

            The response view is permuted globally (before train / test
            splitting) so that the permuted association is absent in both
            partitions simultaneously.
            """
            local_rng = np.random.default_rng(perm_seed)

            # ---- Permute the response view (all N subjects) -----------
            N = view_metadata_k[response_view]["data"].shape[0]
            perm_order = local_rng.permutation(N)

            permuted_metadata_k = []
            for v_idx, meta in enumerate(view_metadata_k):
                if v_idx == response_view:
                    permuted_data = meta["data"][perm_order, :]
                    permuted_metadata_k.append({**meta, "data": permuted_data})
                else:
                    permuted_metadata_k.append(meta)

            # ---- Reproduce the original train / test partition --------
            split_seed = int(
                original_stability_results["split_seeds"][split_idx])
            train_idx, test_idx = train_test_split_by_group(
                combined_group, test_size=test_size, seed=split_seed
            )

            # ---- Leakage-free preprocessing (fit on train only) ------
            train_views_k, test_views_k = get_fold_data(
                train_idx, test_idx, permuted_metadata_k
            )

            # ---- Fit a 1-D CCA model on permuted training data -------
            model_params = {"latent_dimensions": 1}
            if optimal_c is not None:
                model_params["c"] = optimal_c

            try:
                model = rCCA_class(**model_params).fit(train_views_k)

                # Pairwise canonical correlations: response vs. predictors
                corr_mat_train = model.pairwise_correlations(train_views_k)
                corr_mat_test = model.pairwise_correlations(test_views_k)

                other_view_indices = [
                    j for j in range(V) if j != response_view]

                if not other_view_indices:
                    warnings.warn(
                        f"Split {split_idx}, Dim {k + 1}: only one view present; "
                        f"canonical correlation is undefined."
                    )
                    return split_idx, np.nan, np.nan, np.nan, np.nan

                null_corr_tr = np.nanmean([
                    corr_mat_train[response_view, j, 0] for j in other_view_indices
                ])
                null_corr_te = np.nanmean([
                    corr_mat_test[response_view, j, 0] for j in other_view_indices
                ])

                null_r2_tr = compute_r2(model, train_views_k, response_view)
                null_r2_te = compute_r2(model, test_views_k,  response_view)

            except Exception as exc:
                warnings.warn(
                    f"Permutation failed — Split {split_idx}, Dim {k + 1}: {exc}. "
                    f"Returning NaN for this permutation."
                )
                return split_idx, np.nan, np.nan, np.nan, np.nan

            return split_idx, null_corr_tr, null_corr_te, null_r2_tr, null_r2_te

        # --------------------------------------------------------------
        # 3d.  Dispatch permutations in parallel
        # --------------------------------------------------------------
        print(
            f"    Dispatching {n_permutations} permutations × {S} splits "
            f"= {n_permutations * S} jobs (n_jobs={n_jobs}) …"
        )
        raw_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one_permutation)(pseed, sidx)
            for pseed, sidx in tqdm(tasks, desc=f"Dim {k + 1}", total=len(tasks))
        )

        # --------------------------------------------------------------
        # 3e.  Collect null values into (S, B) arrays
        # --------------------------------------------------------------
        null_corr_train_k = np.full((S, n_permutations), np.nan)
        null_corr_test_k = np.full((S, n_permutations), np.nan)
        null_r2_train_k = np.full((S, n_permutations), np.nan)
        null_r2_test_k = np.full((S, n_permutations), np.nan)

        # Count how many permutations have been recorded per split
        # (handles the rare case where a job returns the wrong split index)
        perm_counter = np.zeros(S, dtype=int)

        for sidx, c_tr, c_te, r2_tr, r2_te in raw_results:
            b = perm_counter[sidx]
            if b < n_permutations:
                null_corr_train_k[sidx, b] = c_tr
                null_corr_test_k[sidx,  b] = c_te
                null_r2_train_k[sidx,   b] = r2_tr
                null_r2_test_k[sidx,    b] = r2_te
                perm_counter[sidx] += 1

        # --------------------------------------------------------------
        # 3f.  Compute p-values and CIs for dimension k across all splits
        # --------------------------------------------------------------
        for s in range(S):
            label = f"s{s}_k{k}"
            p_corr_train[s, k],  ci_corr_train[s, k] = _pval_and_ci(
                obs_corr_train[s, k], null_corr_train_k[s], f"CorrTrain_{label}"
            )
            p_corr_test[s, k],   ci_corr_test[s, k] = _pval_and_ci(
                obs_corr_test[s, k],  null_corr_test_k[s],  f"CorrTest_{label}"
            )
            p_r2_train[s, k],    ci_r2_train[s, k] = _pval_and_ci(
                obs_r2_train[s, k],   null_r2_train_k[s],   f"R2Train_{label}"
            )
            p_r2_test[s, k],     ci_r2_test[s, k] = _pval_and_ci(
                obs_r2_test[s, k],    null_r2_test_k[s],    f"R2Test_{label}"
            )

    # ------------------------------------------------------------------
    # 4.  Return all results in a consistent dict
    # ------------------------------------------------------------------
    return {
        "obs_corr_train": obs_corr_train,   # (S, K)
        "p_corr_train":   p_corr_train,
        "ci_corr_train":  ci_corr_train,    # (S, K, 2)
        "obs_corr_test":  obs_corr_test,
        "p_corr_test":    p_corr_test,
        "ci_corr_test":   ci_corr_test,
        "obs_r2_train":   obs_r2_train,
        "p_r2_train":     p_r2_train,
        "ci_r2_train":    ci_r2_train,
        "obs_r2_test":    obs_r2_test,
        "p_r2_test":      p_r2_test,
        "ci_r2_test":     ci_r2_test,
    }