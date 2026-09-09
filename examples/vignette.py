#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:47:05 2026

@author: jc
"""

#!/usr/bin/env python3
"""
ccagg vignette
==============

A complete worked pipeline on simulated data. Runs end to end in a few minutes
with the settings below; the values marked PRODUCTION are what you would use
for a real analysis.

Simulated design: one clinical view (6 factors) and two brain views, sharing a
single latent dimension, measured across 8 sites with a site effect.

    python vignette.py
"""

import numpy as np
import matplotlib.pyplot as plt
 
from cca_zoo.linear import GCCA
 
from ccagg.utils import (
    dummy_code,
    train_test_split_by_group,
    get_elastic_params,
    get_gcca_params,
)
from ccagg.model import (
    parallel_best_param_ensemble_multidim,
    run_rcca_stability_selection_multidim,
)
from ccagg.permutation import (
    permute_score_significance_multidim,
    permute_rgcca_significance_multidim,
)
from ccagg.robust_inference import (
    multisplit_discovery_inference_test,
    omnibus_full_pipeline_permutation_test,
)
from ccagg.validation import leave_one_site_out
from ccagg.cca_extended import DesignElasticCCA
 
import ccagg.visualisation as viz
 
RNG = np.random.default_rng(0)
 
# PRODUCTION: n_splits=100, n_iter=100, cv=10, n_permutations=1000+
FAST = dict(n_splits=20, n_iter=20, cv=5, n_perm=200)
 
 
def has(name):
    """Skip a plot cleanly if that function is not in visualisation.py."""
    return hasattr(viz, name)
 
 
# %%
############
# Simulate
############

# A single latent factor drives 4 of 6 clinical variables, 8 of 60 thickness
# features and 5 of 40 volume features. Everything else is noise. Site adds a
# nuisance offset that the confound regression should remove.
 
def simulate(n=600, n_sites=8, effect=0.55):
    site = RNG.integers(0, n_sites, size=n)
    sex = RNG.integers(0, 2, size=n)
    z = RNG.normal(size=n)                      # shared latent factor
 
    def view(n_feat, n_signal, load):
        X = RNG.normal(size=(n, n_feat))
        X[:, :n_signal] += load * z[:, None]
        X += 0.8 * RNG.normal(size=(1, n_feat)) * site[:, None] / n_sites
        return X
 
    Y = view(6, 4, effect)             # clinical
    X1 = view(60, 8, effect * 0.8)     # thickness
    X2 = view(40, 5, effect * 0.6)     # volume
 
    return Y, X1, X2, site, sex, z
 
 
Y, X1, X2, site, sex, z_true = simulate()
n = Y.shape[0]
 
cov = np.column_stack([dummy_code(site), dummy_code(sex)])
 
view_metadata = [
    {"data": Y,  "confounds": cov, "type": "continuous", "name": "Clinical"},
    {"data": X1, "confounds": cov, "type": "continuous", "name": "Thickness"},
    {"data": X2, "confounds": cov, "type": "continuous", "name": "Volume"},
]
 
view_names = [
    [f"clin_{i}" for i in range(Y.shape[1])],
    [f"ct_{i}" for i in range(X1.shape[1])],
    [f"vol_{i}" for i in range(X2.shape[1])],
]
view_labels = [m["name"] for m in view_metadata]
views = [m["data"] for m in view_metadata]
 
# stratify splits on site x sex, built from the raw labels
combined_group = np.array([f"{s}_{x}" for s, x in zip(site, sex)])
 
# link the clinical view to each brain view, but not the brain views to each
# other
design = np.array([[0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0]])
 
print(f"simulated n = {n}, views = {[v['data'].shape for v in view_metadata]}")
print(f"{len(np.unique(combined_group))} strata for splitting")
 
 
# %%
##############################
# Fit across resampled splits
##############################

# Hyperparameters are sampled per split and tuned by inner CV inside the
# training partition only. Nothing from the held-out 30% enters the fit.
 
results = parallel_best_param_ensemble_multidim(
    DesignElasticCCA,
    sample_param_func=get_elastic_params,
    view_metadata=view_metadata,
    combined_group=combined_group,
    train_test_split_by_group=train_test_split_by_group,
    design=design,
    latent_dimensions=2,
    n_splits=FAST["n_splits"],
    n_iter=FAST["n_iter"],
    cv=FAST["cv"],
    epochs=200,
    n_jobs=4,
)
 
print("\nkeys returned:", sorted(results.keys()))
 
for v, name in enumerate(view_labels):
    freq = np.squeeze(results["average_selection_frequency"][v][:, 0])
    keep = np.asarray(view_names[v])[freq >= 0.9]
    print(f"{name}: {len(keep)}/{len(freq)} features at freq >= 0.9 -> "
          f"{list(keep)[:8]}")
 
# How variable is the selected regularisation across splits? A wide spread with
# comparable performance is what justifies aggregating rather than trusting any
# single tuning solution.
if has("plot_hyperparameter"):
    viz.plot_hyperparameter(results, view_names=view_labels)
    plt.show()
 
# Refit on the stable features only
reduced = run_rcca_stability_selection_multidim(
    results_full=results,
    view_metadata=view_metadata,
    rCCA_class=GCCA,
    train_test_split_by_group=train_test_split_by_group,
    combined_group=combined_group,
    threshold=0.9,
    c_grid=np.logspace(-6, 1, 10),
    response_view=0,
    n_jobs=4,
)
 
 
# %%
###################################
# Inference: 3 complementary tools
###################################
 
# Conditional on the fitted model: permute the scores.
score_perm = permute_score_significance_multidim(
    reduced, response_view=0,
    n_permutations=FAST["n_perm"], random_state=42, n_jobs=4,
)
print("\n[a] median p, held-out correlation:",
      np.median(score_perm["p_corr_test"][:, 0]))
 
# Conditional on the selected features, refitting per permutation.
rgcca_perm = permute_rgcca_significance_multidim(
    view_metadata=view_metadata,
    original_stability_results=reduced,
    rCCA_class=GCCA,
    combined_group=combined_group,
    response_view=0,
    n_permutations=FAST["n_perm"],
    random_state=42, n_jobs=4, test_size=0.3,
    selection_threshold=0.9,
)
print("[b] median p, held-out correlation:",
      np.median(rgcca_perm["p_corr_test"][:, 0]))
 
# Observed statistics across splits, with per-split permutation significance.
if has("plot_permutation_summary"):
    viz.plot_permutation_summary(rgcca_perm, save_path="fig_permtest_summary")
    plt.show()
 
# Train against test: the gap is the optimism the resampling exposes.
if has("plot_train_vs_test"):
    viz.plot_train_vs_test(rgcca_perm, save_path="fig_train_vs_test")
    plt.show()
 
if has("plot_generalisation"):
    viz.plot_generalisation(rgcca_perm, n_dims=2)
    plt.show()
 
# Unconditional: selection and tuning happen in discovery data, the model is
# evaluated unchanged in a matched inference set, and permutation is applied
# there. This is the one to report as the primary test.
ms = multisplit_discovery_inference_test(
    model_class=DesignElasticCCA,
    view_metadata=view_metadata,
    combined_group=combined_group,
    train_test_split_by_group=train_test_split_by_group,
    sample_param_func=get_elastic_params,
    response_view=0,
    inference_size=0.3,
    discovery_split_seeds=np.arange(10),
    discovery_n_splits=10,
    discovery_n_iter=20,
    reduced_cv=5,
    n_permutations=FAST["n_perm"],
    selection_threshold=0.9,
    design=design,
    epochs=200,
    random_state=123,
    selection_rule="max_mean",
    permutation_blocks=combined_group,
    combine_p_method="cauchy",
    n_jobs_splits=4,
    show_progress=True,
)
 
print("\n------- Unconditional inference -------")
print("median held-out corr per dim:", ms["effect_summary"]["median"])
print("combined p (Cauchy):", ms["combined_p_test_corr"])
print("combined p (min-Bonferroni):", ms["combined_p_test_corr_min_bonferroni"])
 
# Omnibus over the whole pipeline, including selection and tuning.
# Expensive: run once, at the end.
# omni = omnibus_full_pipeline_permutation_test(
#     model_class=DesignElasticCCA,
#     view_metadata=view_metadata,
#     combined_group=combined_group,
#     train_test_split_by_group=train_test_split_by_group,
#     sample_param_func=get_elastic_params,
#     response_view=0,
#     permutation_blocks=combined_group,
#     inference_size=0.3,
#     latent_dimensions=1,
#     outer_split_seeds=np.arange(5),
#     discovery_n_splits=10, discovery_n_iter=20, reduced_cv=5,
#     selection_threshold=0.9, design=design, epochs=200,
#     n_perm=20, random_state=123, n_jobs_perm=4, show_progress=True,
# )
# print("observed:", omni["observed_global_stat"], "p:", omni["p_value"])
 
 
# %%
###########################
# Feature characterization
###########################

# Selection frequency, loading and cross-loading say different things. The
# consensus framework combines them; the weight sensitivity shows which
# features stay top-ranked whatever weighting is used.
 
if has("summarize_loadings_multidim2"):
    summary_df = viz.summarize_loadings_multidim2(
        reduced, views, view_labels, view_names
    )
    print("\nfeature summary:", summary_df.shape)
    print(summary_df.head())
 
if has("model_consensus"):
    consensus_df = viz.model_consensus(
        results_filtered=reduced,
        feature_names_per_view=view_names,
        view_labels=view_labels,
        response_view=0,
        data_split="test",
        include_response_view=False,
        min_selection_freq=0.0,
        min_valid_splits=1,
        weights=(0, 0.5, 0.5),      # selection, loading, cross-loading
    )
 
    if has("plot_consensus_features"):
        viz.plot_consensus_features(consensus_df, component="Component_1",
                                    top_n=15)
        plt.show()
 
    # Loading indexes a feature's contribution to the brain pattern;
    # cross-loading indexes its association with the response view. They are
    # complementary, not interchangeable.
    if has("plot_loading_crossloading_scatter"):
        viz.plot_loading_crossloading_scatter(consensus_df,
                                              component="Component_1",
                                              top_n_labels=8)
        plt.show()
 
    if has("plot_metric_profiles_abs"):
        viz.plot_metric_profiles_abs(consensus_df, component="Component_1",
                                     top_n=12)
        plt.show()
 
# No weighting of the three metrics is justified a priori, so scan the grid.
if has("run_consensus_weight_sensitivity"):
    sens = viz.run_consensus_weight_sensitivity(
        results_filtered=reduced,
        feature_names_per_view=view_names,
        view_labels=view_labels,
        response_view=0,
        data_split="test",
        include_response_view=False,
        min_selection_freq=0.0,
        min_valid_splits=1,
        step=0.1,
        min_weight=0.0,
        top_k=20,
    )
    if has("plot_rank_stability_scatter"):
        viz.plot_rank_stability_scatter(sens["summary_df"],
                                        component="Component_1",
                                        top_n_labels=20)
        plt.show()
 
 
# %%
##########################
# Validation accross sites
##########################

# Leave-one-site-out. Drop site from the confounds here: site is the held-out
# unit, so regressing it out first would remove what the test is checking.
 
cov_nosite = dummy_code(sex)
view_metadata_loso = [{**m, "confounds": cov_nosite} for m in view_metadata]
 
loso = leave_one_site_out(
    selector_class=DesignElasticCCA,
    estimator_class=GCCA,
    view_metadata=view_metadata_loso,
    site_labels=site,
    train_test_split_by_group=train_test_split_by_group,
    sample_param_func_selector=get_elastic_params,
    sample_param_func_estimator=get_gcca_params,
    discovery_groups=combined_group,
    response_view=0,
    latent_dimensions=1,
    discovery_n_splits=5,
    discovery_n_iter=20,
    discovery_cv=5,
    reduced_cv=5,
    selection_threshold=0.9,
    selector_design=design,
    epochs_selector=200,
    random_state=42,
    n_permutations=0,          # set >0 for site-level p-values
    n_jobs_sites=2,
    min_test_n=30,
    show_progress=True,
)
 
corr = np.asarray(loso["obs_test_corr_by_site"], dtype=float)
print("\nsites with a valid estimate:", np.isfinite(corr).sum(axis=0))
print("median held-out-site corr:", np.nanmedian(corr, axis=0))
 
 
# %%
#################################
# Retrieve individual predictions
#################################

# average_scores is computed in sample and should not be used as an outcome in
# a downstream model. Build the out-of-fold version from test_scores instead.
 
def out_of_fold(results, combined_group, view=0, dim=0):
    n = len(combined_group)
    acc, cnt = np.zeros(n), np.zeros(n, dtype=int)
    ref = None
    for s, seed in enumerate(results["split_seeds"]):
        if seed is None or results["test_scores"][s] is None:
            continue
        _, test_idx = train_test_split_by_group(combined_group, 0.3, int(seed))
        zs = np.asarray(results["test_scores"][s][view])[:, dim].astype(float)
 
        L = results["test_loadings"][s][view]
        if L is not None:
            L = np.asarray(L)[:, dim]
            if ref is None:
                ref = L
            elif np.corrcoef(ref, L)[0, 1] < 0:
                zs = -zs
 
        sd = np.nanstd(zs)
        if not np.isfinite(sd) or sd < 1e-12:
            continue
        zs = (zs - np.nanmean(zs)) / sd
        ok = np.isfinite(zs)
        acc[test_idx[ok]] += zs[ok]
        cnt[test_idx[ok]] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), cnt
 
 
oof, n_oof = out_of_fold(results, combined_group)
ok = np.isfinite(oof)
 
r_oof = np.corrcoef(oof[ok], z_true[ok])[0, 1]
ins = (view_metadata[0]["data"] @ results["average_weights"][0])[:, 0]
r_ins = np.corrcoef(ins, z_true)[0, 1]

print(f"\nout-of-fold scores for {ok.sum()} participants, "
      f"median {np.median(n_oof):.0f} splits each")
print(f"correlation with the true latent factor:")
print(f"  out-of-fold scores  r = {r_oof:.3f}")
print(f"  average_scores      r = {r_ins:.3f}  (weights fitted on all data)")
 
# Recovery of the simulated factor. No package function covers this: it needs
# ground truth, which only exists in simulation.
fig, ax = plt.subplots(figsize=(3.2, 3.0))
ax.scatter(z_true[ok], oof[ok], s=7, alpha=.4, linewidths=0, c="0.3")
ax.set(xlabel="True latent factor", ylabel="Out-of-fold score")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig_recovery.pdf", bbox_inches="tight")
plt.show()
