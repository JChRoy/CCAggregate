# ccagg — resampling-aggregated multiview CCA

`ccagg` is a Python package for sparse and regularised canonical correlation
analysis of high-dimensional multiview data. Instead of fitting one model on the
full sample, it refits the model over many resampled train/test partitions and
reports what is reproducible across them: out-of-sample canonical correlations,
variable selection frequencies, aggregated weight vectors, and permutation-based
p-values. It extends the linear models of
[cca-zoo](https://github.com/jameschapman19/cca_zoo) with a connectivity design
matrix.

## Description

We consider two or more data blocks, or views, measured on the same
participants. Each view is an *n × p* matrix, and the number and nature of the
variables may differ from one view to another. Canonical correlation analysis
estimates weight vectors such that the resulting canonical variates are
maximally associated across views.

When *p* is large relative to *n*, as is typical for psychometric and
neuroimaging views, two problems follow. The estimated weight vectors are
strongly sample-dependent, so a single fit is difficult to interpret. The
canonical correlations evaluated on the same data used for estimation are
inflated by overfitting, so they are not valid effect sizes.

`ccagg` addresses both problems by repeating the whole estimation over resampled
partitions and summarising the results across them, which separates the
components that are reproducible from those that reflect one particular sample.

## Methods implemented

**Penalised estimators.** Two estimators are provided, both extended with a
design matrix that specifies which pairs of views enter the objective.

- `DesignElasticCCA` is a sparse and regularised CCA. Each weight vector is
  updated by an elastic net regression of its view on the design-weighted sum of
  the canonical variates of the connected views, following the iterative
  penalised least squares formulation of sparse CCA (Mai & Zhang, 2019).
  Because the update is a regression, the within-view
  covariance is whitened and the criterion is correlation-based.
- `DesignSPLS` is a sparse PLS. The weight vector is updated by the cross-product
  with the design-weighted target, projected onto an L1 ball and normalised to
  unit length, which is the penalised matrix decomposition of Witten et al.
  (2009). Because the norm of the weight vector is constrained rather than the
  variance of the variate, the criterion is covariance-based.

The L2 part of the elastic net has an effect comparable to the shrinkage
parameter of regularised generalised CCA (Vinod, 1976; Tenenhaus & Tenenhaus,
2011). With weak regularisation `DesignElasticCCA` targets correlation, and as
the L2 penalty increases it moves toward a covariance criterion. Unlike in
RGCCA, this position is not set by an explicit parameter but follows from the
value of `alpha` selected by cross-validation.

Both classes fall back to a fully connected design, that is all view pairs
active, when no design matrix is supplied. More than two views and more than one
latent dimension are supported.

**Hyperparameter tuning.** Penalty parameters are drawn by random search and
selected by k-fold cross-validation inside each training partition, so tuning
never sees the held-out partition.

**Resampling and out-of-sample evaluation.** The sample is split repeatedly into
training and test partitions, stratified on a grouping variable such as site, or
site by sex. Confound regression and scaling are fitted on the training partition
and applied to the test partition. Weight vectors estimated on the training
partition are projected onto the held-out partition, which gives canonical
correlations that are free of the in-sample upward bias.

**Stability selection.** The selection frequency of each variable across
partitions is recorded. Variables exceeding a user-defined frequency threshold
define the reduced model, which is then refitted with regularised generalised
CCA and re-evaluated out of sample.

**Aggregation across partitions.** Components are matched across partitions by
Hungarian assignment and their sign is aligned, since canonical weights are
identified only up to sign and component order is arbitrary. Weights and scores
are then averaged over partitions.

**Permutation inference.** Empirical null distributions are obtained by
permutation, either conditional on the selected variables or by rerunning the
complete workflow on each permutation. The four available routes are described
below.

## Installation

```bash
git clone https://github.com/JChRoy/CCAggregate.git
cd CCAggregate
pip install -e .
```

Requires Python ≥ 3.9.

## Data format

Every function takes `view_metadata`: a list of dicts, one per view.

```python
view_metadata = [
    {"data": Y,        # (n_samples, n_features)
     "confounds": C,   # (n_samples, n_confounds) or None
     "type": "continuous",   # or "copula" for a rank-based normal transform
     "name": "Clinical"},
    {"data": X1, "confounds": C, "type": "continuous", "name": "Thickness"},
]
```

A `design` matrix controls which view pairs are maximised. To link one
response view to several predictor views without maximising association among
the predictors:

```python
design = np.array([[0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0]])
```

`combined_group` is the vector used to stratify splits (site, or site x sex).
Splits are reproducible from the stored `split_seeds`.

## Minimal example

```python
import numpy as np
from cca_zoo.linear import GCCA
from ccagg.utils import train_test_split_by_group, get_elastic_params
from ccagg.model import (parallel_best_param_ensemble_multidim,
                         run_rcca_stability_selection_multidim)
from ccagg.cca_extended import DesignElasticCCA

results = parallel_best_param_ensemble_multidim(
    DesignElasticCCA,
    sample_param_func=get_elastic_params,
    view_metadata=view_metadata,
    combined_group=groups,
    train_test_split_by_group=train_test_split_by_group,
    design=design,
    latent_dimensions=3,
    n_splits=100, n_iter=100, cv=10, n_jobs=8,
)

reduced = run_rcca_stability_selection_multidim(
    results_full=results,
    view_metadata=view_metadata,
    rCCA_class=GCCA,
    train_test_split_by_group=train_test_split_by_group,
    combined_group=groups,
    threshold=0.9,
    response_view=0,
    n_jobs=8,
)
```

`DesignSPLS` can be substituted for `DesignElasticCCA` if a covariance-based
criterion is preferred, with the matching parameter sampler.

See `examples/vignette.py` for a full worked pipeline on simulated data,
including all inference routes and the plots.

## Inference

| Function | Method |
|---|---|
| `permutation.permute_score_significance_multidim` | Permutes the aggregated canonical scores, keeping the fitted model fixed |
| `permutation.permute_rgcca_significance_multidim` | Permutes the response view and refits the reduced model on each permutation |
| `robust_inference.multisplit_discovery_inference_test` | Selects features and tunes in discovery data, then evaluates and permutes in a matched held-out set, repeated across many splits |
| `robust_inference.omnibus_full_pipeline_permutation_test` | Reruns the entire workflow, that is selection, tuning, fitting and evaluation, on every permutation |

The first two routes are conditional on the selected variables. They test the
observed out-of-sample correlation given that selection, not the procedure that
produced it. The last two include selection and tuning inside the permutation
loop, so they test the pipeline as a whole and are the more conservative choice.

## Output structure

`parallel_best_param_ensemble_multidim` returns a dict. The main entries are:

- `test_scores[split][view]`: canonical variates on the held-out partition
- `train_scores`, `train_loadings`, `test_loadings`, `train_cross_loadings`,
  `test_cross_loadings`
- `average_selection_frequency[view]`: (n_features, K) selection proportions
- `average_weights[view]`, `average_scores[view]`: aligned and averaged across
  splits, computed in sample
- `split_seeds`: one per split; splits are exactly reconstructible from these

For an out-of-sample participant-level score, average each participant's
`test_scores` over the splits where they were held out, rather than using
`average_scores`.

## Caveats

Canonical correlations in high dimensions need large samples to be stable.
Reported effect sizes should come from held-out partitions.

Sign and component order are indeterminate in each split. Alignment uses
Hungarian matching against view 0 of the first split, followed by a sign flip.
If view 0 is sparse or unstable, alignment degrades. Check the per-split loading
correlations before trusting `average_weights`.

In `DesignElasticCCA`, the canonical variates are not rescaled to unit variance
after each weight update. The design weights therefore act on targets whose
scale is not fixed across views. Standardise the views before fitting if the
relative contribution of each view matters for the interpretation.

The Gaussian copula transform, used when `type="copula"`, is applied
independently to the training and test partitions. It imposes a distribution
rather than estimating parameters from the data, so it does not leak information,
but unlike the confound regression and the scaling it is not fitted on the
training partition alone.

## References

> Mai, Q., & Zhang, X. (2019). An iterative penalized least squares approach to
> sparse canonical correlation analysis. *Biometrics*, 75(3), 734–744.
> https://doi.org/10.1111/biom.13043

> Tenenhaus, A., & Tenenhaus, M. (2011). Regularized generalized canonical
> correlation analysis. *Psychometrika*, 76(2), 257–284.
> https://doi.org/10.1007/s11336-011-9206-8

> Tenenhaus, A., Philippe, C., Guillemot, V., Lê Cao, K.-A., Grill, J., &
> Frouin, V. (2014). Variable selection for generalized canonical correlation
> analysis. *Biostatistics*, 15(3), 569–583.
> https://doi.org/10.1093/biostatistics/kxu001

> Vinod, H. D. (1976). Canonical ridge and econometrics of joint production.
> *Journal of Econometrics*, 4(2), 147–166.
> https://doi.org/10.1016/0304-4076(76)90010-5

> Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix
> decomposition, with applications to sparse principal components and canonical
> correlation analysis. *Biostatistics*, 10(3), 515–534.
> https://doi.org/10.1093/biostatistics/kxp008

## Citation

`ccagg` builds on cca-zoo:

> Chapman, J., Wang, H.-T., Wells, L., & Wiesner, J. (2021). CCA-Zoo: A
> collection of Regularized, Deep Learning based, Kernel, and Probabilistic CCA
> methods in a scikit-learn style framework. *Journal of Open Source Software*,
> 6(68), 3823. https://doi.org/10.21105/joss.03823

If you use `ccagg` itself, please cite:

> Roy, J.-C. (2026). CCAggregate (Version v0.1.1).
> Zenodo. https://doi.org/10.5281/zenodo.22676694

## Funding

This work has received funding from the European Union's Marie Skłodowska-Curie
Actions Postdoctoral Fellowship (HORIZON-MSCA-2024-PF-01) under grant agreement
No 101212349 (PRE-EMHPT).
Views and opinions expressed are however those of the author(s) only and do not
necessarily reflect those of the European Union. Neither the European Union nor
the granting authority can be held responsible for them.

## Licence

MIT
