# ccagg — resampling-aggregated multiview CCA

Sparse and regularised canonical correlation analysis across many train/test splits, with stability selection, permutation-based inference and
leave-one-site-out validation. 
Multiview CCA on high-dimensional psychological data is unstable: a single fit gives weights that depend heavily on the sample, and in-sample canonical correlations are
strongly biased upward. 
`ccagg` fits the model repeatedly across resampled splits and reports what survives, rather than reporting one solution.


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

See `examples/vignette.py` for a full worked pipeline on simulated data,
including all three inference routes and the plots.

## Inferences evaluated

| Function | Method |
|---|---|
| `permutation.permute_score_significance_multidim` |  Permutes the aggregated canonical scores, keeping the fitted model fixed |
| `permutation.permute_rgcca_significance_multidim` | Permutes the response view and refits the reduced model on each permutation |
| `robust_inference.multisplit_discovery_inference_test` | Selects features and tunes in discovery data, then evaluates and permutes in a matched held-out set, repeated across many splits |
| `robust_inference.omnibus_full_pipeline_permutation_test` | Reruns the entire workflow — selection, tuning, fitting, evaluation on every permutation |

The first two condition on the selected features and therefore quantify the significance of the observed out-of-sample correlation *given* that selection not of the pipeline. 

## Output structure

`parallel_best_param_ensemble_multidim` returns a dict. The most important ones: 
- `split_seeds`: one per split; splits are exactly reconstructible from these
- `test_scores[split][view]`: canonical variates on the held-out partition
- `train_scores`, `train_loadings`, `test_loadings`, `train_cross_loadings`,
  `test_cross_loadings`
- `average_selection_frequency[view]`: (n_features, K) selection proportions
- `average_weights[view]`, `average_scores[view]`:  aligned and averaged across splits, computed in sample

For an out-of-sample participant-level score, average each participant's `test_scores` over the splits where they were held out, rather than using `average_scores`.

## Some caveats to be aware of
Canonical correlations in high dimensions need large samples to be stable. Reported effect sizes should come from held-out partitions.

Sign and component order are indeterminate per split. Alignment uses Hungarian matching against view 0 of the first split, then a sign flip. If view 0 is
sparse or unstable, alignment degrades. It is better to check the per-split loading correlations before trusting `average_weights`.

The Gaussian copula transform, when `type="copula"`, is applied independently to train and test partitions. This enforces a distribution rather than learning parameters, 
so it does not leak, but it is not fitted on train alone in the way the confound regression and scaling are.

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
[MIT / BSD-3 — choose one and add the file]

