# ccagg/utils.py

"""
Helper functions used across the package.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, norm



#%% Functions for preprocessing 

def dummy_code(
    variable: np.ndarray,
    iscontinuous: bool = False,
    demean: bool = False,
) -> np.ndarray:
    """
    Encode a variable as a numeric covariate matrix.

    Continuous variables are optionally demeaned.
    Categorical variables are one-hot encoded (all levels retained —
    the user is responsible for removing one column to avoid perfect
    multicollinearity where required).

    Parameters
    ----------
    variable : np.ndarray, shape (N,)
    iscontinous : bool
        If True, treat as a continuous variable.
    demean : bool
        Subtract the mean (continuous only).

    Returns
    -------
    np.ndarray, shape (N, 1) for continuous or (N, n_levels) for categorical.
    """
    if iscontinuous:
        x = variable.astype(float).reshape(-1, 1)
        if demean:
            x = x - x.mean()
        return x

    levels  = np.unique(variable)
    encoded = (variable.reshape(-1, 1) == levels).astype(float)
    return encoded

def safe_float(x, default=np.nan):
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default

def gaussian_copula_transform(X):
    """
    Rank-based Gaussian copula transform (Inverse Normal Transformation).
    Forces data to follow a standard normal distribution.
    """
    if X.ndim == 1: X = X.reshape(-1, 1)
    
    X_trans = np.full_like(X, np.nan, dtype=float)
    
    for j in range(X.shape[1]):
        valid = ~np.isnan(X[:, j])
        if np.sum(valid) > 0:
            ranks = rankdata(X[valid, j], method="average")
            # Map to (0, 1) exclusive to avoid inf in norm.ppf
            # (ranks / (N + 1)) is the Van der Waerden formula
            quantiles = ranks / (np.sum(valid) + 1)
            # Apply Inverse CDF
            X_trans[valid, j] = norm.ppf(quantiles)
            
    return X_trans

def subset_view_metadata(view_metadata, idx):
    idx = np.asarray(idx)
    return [
        {
            **meta,
            "data": meta["data"][idx],
            "confounds": (
                meta["confounds"][idx]
                if meta["confounds"] is not None
                else None
            ),
        }
        for meta in view_metadata
    ]


#%% Functions for train/test split 


def train_test_split_by_group(
    group: np.ndarray,
    test_size: float = 0.3,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split subject indices into train and test sets within each group.
    Each group contributes approximately the same proportion of subjects to the train and test partitions.
    

    Parameters
    ----------
    group : np.ndarray, shape (N,)
        Group label for each subject (e.g. site ID).
    test_size : float, default 0.3
        Proportion of groups assigned to the test set.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    train_idx, test_idx : np.ndarray
        Integer indices into the original subject array.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(group)
    train_idx, test_idx = [], []

    for g in unique_groups:
        indices = np.where(group == g)[0]
        indices = rng.permutation(indices)
        split   = int(len(indices) * (1 - test_size))
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])

    return np.array(train_idx), np.array(test_idx)


def get_fold_data(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    view_metadata: list[dict],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extract and preprocess train and test partitions with strict separation.

    Preprocessing (confound regression, z-scoring) is fitted **exclusively**
    on the training partition and then applied to the test partition.  This
    prevents any form of data leakage across the train / test boundary.

    For each view the following steps are applied in order:

    1. Subset rows to ``train_idx`` / ``test_idx``.
    2. Fit a confound regression model on the training rows using the view's
       confound matrix (e.g. site, sex, ICV).
    3. Regress confounds out of both train and test rows using the
       train-fitted model.
    4. Z-score each feature using the training mean and standard deviation;
       apply the same scaling to the test partition.

    Parameters
    ----------
    train_idx : np.ndarray, shape (N_train,)
        Integer indices of training subjects.
    test_idx : np.ndarray, shape (N_test,)
        Integer indices of test subjects.
    view_metadata : list of dict
        One dict per view with keys:

        * "data"      – raw feature matrix
        * "confounds" – confound matrix, shape "(N, C)", or "None"
        * "type"      – "continuous" (reserved for future use).
        * "name"      – human-readable view label (used in warnings).

    Returns
    -------
    train_views : list of np.ndarray
        Preprocessed training data, one array of shape ``(N_train, F_v)``
        per view.
    test_views : list of np.ndarray
        Preprocessed test data, one array of shape ``(N_test, F_v)`` per
        view, transformed using train-fitted parameters only.

    Notes
    -----
    * Features with zero variance in the training set are set to zero in
      both partitions to avoid division by zero during z-scoring.
    * The function never modifies the arrays stored in ``view_metadata``
      in-place; all operations are performed on copies.
    """
    train_views = []
    test_views = []
    
    ols = LinearRegression(n_jobs=1) # Fast OLS
    
    for meta in view_metadata:
        X_raw = meta['data']
        confounds = meta['confounds']
        v_type = meta['type'] # 'continuous' or 'copula'
        
        # 1. Split
        X_tr = X_raw[train_idx].copy()
        X_te = X_raw[test_idx].copy()
        
        # 2. Gaussian Copula 
        if v_type == 'copula':
            # Transform independently (Distribution enforcement, not learning)
            X_tr = gaussian_copula_transform(X_tr)
            X_te = gaussian_copula_transform(X_te)
            
        # 3. Confounder Removal (Strict Learning)
        if confounds is not None:
            C_tr = confounds[train_idx]
            C_te = confounds[test_idx]
            
            ols.fit(C_tr, X_tr)
            X_tr = X_tr - ols.predict(C_tr)
            X_te = X_te - ols.predict(C_te)
            
        # 4. Scaling
        scaler = StandardScaler()
        # Learn mean/sd on train
        X_tr = scaler.fit_transform(X_tr)
        # Apply to test
        X_te = scaler.transform(X_te)
        
        train_views.append(X_tr)
        test_views.append(X_te)
        
    return train_views, test_views


#%% Functions generating random CCA parameters


def get_spls_params(n_iter, seed, n_views, latent_dimensions=1):
    """
    Samples parameters for Sparse CCA using a number of latent dimensions.
    """    
    rng = np.random.default_rng(seed)
    tau_grid = [[round(x, 2) for x in np.linspace(0.1, 0.9, 10)] for _ in range(n_views)]
    param_list = []
    for _ in range(n_iter):
        params = {
            "latent_dimensions": latent_dimensions,
            "tau": [float(rng.choice(tau_grid[v])) for v in range(n_views)]
        }
        param_list.append(params)
        
    return param_list
  
    
def get_elastic_params(n_iter, seed, n_views, latent_dimensions=1):
    """
    Samples parameters for ElasticNet CCA using a number of latent dimensions.
    """
    rng = np.random.default_rng(seed)
    alpha_grid = [[round(x, 4) for x in np.linspace(1e-4, 1e-3, 5)] for _ in range(n_views)]
    l1_ratio_grid = [[round(x, 2) for x in np.linspace(0.3, 0.9, 5)] for _ in range(n_views)]
    param_list = []
    for _ in range(n_iter):
        params = {
            "latent_dimensions": latent_dimensions,
            "alpha": [float(rng.choice(alpha_grid[v])) for v in range(n_views)],
            "l1_ratio": [float(rng.choice(l1_ratio_grid[v])) for v in range(n_views)]
        }
        param_list.append(params)
        
    return param_list


def get_gcca_params(
    n_iter=None,
    seed=None,
    n_views=None,
    latent_dimensions=None,
):
    c_grid = np.logspace(-6, 1, 10)
    return [{"c": float(c)} for c in c_grid]


def make_model_init_kwargs(
    model_class,
    params=None,
    model_kwargs=None,
    design=None,
    random_state=None,
    latent_dimensions=1,
    epochs=200,
    tol=1e-5,
):
    out = {
        **(params or {}),
        **(model_kwargs or {}),
        "latent_dimensions": latent_dimensions,
    }

    if random_state is not None:
        out["random_state"] = int(random_state)

    if design is not None:
        out["design"] = design

    if model_class.__name__ != "GCCA":
        out.setdefault("early_stopping", True)
        out.setdefault("epochs", epochs)
        out.setdefault("tol", tol)

    return out



#%% Functions for robust inference and validation analyses

def block_permutation_index(blocks, rng):
    """
    Permute within exchangeability blocks, e.g. site x sex.
    """
    blocks = np.asarray(blocks)
    perm = np.arange(blocks.shape[0])

    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        perm[idx] = idx[rng.permutation(idx.size)]

    return perm

