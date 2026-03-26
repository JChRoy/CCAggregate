#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:29:24 2026

@author: jc
"""



import numpy as np
import warnings
from typing import Any, Callable, Iterable, Optional, Union
from joblib import Parallel, delayed, parallel_backend

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, ParameterGrid, KFold, ParameterSampler
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample

from cca_zoo.linear import ElasticCCA
from cca_zoo.datasets import JointData
from cca_zoo._utils._checks import _process_parameter
from cca_zoo.linear import (
    SPLS,
    ElasticCCA,
)

from cca_zoo.linear._iterative._base import _BaseIterative
from cca_zoo.linear._iterative._deflation import _DeflationMixin
from cca_zoo._utils._checks import _process_parameter
from cca_zoo.linear._iterative._base import _BaseIterative
from cca_zoo.linear._iterative._deflation import _DeflationMixin
from cca_zoo.linear._pls import PLSMixin
from cca_zoo.linear._search import _delta_search


#%% Multiview SPLS + Design Matrix


class DesignSPLS(_DeflationMixin, _BaseIterative, PLSMixin):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data: bool = True,
        random_state=None,
        tol: float = 1e-3,
        accept_sparse=None,
        epochs: int = 100,
        initialization: Union[str, Callable] = "pls",
        early_stopping: bool = False,
        verbose: bool = True,
        tau=None,
        positive=False,
        design=None,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            tol=tol,
            accept_sparse=accept_sparse,
            epochs=epochs,
            initialization=initialization,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        self.tau = tau
        self.positive = positive
        self.design = design
        self.design_ = None
        self.tau_processed_ = None
        self.positive_processed_ = None
        self._l1_radius_ = None

    @staticmethod
    def _as_2d_scores(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        return x

    @staticmethod
    def _current_score_vector(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] > 0:
            return x[:, -1]
        return x.ravel()

    @staticmethod
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if x.size <= 1 or y.size <= 1:
            return np.nan
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return np.nan

        r = np.corrcoef(x, y)[0, 1]
        return float(r) if np.isfinite(r) else np.nan

    def _build_design(self, n_views: int) -> np.ndarray:
        if self.design is None:
            return (
                np.ones((n_views, n_views), dtype=float)
                - np.eye(n_views, dtype=float)
            )

        design = np.asarray(self.design, dtype=float)
        if design.shape != (n_views, n_views):
            raise ValueError(
                "Design matrix must have shape "
                f"({n_views}, {n_views}), got {design.shape}."
            )
        return design

    def _ensure_design(self, views=None) -> np.ndarray:
        if getattr(self, "design_", None) is not None:
            return self.design_

        n_views = getattr(self, "n_views_", None)
        if n_views is None:
            if views is None:
                raise AttributeError(
                    "design_ is not initialized and n_views_ is unknown. "
                    "Fit the model first or pass views."
                )
            views = list(views)
            n_views = len(views)
            self.n_views_ = n_views

        self.design_ = self._build_design(n_views)
        return self.design_

    def _check_params(self):
        super()._check_params()

        if self.tau is None:
            warnings.warn(
                "tau was not provided; defaulting to 1.0 for each view."
            )

        self.tau_processed_ = _process_parameter(
            "tau",
            self.tau,
            1.0,
            self.n_views_,
        )
        if any((tau < 0 or tau > 1) for tau in self.tau_processed_):
            raise ValueError(
                "All tau values must be between 0 and 1. "
                f"Got tau={self.tau_processed_}."
            )

        self.positive_processed_ = _process_parameter(
            "positive",
            self.positive,
            False,
            self.n_views_,
        )

        self.design_ = self._build_design(self.n_views_)
        self._l1_radius_ = None

    def _update_weights(self, views: Iterable[np.ndarray], i: int) -> np.ndarray:
        views = list(views)
        design = self._ensure_design(views)

        if self._l1_radius_ is None:
            self._l1_radius_ = [
                max(1.0, tau * np.sqrt(w.shape[0]))
                for tau, w in zip(self.tau_processed_, self.weights_)
            ]

        scores = [self._current_score_vector(s) for s in self.transform(views)]

        target_parts = [
            design[i, j] * scores[j]
            for j in range(self.n_views_)
            if j != i and design[i, j] != 0
        ]

        if len(target_parts) == 0:
            prev_w = np.asarray(self.weights_[i], dtype=float)
            if prev_w.ndim == 1:
                return prev_w[:, None]
            return prev_w

        target = np.sum(target_parts, axis=0)
        target = np.asarray(target, dtype=float).ravel()

        X = np.asarray(views[i], dtype=float)
        new_weights = X.T @ target
        new_weights = np.asarray(new_weights, dtype=float).ravel()

        if self.positive_processed_[i]:
            new_weights[new_weights < 0] = 0.0

        if not np.any(np.isfinite(new_weights)) or np.allclose(new_weights, 0):
            return np.zeros(X.shape[1], dtype=float)[:, None]

        new_weights = _delta_search(
            new_weights,
            self._l1_radius_[i],
            tol=self.tol,
        )
        new_weights = np.asarray(new_weights, dtype=float).ravel()

        norm = np.linalg.norm(new_weights)
        if np.isfinite(norm) and norm > 0:
            new_weights = new_weights / norm
        else:
            new_weights = np.zeros_like(new_weights)

        return new_weights[:, None]

    def _objective(self, views: Iterable[np.ndarray]) -> float:
        views = list(views)
        design = self._ensure_design(views)
        transformed_views = [self._as_2d_scores(s) for s in self.transform(views)]

        if len(transformed_views) == 0:
            return 0.0

        n_dims = min(s.shape[1] for s in transformed_views)
        if n_dims == 0:
            return 0.0

        total = 0.0
        for k in range(n_dims):
            for i in range(self.n_views_):
                for j in range(i + 1, self.n_views_):
                    weight = design[i, j]
                    if weight == 0:
                        continue

                    corr = self._safe_corr(
                        transformed_views[i][:, k],
                        transformed_views[j][:, k],
                    )
                    if np.isfinite(corr):
                        total += weight * corr

        return float(total)

    def score(
        self,
        views: Iterable[np.ndarray],
        y: Optional[Any] = None,
        **kwargs,
    ) -> float:
        views = list(views)
        design = self._ensure_design(views)
        transformed_views = [
            self._as_2d_scores(s) for s in self.transform(views, **kwargs)
        ]

        if len(transformed_views) == 0:
            return 0.0

        n_dims = min(s.shape[1] for s in transformed_views)
        if n_dims == 0:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for k in range(n_dims):
            for i in range(self.n_views_):
                for j in range(i + 1, self.n_views_):
                    weight = design[i, j]
                    if weight == 0:
                        continue

                    corr = self._safe_corr(
                        transformed_views[i][:, k],
                        transformed_views[j][:, k],
                    )
                    if np.isfinite(corr):
                        weighted_sum += abs(weight) * abs(corr)
                        total_weight += abs(weight)

        if total_weight == 0:
            return 0.0

        return float(weighted_sum / total_weight)

    def _more_tags(self):
        tags = (
            super()._more_tags()
            if hasattr(super(), "_more_tags")
            else {}
        )
        tags.update({"pls": True, "multiview": True})
        return tags

#%% Multiview Elastic CCA + Design Matrix

class DesignElasticCCA(_DeflationMixin, _BaseIterative):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        epochs=100,
        initialization="cca",
        early_stopping=False,
        verbose=True,
        alpha=None,
        l1_ratio=None,
        positive=None,
        stochastic=False,
        design=None, 
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            tol=tol,
            accept_sparse=accept_sparse,
            epochs=epochs,
            initialization=initialization,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.stochastic = stochastic
        self.design = design

    def _check_params(self):
        self.alpha = _process_parameter("alpha", self.alpha, 1.0, self.n_views_)
        self.l1_ratio = _process_parameter("l1_ratio", self.l1_ratio, 0.5, self.n_views_)
        self.positive = _process_parameter("positive", self.positive, False, self.n_views_)
        self.regressors = initialize_regressors(
            self.alpha,
            self.l1_ratio,
            self.positive,
            self.stochastic,
            self.tol,
            self.random_state,
        )
        # Fill missing design if None
        if self.design is None:
            self.design_ = np.ones((self.n_views_, self.n_views_)) - np.eye(self.n_views_)
        else:
            self.design_ = np.asarray(self.design)
            if self.design_.shape != (self.n_views_, self.n_views_):
                raise ValueError("Design matrix must be square with shape (n_views, n_views)")

    def _update_weights(self, views, i):
        scores = np.stack(self.transform(views))
        target = np.sum(
            [
                self.design_[i, j] * scores[j]
                for j in range(self.n_views_)
                if j != i and self.design_[i, j] != 0
            ],
            axis=0,
        )
        norm = np.linalg.norm(target)
        target = np.zeros_like(target) if not np.isfinite(norm) or norm == 0 else target / norm

        X = views[i]
        reg = self.regressors[i]
        reg.fit(X, target)
        return np.atleast_2d(reg.coef_).T


    def _more_tags(self):
        return {"multiview": True}



#%%
def initialize_regressors(alpha, l1_ratio, positive, stochastic, tol, random_state):
    regressors = []
    for alpha, l1_ratio, positive in zip(alpha, l1_ratio, positive):
        if stochastic:
            regressors.append(
                SGDRegressor(
                    penalty="elasticnet",
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=False,
                    tol=tol,
                    warm_start=True,
                    random_state=random_state,
                )
            )
        elif l1_ratio == 0:
            regressors.append(
                Ridge(
                    alpha=alpha,
                    fit_intercept=False,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                )
            )
        elif l1_ratio == 1:
            regressors.append(
                Lasso(
                    alpha=alpha,
                    fit_intercept=False,
                    warm_start=True,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                    selection="random",
                )
            )
        else:
            regressors.append(
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=False,
                    warm_start=True,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                    selection="random",
                )
            )
    return regressors


def elastic_objective(z, y, w, alpha, l1_ratio):
    n = len(y)
    objective = np.linalg.norm(z - y) ** 2 / (2 * n)
    l1_pen = alpha * l1_ratio * np.linalg.norm(w, ord=1)
    l2_pen = alpha * (1 - l1_ratio) * np.linalg.norm(w, ord=2)
    return objective + l1_pen + l2_pen