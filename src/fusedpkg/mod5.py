
"""
mod2_refactored.py
==================

A cleaned and more efficient refactor of the original `mod2.py` provided in this
conversation. fileciteturn1file0

What changed (high level)
-------------------------
Performance:
- Avoids calling `group_modalities_based_on_betas()` inside lambda-path loops
  (Step 1 and Step 2.1) when only the *variable list* is needed.
- Reuses CVXPy problems via `cp.Parameter` + `warm_start=True`:
  - group-lasso problem on full data reused across lambdas
  - fused / g_fused problem on full data reused across fused lambdas per variable list
  - optional caches for per-fold group-lasso and per-fold fused problems used in CV
- Reuses KFold splits (with optional random_state for reproducibility).
- Uses NumPy for prediction and deviance computations in inner loops.

Correctness / robustness:
- Fixes boolean bug using `|` instead of `or`.
- Fixes "no variables kept" baseline prediction (offset-aware Poisson intercept-only).
- Makes Poisson deviance consistent with predictions produced by `exp(eta)`
  (predictions are mean counts μ, not rates).

Style / maintainability:
- Clear docstrings, type hints, and explicit naming.
- Consolidated penalty construction.
- Reduced unused imports / dead code.
- Centralized solver choice.

Notes
-----
- This module still depends on your project helpers:
  `choose_ref_modality` and `reorder_df_columns` from
  `mypkg.additional_functions.mod1`.
- The penalty types accepted are the same as in your original code:
  values should be "fused" or "g_fused" for each categorical variable key.

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from mypkg.additional_functions.mod1 import choose_ref_modality, reorder_df_columns


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

_INDENT_LEVEL = 0


def time_it(func):
    """Decorator printing execution time with indentation for nested calls."""
    def wrapper(*args, **kwargs):
        global _INDENT_LEVEL
        _INDENT_LEVEL += 1
        tabs = "\t" * (_INDENT_LEVEL - 1)
        t0 = time.time()
        print(f"{tabs}Executing <{func.__name__}>")
        out = func(*args, **kwargs)
        dt = time.time() - t0
        mins = int(dt // 60)
        secs = dt % 60
        print(f"{tabs}Function <{func.__name__}> execution time : {mins:.0f} minutes and {secs:.0f} seconds")
        _INDENT_LEVEL -= 1
        return out
    return wrapper


def get_onehot_columns(df: pd.DataFrame, prefixes: Sequence[str]) -> List[str]:
    """Return columns whose name starts with any prefix in `prefixes`."""
    prefixes = tuple(prefixes)
    return [c for c in df.columns if c.startswith(prefixes)]


# -----------------------------------------------------------------------------
# Deviance (Poisson)
# -----------------------------------------------------------------------------

def poisson_deviance_mu(mu_hat: Iterable[float], y: Iterable[float], exposure: Optional[Iterable[float]] = None) -> float:
    """Compute Poisson deviance contribution given predicted mean counts μ.

    Parameters
    ----------
    mu_hat:
        Predicted mean counts (μ_i). In your code these are produced by exp(η_i).
    y:
        Observed counts.
    exposure:
        Optional exposure weights for scaling the final deviance
        (returned value is sum(dev) / sum(exposure)). If None, returns mean deviance.

    Returns
    -------
    float
        Scaled deviance (without the conventional factor 2, matching your original style).

    Notes
    -----
    This is consistent with:
        dev_i = y_i * log(y_i / mu_i) - (y_i - mu_i)
    with the convention 0*log(0/mu)=0.
    """
    mu_hat = np.asarray(mu_hat, dtype=float)
    y = np.asarray(y, dtype=float)

    eps = 1e-12
    mu_hat = np.clip(mu_hat, eps, None)

    mask = y > 0
    term = np.zeros_like(y)
    term[mask] = y[mask] * (np.log(y[mask]) - np.log(mu_hat[mask]))
    dev = term - (y - mu_hat)

    if exposure is None:
        return float(dev.mean())
    exposure = np.asarray(exposure, dtype=float)
    exposure = np.clip(exposure, eps, None)
    return float(dev.sum() / exposure.sum())


def offset_to_exposure(offset: np.ndarray) -> np.ndarray:
    """Convert log-exposure offset to exposure = exp(offset)."""
    return np.exp(np.asarray(offset, dtype=float))


# -----------------------------------------------------------------------------
# GLM building blocks
# -----------------------------------------------------------------------------

def _log_likelihood(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    offset_arr: np.ndarray,
    beta: cp.Expression,
    intercept: cp.Expression,
) -> cp.Expression:
    """Return a concave log-likelihood expression for supported families."""
    eta = X @ beta + intercept + offset_arr
    fam = family.lower()

    if fam == "poisson":
        return cp.sum(cp.multiply(y, eta) - cp.exp(eta))
    if fam == "gaussian":
        return -0.5 * cp.sum_squares(y - eta)

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def _build_one_hot_groups(input_var_onehot: List[str], penalty_types: Dict[str, str]) -> Dict[str, List[int]]:
    """Map each variable key -> indices of its one-hot columns in `input_var_onehot`."""
    groups: Dict[str, List[int]] = {}
    for key in penalty_types:
        idx = [i for i, col in enumerate(input_var_onehot) if col.startswith(key + "_")]
        if idx:
            groups[key] = idx
    return groups


@dataclass(frozen=True)
class SolverConfig:
    solver: str = "CLARABEL"
    warm_start: bool = True
    verbose: bool = False


def build_group_lasso_problem(
    data: pd.DataFrame,
    input_var: Sequence[str],
    target_var: str,
    offset_var: str,
    penalty_types: Dict[str, str],
    family: str = "Poisson",
) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str], Dict[str, List[int]]]:
    """Build group-lasso GLM problem with lambda as a Parameter (for reuse)."""
    input_var_onehot = get_onehot_columns(data, input_var)
    X = data[input_var_onehot].to_numpy(dtype=float)
    y = data[target_var].to_numpy(dtype=float)
    offset_arr = data[offset_var].to_numpy(dtype=float)

    beta = cp.Variable(X.shape[1])
    intercept = cp.Variable()
    lam = cp.Parameter(nonneg=True, name="lambda_group")

    ll = _log_likelihood(family, X, y, offset_arr, beta, intercept)

    ptypes = {k: v for k, v in penalty_types.items() if any(k.startswith(prefix) for prefix in input_var)}
    one_hot_groups = _build_one_hot_groups(input_var_onehot, ptypes)

    group_pen = 0
    for idxs in one_hot_groups.values():
        w = np.sqrt(len(idxs))
        # Matches your original: l1 norm of group coefficients (not l2 group-lasso)
        group_pen += cp.norm1(w * beta[idxs])

    prob = cp.Problem(cp.Minimize(-ll + lam * group_pen))
    return prob, beta, intercept, lam, input_var_onehot, one_hot_groups


def _compute_ref_counts(n: int, col_sums: np.ndarray, group_idxs: List[int]) -> float:
    """Reference modality count = n - sum(one-hot counts)."""
    return float(n - col_sums[group_idxs].sum())


def build_fused_problem(
    data: pd.DataFrame,
    input_var: Sequence[str],
    target_var: str,
    offset_var: str,
    penalty_types: Dict[str, str],
    family: str = "Poisson",
) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]:
    """Build fused / g_fused GLM problem with lambda as a Parameter (for reuse)."""
    input_var_onehot = get_onehot_columns(data, input_var)
    X = data[input_var_onehot].to_numpy(dtype=float)
    y = data[target_var].to_numpy(dtype=float)
    offset_arr = data[offset_var].to_numpy(dtype=float)

    beta = cp.Variable(X.shape[1])
    intercept = cp.Variable()
    lam = cp.Parameter(nonneg=True, name="lambda_fused")

    ll = _log_likelihood(family, X, y, offset_arr, beta, intercept)

    ptypes = {k: v for k, v in penalty_types.items() if any(k.startswith(prefix) for prefix in input_var)}
    one_hot_groups = _build_one_hot_groups(input_var_onehot, ptypes)

    # Precompute sums for weights once
    col_sums = data[input_var_onehot].sum(axis=0).to_numpy(dtype=float)
    n = float(len(data))

    g_fused_graph_size = 0
    for key, idxs in one_hot_groups.items():
        if ptypes.get(key) == "g_fused":
            g_fused_graph_size += len(idxs) + 1
    g_fused_graph_size = max(g_fused_graph_size, 1)  # avoid /0

    fused_pen = 0
    g_fused_pen = 0

    for key, idxs in one_hot_groups.items():
        ref_count = _compute_ref_counts(len(data), col_sums, idxs)

        if ptypes.get(key) == "fused":
            # Reference modality -> 0
            w0 = np.sqrt((col_sums[idxs[0]] + ref_count) / n)
            fused_pen += cp.norm1(w0 * (beta[idxs[0]] - 0))

            # Sequential differences (in column order)
            for i in range(1, len(idxs)):
                w = np.sqrt((col_sums[idxs[i]] + col_sums[idxs[i - 1]]) / n)
                fused_pen += cp.norm1(w * (beta[idxs[i]] - beta[idxs[i - 1]]))

        elif ptypes.get(key) == "g_fused":
            scale = len(idxs) / g_fused_graph_size
            for i in range(len(idxs)):
                w0 = scale * np.sqrt((col_sums[idxs[i]] + ref_count) / n)
                g_fused_pen += cp.norm1(w0 * (beta[idxs[i]] - 0))

                for j in range(i, len(idxs)):
                    w = scale * np.sqrt((col_sums[idxs[i]] + col_sums[idxs[j]]) / n)
                    g_fused_pen += cp.norm1(w * (beta[idxs[i]] - beta[idxs[j]]))

        else:
            # Unpenalized variable; do nothing
            pass

    prob = cp.Problem(cp.Minimize(-ll + lam * (fused_pen + g_fused_pen)))
    return prob, beta, intercept, lam, input_var_onehot


def solve_problem(prob: cp.Problem, cfg: SolverConfig) -> None:
    """Solve a CVXPy problem with configured solver."""
    prob.solve(solver=getattr(cp, cfg.solver), warm_start=cfg.warm_start, verbose=cfg.verbose)


def coef_vector(intercept: cp.Variable, beta: cp.Variable) -> np.ndarray:
    """Return coefficients as [intercept, beta...]."""
    return np.concatenate(([float(intercept.value)], np.asarray(beta.value, dtype=float)))


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def get_data_ready_for_glm(
    data: pd.DataFrame,
    input_variables: Sequence[str],
    penalty_types: Dict[str, str],
    target: str,
    offset: str,
    method: str,
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """One-hot encode, reorder columns, attach original (non-dummy) inputs, target and offset."""
    data_onehot, ref_modality_dict, input_var_onehot = choose_ref_modality(data, input_variables, method)
    data_onehot = reorder_df_columns(data_onehot, penalty_types, input_variables)

    data_onehot = pd.concat(
        [
            data_onehot,
            data[input_variables].rename(columns={col: "non_dum_" + col for col in input_variables}),
        ],
        axis=1,
    )
    data_onehot[target] = data[target]
    data_onehot[offset] = data[offset]
    return data_onehot, ref_modality_dict, input_var_onehot


def group_modalities_based_on_betas(
    data: pd.DataFrame,
    modality_var_df: pd.DataFrame,
    beta_list: Sequence[float],
    input_variables: Sequence[str],
    ref_modalities: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Group categorical modalities with identical rounded coefficients.

    Returns:
        - updated data with <var>_grouped columns
        - list of kept variables as <var>_grouped
        - list of removed variables
    """
    chosen_model = np.asarray(beta_list[1:], dtype=float)  # remove intercept

    tmp = modality_var_df[modality_var_df["variable"].isin(input_variables)].copy()
    tmp["betas"] = [str(round(v, 6)) for v in chosen_model]

    grouped = (
        tmp.groupby(["variable", "betas"])["modality"]
        .apply(lambda x: "_".join(x.astype(str)))
        .reset_index()
        .rename(columns={"modality": "modality_grouped"})
    )
    tmp = tmp.merge(grouped, how="left", on=["variable", "betas"])

    kept, removed = [], []

    for var in input_variables:
        original_var = "non_dum_" + var
        one = tmp[tmp["variable"] == var].copy()

        if (np.abs(one["betas"].astype(float)) < 1e-6).all():
            removed.append(var)
            continue

        # Map original modality -> grouped label
        data[original_var] = data[original_var].astype(str)
        one["modality"] = one["modality"].astype(str)

        data = data.merge(one[["modality", "modality_grouped"]], how="left", left_on=original_var, right_on="modality")
        ref_mod = ref_modalities[var].replace(var + "_", "")
        data["modality_grouped"] = data["modality_grouped"].fillna(ref_mod)

        data.rename(columns={"modality_grouped": var + "_grouped"}, inplace=True)
        data.drop(columns=["modality"], inplace=True)
        kept.append(var)

    kept_grouped = [v + "_grouped" for v in kept]
    return data, kept_grouped, removed


def variables_kept_from_beta(
    beta_vec: np.ndarray,
    input_var_onehot: List[str],
    input_variables: Sequence[str],
    tol: float = 1e-6,
) -> List[str]:
    """Return variables whose one-hot coefficients are not all ~0."""
    beta_vec = np.asarray(beta_vec, dtype=float)
    kept: List[str] = []
    for var in input_variables:
        idx = [i for i, col in enumerate(input_var_onehot) if col.startswith(var + "_")]
        if idx and np.any(np.abs(beta_vec[idx]) > tol):
            kept.append(var)
    return kept


# -----------------------------------------------------------------------------
# Grid Search
# -----------------------------------------------------------------------------

class GridSearch_Generalised:
    """Grid search for (group_lambda, fused_lambda) with 2-stage procedure.

    Parameters mirror your original code:
    - Either provide (parcimony_step & smoothness_step) + var_nb_min/var_nb_max for auto-curve
    - Or provide explicit (lbd_group, lbd_fused) to evaluate directly.

    New optional parameters:
    - random_state for reproducible CV
    - solver configuration
    """

    def __init__(
        self,
        family: str,
        var_nb_min: Optional[int] = None,
        var_nb_max: Optional[int] = None,
        parcimony_step: Optional[int] = None,
        smoothness_step: Optional[int] = None,
        lbd_fused: Optional[object] = None,
        lbd_group: Optional[object] = None,
        random_state: Optional[int] = 0,
        solver: str = "CLARABEL",
        warm_start: bool = True,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        self.family = family
        self.var_nb_min = var_nb_min
        self.var_nb_max = var_nb_max
        self.parcimony_step = parcimony_step
        self.smoothness_step = smoothness_step
        self.lbd_group = lbd_group
        self.lbd_fused = lbd_fused

        self.random_state = random_state
        self.solver_cfg = SolverConfig(solver=solver, warm_start=warm_start, verbose=verbose)
        self.n_jobs = int(n_jobs) if n_jobs is not None else 1

        group1_any = (parcimony_step is not None) or (smoothness_step is not None)
        group1_full = (parcimony_step is not None) and (smoothness_step is not None)

        group2_any = (lbd_group is not None) or (lbd_fused is not None)
        group2_full = (lbd_group is not None) and (lbd_fused is not None)

        group3 = (var_nb_min is not None) or (var_nb_max is not None)

        if group1_full == group2_full:
            raise ValueError("Provide either (parcimony & smoothness) OR (group & fused) values, but not both.")

        if group1_any and not group1_full:
            raise ValueError("Both parcimony and smoothness values must be provided together.")
        if group2_any and not group2_full:
            raise ValueError("Both group and fused values must be provided together.")

        if group3 and group2_any:
            raise ValueError("var_nb_min/var_nb_max cannot be used when (group, fused) are fixed.")

        # caches
        self._group_full_cache = None
        self._fused_full_cache: Dict[Tuple[str, ...], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}
        self._group_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}
        self._fused_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}

    # -------------------------
    # Parallel fold workers
    # -------------------------

    def _cv_fold_deviance(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        kept_vars: List[str],
        fused_lambda_val: float,
    ) -> Tuple[float, float]:
        """Compute (train_dev, test_dev) for one fold at a fixed fused_lambda.

        Notes
        -----
        This method is designed to be safe for *threaded* parallelism:
        each fold uses its own cached CVXPy Problem/Variables/Parameters.
        """
        train_df = self.data_onehot.iloc[train_idx]
        test_df = self.data_onehot.iloc[test_idx]

        y_train = train_df[self.target].to_numpy(dtype=float)
        y_test = test_df[self.target].to_numpy(dtype=float)
        off_train = train_df[self.offset].to_numpy(dtype=float)
        off_test = test_df[self.offset].to_numpy(dtype=float)

        if kept_vars:
            # Use per-fold cached fused problem (created outside the threads)
            prob, beta, intercept, lam, _ = self._get_fused_fold_problem(fold_id, train_idx, kept_vars)
            lam.value = float(fused_lambda_val)
            solve_problem(prob, self.solver_cfg)

            b = np.asarray(beta.value, dtype=float)
            onehot_cols = get_onehot_columns(train_df, kept_vars)  # stable order; cheap
            X_train = train_df[onehot_cols].to_numpy(dtype=float)
            X_test = test_df[onehot_cols].to_numpy(dtype=float)

            mu_train = np.exp(X_train @ b + float(intercept.value) + off_train)
            mu_test = np.exp(X_test @ b + float(intercept.value) + off_test)
        else:
            mu_train, mu_test = self._intercept_only_poisson_predictions(y_train, off_train, off_test)

        exp_train = np.exp(off_train)
        exp_test = np.exp(off_test)
        train_dev = poisson_deviance_mu(mu_train, y_train, exposure=exp_train)
        test_dev = poisson_deviance_mu(mu_test, y_test, exposure=exp_test)
        return train_dev, test_dev


    @time_it
    def fit(self, data: pd.DataFrame, penalty_types: Dict[str, str], input_variables: List[str], target: str, offset: str, n_k_fold: int = 5):
        self.data = data
        self.penalty_types = penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.target = target
        self.offset = offset

        self.data_onehot, self.ref_modality_dict, _ = get_data_ready_for_glm(
            self.data, self.input_variables, self.penalty_types, target, offset, "First"
        )

        # Precompute CV splits once
        self.cv_splits = list(KFold(n_splits=self.n_k_fold, shuffle=True, random_state=self.random_state).split(self.data_onehot))

        # Build modality mapping once (aligns with one-hot column order)
        modality_arr, variables_arr = [], []
        for var in self.input_variables:
            for col in self.data_onehot.columns:
                if col.startswith(var + "_"):
                    modality_arr.append(col.replace(var + "_", ""))
                    variables_arr.append(var)
        self.modality_var_df = pd.DataFrame({"variable": variables_arr, "modality": modality_arr})

        if (self.lbd_group is None) and (self.lbd_fused is None):
            self.get_lambda_curve()
        else:
            results = self._init_results_table(1)

            if isinstance(self.lbd_group, (list, tuple)):
                if len(self.lbd_group) != len(self.lbd_fused):
                    raise ValueError(f"Fused and Group arrays must have the same size, got {len(self.lbd_group)} and {len(self.lbd_fused)}")

                for i, (g, f) in enumerate(zip(self.lbd_group, self.lbd_fused)):
                    kept_per_fold = self._precompute_group_selection_per_fold(self.input_variables, float(g), fold_cache=True)
                    results = self.crossval_pair_lambdas(results, self.input_variables, float(g), float(f), i, kept_per_fold=kept_per_fold)
            else:
                kept_per_fold = self._precompute_group_selection_per_fold(self.input_variables, float(self.lbd_group), fold_cache=True)
                results = self.crossval_pair_lambdas(results, self.input_variables, float(self.lbd_group), float(self.lbd_fused), 0, kept_per_fold=kept_per_fold)

            self.lambda_curve = results

    # -------------------------
    # Core algorithm helpers
    # -------------------------

    def _init_results_table(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "group_lambda",
                "fused_lambda",
                "variable_number",
                "modalities_number",
                "var_mod_details",
                "Deviance_cv_train",
                "Deviance_cv_test",
                "variables",
                "betas",
            ],
            data=np.empty((n_rows, 9), dtype=object),
        )

    def _get_group_full_problem(self):
        """Cache group-lasso problem on full data for reuse across lambdas."""
        if self._group_full_cache is not None:
            return self._group_full_cache

        prob, beta, intercept, lam, onehot_cols, _ = build_group_lasso_problem(
            self.data_onehot,
            self.input_variables,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._group_full_cache = (prob, beta, intercept, lam, onehot_cols)
        return self._group_full_cache

    def _get_fused_full_problem(self, inputs: List[str]):
        """Cache fused problem on full data per variable list."""
        key = tuple(sorted(inputs))
        if key in self._fused_full_cache:
            return self._fused_full_cache[key]

        prob, beta, intercept, lam, onehot_cols = build_fused_problem(
            self.data_onehot,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._fused_full_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._fused_full_cache[key]

    def _get_group_fold_problem(self, fold_id: int, train_idx: np.ndarray, inputs: List[str]):
        """Cache group-lasso problem per fold and input set."""
        key = (fold_id, tuple(sorted(inputs)))
        if key in self._group_fold_cache:
            return self._group_fold_cache[key]

        X_train = self.data_onehot.iloc[train_idx]
        prob, beta, intercept, lam, onehot_cols, _ = build_group_lasso_problem(
            X_train, inputs, self.target, self.offset, self.penalty_types, family=self.family
        )
        self._group_fold_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._group_fold_cache[key]

    def _get_fused_fold_problem(self, fold_id: int, train_idx: np.ndarray, kept_vars: List[str]):
        """Cache fused problem per fold and kept variable set."""
        key = (fold_id, tuple(sorted(kept_vars)))
        if key in self._fused_fold_cache:
            return self._fused_fold_cache[key]

        X_train = self.data_onehot.iloc[train_idx]
        prob, beta, intercept, lam, onehot_cols = build_fused_problem(
            X_train, kept_vars, self.target, self.offset, self.penalty_types, family=self.family
        )
        self._fused_fold_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._fused_fold_cache[key]

    def _intercept_only_poisson_predictions(self, y_train: np.ndarray, offset_train: np.ndarray, offset_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Offset-aware intercept-only Poisson baseline: μ = rate * exp(offset)."""
        exp_train = np.exp(offset_train)
        exp_test = np.exp(offset_test)
        rate = y_train.sum() / max(exp_train.sum(), 1e-12)
        return rate * exp_train, rate * exp_test

    def _precompute_group_selection_per_fold(self, inputs: List[str], group_lambda: float, fold_cache: bool = True) -> List[Tuple[int, np.ndarray, np.ndarray, List[str]]]:
        """Compute kept variables per fold for a fixed group_lambda (reused across fused lambdas).

        Returns list of (fold_id, train_idx, test_idx, kept_vars).
        """
        if not fold_cache:
            # Keep the non-cached path simple and sequential (safe, but slower).
            kept_per_fold: List[Tuple[int, np.ndarray, np.ndarray, List[str]]] = []
            for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits):
                X_train = self.data_onehot.iloc[train_idx]
                prob, beta, intercept, lam, onehot_cols, _ = build_group_lasso_problem(
                    X_train, inputs, self.target, self.offset, self.penalty_types, family=self.family
                )
                lam.value = float(group_lambda)
                solve_problem(prob, self.solver_cfg)
                kept_vars = variables_kept_from_beta(np.asarray(beta.value, dtype=float), onehot_cols, inputs)
                kept_per_fold.append((fold_id, train_idx, test_idx, kept_vars))
            return kept_per_fold

        # Cached path: optionally parallelize across folds with threads.
        # Important: we pre-create each fold's CVXPy objects before running threads to avoid
        # concurrent writes to the cache dict.
        for fold_id, (train_idx, _) in enumerate(self.cv_splits):
            self._get_group_fold_problem(fold_id, train_idx, inputs)

        def _solve_one_fold(fold_id: int, train_idx: np.ndarray, test_idx: np.ndarray):
            prob, beta, _, lam, onehot_cols = self._get_group_fold_problem(fold_id, train_idx, inputs)
            lam.value = float(group_lambda)
            solve_problem(prob, self.solver_cfg)
            b = np.asarray(beta.value, dtype=float)
            kept_vars = variables_kept_from_beta(b, onehot_cols, inputs)
            return fold_id, train_idx, test_idx, kept_vars

        if self.n_jobs and self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futs = [
                    ex.submit(_solve_one_fold, fold_id, train_idx, test_idx)
                    for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
                ]
                out = [f.result() for f in futs]
        else:
            out = [
                _solve_one_fold(fold_id, train_idx, test_idx)
                for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
            ]

        # Keep ordering stable by fold_id
        out.sort(key=lambda t: t[0])
        return out

    # -------------------------
    # Training / evaluation
    # -------------------------

    @time_it
    def fit_one_lambda(self, data: pd.DataFrame, inputs: List[str], lbd_fused: float, lbd_group: float):
        """Fit model at one (group_lambda, fused_lambda) and return final coefficients."""
        # Group-lasso on full data (cached)
        prob_g, beta_g, intercept_g, lam_g, onehot_cols_g = self._get_group_full_problem()
        lam_g.value = float(lbd_group)
        solve_problem(prob_g, self.solver_cfg)

        kept_vars = variables_kept_from_beta(np.asarray(beta_g.value, dtype=float), onehot_cols_g, inputs)

        # Fused model on full data for these vars (cached)
        if kept_vars:
            prob_f, beta_f, intercept_f, lam_f, onehot_cols_f = self._get_fused_full_problem(kept_vars)
            lam_f.value = float(lbd_fused)
            solve_problem(prob_f, self.solver_cfg)
            temp_glm = coef_vector(intercept_f, beta_f)

            # Group modalities (this is the expensive part; do it only here)
            temp_data, kept_grouped, _ = group_modalities_based_on_betas(
                data.copy(), self.modality_var_df, temp_glm, kept_vars, self.ref_modality_dict
            )

            if kept_grouped:
                temp_data_onehot, temp_ref_modality_dict, _ = get_data_ready_for_glm(
                    temp_data, kept_grouped, self.penalty_types, self.target, self.offset, "First"
                )

                # Unpenalized refit on grouped variables
                betas = self._fit_glm_no_pen(temp_data_onehot, kept_grouped)
                return len(temp_ref_modality_dict), temp_ref_modality_dict, betas, kept_grouped

        # No variables kept
        betas = np.array([data[self.target].mean()], dtype=float)
        return 0, {}, betas, []

    def _fit_glm_no_pen(self, data_onehot: pd.DataFrame, inputs: List[str]) -> np.ndarray:
        """Unpenalized Poisson GLM (CVXPy) on final grouped design."""
        onehot_cols = get_onehot_columns(data_onehot, inputs)
        X = data_onehot[onehot_cols].to_numpy(dtype=float)
        y = data_onehot[self.target].to_numpy(dtype=float)
        offset_arr = data_onehot[self.offset].to_numpy(dtype=float)

        beta = cp.Variable(X.shape[1])
        intercept = cp.Variable()
        ll = _log_likelihood(self.family, X, y, offset_arr, beta, intercept)

        prob = cp.Problem(cp.Minimize(-ll))
        solve_problem(prob, self.solver_cfg)
        return coef_vector(intercept, beta)

    @time_it
    def crossval_pair_lambdas(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        group_lambda_val: float,
        fused_lambda_val: float,
        counter: int,
        kept_per_fold: Optional[List[Tuple[int, np.ndarray, np.ndarray, List[str]]]] = None,
    ) -> pd.DataFrame:
        """Evaluate one (group_lambda, fused_lambda) pair via K-fold CV and store results."""
        print(f"Model with group lambda = {group_lambda_val} and fused lambda = {fused_lambda_val}")

        if kept_per_fold is None:
            kept_per_fold = self._precompute_group_selection_per_fold(inputs, group_lambda_val, fold_cache=True)

        err_train = 0.0
        err_test = 0.0

        # If using threads, ensure fused fold problems are created before the threads
        # (avoid concurrent cache writes).
        if self.n_jobs and self.n_jobs > 1:
            for fold_id, train_idx, _, kept_vars in kept_per_fold:
                if kept_vars:
                    self._get_fused_fold_problem(fold_id, train_idx, kept_vars)

            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futs = [
                    ex.submit(self._cv_fold_deviance, fold_id, train_idx, test_idx, kept_vars, fused_lambda_val)
                    for fold_id, train_idx, test_idx, kept_vars in kept_per_fold
                ]
                for f in futs:
                    tr, te = f.result()
                    err_train += tr
                    err_test += te
        else:
            for fold_id, train_idx, test_idx, kept_vars in kept_per_fold:
                tr, te = self._cv_fold_deviance(fold_id, train_idx, test_idx, kept_vars, fused_lambda_val)
                err_train += tr
                err_test += te

        err_train /= self.n_k_fold
        err_test /= self.n_k_fold

        results.at[counter, "fused_lambda"] = fused_lambda_val
        results.at[counter, "group_lambda"] = group_lambda_val
        results.at[counter, "Deviance_cv_train"] = err_train
        results.at[counter, "Deviance_cv_test"] = err_test

        # Final fit on full data
        nb_mod, mod_dict, betas, kept = self.fit_one_lambda(self.data_onehot, inputs, fused_lambda_val, group_lambda_val)
        results.at[counter, "modalities_number"] = nb_mod
        results.at[counter, "var_mod_details"] = mod_dict
        results.at[counter, "betas"] = betas
        results.at[counter, "variables"] = " ".join(map(str, kept))
        results.at[counter, "variable_number"] = len(kept)

        results = results.sort_values(by=["group_lambda", "fused_lambda"], ascending=True).reset_index(drop=True)
        return results

    # -------------------------
    # Lambda curve exploration
    # -------------------------

    @time_it
    def get_lambda_curve(self):
        """Run your 2-stage lambda curve exploration (cleaned & faster)."""

        if self.parcimony_step is None or self.smoothness_step is None:
            raise ValueError("parcimony_step and smoothness_step must be provided for lambda curve exploration.")

        if self.var_nb_min is None:
            raise ValueError("var_nb_min must be provided for lambda curve exploration.")

        # ---------------- Step 1: group lambda range ----------------
        print("----------------------------------")
        print("Step 1 : Get range of Group Lambdas")
        print("----------------------------------\n")

        max_lambda_group_reached = False
        lambda_group_temp = 0.1
        counter = 0

        grouped_lasso_table = pd.DataFrame(columns=["lambda", "var_nb", "variables"], data=np.empty((1, 3), dtype=object))

        prob, beta, intercept, lam, onehot_cols = self._get_group_full_problem()

        while not max_lambda_group_reached:
            print(f"Running group GLMs until reaching maximum lambda : {lambda_group_temp}")

            lam.value = float(lambda_group_temp)
            solve_problem(prob, self.solver_cfg)

            kept_vars = variables_kept_from_beta(np.asarray(beta.value, dtype=float), onehot_cols, self.input_variables)
            grouped_lasso_table.at[counter, "lambda"] = lambda_group_temp
            grouped_lasso_table.at[counter, "var_nb"] = len(kept_vars)
            grouped_lasso_table.at[counter, "variables"] = kept_vars
            grouped_lasso_table = grouped_lasso_table.sort_values("lambda", ascending=True).reset_index(drop=True)

            # FIX: boolean logic (was using bitwise |)
            if (len(kept_vars) == 0) or (len(kept_vars) <= self.var_nb_min):
                max_lambda_group_reached = True
                max_lambda_group = float(grouped_lasso_table["lambda"].max())

            counter += 1
            lambda_group_temp *= 10

        # Fill intermediate lambdas linearly
        jump = (max_lambda_group - float(grouped_lasso_table.loc[0, "lambda"])) / float(self.parcimony_step)
        tested_lambda_list = [float(grouped_lasso_table.loc[0, "lambda"]) + jump * step for step in range(1, self.parcimony_step)]

        for lambda_group_temp in tested_lambda_list:
            print(f"Running group GLMs with lambda = {lambda_group_temp}")
            lam.value = float(lambda_group_temp)
            solve_problem(prob, self.solver_cfg)

            kept_vars = variables_kept_from_beta(np.asarray(beta.value, dtype=float), onehot_cols, self.input_variables)
            grouped_lasso_table.at[counter, "lambda"] = lambda_group_temp
            grouped_lasso_table.at[counter, "var_nb"] = len(kept_vars)
            grouped_lasso_table.at[counter, "variables"] = kept_vars
            grouped_lasso_table = grouped_lasso_table.sort_values("lambda", ascending=True).reset_index(drop=True)
            counter += 1

        # Deduplicate variable lists
        grouped_lasso_table["variables_tuple"] = grouped_lasso_table["variables"].apply(frozenset)
        grouped_lasso_uniques = grouped_lasso_table.groupby("variables_tuple")["lambda"].mean().reset_index()
        grouped_lasso_uniques["variables"] = grouped_lasso_uniques["variables_tuple"].apply(lambda x: sorted(list(x)))
        grouped_lasso_uniques = grouped_lasso_uniques.drop(columns="variables_tuple")

        variable_lists = [lst for lst in grouped_lasso_uniques["variables"] if lst]

        print("Group GLMs gives the following variable lists :")
        for arr in variable_lists:
            print(arr)

        # ---------------- Step 2: fused range per variable list ----------------
        print("\n----------------------------------")
        print("Step 2 : Iterate on each variable list\n")

        results = self._init_results_table(1)
        k = 0

        for m in range(len(grouped_lasso_uniques)):
            input_list_temp = grouped_lasso_uniques.loc[m, "variables"]
            if not input_list_temp:
                print("Skipping empty variable list.")
                continue

            group_lambda_temp = float(grouped_lasso_uniques.loc[m, "lambda"])
            print(f"Fused GLMs on this variable list : {input_list_temp}")

            # Step 2.1: find fused lambda max where all vars drop
            print("\n Step 2.1 : For each variable list, get range of Fused Lambdas\n")
            max_lambda_reached = False
            lambda_temp = 0.1
            row = 0

            tmp = pd.DataFrame(columns=["fused_lambda", "variable_number"], data=np.empty((1, 2), dtype=object))

            prob_f, beta_f, intercept_f, lam_f, onehot_cols_f = self._get_fused_full_problem(input_list_temp)

            while not max_lambda_reached:
                lam_f.value = float(lambda_temp)
                solve_problem(prob_f, self.solver_cfg)

                kept_vars = variables_kept_from_beta(np.asarray(beta_f.value, dtype=float), onehot_cols_f, input_list_temp)
                tmp.at[row, "fused_lambda"] = float(lambda_temp)
                tmp.at[row, "variable_number"] = len(kept_vars)

                if len(kept_vars) == 0:
                    max_lambda_reached = True
                    lambda_max = float(lambda_temp)

                lambda_temp *= 10
                row += 1

            jump = (lambda_max - float(tmp.loc[0, "fused_lambda"])) / float(self.smoothness_step)
            tested_fused_lambda_list = [float(tmp.loc[0, "fused_lambda"]) + jump * step for step in range(0, self.smoothness_step + 1)]

            # Step 2.2: CV for candidate fused lambdas (reuse group selection per fold)
            kept_per_fold = self._precompute_group_selection_per_fold(input_list_temp, group_lambda_temp, fold_cache=True)

            for fused_lambda_temp in tested_fused_lambda_list:
                print("\n Step 2.2 : For selected group and fused lambdas, get cross-val")
                results = self.crossval_pair_lambdas(
                    results,
                    input_list_temp,
                    group_lambda_temp,
                    float(fused_lambda_temp),
                    k,
                    kept_per_fold=kept_per_fold,
                )
                k += 1

            print(f"End of curve for {input_list_temp}")

        self.lambda_curve = results

    # -------------------------
    # Plotting (unchanged)
    # -------------------------

    def plot_curve(self):
        """Scatter plot: variable_number vs Deviance_cv_test, colored by group_lambda."""
        categories = self.lambda_curve["group_lambda"].unique()
        colors = plt.cm.tab10(range(len(categories)))
        plt.figure(figsize=(8, 6))

        for cat, color in zip(categories, colors):
            subset = self.lambda_curve[self.lambda_curve["group_lambda"] == cat]
            plt.scatter(subset["variable_number"], subset["Deviance_cv_test"], color=color, label=str(np.around(cat)))

            for _, row in subset.iterrows():
                label = f"{np.around(row['group_lambda'])} / {np.around(row['fused_lambda'])}"
                plt.text(row["variable_number"], row["Deviance_cv_test"], label, fontsize=8, ha="center", va="bottom")

        plt.xlabel("Final Variable Number")
        plt.ylabel("Deviance cv test")
        plt.title("Scatter plot with discrete colors for group lambda value")
        plt.legend(title="Group Lambda")
        plt.tight_layout()
        plt.show()


# =============================================================================
# Single-penalty grid searches
# =============================================================================


def _predict_mean_from_linear_predictor(family: str, eta: np.ndarray) -> np.ndarray:
    """Return the mean prediction given linear predictor eta."""
    fam = family.lower()
    if fam == "poisson":
        return np.exp(eta)
    if fam == "gaussian":
        return eta
    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def _loss_from_predictions(
    family: str,
    y: np.ndarray,
    pred: np.ndarray,
    offset: Optional[np.ndarray] = None,
) -> float:
    """Return the per-sample loss used in CV.

    - Poisson: scaled deviance (matching this module's conventions)
    - Gaussian: mean squared error
    """
    fam = family.lower()
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if fam == "poisson":
        exposure = None if offset is None else np.exp(np.asarray(offset, dtype=float))
        return poisson_deviance_mu(pred, y, exposure=exposure)

    if fam == "gaussian":
        return float(np.mean((y - pred) ** 2))

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def _intercept_only_predictions(
    family: str,
    y_train: np.ndarray,
    offset_train: np.ndarray,
    offset_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intercept-only baseline for supported families (offset-aware)."""
    fam = family.lower()
    y_train = np.asarray(y_train, dtype=float)
    off_train = np.asarray(offset_train, dtype=float)
    off_test = np.asarray(offset_test, dtype=float)

    if fam == "poisson":
        exp_train = np.exp(off_train)
        exp_test = np.exp(off_test)
        rate = y_train.sum() / max(exp_train.sum(), 1e-12)
        return rate * exp_train, rate * exp_test

    if fam == "gaussian":
        # eta = intercept + offset, so best intercept is mean(y - offset)
        intercept = float(np.mean(y_train - off_train))
        return intercept + off_train, intercept + off_test

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def _beta_subvector_for_vars(
    beta_full: np.ndarray,
    full_onehot_cols: List[str],
    data_onehot: pd.DataFrame,
    kept_vars: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Extract beta subvector aligned with one-hot columns for `kept_vars`.

    This is required before calling `group_modalities_based_on_betas`:
    that routine assumes the provided beta list aligns *exactly* with the
    modalities for the variables passed in.
    """
    kept_onehot_cols = get_onehot_columns(data_onehot, kept_vars)
    col_to_idx = {c: i for i, c in enumerate(full_onehot_cols)}
    idx = [col_to_idx[c] for c in kept_onehot_cols]
    return np.asarray(beta_full, dtype=float)[idx], kept_onehot_cols


class GridSearch_Group:
    """Grid search for a *group-only* penalized GLM.

    This mirrors the public API of `GridSearch_Generalised`, but it evaluates
    only the group penalty (no fused stage).

    Outputs are stored in `self.lambda_curve` with the same schema as
    `GridSearch_Generalised` (with `fused_lambda` set to 0.0).
    """

    def __init__(
        self,
        family: str,
        var_nb_min: Optional[int] = None,
        var_nb_max: Optional[int] = None,
        parcimony_step: Optional[int] = None,
        lbd_group: Optional[object] = None,
        random_state: Optional[int] = 0,
        solver: str = "CLARABEL",
        warm_start: bool = True,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        self.family = family
        self.var_nb_min = var_nb_min
        self.var_nb_max = var_nb_max
        self.parcimony_step = parcimony_step
        self.lbd_group = lbd_group

        self.random_state = random_state
        self.solver_cfg = SolverConfig(solver=solver, warm_start=warm_start, verbose=verbose)
        self.n_jobs = int(n_jobs) if n_jobs is not None else 1

        auto_any = parcimony_step is not None
        fixed_any = lbd_group is not None

        if auto_any == fixed_any:
            raise ValueError("Provide either parcimony_step (auto curve) OR lbd_group (fixed), but not both.")

        if auto_any and (self.var_nb_min is None):
            raise ValueError("var_nb_min must be provided for lambda curve exploration.")

        # caches
        self._group_full_cache = None
        self._group_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}

    # -------------------------
    # Public API
    # -------------------------

    @time_it
    def fit(
        self,
        data: pd.DataFrame,
        penalty_types: Dict[str, str],
        input_variables: List[str],
        target: str,
        offset: str,
        n_k_fold: int = 5,
    ):
        self.data = data
        self.penalty_types = penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.target = target
        self.offset = offset

        self.data_onehot, self.ref_modality_dict, _ = get_data_ready_for_glm(
            self.data, self.input_variables, self.penalty_types, target, offset, "First"
        )

        self.cv_splits = list(KFold(n_splits=self.n_k_fold, shuffle=True, random_state=self.random_state).split(self.data_onehot))

        modality_arr, variables_arr = [], []
        for var in self.input_variables:
            for col in self.data_onehot.columns:
                if col.startswith(var + "_"):
                    modality_arr.append(col.replace(var + "_", ""))
                    variables_arr.append(var)
        self.modality_var_df = pd.DataFrame({"variable": variables_arr, "modality": modality_arr})

        if self.lbd_group is None:
            self.get_lambda_curve()
            return

        # Fixed evaluation(s)
        if isinstance(self.lbd_group, (list, tuple, np.ndarray)):
            lambdas = [float(x) for x in self.lbd_group]
        else:
            lambdas = [float(self.lbd_group)]

        results = self._init_results_table(len(lambdas))
        for i, g in enumerate(lambdas):
            results = self.crossval_group_lambda(results, self.input_variables, g, i)
        self.lambda_curve = results

    # -------------------------
    # Internals
    # -------------------------

    def _init_results_table(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "group_lambda",
                "fused_lambda",
                "variable_number",
                "modalities_number",
                "var_mod_details",
                "Deviance_cv_train",
                "Deviance_cv_test",
                "variables",
                "betas",
            ],
            data=np.empty((n_rows, 9), dtype=object),
        )

    def _get_group_full_problem(self):
        if self._group_full_cache is not None:
            return self._group_full_cache

        prob, beta, intercept, lam, onehot_cols, _ = build_group_lasso_problem(
            self.data_onehot,
            self.input_variables,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._group_full_cache = (prob, beta, intercept, lam, onehot_cols)
        return self._group_full_cache

    def _get_group_fold_problem(self, fold_id: int, train_idx: np.ndarray, inputs: List[str]):
        key = (fold_id, tuple(sorted(inputs)))
        if key in self._group_fold_cache:
            return self._group_fold_cache[key]

        X_train = self.data_onehot.iloc[train_idx]
        prob, beta, intercept, lam, onehot_cols, _ = build_group_lasso_problem(
            X_train, inputs, self.target, self.offset, self.penalty_types, family=self.family
        )
        self._group_fold_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._group_fold_cache[key]

    def _cv_fold_loss_group(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        inputs: List[str],
        group_lambda_val: float,
    ) -> Tuple[float, float]:
        train_df = self.data_onehot.iloc[train_idx]
        test_df = self.data_onehot.iloc[test_idx]

        y_train = train_df[self.target].to_numpy(dtype=float)
        y_test = test_df[self.target].to_numpy(dtype=float)
        off_train = train_df[self.offset].to_numpy(dtype=float)
        off_test = test_df[self.offset].to_numpy(dtype=float)

        prob, beta, intercept, lam, onehot_cols = self._get_group_fold_problem(fold_id, train_idx, inputs)
        lam.value = float(group_lambda_val)
        solve_problem(prob, self.solver_cfg)

        b = np.asarray(beta.value, dtype=float)
        X_train = train_df[onehot_cols].to_numpy(dtype=float)
        X_test = test_df[onehot_cols].to_numpy(dtype=float)

        eta_train = X_train @ b + float(intercept.value) + off_train
        eta_test = X_test @ b + float(intercept.value) + off_test

        pred_train = _predict_mean_from_linear_predictor(self.family, eta_train)
        pred_test = _predict_mean_from_linear_predictor(self.family, eta_test)

        train_loss = _loss_from_predictions(self.family, y_train, pred_train, offset=off_train)
        test_loss = _loss_from_predictions(self.family, y_test, pred_test, offset=off_test)
        return train_loss, test_loss

    def _fit_glm_no_pen(self, data_onehot: pd.DataFrame, inputs: List[str]) -> np.ndarray:
        onehot_cols = get_onehot_columns(data_onehot, inputs)
        X = data_onehot[onehot_cols].to_numpy(dtype=float)
        y = data_onehot[self.target].to_numpy(dtype=float)
        offset_arr = data_onehot[self.offset].to_numpy(dtype=float)

        beta = cp.Variable(X.shape[1])
        intercept = cp.Variable()
        ll = _log_likelihood(self.family, X, y, offset_arr, beta, intercept)

        prob = cp.Problem(cp.Minimize(-ll))
        solve_problem(prob, self.solver_cfg)
        return coef_vector(intercept, beta)

    @time_it
    def fit_one_lambda(self, data: pd.DataFrame, inputs: List[str], lbd_group: float):
        """Fit a group-only model at one lambda and return final coefficients."""
        prob_g, beta_g, intercept_g, lam_g, onehot_cols_g = self._get_group_full_problem()
        lam_g.value = float(lbd_group)
        solve_problem(prob_g, self.solver_cfg)

        kept_vars = variables_kept_from_beta(np.asarray(beta_g.value, dtype=float), onehot_cols_g, inputs)

        if kept_vars:
            beta_sub, _ = _beta_subvector_for_vars(
                np.asarray(beta_g.value, dtype=float),
                onehot_cols_g,
                self.data_onehot,
                kept_vars,
            )
            temp_glm = np.concatenate(([float(intercept_g.value)], beta_sub))

            temp_data, kept_grouped, _ = group_modalities_based_on_betas(
                data.copy(), self.modality_var_df, temp_glm, kept_vars, self.ref_modality_dict
            )

            if kept_grouped:
                temp_data_onehot, temp_ref_modality_dict, _ = get_data_ready_for_glm(
                    temp_data, kept_grouped, self.penalty_types, self.target, self.offset, "First"
                )
                betas = self._fit_glm_no_pen(temp_data_onehot, kept_grouped)
                return len(temp_ref_modality_dict), temp_ref_modality_dict, betas, kept_grouped

        # No variables kept
        if self.family.lower() == "poisson":
            betas = np.array([data[self.target].mean()], dtype=float)
        else:
            # eta = intercept + offset, so baseline intercept is mean(y - offset)
            betas = np.array([float(np.mean(data[self.target].to_numpy(dtype=float) - data[self.offset].to_numpy(dtype=float)))], dtype=float)
        return 0, {}, betas, []

    @time_it
    def crossval_group_lambda(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        group_lambda_val: float,
        counter: int,
    ) -> pd.DataFrame:
        print(f"Group-only model with group lambda = {group_lambda_val}")

        err_train = 0.0
        err_test = 0.0

        # Ensure fold problems exist before threaded execution
        if self.n_jobs and self.n_jobs > 1:
            for fold_id, (train_idx, _) in enumerate(self.cv_splits):
                self._get_group_fold_problem(fold_id, train_idx, inputs)

            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futs = [
                    ex.submit(self._cv_fold_loss_group, fold_id, train_idx, test_idx, inputs, group_lambda_val)
                    for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
                ]
                for f in futs:
                    tr, te = f.result()
                    err_train += tr
                    err_test += te
        else:
            for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits):
                tr, te = self._cv_fold_loss_group(fold_id, train_idx, test_idx, inputs, group_lambda_val)
                err_train += tr
                err_test += te

        err_train /= self.n_k_fold
        err_test /= self.n_k_fold

        results.at[counter, "group_lambda"] = group_lambda_val
        results.at[counter, "fused_lambda"] = 0.0
        results.at[counter, "Deviance_cv_train"] = err_train
        results.at[counter, "Deviance_cv_test"] = err_test

        nb_mod, mod_dict, betas, kept = self.fit_one_lambda(self.data_onehot, inputs, group_lambda_val)
        results.at[counter, "modalities_number"] = nb_mod
        results.at[counter, "var_mod_details"] = mod_dict
        results.at[counter, "betas"] = betas
        results.at[counter, "variables"] = " ".join(map(str, kept))
        results.at[counter, "variable_number"] = len(kept)

        results = results.sort_values(by=["group_lambda"], ascending=True).reset_index(drop=True)
        return results

    @time_it
    def get_lambda_curve(self):
        """Explore a range of group lambdas and evaluate each via CV."""
        if self.parcimony_step is None:
            raise ValueError("parcimony_step must be provided for lambda curve exploration.")

        print("----------------------------------")
        print("Group-only: Get range of Group Lambdas")
        print("----------------------------------\n")

        max_lambda_reached = False
        lambda_temp = 0.1

        prob, beta, intercept, lam, onehot_cols = self._get_group_full_problem()

        while not max_lambda_reached:
            print(f"Running group GLMs until reaching maximum lambda : {lambda_temp}")
            lam.value = float(lambda_temp)
            solve_problem(prob, self.solver_cfg)

            kept_vars = variables_kept_from_beta(np.asarray(beta.value, dtype=float), onehot_cols, self.input_variables)

            if (len(kept_vars) == 0) or (len(kept_vars) <= int(self.var_nb_min)):
                max_lambda_reached = True
                lambda_max = float(lambda_temp)
            else:
                lambda_temp *= 10

        lambda_min = 0.1
        jump = (lambda_max - lambda_min) / float(self.parcimony_step)
        tested_lambda_list = [lambda_min + jump * step for step in range(0, self.parcimony_step + 1)]

        results = self._init_results_table(len(tested_lambda_list))
        for i, g in enumerate(tested_lambda_list):
            results = self.crossval_group_lambda(results, self.input_variables, float(g), i)

        self.lambda_curve = results

    def plot_curve(self):
        """Scatter plot: variable_number vs CV loss, with lambda labels."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.lambda_curve["variable_number"], self.lambda_curve["Deviance_cv_test"])

        for _, row in self.lambda_curve.iterrows():
            label = f"{np.around(row['group_lambda'])}"
            plt.text(row["variable_number"], row["Deviance_cv_test"], label, fontsize=8, ha="center", va="bottom")

        plt.xlabel("Final Variable Number")
        plt.ylabel("CV loss (Poisson deviance or Gaussian MSE)")
        plt.title("Group-only grid search")
        plt.tight_layout()
        plt.show()


class GridSearch_Fused:
    """Grid search for a *fused-only* penalized GLM.

    This mirrors the public API of `GridSearch_Generalised`, but it evaluates
    only the fused (or g_fused) penalty (no group selection stage).

    Outputs are stored in `self.lambda_curve` with the same schema as
    `GridSearch_Generalised` (with `group_lambda` set to 0.0).
    """

    def __init__(
        self,
        family: str,
        smoothness_step: Optional[int] = None,
        lbd_fused: Optional[object] = None,
        random_state: Optional[int] = 0,
        solver: str = "CLARABEL",
        warm_start: bool = True,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        self.family = family
        self.smoothness_step = smoothness_step
        self.lbd_fused = lbd_fused

        self.random_state = random_state
        self.solver_cfg = SolverConfig(solver=solver, warm_start=warm_start, verbose=verbose)
        self.n_jobs = int(n_jobs) if n_jobs is not None else 1

        auto_any = smoothness_step is not None
        fixed_any = lbd_fused is not None

        if auto_any == fixed_any:
            raise ValueError("Provide either smoothness_step (auto curve) OR lbd_fused (fixed), but not both.")

        # caches
        self._fused_full_cache: Dict[Tuple[str, ...], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}
        self._fused_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}

    # -------------------------
    # Public API
    # -------------------------

    @time_it
    def fit(
        self,
        data: pd.DataFrame,
        penalty_types: Dict[str, str],
        input_variables: List[str],
        target: str,
        offset: str,
        n_k_fold: int = 5,
    ):
        self.data = data
        self.penalty_types = penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.target = target
        self.offset = offset

        self.data_onehot, self.ref_modality_dict, _ = get_data_ready_for_glm(
            self.data, self.input_variables, self.penalty_types, target, offset, "First"
        )

        self.cv_splits = list(KFold(n_splits=self.n_k_fold, shuffle=True, random_state=self.random_state).split(self.data_onehot))

        modality_arr, variables_arr = [], []
        for var in self.input_variables:
            for col in self.data_onehot.columns:
                if col.startswith(var + "_"):
                    modality_arr.append(col.replace(var + "_", ""))
                    variables_arr.append(var)
        self.modality_var_df = pd.DataFrame({"variable": variables_arr, "modality": modality_arr})

        if self.lbd_fused is None:
            self.get_lambda_curve()
            return

        # Fixed evaluation(s)
        if isinstance(self.lbd_fused, (list, tuple, np.ndarray)):
            lambdas = [float(x) for x in self.lbd_fused]
        else:
            lambdas = [float(self.lbd_fused)]

        results = self._init_results_table(len(lambdas))
        for i, f in enumerate(lambdas):
            results = self.crossval_fused_lambda(results, self.input_variables, f, i)
        self.lambda_curve = results

    # -------------------------
    # Internals
    # -------------------------

    def _init_results_table(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "group_lambda",
                "fused_lambda",
                "variable_number",
                "modalities_number",
                "var_mod_details",
                "Deviance_cv_train",
                "Deviance_cv_test",
                "variables",
                "betas",
            ],
            data=np.empty((n_rows, 9), dtype=object),
        )

    def _get_fused_full_problem(self, inputs: List[str]):
        key = tuple(sorted(inputs))
        if key in self._fused_full_cache:
            return self._fused_full_cache[key]

        prob, beta, intercept, lam, onehot_cols = build_fused_problem(
            self.data_onehot,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._fused_full_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._fused_full_cache[key]

    def _get_fused_fold_problem(self, fold_id: int, train_idx: np.ndarray, inputs: List[str]):
        key = (fold_id, tuple(sorted(inputs)))
        if key in self._fused_fold_cache:
            return self._fused_fold_cache[key]

        X_train = self.data_onehot.iloc[train_idx]
        prob, beta, intercept, lam, onehot_cols = build_fused_problem(
            X_train, inputs, self.target, self.offset, self.penalty_types, family=self.family
        )
        self._fused_fold_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._fused_fold_cache[key]

    def _cv_fold_loss_fused(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        inputs: List[str],
        fused_lambda_val: float,
    ) -> Tuple[float, float]:
        train_df = self.data_onehot.iloc[train_idx]
        test_df = self.data_onehot.iloc[test_idx]

        y_train = train_df[self.target].to_numpy(dtype=float)
        y_test = test_df[self.target].to_numpy(dtype=float)
        off_train = train_df[self.offset].to_numpy(dtype=float)
        off_test = test_df[self.offset].to_numpy(dtype=float)

        prob, beta, intercept, lam, onehot_cols = self._get_fused_fold_problem(fold_id, train_idx, inputs)
        lam.value = float(fused_lambda_val)
        solve_problem(prob, self.solver_cfg)

        b = np.asarray(beta.value, dtype=float)
        X_train = train_df[onehot_cols].to_numpy(dtype=float)
        X_test = test_df[onehot_cols].to_numpy(dtype=float)

        eta_train = X_train @ b + float(intercept.value) + off_train
        eta_test = X_test @ b + float(intercept.value) + off_test

        pred_train = _predict_mean_from_linear_predictor(self.family, eta_train)
        pred_test = _predict_mean_from_linear_predictor(self.family, eta_test)

        train_loss = _loss_from_predictions(self.family, y_train, pred_train, offset=off_train)
        test_loss = _loss_from_predictions(self.family, y_test, pred_test, offset=off_test)
        return train_loss, test_loss

    def _fit_glm_no_pen(self, data_onehot: pd.DataFrame, inputs: List[str]) -> np.ndarray:
        onehot_cols = get_onehot_columns(data_onehot, inputs)
        X = data_onehot[onehot_cols].to_numpy(dtype=float)
        y = data_onehot[self.target].to_numpy(dtype=float)
        offset_arr = data_onehot[self.offset].to_numpy(dtype=float)

        beta = cp.Variable(X.shape[1])
        intercept = cp.Variable()
        ll = _log_likelihood(self.family, X, y, offset_arr, beta, intercept)

        prob = cp.Problem(cp.Minimize(-ll))
        solve_problem(prob, self.solver_cfg)
        return coef_vector(intercept, beta)

    @time_it
    def fit_one_lambda(self, data: pd.DataFrame, inputs: List[str], lbd_fused: float):
        """Fit a fused-only model at one lambda and return final coefficients."""
        prob_f, beta_f, intercept_f, lam_f, onehot_cols_f = self._get_fused_full_problem(inputs)
        lam_f.value = float(lbd_fused)
        solve_problem(prob_f, self.solver_cfg)

        kept_vars = variables_kept_from_beta(np.asarray(beta_f.value, dtype=float), onehot_cols_f, inputs)

        if kept_vars:
            beta_sub, _ = _beta_subvector_for_vars(
                np.asarray(beta_f.value, dtype=float),
                onehot_cols_f,
                self.data_onehot,
                kept_vars,
            )
            temp_glm = np.concatenate(([float(intercept_f.value)], beta_sub))

            temp_data, kept_grouped, _ = group_modalities_based_on_betas(
                data.copy(), self.modality_var_df, temp_glm, kept_vars, self.ref_modality_dict
            )

            if kept_grouped:
                temp_data_onehot, temp_ref_modality_dict, _ = get_data_ready_for_glm(
                    temp_data, kept_grouped, self.penalty_types, self.target, self.offset, "First"
                )
                betas = self._fit_glm_no_pen(temp_data_onehot, kept_grouped)
                return len(temp_ref_modality_dict), temp_ref_modality_dict, betas, kept_grouped

        # No variables kept
        if self.family.lower() == "poisson":
            betas = np.array([data[self.target].mean()], dtype=float)
        else:
            betas = np.array([float(np.mean(data[self.target].to_numpy(dtype=float) - data[self.offset].to_numpy(dtype=float)))], dtype=float)
        return 0, {}, betas, []

    @time_it
    def crossval_fused_lambda(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        fused_lambda_val: float,
        counter: int,
    ) -> pd.DataFrame:
        print(f"Fused-only model with fused lambda = {fused_lambda_val}")

        err_train = 0.0
        err_test = 0.0

        if self.n_jobs and self.n_jobs > 1:
            for fold_id, (train_idx, _) in enumerate(self.cv_splits):
                self._get_fused_fold_problem(fold_id, train_idx, inputs)

            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futs = [
                    ex.submit(self._cv_fold_loss_fused, fold_id, train_idx, test_idx, inputs, fused_lambda_val)
                    for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
                ]
                for f in futs:
                    tr, te = f.result()
                    err_train += tr
                    err_test += te
        else:
            for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits):
                tr, te = self._cv_fold_loss_fused(fold_id, train_idx, test_idx, inputs, fused_lambda_val)
                err_train += tr
                err_test += te

        err_train /= self.n_k_fold
        err_test /= self.n_k_fold

        results.at[counter, "group_lambda"] = 0.0
        results.at[counter, "fused_lambda"] = fused_lambda_val
        results.at[counter, "Deviance_cv_train"] = err_train
        results.at[counter, "Deviance_cv_test"] = err_test

        nb_mod, mod_dict, betas, kept = self.fit_one_lambda(self.data_onehot, inputs, fused_lambda_val)
        results.at[counter, "modalities_number"] = nb_mod
        results.at[counter, "var_mod_details"] = mod_dict
        results.at[counter, "betas"] = betas
        results.at[counter, "variables"] = " ".join(map(str, kept))
        results.at[counter, "variable_number"] = len(kept)

        results = results.sort_values(by=["fused_lambda"], ascending=True).reset_index(drop=True)
        return results

    @time_it
    def get_lambda_curve(self):
        """Explore a range of fused lambdas and evaluate each via CV."""
        if self.smoothness_step is None:
            raise ValueError("smoothness_step must be provided for lambda curve exploration.")

        print("----------------------------------")
        print("Fused-only: Get range of Fused Lambdas")
        print("----------------------------------\n")

        max_lambda_reached = False
        lambda_temp = 0.1

        prob_f, beta_f, intercept_f, lam_f, onehot_cols_f = self._get_fused_full_problem(self.input_variables)

        while not max_lambda_reached:
            print(f"Running fused GLMs until reaching maximum lambda : {lambda_temp}")
            lam_f.value = float(lambda_temp)
            solve_problem(prob_f, self.solver_cfg)

            kept_vars = variables_kept_from_beta(np.asarray(beta_f.value, dtype=float), onehot_cols_f, self.input_variables)
            if len(kept_vars) == 0:
                max_lambda_reached = True
                lambda_max = float(lambda_temp)
            else:
                lambda_temp *= 10

        lambda_min = 0.1
        jump = (lambda_max - lambda_min) / float(self.smoothness_step)
        tested_fused_lambda_list = [lambda_min + jump * step for step in range(0, self.smoothness_step + 1)]

        results = self._init_results_table(len(tested_fused_lambda_list))
        for i, f in enumerate(tested_fused_lambda_list):
            results = self.crossval_fused_lambda(results, self.input_variables, float(f), i)

        self.lambda_curve = results

    def plot_curve(self):
        """Scatter plot: variable_number vs CV loss, with lambda labels."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.lambda_curve["variable_number"], self.lambda_curve["Deviance_cv_test"])

        for _, row in self.lambda_curve.iterrows():
            label = f"{np.around(row['fused_lambda'])}"
            plt.text(row["variable_number"], row["Deviance_cv_test"], label, fontsize=8, ha="center", va="bottom")

        plt.xlabel("Final Variable Number")
        plt.ylabel("CV loss (Poisson deviance or Gaussian MSE)")
        plt.title("Fused-only grid search")
        plt.tight_layout()
        plt.show()


# =============================================================================
# Single-penalty grid searches
# =============================================================================


def _predict_mean_from_linear_predictor(family: str, eta: np.ndarray) -> np.ndarray:
    """Return the mean prediction given linear predictor eta."""
    fam = family.lower()
    if fam == "poisson":
        return np.exp(eta)
    if fam == "gaussian":
        return eta
    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def _cv_loss(
    family: str,
    y: np.ndarray,
    pred: np.ndarray,
    offset_arr: Optional[np.ndarray] = None,
) -> float:
    """Cross-validation loss consistent with the fitted family.

    - Poisson: scaled deviance (matching poisson_deviance_mu)
    - Gaussian: mean squared error
    """
    fam = family.lower()
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if fam == "poisson":
        exposure = None if offset_arr is None else np.exp(np.asarray(offset_arr, dtype=float))
        return poisson_deviance_mu(pred, y, exposure=exposure)

    if fam == "gaussian":
        return float(np.mean((y - pred) ** 2))

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def _intercept_only_predictions(
    family: str,
    y_train: np.ndarray,
    offset_train: np.ndarray,
    offset_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intercept-only baseline for supported families (offset-aware)."""
    fam = family.lower()
    y_train = np.asarray(y_train, dtype=float)
    off_train = np.asarray(offset_train, dtype=float)
    off_test = np.asarray(offset_test, dtype=float)

    if fam == "poisson":
        exp_train = np.exp(off_train)
        exp_test = np.exp(off_test)
        rate = y_train.sum() / max(exp_train.sum(), 1e-12)
        return rate * exp_train, rate * exp_test

    if fam == "gaussian":
        # Minimize MSE for y ~= intercept + offset
        intercept = float(np.mean(y_train - off_train))
        return intercept + off_train, intercept + off_test

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def _subset_beta_for_vars(
    beta_full: np.ndarray,
    onehot_cols_full: List[str],
    data_onehot: pd.DataFrame,
    vars_subset: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Extract beta coefficients corresponding to `vars_subset` in stable one-hot order.

    Returns
    -------
    beta_sub : np.ndarray
        Coefficient vector aligned with `subset_cols`.
    subset_cols : list[str]
        One-hot columns corresponding to vars_subset.
    """
    subset_cols = get_onehot_columns(data_onehot, vars_subset)
    col_to_idx = {c: i for i, c in enumerate(onehot_cols_full)}
    idxs = [col_to_idx[c] for c in subset_cols]
    return np.asarray(beta_full, dtype=float)[idxs], subset_cols


class GridSearch_Group:
    """Grid search using **only** the group penalty.

    This class mirrors the interface of `GridSearch_Generalised` but removes the
    fused stage entirely.

    Supported usage modes
    ---------------------
    1) Automatic curve:
        Provide `parcimony_step` and `var_nb_min` (optionally `var_nb_max`).
        The algorithm searches a reasonable range of group lambdas, then
        evaluates the selected lambdas via K-fold CV.

    2) Fixed lambdas:
        Provide `lbd_group` as a float or a list/tuple of floats.

    Notes
    -----
    - The CV loss is Poisson deviance (for Poisson family) or MSE (for Gaussian).
    - Final model uses the group-penalized fit, then (optionally) groups
      modalities using `group_modalities_based_on_betas` and refits an
      unpenalized GLM on grouped variables, matching the "Generalised" pipeline.
    """

    def __init__(
        self,
        family: str,
        var_nb_min: Optional[int] = None,
        var_nb_max: Optional[int] = None,
        parcimony_step: Optional[int] = None,
        lbd_group: Optional[object] = None,
        random_state: Optional[int] = 0,
        solver: str = "CLARABEL",
        warm_start: bool = True,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        self.family = family
        self.var_nb_min = var_nb_min
        self.var_nb_max = var_nb_max
        self.parcimony_step = parcimony_step
        self.lbd_group = lbd_group

        self.random_state = random_state
        self.solver_cfg = SolverConfig(solver=solver, warm_start=warm_start, verbose=verbose)
        self.n_jobs = int(n_jobs) if n_jobs is not None else 1

        curve_mode = parcimony_step is not None
        fixed_mode = lbd_group is not None

        if curve_mode == fixed_mode:
            raise ValueError("Provide either parcimony_step (curve) OR lbd_group (fixed), but not both.")
        if curve_mode and (var_nb_min is None):
            raise ValueError("var_nb_min must be provided when using parcimony_step.")

        # caches
        self._group_full_cache = None
        self._group_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}

    @time_it
    def fit(
        self,
        data: pd.DataFrame,
        penalty_types: Dict[str, str],
        input_variables: List[str],
        target: str,
        offset: str,
        n_k_fold: int = 5,
    ):
        self.data = data
        self.penalty_types = penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.target = target
        self.offset = offset

        self.data_onehot, self.ref_modality_dict, _ = get_data_ready_for_glm(
            self.data, self.input_variables, self.penalty_types, target, offset, "First"
        )

        self.cv_splits = list(KFold(n_splits=self.n_k_fold, shuffle=True, random_state=self.random_state).split(self.data_onehot))

        # Build modality mapping once (aligns with one-hot column order)
        modality_arr, variables_arr = [], []
        for var in self.input_variables:
            for col in self.data_onehot.columns:
                if col.startswith(var + "_"):
                    modality_arr.append(col.replace(var + "_", ""))
                    variables_arr.append(var)
        self.modality_var_df = pd.DataFrame({"variable": variables_arr, "modality": modality_arr})

        if self.lbd_group is None:
            self.get_lambda_curve()
            return

        # Fixed lambdas evaluation
        if isinstance(self.lbd_group, (list, tuple)):
            lambdas = [float(x) for x in self.lbd_group]
        else:
            lambdas = [float(self.lbd_group)]

        results = self._init_results_table(len(lambdas))
        for i, g in enumerate(lambdas):
            results = self.crossval_group_lambda(results, self.input_variables, g, i)

        self.lambda_curve = results

    # -------------------------
    # Tables and caching
    # -------------------------

    def _init_results_table(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "group_lambda",
                "fused_lambda",
                "variable_number",
                "modalities_number",
                "var_mod_details",
                "Deviance_cv_train",
                "Deviance_cv_test",
                "variables",
                "betas",
            ],
            data=np.empty((n_rows, 9), dtype=object),
        )

    def _get_group_full_problem(self):
        if self._group_full_cache is not None:
            return self._group_full_cache

        prob, beta, intercept, lam, onehot_cols, _ = build_group_lasso_problem(
            self.data_onehot,
            self.input_variables,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._group_full_cache = (prob, beta, intercept, lam, onehot_cols)
        return self._group_full_cache

    def _get_group_fold_problem(self, fold_id: int, train_idx: np.ndarray, inputs: List[str]):
        key = (fold_id, tuple(sorted(inputs)))
        if key in self._group_fold_cache:
            return self._group_fold_cache[key]

        X_train = self.data_onehot.iloc[train_idx]
        prob, beta, intercept, lam, onehot_cols, _ = build_group_lasso_problem(
            X_train, inputs, self.target, self.offset, self.penalty_types, family=self.family
        )
        self._group_fold_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._group_fold_cache[key]

    # -------------------------
    # Core CV
    # -------------------------

    def _cv_fold_loss_group(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        inputs: List[str],
        group_lambda_val: float,
    ) -> Tuple[float, float]:
        train_df = self.data_onehot.iloc[train_idx]
        test_df = self.data_onehot.iloc[test_idx]

        y_train = train_df[self.target].to_numpy(dtype=float)
        y_test = test_df[self.target].to_numpy(dtype=float)
        off_train = train_df[self.offset].to_numpy(dtype=float)
        off_test = test_df[self.offset].to_numpy(dtype=float)

        prob, beta, intercept, lam, onehot_cols = self._get_group_fold_problem(fold_id, train_idx, inputs)
        lam.value = float(group_lambda_val)
        solve_problem(prob, self.solver_cfg)

        b = np.asarray(beta.value, dtype=float)
        X_train = train_df[onehot_cols].to_numpy(dtype=float)
        X_test = test_df[onehot_cols].to_numpy(dtype=float)

        eta_train = X_train @ b + float(intercept.value) + off_train
        eta_test = X_test @ b + float(intercept.value) + off_test

        pred_train = _predict_mean_from_linear_predictor(self.family, eta_train)
        pred_test = _predict_mean_from_linear_predictor(self.family, eta_test)

        return (
            _cv_loss(self.family, y_train, pred_train, offset_arr=off_train),
            _cv_loss(self.family, y_test, pred_test, offset_arr=off_test),
        )

    @time_it
    def crossval_group_lambda(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        group_lambda_val: float,
        counter: int,
    ) -> pd.DataFrame:
        print(f"Model with group lambda = {group_lambda_val}")

        # Pre-create fold problems if parallel
        if self.n_jobs and self.n_jobs > 1:
            for fold_id, (train_idx, _) in enumerate(self.cv_splits):
                self._get_group_fold_problem(fold_id, train_idx, inputs)

        err_train = 0.0
        err_test = 0.0

        if self.n_jobs and self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futs = [
                    ex.submit(self._cv_fold_loss_group, fold_id, train_idx, test_idx, inputs, group_lambda_val)
                    for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
                ]
                for f in futs:
                    tr, te = f.result()
                    err_train += tr
                    err_test += te
        else:
            for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits):
                tr, te = self._cv_fold_loss_group(fold_id, train_idx, test_idx, inputs, group_lambda_val)
                err_train += tr
                err_test += te

        err_train /= self.n_k_fold
        err_test /= self.n_k_fold

        results.at[counter, "group_lambda"] = group_lambda_val
        results.at[counter, "fused_lambda"] = 0.0
        results.at[counter, "Deviance_cv_train"] = err_train
        results.at[counter, "Deviance_cv_test"] = err_test

        # Final fit on full data
        nb_mod, mod_dict, betas, kept = self.fit_one_lambda(group_lambda_val)
        results.at[counter, "modalities_number"] = nb_mod
        results.at[counter, "var_mod_details"] = mod_dict
        results.at[counter, "betas"] = betas
        results.at[counter, "variables"] = " ".join(map(str, kept))
        results.at[counter, "variable_number"] = len(kept)

        results = results.sort_values(by=["group_lambda"], ascending=True).reset_index(drop=True)
        return results

    # -------------------------
    # Final model fit
    # -------------------------

    @time_it
    def fit_one_lambda(self, lbd_group: float):
        prob_g, beta_g, intercept_g, lam_g, onehot_cols_g = self._get_group_full_problem()
        lam_g.value = float(lbd_group)
        solve_problem(prob_g, self.solver_cfg)

        kept_vars = variables_kept_from_beta(np.asarray(beta_g.value, dtype=float), onehot_cols_g, self.input_variables)

        if kept_vars:
            beta_sub, _ = _subset_beta_for_vars(np.asarray(beta_g.value, dtype=float), onehot_cols_g, self.data_onehot, kept_vars)
            temp_glm = np.concatenate(([float(intercept_g.value)], beta_sub))

            temp_data, kept_grouped, _ = group_modalities_based_on_betas(
                self.data_onehot.copy(), self.modality_var_df, temp_glm, kept_vars, self.ref_modality_dict
            )

            if kept_grouped:
                temp_data_onehot, temp_ref_modality_dict, _ = get_data_ready_for_glm(
                    temp_data, kept_grouped, self.penalty_types, self.target, self.offset, "First"
                )
                betas = self._fit_glm_no_pen(temp_data_onehot, kept_grouped)
                return len(temp_ref_modality_dict), temp_ref_modality_dict, betas, kept_grouped

        # No variables kept
        y = self.data_onehot[self.target].to_numpy(dtype=float)
        off = self.data_onehot[self.offset].to_numpy(dtype=float)
        mu_train, _ = _intercept_only_predictions(self.family, y, off, off)
        if self.family.lower() == "poisson":
            # Return intercept in the same style as coef_vector: log(rate)
            rate = float(y.sum() / max(np.exp(off).sum(), 1e-12))
            intercept = float(np.log(max(rate, 1e-12)))
            return 0, {}, np.array([intercept], dtype=float), []
        return 0, {}, np.array([float(np.mean(y - off))], dtype=float), []

    def _fit_glm_no_pen(self, data_onehot: pd.DataFrame, inputs: List[str]) -> np.ndarray:
        onehot_cols = get_onehot_columns(data_onehot, inputs)
        X = data_onehot[onehot_cols].to_numpy(dtype=float)
        y = data_onehot[self.target].to_numpy(dtype=float)
        offset_arr = data_onehot[self.offset].to_numpy(dtype=float)

        beta = cp.Variable(X.shape[1])
        intercept = cp.Variable()
        ll = _log_likelihood(self.family, X, y, offset_arr, beta, intercept)

        prob = cp.Problem(cp.Minimize(-ll))
        solve_problem(prob, self.solver_cfg)
        return coef_vector(intercept, beta)

    # -------------------------
    # Lambda curve exploration
    # -------------------------

    @time_it
    def get_lambda_curve(self):
        if self.parcimony_step is None:
            raise ValueError("parcimony_step must be provided for lambda curve exploration.")

        print("----------------------------------")
        print("Group-only: Get range of Group Lambdas")
        print("----------------------------------\n")

        max_lambda_group_reached = False
        lambda_group_temp = 0.1

        prob, beta, intercept, lam, onehot_cols = self._get_group_full_problem()

        while not max_lambda_group_reached:
            print(f"Running group GLMs until reaching maximum lambda : {lambda_group_temp}")
            lam.value = float(lambda_group_temp)
            solve_problem(prob, self.solver_cfg)

            kept_vars = variables_kept_from_beta(np.asarray(beta.value, dtype=float), onehot_cols, self.input_variables)

            if (len(kept_vars) == 0) or (self.var_nb_min is not None and len(kept_vars) <= self.var_nb_min):
                max_lambda_group_reached = True
                lambda_max = float(lambda_group_temp)
            else:
                lambda_group_temp *= 10

        lambda_min = 0.1
        jump = (lambda_max - lambda_min) / float(self.parcimony_step)
        tested_lambda_list = [lambda_min + jump * step for step in range(0, self.parcimony_step + 1)]

        results = self._init_results_table(len(tested_lambda_list))
        for i, g in enumerate(tested_lambda_list):
            results = self.crossval_group_lambda(results, self.input_variables, float(g), i)

        self.lambda_curve = results

    def plot_curve(self):
        """Scatter plot: variable_number vs CV loss, with lambda labels."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.lambda_curve["variable_number"], self.lambda_curve["Deviance_cv_test"])

        for _, row in self.lambda_curve.iterrows():
            label = f"{np.around(row['group_lambda'])}"
            plt.text(row["variable_number"], row["Deviance_cv_test"], label, fontsize=8, ha="center", va="bottom")

        plt.xlabel("Final Variable Number")
        plt.ylabel("CV loss (Poisson deviance or Gaussian MSE)")
        plt.title("Group-only grid search")
        plt.tight_layout()
        plt.show()


class GridSearch_Fused:
    """Grid search using **only** the fused (or g_fused) penalty.

    This class mirrors `GridSearch_Generalised` but removes the group selection stage.

    Supported usage modes
    ---------------------
    1) Automatic curve:
        Provide `smoothness_step`.

    2) Fixed lambdas:
        Provide `lbd_fused` as a float or a list/tuple of floats.

    Notes
    -----
    - The penalty type for each variable is read from `penalty_types` as in the
      original code ("fused" or "g_fused").
    """

    def __init__(
        self,
        family: str,
        smoothness_step: Optional[int] = None,
        lbd_fused: Optional[object] = None,
        random_state: Optional[int] = 0,
        solver: str = "CLARABEL",
        warm_start: bool = True,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        self.family = family
        self.smoothness_step = smoothness_step
        self.lbd_fused = lbd_fused

        self.random_state = random_state
        self.solver_cfg = SolverConfig(solver=solver, warm_start=warm_start, verbose=verbose)
        self.n_jobs = int(n_jobs) if n_jobs is not None else 1

        curve_mode = smoothness_step is not None
        fixed_mode = lbd_fused is not None

        if curve_mode == fixed_mode:
            raise ValueError("Provide either smoothness_step (curve) OR lbd_fused (fixed), but not both.")

        self._fused_full_cache: Dict[Tuple[str, ...], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}
        self._fused_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}

    @time_it
    def fit(
        self,
        data: pd.DataFrame,
        penalty_types: Dict[str, str],
        input_variables: List[str],
        target: str,
        offset: str,
        n_k_fold: int = 5,
    ):
        self.data = data
        self.penalty_types = penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.target = target
        self.offset = offset

        self.data_onehot, self.ref_modality_dict, _ = get_data_ready_for_glm(
            self.data, self.input_variables, self.penalty_types, target, offset, "First"
        )

        self.cv_splits = list(KFold(n_splits=self.n_k_fold, shuffle=True, random_state=self.random_state).split(self.data_onehot))

        modality_arr, variables_arr = [], []
        for var in self.input_variables:
            for col in self.data_onehot.columns:
                if col.startswith(var + "_"):
                    modality_arr.append(col.replace(var + "_", ""))
                    variables_arr.append(var)
        self.modality_var_df = pd.DataFrame({"variable": variables_arr, "modality": modality_arr})

        if self.lbd_fused is None:
            self.get_lambda_curve()
            return

        if isinstance(self.lbd_fused, (list, tuple)):
            lambdas = [float(x) for x in self.lbd_fused]
        else:
            lambdas = [float(self.lbd_fused)]

        results = self._init_results_table(len(lambdas))
        for i, f in enumerate(lambdas):
            results = self.crossval_fused_lambda(results, self.input_variables, f, i)

        self.lambda_curve = results

    def _init_results_table(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "group_lambda",
                "fused_lambda",
                "variable_number",
                "modalities_number",
                "var_mod_details",
                "Deviance_cv_train",
                "Deviance_cv_test",
                "variables",
                "betas",
            ],
            data=np.empty((n_rows, 9), dtype=object),
        )

    def _get_fused_full_problem(self, inputs: List[str]):
        key = tuple(sorted(inputs))
        if key in self._fused_full_cache:
            return self._fused_full_cache[key]

        prob, beta, intercept, lam, onehot_cols = build_fused_problem(
            self.data_onehot,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._fused_full_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._fused_full_cache[key]

    def _get_fused_fold_problem(self, fold_id: int, train_idx: np.ndarray, inputs: List[str]):
        key = (fold_id, tuple(sorted(inputs)))
        if key in self._fused_fold_cache:
            return self._fused_fold_cache[key]

        X_train = self.data_onehot.iloc[train_idx]
        prob, beta, intercept, lam, onehot_cols = build_fused_problem(
            X_train, inputs, self.target, self.offset, self.penalty_types, family=self.family
        )
        self._fused_fold_cache[key] = (prob, beta, intercept, lam, onehot_cols)
        return self._fused_fold_cache[key]

    def _cv_fold_loss_fused(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        inputs: List[str],
        fused_lambda_val: float,
    ) -> Tuple[float, float]:
        train_df = self.data_onehot.iloc[train_idx]
        test_df = self.data_onehot.iloc[test_idx]

        y_train = train_df[self.target].to_numpy(dtype=float)
        y_test = test_df[self.target].to_numpy(dtype=float)
        off_train = train_df[self.offset].to_numpy(dtype=float)
        off_test = test_df[self.offset].to_numpy(dtype=float)

        prob, beta, intercept, lam, onehot_cols = self._get_fused_fold_problem(fold_id, train_idx, inputs)
        lam.value = float(fused_lambda_val)
        solve_problem(prob, self.solver_cfg)

        b = np.asarray(beta.value, dtype=float)
        X_train = train_df[onehot_cols].to_numpy(dtype=float)
        X_test = test_df[onehot_cols].to_numpy(dtype=float)

        eta_train = X_train @ b + float(intercept.value) + off_train
        eta_test = X_test @ b + float(intercept.value) + off_test

        pred_train = _predict_mean_from_linear_predictor(self.family, eta_train)
        pred_test = _predict_mean_from_linear_predictor(self.family, eta_test)

        return (
            _cv_loss(self.family, y_train, pred_train, offset_arr=off_train),
            _cv_loss(self.family, y_test, pred_test, offset_arr=off_test),
        )

    @time_it
    def crossval_fused_lambda(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        fused_lambda_val: float,
        counter: int,
    ) -> pd.DataFrame:
        print(f"Model with fused lambda = {fused_lambda_val}")

        if self.n_jobs and self.n_jobs > 1:
            for fold_id, (train_idx, _) in enumerate(self.cv_splits):
                self._get_fused_fold_problem(fold_id, train_idx, inputs)

        err_train = 0.0
        err_test = 0.0

        if self.n_jobs and self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futs = [
                    ex.submit(self._cv_fold_loss_fused, fold_id, train_idx, test_idx, inputs, fused_lambda_val)
                    for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
                ]
                for f in futs:
                    tr, te = f.result()
                    err_train += tr
                    err_test += te
        else:
            for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits):
                tr, te = self._cv_fold_loss_fused(fold_id, train_idx, test_idx, inputs, fused_lambda_val)
                err_train += tr
                err_test += te

        err_train /= self.n_k_fold
        err_test /= self.n_k_fold

        results.at[counter, "group_lambda"] = 0.0
        results.at[counter, "fused_lambda"] = fused_lambda_val
        results.at[counter, "Deviance_cv_train"] = err_train
        results.at[counter, "Deviance_cv_test"] = err_test

        nb_mod, mod_dict, betas, kept = self.fit_one_lambda(fused_lambda_val)
        results.at[counter, "modalities_number"] = nb_mod
        results.at[counter, "var_mod_details"] = mod_dict
        results.at[counter, "betas"] = betas
        results.at[counter, "variables"] = " ".join(map(str, kept))
        results.at[counter, "variable_number"] = len(kept)

        results = results.sort_values(by=["fused_lambda"], ascending=True).reset_index(drop=True)
        return results

    @time_it
    def fit_one_lambda(self, lbd_fused: float):
        prob_f, beta_f, intercept_f, lam_f, onehot_cols_f = self._get_fused_full_problem(self.input_variables)
        lam_f.value = float(lbd_fused)
        solve_problem(prob_f, self.solver_cfg)

        kept_vars = variables_kept_from_beta(np.asarray(beta_f.value, dtype=float), onehot_cols_f, self.input_variables)

        if kept_vars:
            beta_sub, _ = _subset_beta_for_vars(np.asarray(beta_f.value, dtype=float), onehot_cols_f, self.data_onehot, kept_vars)
            temp_glm = np.concatenate(([float(intercept_f.value)], beta_sub))

            temp_data, kept_grouped, _ = group_modalities_based_on_betas(
                self.data_onehot.copy(), self.modality_var_df, temp_glm, kept_vars, self.ref_modality_dict
            )

            if kept_grouped:
                temp_data_onehot, temp_ref_modality_dict, _ = get_data_ready_for_glm(
                    temp_data, kept_grouped, self.penalty_types, self.target, self.offset, "First"
                )

                betas = self._fit_glm_no_pen(temp_data_onehot, kept_grouped)
                return len(temp_ref_modality_dict), temp_ref_modality_dict, betas, kept_grouped

        # No variables kept
        y = self.data_onehot[self.target].to_numpy(dtype=float)
        off = self.data_onehot[self.offset].to_numpy(dtype=float)
        if self.family.lower() == "poisson":
            rate = float(y.sum() / max(np.exp(off).sum(), 1e-12))
            intercept = float(np.log(max(rate, 1e-12)))
            return 0, {}, np.array([intercept], dtype=float), []
        return 0, {}, np.array([float(np.mean(y - off))], dtype=float), []

    def _fit_glm_no_pen(self, data_onehot: pd.DataFrame, inputs: List[str]) -> np.ndarray:
        onehot_cols = get_onehot_columns(data_onehot, inputs)
        X = data_onehot[onehot_cols].to_numpy(dtype=float)
        y = data_onehot[self.target].to_numpy(dtype=float)
        offset_arr = data_onehot[self.offset].to_numpy(dtype=float)

        beta = cp.Variable(X.shape[1])
        intercept = cp.Variable()
        ll = _log_likelihood(self.family, X, y, offset_arr, beta, intercept)

        prob = cp.Problem(cp.Minimize(-ll))
        solve_problem(prob, self.solver_cfg)
        return coef_vector(intercept, beta)

    @time_it
    def get_lambda_curve(self):
        """Explore a range of fused lambdas and evaluate each via CV."""
        if self.smoothness_step is None:
            raise ValueError("smoothness_step must be provided for lambda curve exploration.")

        print("----------------------------------")
        print("Fused-only: Get range of Fused Lambdas")
        print("----------------------------------\n")

        max_lambda_reached = False
        lambda_temp = 0.1

        prob_f, beta_f, intercept_f, lam_f, onehot_cols_f = self._get_fused_full_problem(self.input_variables)

        while not max_lambda_reached:
            print(f"Running fused GLMs until reaching maximum lambda : {lambda_temp}")
            lam_f.value = float(lambda_temp)
            solve_problem(prob_f, self.solver_cfg)

            kept_vars = variables_kept_from_beta(np.asarray(beta_f.value, dtype=float), onehot_cols_f, self.input_variables)
            if len(kept_vars) == 0:
                max_lambda_reached = True
                lambda_max = float(lambda_temp)
            else:
                lambda_temp *= 10

        lambda_min = 0.1
        jump = (lambda_max - lambda_min) / float(self.smoothness_step)
        tested_fused_lambda_list = [lambda_min + jump * step for step in range(0, self.smoothness_step + 1)]

        results = self._init_results_table(len(tested_fused_lambda_list))
        for i, f in enumerate(tested_fused_lambda_list):
            results = self.crossval_fused_lambda(results, self.input_variables, float(f), i)

        self.lambda_curve = results

    def plot_curve(self):
        """Scatter plot: variable_number vs CV loss, with lambda labels."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.lambda_curve["variable_number"], self.lambda_curve["Deviance_cv_test"])

        for _, row in self.lambda_curve.iterrows():
            label = f"{np.around(row['fused_lambda'])}"
            plt.text(row["variable_number"], row["Deviance_cv_test"], label, fontsize=8, ha="center", va="bottom")

        plt.xlabel("Final Variable Number")
        plt.ylabel("CV loss (Poisson deviance or Gaussian MSE)")
        plt.title("Fused-only grid search")
        plt.tight_layout()
        plt.show()
