
"""
mod2_clean.py
=============

Cleaned and documented utilities to fit GLMs with group and (generalized) fused
penalties using CVXPY, plus a grid-search helper for cross-validation over
(group_lambda, fused_lambda) pairs.


Dependencies
------------
- numpy
- pandas
- cvxpy (with a compatible solver installed, e.g., CLARABEL)
- scikit-learn (for KFold)

Note: This module avoids any project-specific I/O and can be imported directly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# If these helpers live elsewhere in your codebase, keep the import.
# They are not reimplemented here.
from mypkg.additional_functions.mod1 import (
    choose_ref_modality,
    reorder_df_columns,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def time_it(f):
    """Simple timing decorator that indents nested calls for readability."""
    time_it.active = 0

    def tt(*args, **kwargs):
        time_it.active += 1
        t0 = time.time()
        tabs = '\t' * (time_it.active - 1)
        name = f.__name__
        print(f"{tabs}Executing <{name}>")
        res = f(*args, **kwargs)
        elapsed = time.time() - t0
        mins = int(elapsed // 60)
        secs = elapsed % 60
        print(f"{tabs}Function <{name}> execution time : {mins:.0f} minutes and {secs:.0f} seconds")
        time_it.active -= 1
        return res

    return tt


def get_onehot_columns(df: pd.DataFrame, prefixes: Sequence[str]) -> List[str]:
    """Return one-hot column names whose prefix is in *prefixes*."""
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


def Compute_Poisson_Deviance(lambdahat: Iterable[float], N: Iterable[float], v: Iterable[float]) -> float:
    """Vectorized, numerically safe Poisson deviance (scaled by total exposure).

    Parameters
    ----------
    lambdahat : array-like
        Predicted Poisson rates (\hat{\lambda}).
    N : array-like
        Observed counts.
    v : array-like
        Exposure (same length as N).

    Returns
    -------
    float
        Sum_i D_i / sum_i v_i where
        D_i = n_i * log(n_i / (mu_i)) - (n_i - mu_i) and mu_i = v_i * lambdahat_i.
        Convention 0 * log(0 / mu) = 0.
    """
    lambdahat = np.asarray(lambdahat, dtype=float)
    N = np.asarray(N, dtype=float)
    v = np.asarray(v, dtype=float)

    # Guard against zeros to avoid log(0) and division by zero
    eps = 1e-12
    lambdahat = np.clip(lambdahat, eps, None)
    v = np.clip(v, eps, None)

    mu = lambdahat * v

    mask = N > 0
    nlogn = np.zeros_like(N)
    nlogn[mask] = N[mask] * (np.log(N[mask]) - np.log(mu[mask]))

    dev = nlogn - (N - mu)
    return float(dev.sum() / v.sum())


# ---------------------------------------------------------------------------
# Penalty builders
# ---------------------------------------------------------------------------

def _build_one_hot_groups(data: pd.DataFrame, input_var_onehot: List[str], penalty_types: Dict[str, str]) -> Dict[str, List[int]]:
    """Map each group prefix to indices of its one-hot columns in *input_var_onehot*."""
    one_hot_groups = {
        key: [i for i, x in enumerate(input_var_onehot) if x.startswith(key + "_")]
        for key in penalty_types
    }
    # remove empty groups just in case
    return {k: v for k, v in one_hot_groups.items() if v}


def _poisson_loglik(X: np.ndarray, y: np.ndarray, offset_arr: np.ndarray, beta: cp.Expression, intercept: cp.Expression) -> cp.Expression:
    eta = X @ beta + intercept + offset_arr
    return cp.sum(cp.multiply(y, eta) - cp.exp(eta))


def compute_glm_no_pen(
    data: pd.DataFrame,
    input_var: Sequence[str],
    target_var: str,
    offset_var: str,
    family: str = "Poisson",
    solver: str = "CLARABEL",
) -> np.ndarray:
    """Fit an unpenalized GLM via CVXPY and return [intercept, betas...]."""
    input_var_onehot = get_onehot_columns(data, input_var)
    X = data[input_var_onehot].to_numpy()
    y = data[target_var].to_numpy()
    offset_arr = data[offset_var].to_numpy()

    intercept = cp.Variable()
    beta = cp.Variable(X.shape[1])

    if family.lower() == "poisson":
        log_likelihood = _poisson_loglik(X, y, offset_arr, beta, intercept)
    elif family.lower() == "gaussian":
        log_likelihood = -0.5 * cp.sum_squares(y - (X @ beta + intercept + offset_arr))
    else:
        raise ValueError(f"Unsupported family: {family}")

    prob = cp.Problem(cp.Minimize(-log_likelihood))
    prob.solve(solver=getattr(cp, solver))

    return np.concatenate(([intercept.value], beta.value))


def compute_group_lasso_glm(
    data: pd.DataFrame,
    input_var: Sequence[str],
    target_var: str,
    offset_var: str,
    penalty_types: Dict[str, str],
    lambda_group: float = 1.0,
    family: str = 'Poisson',
    solver: str = "CLARABEL",
) -> np.ndarray:
    """Group-lasso GLM where each categorical variable is a group of one-hots.

    Returns [intercept, betas...].
    """
    input_var_onehot = get_onehot_columns(data, input_var)
    X = data[input_var_onehot].to_numpy()
    y = data[target_var].to_numpy()
    offset_arr = data[offset_var].to_numpy()

    beta = cp.Variable(X.shape[1])
    intercept = cp.Variable()

    if family.lower() == "poisson":
        log_likelihood = _poisson_loglik(X, y, offset_arr, beta, intercept)
    elif family.lower() == "gaussian":
        log_likelihood = -0.5 * cp.sum_squares(y - (X @ beta + intercept + offset_arr))
    else:
        raise ValueError(f"Unsupported family: {family}")

    # Only keep groups present in input_var
    ptypes = {k: v for k, v in penalty_types.items() if any(k.startswith(prefix) for prefix in input_var)}
    one_hot_groups = _build_one_hot_groups(data, input_var_onehot, ptypes)

    group_penalty = 0
    for idxs in one_hot_groups.values():
        w = np.sqrt(len(idxs))
        group_penalty += cp.norm1(w * beta[idxs])

    objective = cp.Minimize(-log_likelihood + lambda_group * group_penalty)
    prob = cp.Problem(objective)
    prob.solve(solver=getattr(cp, solver))

    return np.concatenate(([intercept.value], beta.value))


def custom_glm_with_fused_penalty(
    data: pd.DataFrame,
    input_var: Sequence[str],
    target_var: str,
    offset_var: str,
    penalty_types: Dict[str, str],
    lambda_fused: float = 1.0,
    family: str = 'Poisson',
    solver: str = "CLARABEL",
) -> np.ndarray:
    """GLM with per-variable fused and generalized fused penalties.

    This mirrors the original implementation but with clearer structure.
    Returns [intercept, betas...].
    """
    input_var_onehot = get_onehot_columns(data, input_var)
    X = data[input_var_onehot].to_numpy()
    y = data[target_var].to_numpy()
    offset_arr = data[offset_var].to_numpy()

    beta = cp.Variable(X.shape[1])
    intercept = cp.Variable()

    if family.lower() == "poisson":
        log_likelihood = _poisson_loglik(X, y, offset_arr, beta, intercept)
    elif family.lower() == "gaussian":
        log_likelihood = -0.5 * cp.sum_squares(y - (X @ beta + intercept + offset_arr))
    else:
        raise ValueError(f"Unsupported family: {family}")

    # Restrict penalty types to variables actually in *input_var*
    ptypes = {k: v for k, v in penalty_types.items() if any(k.startswith(prefix) for prefix in input_var)}
    one_hot_groups = _build_one_hot_groups(data, input_var_onehot, ptypes)

    g_fused_penalty = 0
    fused_penalty = 0
    g_fused_graph_size = 0

    # Precompute per-group reference modality length and graph size
    ref_mod_len_per_group: Dict[str, int] = {}
    for key, val in one_hot_groups.items():
        if ptypes.get(key) == "g_fused":
            g_fused_graph_size += len(val) + 1

        ref_mod_length = len(data)
        for col in input_var_onehot:
            if col.startswith(key):
                ref_mod_length -= int(data[col].sum())
        ref_mod_len_per_group[key] = ref_mod_length

    # Build fused and generalized fused penalties
    for key, val in one_hot_groups.items():
        ref_mod_length = ref_mod_len_per_group[key]

        if ptypes.get(key) == "fused":
            # reference to 0 (reference modality)
            w = ((int(data.iloc[:, val[0]].sum()) + ref_mod_length) / len(data)) ** 0.5
            fused_penalty += cp.norm1(w * (beta[val[0]] - 0))
            for i in range(1, len(val)):
                w = ((int(data.iloc[:, val[i]].sum()) + int(data.iloc[:, val[i-1]].sum())) / len(data)) ** 0.5
                fused_penalty += cp.norm1(w * (beta[val[i]] - beta[val[i - 1]]))

        elif ptypes.get(key) == "g_fused":
            for k in range(len(val)):
                w = (len(val) / g_fused_graph_size) * ((int(data.iloc[:, val[k]].sum()) + ref_mod_length) / len(data)) ** 0.5
                g_fused_penalty += cp.norm1(w * (beta[val[k]] - 0))
                for j in range(k, len(val)):
                    w = (len(val) / g_fused_graph_size) * ((int(data.iloc[:, val[k]].sum()) + int(data.iloc[:, val[j]].sum())) / len(data)) ** 0.5
                    g_fused_penalty += cp.norm1(w * (beta[val[k]] - beta[val[j]]))

    objective = cp.Minimize(-log_likelihood + lambda_fused * (g_fused_penalty + fused_penalty))
    prob = cp.Problem(objective)
    prob.solve(solver=getattr(cp, solver))

    return np.concatenate(([intercept.value], beta.value))


# ---------------------------------------------------------------------------
# Data prep and post-processing
# ---------------------------------------------------------------------------

def get_data_ready_for_glm(
    data: pd.DataFrame,
    input_variables: Sequence[str],
    penalty_types: Dict[str, str],
    target: str,
    offset: str,
    method: str,
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """One-hot + reorder + keep non-dummy originals + attach target & offset."""
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
    """Group/merge modalities with identical rounded betas; drop all-zero vars."""
    chosen_model_temp = beta_list[1:]  # strip intercept

    temp_modality_var_df = modality_var_df[modality_var_df['variable'].isin(input_variables)].copy()
    temp_modality_var_df["betas"] = [str(round(i, 6)) for i in chosen_model_temp]

    grouped = (
        temp_modality_var_df.groupby(['variable', 'betas'])['modality']
        .apply(lambda x: '_'.join(x))
        .reset_index()
        .rename(columns={"modality": "modality_grouped"})
    )

    temp_modality_var_df = temp_modality_var_df.merge(grouped, how="left", on=["variable", "betas"])

    kept_variables: List[str] = []
    removed_variables: List[str] = []

    for var in input_variables:
        original_var = "non_dum_" + var
        one = temp_modality_var_df[temp_modality_var_df["variable"] == var].copy()

        if (abs(one["betas"].astype(float)) < 1e-6).all():
            removed_variables.append(var)
            continue

        # Map original modality -> grouped modality
        data[original_var] = data[original_var].astype(str)
        one["modality"] = one["modality"].astype(str)

        data = pd.merge(
            data,
            one[["modality", "modality_grouped"]],
            how="left",
            left_on=original_var,
            right_on="modality",
        )

        # Fill NA with reference modality
        ref_mod = ref_modalities[var].replace(var + "_", "")
        data["modality_grouped"] = data["modality_grouped"].fillna(ref_mod)

        data.rename(columns={"modality_grouped": var + "_grouped"}, inplace=True)
        data.drop(columns=["modality"], inplace=True)

        kept_variables.append(var)

    input_variables_fused = [v + "_grouped" for v in kept_variables]
    return data, input_variables_fused, removed_variables


def keep_variables_after_group_lasso(
    data: pd.DataFrame,
    modality_var_df: pd.DataFrame,
    beta_list: Sequence[float],
    input_variables: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Return variables kept/removed after a group-lasso step based on betas."""
    chosen_model_temp = beta_list[1:]  # strip intercept

    temp_modality_var_df = modality_var_df[modality_var_df['variable'].isin(input_variables)].copy()
    temp_modality_var_df["betas"] = [str(round(i, 6)) for i in chosen_model_temp]

    grouped = (
        temp_modality_var_df.groupby(['variable', 'betas'])['modality']
        .apply(lambda x: '_'.join(x))
        .reset_index()
        .rename(columns={"modality": "modality_grouped"})
    )

    temp_modality_var_df = temp_modality_var_df.merge(grouped, how="left", on=["variable", "betas"])

    kept_variables: List[str] = []
    removed_variables: List[str] = []

    for var in input_variables:
        one = temp_modality_var_df[temp_modality_var_df["variable"] == var]
        if (abs(one["betas"].astype(float)) < 1e-6).all():
            removed_variables.append(var)
        else:
            kept_variables.append(var)

    return kept_variables, removed_variables


# ---------------------------------------------------------------------------
# Grid Search (Generalised)
# ---------------------------------------------------------------------------

@dataclass
class GridSearchSettings:
    family: str
    var_nb_min: Optional[int] = None
    var_nb_max: Optional[int] = None
    parcimony_step: Optional[int] = None
    smoothness_step: Optional[int] = None
    lbd_fused: Optional[float] = None
    lbd_group: Optional[float] = None
    random_state: Optional[int] = 0  # reproducible CV by default


class GridSearch_Generalised:
    """Cross-validate over (group_lambda, fused_lambda) with optional bounds.
    """

    def __init__(self,family,var_nb_min=None,var_nb_max=None,parcimony_step=None,smoothness_step=None,lbd_fused=None,lbd_group=None,random_state=0):
        
        self.family = family
        self.var_nb_min = var_nb_min
        self.var_nb_max = var_nb_max
        self.parcimony_step = parcimony_step
        self.smoothness_step = smoothness_step
        self.lbd_group = lbd_group
        self.lbd_fused = lbd_fused
        self.random_state = random_state

        group1_any = (parcimony_step is not None) or (smoothness_step is not None)
        group1_full = (parcimony_step is not None) and (smoothness_step is not None)
    
        # Group 2 checks
        group2_any = (lbd_group is not None) or (lbd_fused is not None)
        group2_full = (lbd_group is not None) and (lbd_fused is not None)
        
        group3 = (var_nb_min is not None) or (var_nb_max is not None)
        # Errors:

        # 1. Must provide exactly one full group
        if group1_full == group2_full:  # Both True or both False
            raise ValueError("Provide either (parcimony & smoothness) OR (group & fused) values, but not both.")
            
        # 2. Partial groups
        if group1_any and not group1_full:
            raise ValueError("Both parcimony and smoothness values must be provided together.")
        if group2_any and not group2_full:
            raise ValueError("Both group and fused values must be provided together.")
    
        # 2. Can't use limits on variables for a given lambda
        if group3 and group2_any:  # Both True or both False
            raise ValueError("Input var_nb_min and var_nb_max doesn't work with a given set of lambdas.")

    @time_it
    def fit(self, data: pd.DataFrame, penalty_types: Dict[str, str], input_variables: List[str], target: str, offset: str, n_k_fold: int = 5):
        self.data = data
        self.penalty_types = penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.data_onehot, self.ref_modality_dict, self.input_var_onehot = get_data_ready_for_glm(
            self.data, self.input_variables, self.penalty_types, target, offset, "First"
        )

        # Prepare CV splits once (reused everywhere)
        self.cv_splits = list(KFold(n_splits=self.n_k_fold, shuffle=True, random_state=self.random_state).split(self.data_onehot))

        # Build modality mapping once
        modality_arr, variables_arr = [], []
        for val in self.input_variables:
            for x in self.data_onehot.columns:
                if x.startswith(val + "_"):
                    modality_arr.append(x.replace(val + "_", ""))
                    variables_arr.append(val)
        self.modality_var_df = pd.DataFrame(data={'variable': variables_arr, 'modality': modality_arr})
        self.fused_groups_dict = {val: [i for i, x in enumerate(self.data_onehot.columns) if (x.startswith(val + "_"))] for val in self.input_variables}
        self.target = target
        self.offset = offset

        # If explicit (group, fused) are provided, just evaluate them
        if (self.lbd_group is not None) and (self.lbd_fused is not None):
            results = pd.DataFrame(
                columns=['group_lambda', 'fused_lambda', 'variable_number', 'modalities_number', 'var_mod_details', 'Deviance_cv_train', 'Deviance_cv_test', 'variables', 'betas'],
                data=np.empty((1, 9), dtype=object)
            )

            if isinstance(self.lbd_group, (list, tuple)):
                if len(self.lbd_group) != len(self.lbd_fused):
                    raise ValueError(f"Fused and Group arrays must have the same size, got {len(self.lbd_group)} and {len(self.lbd_fused)}")
                for i, (g, f) in enumerate(zip(self.lbd_group, self.lbd_fused)):
                    print(f"Running for fused value = {f} and group value = {g}")
                    kept_per_fold = self._precompute_group_selection_per_fold(input_variables, g)
                    results = self.crossval_pair_lambdas(results, input_variables, g, f, i, kept_per_fold=kept_per_fold)
            else:
                kept_per_fold = self._precompute_group_selection_per_fold(input_variables, float(self.lbd_group))
                results = self.crossval_pair_lambdas(results, input_variables, float(self.lbd_group), float(self.lbd_fused), 0, kept_per_fold=kept_per_fold)

            self.lambda_curve = results
        else:
            # Otherwise, run your existing lambda-curve routine (not included here).
            raise NotImplementedError("Lambda-curve exploration is not included in this cleaned module.")

    def _intercept_only_poisson(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Offset-aware intercept-only Poisson model for the 'no variables kept' case."""
        # μ = exp(intercept) * exp(offset) ⇒ aggregate rate estimated on train
        mu_rate = X_train[self.target].sum() / np.exp(X_train[self.offset]).sum()
        pred_train = mu_rate * np.exp(X_train[self.offset].to_numpy())
        pred_test = mu_rate * np.exp(X_test[self.offset].to_numpy())
        return pred_train, pred_test

    def _precompute_group_selection_per_fold(self, inputs: List[str], group_lambda_val: float):
        """Run group-lasso once per CV fold and store kept variables for reuse."""
        kept_per_fold = []
        for train_idx, test_idx in self.cv_splits:
            X_train_cv = self.data_onehot.iloc[train_idx]

            grouped_model_temp = compute_group_lasso_glm(
                X_train_cv,
                inputs,
                self.target,
                self.offset,
                self.penalty_types,
                group_lambda=group_lambda_val,
                family=self.family,
            )

            temp_var_kept, _ = keep_variables_after_group_lasso(
                X_train_cv,
                self.modality_var_df,
                grouped_model_temp,
                inputs,
            )
            kept_per_fold.append((train_idx, test_idx, temp_var_kept))

        return kept_per_fold

    @time_it
    def fit_one_lambda(self, data: pd.DataFrame, inputs: List[str], lbd_fused: float, lbd_group: float):
        """Fit group-lasso to select variables, then fused-penalty GLM and refit intercept-only GLM on grouped data."""
        grouped_model_temp = compute_group_lasso_glm(
            data,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            lambda_group=lbd_group,
            family=self.family,
        )

        temp_var_kept, _ = keep_variables_after_group_lasso(data, self.modality_var_df, grouped_model_temp, inputs)

        temp_glm = custom_glm_with_fused_penalty(
            data,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            lambda_fused=lbd_fused,
            family=self.family,
        )

        temp_data, temp_var_kept, _ = group_modalities_based_on_betas(
            data,
            self.modality_var_df,
            temp_glm,
            inputs,
            self.ref_modality_dict,
        )

        if temp_var_kept:
            temp_data_onehot, temp_ref_modality_dict, _ = get_data_ready_for_glm(
                temp_data, temp_var_kept, self.penalty_types, self.target, self.offset, "First"
            )
            betas_temp = compute_glm_no_pen(temp_data_onehot, temp_var_kept, self.target, self.offset, family=self.family)
            betas = betas_temp
            return len(temp_ref_modality_dict), temp_ref_modality_dict, betas, temp_var_kept
        else:
            print("NO VARIABLES kept after grouping/fusion.")
            betas = np.array([data[self.target].mean()])
            return 0, {}, betas, []

    @time_it
    def crossval_pair_lambdas(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        group_lambda_val: float,
        fused_lambda_val: float,
        counter: int,
        kept_per_fold: Optional[List[Tuple[np.ndarray, np.ndarray, List[str]]]] = None,
    ) -> pd.DataFrame:
        """Evaluate one (group_lambda, fused_lambda) pair via K-fold CV and store results."""

        print(f"Model with group lambda = {group_lambda_val} and fused lambda = {fused_lambda_val}")

        if kept_per_fold is None:
            kept_per_fold = self._precompute_group_selection_per_fold(inputs, group_lambda_val)

        error_train_total = 0.0
        error_test_total = 0.0

        for (train_idx, test_idx, temp_var_kept) in kept_per_fold:
            X_train_cv = self.data_onehot.iloc[train_idx]
            X_test_cv = self.data_onehot.iloc[test_idx]

            if temp_var_kept:
                temp_betas = custom_glm_with_fused_penalty(
                    X_train_cv,
                    temp_var_kept,
                    self.target,
                    self.offset,
                    self.penalty_types,
                    lambda_fused=fused_lambda_val,
                    family=self.family,
                )
                kept_onehot = get_onehot_columns(X_train_cv, temp_var_kept)
                predict_train = np.exp(
                    X_train_cv[kept_onehot].to_numpy() @ temp_betas[1:] + temp_betas[0] + X_train_cv[self.offset].to_numpy()
                )
                predict_test = np.exp(
                    X_test_cv[kept_onehot].to_numpy() @ temp_betas[1:] + temp_betas[0] + X_test_cv[self.offset].to_numpy()
                )
            else:
                # Offset-aware intercept-only predictions
                predict_train, predict_test = self._intercept_only_poisson(X_train_cv, X_test_cv)

            error_train = Compute_Poisson_Deviance(predict_train, np.array(X_train_cv[self.target]), np.array(X_train_cv[self.offset]))
            error_test = Compute_Poisson_Deviance(predict_test, np.array(X_test_cv[self.target]), np.array(X_test_cv[self.offset]))
            error_train_total += error_train
            error_test_total += error_test

        error_train_mean = error_train_total / self.n_k_fold
        error_test_mean = error_test_total / self.n_k_fold

        results.at[counter, 'fused_lambda'] = fused_lambda_val
        results.at[counter, 'group_lambda'] = group_lambda_val
        results.at[counter, "Deviance_cv_train"] = error_train_mean
        results.at[counter, "Deviance_cv_test"] = error_test_mean

        # FINAL MODEL on full data
        nb_modalities, total_modality_dict, betas, kept_var = self.fit_one_lambda(self.data_onehot, inputs, fused_lambda_val, group_lambda_val)
        results.at[counter, "modalities_number"] = nb_modalities
        results.at[counter, "var_mod_details"] = total_modality_dict
        results.at[counter, 'betas'] = betas
        results.at[counter, "variables"] = " ".join(map(str, kept_var))
        results.at[counter, "variable_number"] = len(kept_var)

        results = results.sort_values(by=['group_lambda', 'fused_lambda'], ascending=True).reset_index(drop=True)
        return results
