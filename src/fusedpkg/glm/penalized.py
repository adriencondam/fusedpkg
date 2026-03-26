from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from .preprocessing import get_onehot_column_names


def build_glm_log_likelihood(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    offset_array: np.ndarray,
    beta: cp.Expression,
    intercept: cp.Expression,
) -> cp.Expression:
    """Return the concave log-likelihood expression for the requested family."""
    eta = X @ beta + intercept + offset_array
    family_name = family.lower()

    if family_name == "poisson":
        return cp.sum(cp.multiply(y, eta) - cp.exp(eta))
    if family_name == "gaussian":
        return -0.5 * cp.sum_squares(y - eta)

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def _group_onehot_columns_by_variable(
    input_var_onehot: Sequence[str],
    penalty_types: Dict[str, str],
) -> Dict[str, List[int]]:
    grouped_columns: Dict[str, List[int]] = {}
    for variable in penalty_types:
        indices = [index for index, column in enumerate(input_var_onehot) if column.startswith(f"{variable}_")]
        if indices:
            grouped_columns[variable] = indices
    return grouped_columns


@dataclass(frozen=True)
class SolverConfig:
    solver: str = "CLARABEL"
    warm_start: bool = True
    verbose: bool = False


def build_group_penalized_glm(
    data: pd.DataFrame,
    input_variables: Sequence[str],
    target: str,
    offset: str,
    penalty_types: Dict[str, str],
    family: str = "Poisson",
) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str], Dict[str, List[int]]]:
    """Build the reusable group-penalized GLM optimization objects."""
    input_var_onehot = get_onehot_column_names(data, input_variables)
    X = data[input_var_onehot].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    offset_array = data[offset].to_numpy(dtype=float)

    beta = cp.Variable(X.shape[1])
    intercept = cp.Variable()
    group_lambda = cp.Parameter(nonneg=True, name="group_lambda")

    log_likelihood = build_glm_log_likelihood(family, X, y, offset_array, beta, intercept)
    selected_penalty_types = {
        variable: penalty for variable, penalty in penalty_types.items() if any(variable.startswith(prefix) for prefix in input_variables)
    }
    grouped_columns = _group_onehot_columns_by_variable(input_var_onehot, selected_penalty_types)

    group_penalty = 0
    for indices in grouped_columns.values():
        weight = np.sqrt(len(indices))
        group_penalty += cp.norm1(weight * beta[indices])

    problem = cp.Problem(cp.Minimize(-log_likelihood + group_lambda * group_penalty))
    return problem, beta, intercept, group_lambda, input_var_onehot, grouped_columns


def _reference_count(row_count: int, column_sums: np.ndarray, group_indices: List[int]) -> float:
    return float(row_count - column_sums[group_indices].sum())


def build_fused_penalized_glm(
    data: pd.DataFrame,
    input_variables: Sequence[str],
    target: str,
    offset: str,
    penalty_types: Dict[str, str],
    family: str = "Poisson",
) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]:
    """Build the reusable fused-penalized GLM optimization objects."""
    input_var_onehot = get_onehot_column_names(data, input_variables)
    X = data[input_var_onehot].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    offset_array = data[offset].to_numpy(dtype=float)

    beta = cp.Variable(X.shape[1])
    intercept = cp.Variable()
    fused_lambda = cp.Parameter(nonneg=True, name="fused_lambda")

    log_likelihood = build_glm_log_likelihood(family, X, y, offset_array, beta, intercept)
    selected_penalty_types = {
        variable: penalty for variable, penalty in penalty_types.items() if any(variable.startswith(prefix) for prefix in input_variables)
    }
    grouped_columns = _group_onehot_columns_by_variable(input_var_onehot, selected_penalty_types)

    column_sums = data[input_var_onehot].sum(axis=0).to_numpy(dtype=float)
    row_count = float(len(data))

    generalized_graph_size = 0
    for variable, indices in grouped_columns.items():
        if selected_penalty_types.get(variable) == "g_fused":
            generalized_graph_size += len(indices) + 1
    generalized_graph_size = max(generalized_graph_size, 1)

    fused_penalty = 0
    generalized_fused_penalty = 0

    for variable, indices in grouped_columns.items():
        reference_count = _reference_count(len(data), column_sums, indices)
        penalty_name = selected_penalty_types.get(variable)

        if penalty_name == "fused":
            first_weight = np.sqrt((column_sums[indices[0]] + reference_count) / row_count)
            fused_penalty += cp.norm1(first_weight * beta[indices[0]])
            for index in range(1, len(indices)):
                weight = np.sqrt((column_sums[indices[index]] + column_sums[indices[index - 1]]) / row_count)
                fused_penalty += cp.norm1(weight * (beta[indices[index]] - beta[indices[index - 1]]))
        elif penalty_name == "g_fused":
            scale = len(indices) / generalized_graph_size
            for left_index in range(len(indices)):
                reference_weight = scale * np.sqrt((column_sums[indices[left_index]] + reference_count) / row_count)
                generalized_fused_penalty += cp.norm1(reference_weight * beta[indices[left_index]])
                for right_index in range(left_index, len(indices)):
                    weight = scale * np.sqrt((column_sums[indices[left_index]] + column_sums[indices[right_index]]) / row_count)
                    generalized_fused_penalty += cp.norm1(
                        weight * (beta[indices[left_index]] - beta[indices[right_index]])
                    )

    problem = cp.Problem(cp.Minimize(-log_likelihood + fused_lambda * (fused_penalty + generalized_fused_penalty)))
    return problem, beta, intercept, fused_lambda, input_var_onehot


def solve_cvxpy_problem(problem: cp.Problem, solver_config: SolverConfig) -> None:
    """Solve the supplied CVXPY problem with the configured solver."""
    problem.solve(
        solver=getattr(cp, solver_config.solver),
        warm_start=solver_config.warm_start,
        verbose=solver_config.verbose,
    )


def combine_intercept_and_coefficients(intercept: cp.Variable, beta: cp.Variable) -> np.ndarray:
    """Return coefficients as [intercept, beta...]."""
    return np.concatenate(([float(intercept.value)], np.asarray(beta.value, dtype=float)))
