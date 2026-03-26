from __future__ import annotations

import time
from functools import wraps
from typing import List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from .glm_preprocessing import get_onehot_column_names
from .penalized_glm import (
    SolverConfig,
    build_glm_log_likelihood,
    combine_intercept_and_coefficients,
    solve_cvxpy_problem,
)

_INDENT_LEVEL = 0


def timed(func):
    """Print nested execution timings for long-running grid-search steps."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _INDENT_LEVEL
        _INDENT_LEVEL += 1
        tabs = "\t" * (_INDENT_LEVEL - 1)
        start = time.time()
        print(f"{tabs}Executing <{func.__name__}>")
        result = func(*args, **kwargs)
        duration = time.time() - start
        minutes = int(duration // 60)
        seconds = duration % 60
        print(f"{tabs}Function <{func.__name__}> execution time : {minutes:.0f} minutes and {seconds:.0f} seconds")
        _INDENT_LEVEL -= 1
        return result

    return wrapper


def create_results_table(row_count: int) -> pd.DataFrame:
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
        data=np.empty((row_count, 9), dtype=object),
    )


def poisson_deviance_from_mean(
    predicted_mean: Sequence[float],
    y: Sequence[float],
    exposure: Optional[Sequence[float]] = None,
) -> float:
    """Compute the scaled Poisson deviance from predicted means."""
    predicted_mean_array = np.asarray(predicted_mean, dtype=float)
    y_array = np.asarray(y, dtype=float)

    epsilon = 1e-12
    predicted_mean_array = np.clip(predicted_mean_array, epsilon, None)

    mask = y_array > 0
    term = np.zeros_like(y_array)
    term[mask] = y_array[mask] * (np.log(y_array[mask]) - np.log(predicted_mean_array[mask]))
    deviance = term - (y_array - predicted_mean_array)

    if exposure is None:
        return float(deviance.mean())

    exposure_array = np.asarray(exposure, dtype=float)
    exposure_array = np.clip(exposure_array, epsilon, None)
    return float(deviance.sum() / exposure_array.sum())


def predict_mean_from_linear_predictor(family: str, eta: np.ndarray) -> np.ndarray:
    family_name = family.lower()
    if family_name == "poisson":
        return np.exp(eta)
    if family_name == "gaussian":
        return eta
    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def compute_cv_loss(
    family: str,
    y: np.ndarray,
    prediction: np.ndarray,
    offset_array: Optional[np.ndarray] = None,
) -> float:
    family_name = family.lower()
    y_array = np.asarray(y, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)

    if family_name == "poisson":
        exposure = None if offset_array is None else np.exp(np.asarray(offset_array, dtype=float))
        return poisson_deviance_from_mean(prediction_array, y_array, exposure=exposure)
    if family_name == "gaussian":
        return float(np.mean((y_array - prediction_array) ** 2))

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def intercept_only_predictions(
    family: str,
    y_train: np.ndarray,
    offset_train: np.ndarray,
    offset_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    family_name = family.lower()
    y_train_array = np.asarray(y_train, dtype=float)
    train_offset = np.asarray(offset_train, dtype=float)
    test_offset = np.asarray(offset_test, dtype=float)

    if family_name == "poisson":
        train_exposure = np.exp(train_offset)
        test_exposure = np.exp(test_offset)
        rate = y_train_array.sum() / max(train_exposure.sum(), 1e-12)
        return rate * train_exposure, rate * test_exposure
    if family_name == "gaussian":
        intercept = float(np.mean(y_train_array - train_offset))
        return intercept + train_offset, intercept + test_offset

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")


def subset_coefficients_for_variables(
    beta_full: np.ndarray,
    full_onehot_columns: Sequence[str],
    data_onehot: pd.DataFrame,
    variables_subset: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Extract coefficients aligned with the one-hot columns for the selected variables."""
    subset_columns = get_onehot_column_names(data_onehot, variables_subset)
    column_to_index = {column: index for index, column in enumerate(full_onehot_columns)}
    indices = [column_to_index[column] for column in subset_columns]
    return np.asarray(beta_full, dtype=float)[indices], subset_columns


def fit_unpenalized_glm(
    data_onehot: pd.DataFrame,
    inputs: Sequence[str],
    target: str,
    offset: str,
    family: str,
    solver_config: SolverConfig,
) -> np.ndarray:
    """Refit an unpenalized GLM on the grouped design matrix."""
    onehot_columns = get_onehot_column_names(data_onehot, inputs)
    X = data_onehot[onehot_columns].to_numpy(dtype=float)
    y = data_onehot[target].to_numpy(dtype=float)
    offset_array = data_onehot[offset].to_numpy(dtype=float)

    beta = cp.Variable(X.shape[1])
    intercept = cp.Variable()
    log_likelihood = build_glm_log_likelihood(family, X, y, offset_array, beta, intercept)
    problem = cp.Problem(cp.Minimize(-log_likelihood))
    solve_cvxpy_problem(problem, solver_config)
    return combine_intercept_and_coefficients(intercept, beta)


def build_intercept_only_coefficients(
    family: str,
    y: np.ndarray,
    offset_array: np.ndarray,
) -> np.ndarray:
    family_name = family.lower()
    y_array = np.asarray(y, dtype=float)
    offsets = np.asarray(offset_array, dtype=float)

    if family_name == "poisson":
        rate = float(y_array.sum() / max(np.exp(offsets).sum(), 1e-12))
        return np.array([float(np.log(max(rate, 1e-12)))], dtype=float)
    if family_name == "gaussian":
        return np.array([float(np.mean(y_array - offsets))], dtype=float)

    raise ValueError(f"Unsupported family: {family!r} (supported: Poisson, Gaussian)")
