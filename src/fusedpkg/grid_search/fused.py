from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..glm.preprocessing import (
    build_modality_lookup,
    group_modalities_by_coefficient,
    prepare_categorical_glm_data,
    select_active_variables,
)
from .common import (
    build_intercept_only_coefficients,
    compute_cv_loss,
    create_results_table,
    fit_unpenalized_glm,
    predict_mean_from_linear_predictor,
    subset_coefficients_for_variables,
    timed,
)
from ..glm.penalized import SolverConfig, build_fused_penalized_glm, solve_cvxpy_problem


class GridSearch_Fused:
    """Grid search using only the fused or generalized fused penalty."""

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
            raise ValueError("Provide either smoothness_step or lbd_fused, but not both.")

        self._fused_full_cache: Dict[Tuple[str, ...], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}
        self._fused_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}

    @timed
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
        self.target = target
        self.offset = offset
        self.n_k_fold = n_k_fold

        self.data_onehot, self.reference_modalities, _ = prepare_categorical_glm_data(
            self.data,
            self.input_variables,
            self.penalty_types,
            self.target,
            self.offset,
            "First",
        )
        self.cv_splits = list(
            KFold(n_splits=self.n_k_fold, shuffle=True, random_state=self.random_state).split(self.data_onehot)
        )
        self.modality_lookup = build_modality_lookup(self.data_onehot, self.input_variables)

        if self.lbd_fused is None:
            self.get_lambda_curve()
            return

        if isinstance(self.lbd_fused, (list, tuple, np.ndarray)):
            lambdas = [float(value) for value in self.lbd_fused]
        else:
            lambdas = [float(self.lbd_fused)]

        results = create_results_table(len(lambdas))
        for index, lambda_value in enumerate(lambdas):
            results = self.crossval_fused_lambda(results, self.input_variables, lambda_value, index)
        self.lambda_curve = results

    def _get_fused_full_fit(self, inputs: List[str]):
        cache_key = tuple(sorted(inputs))
        if cache_key in self._fused_full_cache:
            return self._fused_full_cache[cache_key]

        full_fit = build_fused_penalized_glm(
            self.data_onehot,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._fused_full_cache[cache_key] = full_fit
        return full_fit

    def _get_fused_fold_fit(self, fold_id: int, train_idx: np.ndarray, inputs: List[str]):
        cache_key = (fold_id, tuple(sorted(inputs)))
        if cache_key in self._fused_fold_cache:
            return self._fused_fold_cache[cache_key]

        train_data = self.data_onehot.iloc[train_idx]
        fold_fit = build_fused_penalized_glm(
            train_data,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._fused_fold_cache[cache_key] = fold_fit
        return fold_fit

    def _cv_fold_loss_fused(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        inputs: List[str],
        fused_lambda: float,
    ) -> Tuple[float, float]:
        train_df = self.data_onehot.iloc[train_idx]
        test_df = self.data_onehot.iloc[test_idx]

        y_train = train_df[self.target].to_numpy(dtype=float)
        y_test = test_df[self.target].to_numpy(dtype=float)
        train_offset = train_df[self.offset].to_numpy(dtype=float)
        test_offset = test_df[self.offset].to_numpy(dtype=float)

        problem, beta, intercept, lambda_parameter, onehot_columns = self._get_fused_fold_fit(fold_id, train_idx, inputs)
        lambda_parameter.value = float(fused_lambda)
        solve_cvxpy_problem(problem, self.solver_cfg)

        coefficients = np.asarray(beta.value, dtype=float)
        X_train = train_df[onehot_columns].to_numpy(dtype=float)
        X_test = test_df[onehot_columns].to_numpy(dtype=float)
        eta_train = X_train @ coefficients + float(intercept.value) + train_offset
        eta_test = X_test @ coefficients + float(intercept.value) + test_offset

        train_prediction = predict_mean_from_linear_predictor(self.family, eta_train)
        test_prediction = predict_mean_from_linear_predictor(self.family, eta_test)
        return (
            compute_cv_loss(self.family, y_train, train_prediction, offset_array=train_offset),
            compute_cv_loss(self.family, y_test, test_prediction, offset_array=test_offset),
        )

    @timed
    def fit_one_lambda(self, fused_lambda: float):
        problem, beta, intercept, lambda_parameter, onehot_columns = self._get_fused_full_fit(self.input_variables)
        lambda_parameter.value = float(fused_lambda)
        solve_cvxpy_problem(problem, self.solver_cfg)

        kept_variables = select_active_variables(np.asarray(beta.value, dtype=float), onehot_columns, self.input_variables)
        if kept_variables:
            beta_subset, _ = subset_coefficients_for_variables(
                np.asarray(beta.value, dtype=float),
                onehot_columns,
                self.data_onehot,
                kept_variables,
            )
            coefficient_vector = np.concatenate(([float(intercept.value)], beta_subset))

            grouped_data, kept_grouped_variables, _ = group_modalities_by_coefficient(
                self.data_onehot.copy(),
                self.modality_lookup,
                coefficient_vector,
                kept_variables,
                self.reference_modalities,
            )
            if kept_grouped_variables:
                grouped_onehot, grouped_reference_modalities, _ = prepare_categorical_glm_data(
                    grouped_data,
                    kept_grouped_variables,
                    self.penalty_types,
                    self.target,
                    self.offset,
                    "First",
                )
                final_coefficients = fit_unpenalized_glm(
                    grouped_onehot,
                    kept_grouped_variables,
                    self.target,
                    self.offset,
                    self.family,
                    self.solver_cfg,
                )
                return len(grouped_reference_modalities), grouped_reference_modalities, final_coefficients, kept_grouped_variables

        baseline_coefficients = build_intercept_only_coefficients(
            self.family,
            self.data_onehot[self.target].to_numpy(dtype=float),
            self.data_onehot[self.offset].to_numpy(dtype=float),
        )
        return 0, {}, baseline_coefficients, []

    @timed
    def crossval_fused_lambda(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        fused_lambda: float,
        counter: int,
    ) -> pd.DataFrame:
        print(f"Model with fused lambda = {fused_lambda}")

        if self.n_jobs > 1:
            for fold_id, (train_idx, _) in enumerate(self.cv_splits):
                self._get_fused_fold_fit(fold_id, train_idx, inputs)

        train_error = 0.0
        test_error = 0.0

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(self._cv_fold_loss_fused, fold_id, train_idx, test_idx, inputs, fused_lambda)
                    for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
                ]
                for future in futures:
                    fold_train, fold_test = future.result()
                    train_error += fold_train
                    test_error += fold_test
        else:
            for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits):
                fold_train, fold_test = self._cv_fold_loss_fused(fold_id, train_idx, test_idx, inputs, fused_lambda)
                train_error += fold_train
                test_error += fold_test

        results.at[counter, "group_lambda"] = 0.0
        results.at[counter, "fused_lambda"] = fused_lambda
        results.at[counter, "Deviance_cv_train"] = train_error / self.n_k_fold
        results.at[counter, "Deviance_cv_test"] = test_error / self.n_k_fold

        modality_count, modality_details, coefficients, kept_variables = self.fit_one_lambda(fused_lambda)
        results.at[counter, "modalities_number"] = modality_count
        results.at[counter, "var_mod_details"] = modality_details
        results.at[counter, "betas"] = coefficients
        results.at[counter, "variables"] = " ".join(map(str, kept_variables))
        results.at[counter, "variable_number"] = len(kept_variables)
        return results.sort_values(by=["fused_lambda"], ascending=True).reset_index(drop=True)

    @timed
    def get_lambda_curve(self):
        if self.smoothness_step is None:
            raise ValueError("smoothness_step must be provided for lambda-curve exploration.")

        print("----------------------------------")
        print("Fused-only: Get range of Fused Lambdas")
        print("----------------------------------\n")

        max_lambda_reached = False
        lambda_candidate = 0.1
        problem, beta, _, lambda_parameter, onehot_columns = self._get_fused_full_fit(self.input_variables)

        while not max_lambda_reached:
            print(f"Running fused GLMs until reaching maximum lambda : {lambda_candidate}")
            lambda_parameter.value = float(lambda_candidate)
            solve_cvxpy_problem(problem, self.solver_cfg)

            kept_variables = select_active_variables(np.asarray(beta.value, dtype=float), onehot_columns, self.input_variables)
            if len(kept_variables) == 0:
                max_lambda_reached = True
                max_fused_lambda = float(lambda_candidate)
            else:
                lambda_candidate *= 10

        lambda_min = 0.1
        step_size = (max_fused_lambda - lambda_min) / float(self.smoothness_step)
        candidate_lambdas = [lambda_min + step_size * step for step in range(0, self.smoothness_step + 1)]

        results = create_results_table(len(candidate_lambdas))
        for index, lambda_value in enumerate(candidate_lambdas):
            results = self.crossval_fused_lambda(results, self.input_variables, float(lambda_value), index)
        self.lambda_curve = results

    def plot_curve(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.lambda_curve["variable_number"], self.lambda_curve["Deviance_cv_test"])
        for _, row in self.lambda_curve.iterrows():
            plt.text(
                row["variable_number"],
                row["Deviance_cv_test"],
                f"{np.around(row['fused_lambda'])}",
                fontsize=8,
                ha="center",
                va="bottom",
            )
        plt.xlabel("Final Variable Number")
        plt.ylabel("CV loss (Poisson deviance or Gaussian MSE)")
        plt.title("Fused-only grid search")
        plt.tight_layout()
        plt.show()
