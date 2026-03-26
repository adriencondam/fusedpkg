from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .glm_preprocessing import (
    build_modality_lookup,
    group_modalities_by_coefficient,
    prepare_categorical_glm_data,
    select_active_variables,
)
from .grid_search_common import (
    build_intercept_only_coefficients,
    compute_cv_loss,
    create_results_table,
    fit_unpenalized_glm,
    predict_mean_from_linear_predictor,
    subset_coefficients_for_variables,
    timed,
)
from .penalized_glm import SolverConfig, build_group_penalized_glm, solve_cvxpy_problem


class GridSearch_Group:
    """Grid search using only the group penalty."""

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
            raise ValueError("Provide either parcimony_step or lbd_group, but not both.")
        if curve_mode and (var_nb_min is None):
            raise ValueError("var_nb_min must be provided when parcimony_step is used.")

        self._group_full_cache = None
        self._group_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}

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

        if self.lbd_group is None:
            self.get_lambda_curve()
            return

        if isinstance(self.lbd_group, (list, tuple, np.ndarray)):
            lambdas = [float(value) for value in self.lbd_group]
        else:
            lambdas = [float(self.lbd_group)]

        results = create_results_table(len(lambdas))
        for index, lambda_value in enumerate(lambdas):
            results = self.crossval_group_lambda(results, self.input_variables, lambda_value, index)
        self.lambda_curve = results

    def _get_group_full_fit(self):
        if self._group_full_cache is not None:
            return self._group_full_cache

        full_fit = build_group_penalized_glm(
            self.data_onehot,
            self.input_variables,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        problem, beta, intercept, group_lambda, onehot_columns, _ = full_fit
        self._group_full_cache = (problem, beta, intercept, group_lambda, onehot_columns)
        return self._group_full_cache

    def _get_group_fold_fit(self, fold_id: int, train_idx: np.ndarray, inputs: List[str]):
        cache_key = (fold_id, tuple(sorted(inputs)))
        if cache_key in self._group_fold_cache:
            return self._group_fold_cache[cache_key]

        train_data = self.data_onehot.iloc[train_idx]
        fold_fit = build_group_penalized_glm(
            train_data,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        problem, beta, intercept, group_lambda, onehot_columns, _ = fold_fit
        self._group_fold_cache[cache_key] = (problem, beta, intercept, group_lambda, onehot_columns)
        return self._group_fold_cache[cache_key]

    def _cv_fold_loss_group(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        inputs: List[str],
        group_lambda: float,
    ) -> Tuple[float, float]:
        train_df = self.data_onehot.iloc[train_idx]
        test_df = self.data_onehot.iloc[test_idx]

        y_train = train_df[self.target].to_numpy(dtype=float)
        y_test = test_df[self.target].to_numpy(dtype=float)
        train_offset = train_df[self.offset].to_numpy(dtype=float)
        test_offset = test_df[self.offset].to_numpy(dtype=float)

        problem, beta, intercept, lambda_parameter, onehot_columns = self._get_group_fold_fit(fold_id, train_idx, inputs)
        lambda_parameter.value = float(group_lambda)
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
    def fit_one_lambda(self, group_lambda: float):
        problem, beta, intercept, lambda_parameter, onehot_columns = self._get_group_full_fit()
        lambda_parameter.value = float(group_lambda)
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
    def crossval_group_lambda(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        group_lambda: float,
        counter: int,
    ) -> pd.DataFrame:
        print(f"Model with group lambda = {group_lambda}")

        if self.n_jobs > 1:
            for fold_id, (train_idx, _) in enumerate(self.cv_splits):
                self._get_group_fold_fit(fold_id, train_idx, inputs)

        train_error = 0.0
        test_error = 0.0

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(self._cv_fold_loss_group, fold_id, train_idx, test_idx, inputs, group_lambda)
                    for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
                ]
                for future in futures:
                    fold_train, fold_test = future.result()
                    train_error += fold_train
                    test_error += fold_test
        else:
            for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits):
                fold_train, fold_test = self._cv_fold_loss_group(fold_id, train_idx, test_idx, inputs, group_lambda)
                train_error += fold_train
                test_error += fold_test

        results.at[counter, "group_lambda"] = group_lambda
        results.at[counter, "fused_lambda"] = 0.0
        results.at[counter, "Deviance_cv_train"] = train_error / self.n_k_fold
        results.at[counter, "Deviance_cv_test"] = test_error / self.n_k_fold

        modality_count, modality_details, coefficients, kept_variables = self.fit_one_lambda(group_lambda)
        results.at[counter, "modalities_number"] = modality_count
        results.at[counter, "var_mod_details"] = modality_details
        results.at[counter, "betas"] = coefficients
        results.at[counter, "variables"] = " ".join(map(str, kept_variables))
        results.at[counter, "variable_number"] = len(kept_variables)
        return results.sort_values(by=["group_lambda"], ascending=True).reset_index(drop=True)

    @timed
    def get_lambda_curve(self):
        if self.parcimony_step is None:
            raise ValueError("parcimony_step must be provided for lambda-curve exploration.")

        print("----------------------------------")
        print("Group-only: Get range of Group Lambdas")
        print("----------------------------------\n")

        max_lambda_reached = False
        lambda_candidate = 0.1
        problem, beta, _, lambda_parameter, onehot_columns = self._get_group_full_fit()

        while not max_lambda_reached:
            print(f"Running group GLMs until reaching maximum lambda : {lambda_candidate}")
            lambda_parameter.value = float(lambda_candidate)
            solve_cvxpy_problem(problem, self.solver_cfg)

            kept_variables = select_active_variables(np.asarray(beta.value, dtype=float), onehot_columns, self.input_variables)
            if (len(kept_variables) == 0) or (len(kept_variables) <= int(self.var_nb_min)):
                max_lambda_reached = True
                max_group_lambda = float(lambda_candidate)
            else:
                lambda_candidate *= 10

        lambda_min = 0.1
        step_size = (max_group_lambda - lambda_min) / float(self.parcimony_step)
        candidate_lambdas = [lambda_min + step_size * step for step in range(0, self.parcimony_step + 1)]

        results = create_results_table(len(candidate_lambdas))
        for index, lambda_value in enumerate(candidate_lambdas):
            results = self.crossval_group_lambda(results, self.input_variables, float(lambda_value), index)
        self.lambda_curve = results

    def plot_curve(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.lambda_curve["variable_number"], self.lambda_curve["Deviance_cv_test"])
        for _, row in self.lambda_curve.iterrows():
            plt.text(
                row["variable_number"],
                row["Deviance_cv_test"],
                f"{np.around(row['group_lambda'])}",
                fontsize=8,
                ha="center",
                va="bottom",
            )
        plt.xlabel("Final Variable Number")
        plt.ylabel("CV loss (Poisson deviance or Gaussian MSE)")
        plt.title("Group-only grid search")
        plt.tight_layout()
        plt.show()
