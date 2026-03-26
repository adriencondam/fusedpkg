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
    get_onehot_column_names,
    group_modalities_by_coefficient,
    prepare_categorical_glm_data,
    select_active_variables,
)
from .grid_search_common import (
    build_intercept_only_coefficients,
    compute_cv_loss,
    create_results_table,
    fit_unpenalized_glm,
    intercept_only_predictions,
    predict_mean_from_linear_predictor,
    timed,
)
from .penalized_glm import (
    SolverConfig,
    build_fused_penalized_glm,
    build_group_penalized_glm,
    combine_intercept_and_coefficients,
    solve_cvxpy_problem,
)


class GridSearch_Generalised:
    """Two-stage grid search over group and fused penalties."""

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

        curve_inputs_supplied = (parcimony_step is not None) or (smoothness_step is not None)
        curve_inputs_complete = (parcimony_step is not None) and (smoothness_step is not None)
        fixed_inputs_supplied = (lbd_group is not None) or (lbd_fused is not None)
        fixed_inputs_complete = (lbd_group is not None) and (lbd_fused is not None)
        variable_bounds_supplied = (var_nb_min is not None) or (var_nb_max is not None)

        if curve_inputs_complete == fixed_inputs_complete:
            raise ValueError("Provide either (parcimony_step, smoothness_step) or (lbd_group, lbd_fused), but not both.")
        if curve_inputs_supplied and not curve_inputs_complete:
            raise ValueError("parcimony_step and smoothness_step must be provided together.")
        if fixed_inputs_supplied and not fixed_inputs_complete:
            raise ValueError("lbd_group and lbd_fused must be provided together.")
        if variable_bounds_supplied and fixed_inputs_supplied:
            raise ValueError("var_nb_min/var_nb_max cannot be used when lbd_group and lbd_fused are fixed.")

        self._group_full_cache = None
        self._fused_full_cache: Dict[Tuple[str, ...], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}
        self._group_fold_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, List[str]]] = {}
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

        if (self.lbd_group is None) and (self.lbd_fused is None):
            self.get_lambda_curve()
            return

        if isinstance(self.lbd_group, (list, tuple, np.ndarray)):
            group_lambdas = [float(value) for value in self.lbd_group]
            fused_lambdas = [float(value) for value in self.lbd_fused]
            if len(group_lambdas) != len(fused_lambdas):
                raise ValueError(
                    f"lbd_group and lbd_fused must have the same size, got {len(group_lambdas)} and {len(fused_lambdas)}."
                )
        else:
            group_lambdas = [float(self.lbd_group)]
            fused_lambdas = [float(self.lbd_fused)]

        results = create_results_table(len(group_lambdas))
        for index, (group_lambda, fused_lambda) in enumerate(zip(group_lambdas, fused_lambdas)):
            kept_per_fold = self._precompute_group_selection_per_fold(self.input_variables, group_lambda, use_cache=True)
            results = self.crossval_pair_lambdas(
                results,
                self.input_variables,
                group_lambda,
                fused_lambda,
                index,
                kept_per_fold=kept_per_fold,
            )
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

    def _get_fused_fold_fit(self, fold_id: int, train_idx: np.ndarray, kept_vars: List[str]):
        cache_key = (fold_id, tuple(sorted(kept_vars)))
        if cache_key in self._fused_fold_cache:
            return self._fused_fold_cache[cache_key]

        train_data = self.data_onehot.iloc[train_idx]
        fold_fit = build_fused_penalized_glm(
            train_data,
            kept_vars,
            self.target,
            self.offset,
            self.penalty_types,
            family=self.family,
        )
        self._fused_fold_cache[cache_key] = fold_fit
        return fold_fit

    def _cv_fold_loss_generalized(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        kept_vars: List[str],
        fused_lambda: float,
    ) -> Tuple[float, float]:
        train_df = self.data_onehot.iloc[train_idx]
        test_df = self.data_onehot.iloc[test_idx]

        y_train = train_df[self.target].to_numpy(dtype=float)
        y_test = test_df[self.target].to_numpy(dtype=float)
        train_offset = train_df[self.offset].to_numpy(dtype=float)
        test_offset = test_df[self.offset].to_numpy(dtype=float)

        if kept_vars:
            problem, beta, intercept, lambda_parameter, _ = self._get_fused_fold_fit(fold_id, train_idx, kept_vars)
            lambda_parameter.value = float(fused_lambda)
            solve_cvxpy_problem(problem, self.solver_cfg)

            coefficients = np.asarray(beta.value, dtype=float)
            onehot_columns = get_onehot_column_names(train_df, kept_vars)
            X_train = train_df[onehot_columns].to_numpy(dtype=float)
            X_test = test_df[onehot_columns].to_numpy(dtype=float)
            eta_train = X_train @ coefficients + float(intercept.value) + train_offset
            eta_test = X_test @ coefficients + float(intercept.value) + test_offset
            train_prediction = predict_mean_from_linear_predictor(self.family, eta_train)
            test_prediction = predict_mean_from_linear_predictor(self.family, eta_test)
        else:
            train_prediction, test_prediction = intercept_only_predictions(
                self.family,
                y_train,
                train_offset,
                test_offset,
            )

        train_loss = compute_cv_loss(self.family, y_train, train_prediction, offset_array=train_offset)
        test_loss = compute_cv_loss(self.family, y_test, test_prediction, offset_array=test_offset)
        return train_loss, test_loss

    def _precompute_group_selection_per_fold(
        self,
        inputs: List[str],
        group_lambda: float,
        use_cache: bool = True,
    ) -> List[Tuple[int, np.ndarray, np.ndarray, List[str]]]:
        if not use_cache:
            kept_per_fold: List[Tuple[int, np.ndarray, np.ndarray, List[str]]] = []
            for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits):
                train_data = self.data_onehot.iloc[train_idx]
                problem, beta, _, lambda_parameter, onehot_columns, _ = build_group_penalized_glm(
                    train_data,
                    inputs,
                    self.target,
                    self.offset,
                    self.penalty_types,
                    family=self.family,
                )
                lambda_parameter.value = float(group_lambda)
                solve_cvxpy_problem(problem, self.solver_cfg)
                kept_variables = select_active_variables(np.asarray(beta.value, dtype=float), onehot_columns, inputs)
                kept_per_fold.append((fold_id, train_idx, test_idx, kept_variables))
            return kept_per_fold

        for fold_id, (train_idx, _) in enumerate(self.cv_splits):
            self._get_group_fold_fit(fold_id, train_idx, inputs)

        def solve_fold(fold_id: int, train_idx: np.ndarray, test_idx: np.ndarray):
            problem, beta, _, lambda_parameter, onehot_columns = self._get_group_fold_fit(fold_id, train_idx, inputs)
            lambda_parameter.value = float(group_lambda)
            solve_cvxpy_problem(problem, self.solver_cfg)
            kept_variables = select_active_variables(np.asarray(beta.value, dtype=float), onehot_columns, inputs)
            return fold_id, train_idx, test_idx, kept_variables

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(solve_fold, fold_id, train_idx, test_idx)
                    for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
                ]
                results = [future.result() for future in futures]
        else:
            results = [
                solve_fold(fold_id, train_idx, test_idx)
                for fold_id, (train_idx, test_idx) in enumerate(self.cv_splits)
            ]

        results.sort(key=lambda item: item[0])
        return results

    @timed
    def fit_one_lambda(
        self,
        inputs: List[str],
        fused_lambda: float,
        group_lambda: float,
    ):
        problem, beta, intercept, lambda_parameter, onehot_columns = self._get_group_full_fit()
        lambda_parameter.value = float(group_lambda)
        solve_cvxpy_problem(problem, self.solver_cfg)

        kept_variables = select_active_variables(np.asarray(beta.value, dtype=float), onehot_columns, inputs)
        if kept_variables:
            fused_problem, fused_beta, fused_intercept, fused_lambda_parameter, _ = self._get_fused_full_fit(kept_variables)
            fused_lambda_parameter.value = float(fused_lambda)
            solve_cvxpy_problem(fused_problem, self.solver_cfg)
            coefficient_vector = combine_intercept_and_coefficients(fused_intercept, fused_beta)

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
    def crossval_pair_lambdas(
        self,
        results: pd.DataFrame,
        inputs: List[str],
        group_lambda: float,
        fused_lambda: float,
        counter: int,
        kept_per_fold: Optional[List[Tuple[int, np.ndarray, np.ndarray, List[str]]]] = None,
    ) -> pd.DataFrame:
        print(f"Model with group lambda = {group_lambda} and fused lambda = {fused_lambda}")

        if kept_per_fold is None:
            kept_per_fold = self._precompute_group_selection_per_fold(inputs, group_lambda, use_cache=True)

        if self.n_jobs > 1:
            for fold_id, train_idx, _, kept_variables in kept_per_fold:
                if kept_variables:
                    self._get_fused_fold_fit(fold_id, train_idx, kept_variables)

        train_error = 0.0
        test_error = 0.0

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(
                        self._cv_fold_loss_generalized,
                        fold_id,
                        train_idx,
                        test_idx,
                        kept_variables,
                        fused_lambda,
                    )
                    for fold_id, train_idx, test_idx, kept_variables in kept_per_fold
                ]
                for future in futures:
                    fold_train, fold_test = future.result()
                    train_error += fold_train
                    test_error += fold_test
        else:
            for fold_id, train_idx, test_idx, kept_variables in kept_per_fold:
                fold_train, fold_test = self._cv_fold_loss_generalized(
                    fold_id,
                    train_idx,
                    test_idx,
                    kept_variables,
                    fused_lambda,
                )
                train_error += fold_train
                test_error += fold_test

        results.at[counter, "group_lambda"] = group_lambda
        results.at[counter, "fused_lambda"] = fused_lambda
        results.at[counter, "Deviance_cv_train"] = train_error / self.n_k_fold
        results.at[counter, "Deviance_cv_test"] = test_error / self.n_k_fold

        modality_count, modality_details, coefficients, kept_variables = self.fit_one_lambda(inputs, fused_lambda, group_lambda)
        results.at[counter, "modalities_number"] = modality_count
        results.at[counter, "var_mod_details"] = modality_details
        results.at[counter, "betas"] = coefficients
        results.at[counter, "variables"] = " ".join(map(str, kept_variables))
        results.at[counter, "variable_number"] = len(kept_variables)
        return results.sort_values(by=["group_lambda", "fused_lambda"], ascending=True).reset_index(drop=True)

    @timed
    def get_lambda_curve(self):
        if self.parcimony_step is None or self.smoothness_step is None:
            raise ValueError("parcimony_step and smoothness_step must be provided for lambda-curve exploration.")
        if self.var_nb_min is None:
            raise ValueError("var_nb_min must be provided for lambda-curve exploration.")

        print("----------------------------------")
        print("Step 1 : Get range of Group Lambdas")
        print("----------------------------------\n")

        grouped_table = pd.DataFrame(columns=["lambda", "var_nb", "variables"], data=np.empty((1, 3), dtype=object))
        counter = 0
        lambda_candidate = 0.1
        max_lambda_reached = False

        problem, beta, _, lambda_parameter, onehot_columns = self._get_group_full_fit()

        while not max_lambda_reached:
            print(f"Running group GLMs until reaching maximum lambda : {lambda_candidate}")
            lambda_parameter.value = float(lambda_candidate)
            solve_cvxpy_problem(problem, self.solver_cfg)

            kept_variables = select_active_variables(np.asarray(beta.value, dtype=float), onehot_columns, self.input_variables)
            grouped_table.at[counter, "lambda"] = lambda_candidate
            grouped_table.at[counter, "var_nb"] = len(kept_variables)
            grouped_table.at[counter, "variables"] = kept_variables
            grouped_table = grouped_table.sort_values("lambda", ascending=True).reset_index(drop=True)

            if (len(kept_variables) == 0) or (len(kept_variables) <= self.var_nb_min):
                max_lambda_reached = True
                max_group_lambda = float(grouped_table["lambda"].max())

            counter += 1
            lambda_candidate *= 10

        lambda_min = float(grouped_table.loc[0, "lambda"])
        step_size = (max_group_lambda - lambda_min) / float(self.parcimony_step)
        intermediate_group_lambdas = [lambda_min + step_size * step for step in range(1, self.parcimony_step)]

        for lambda_candidate in intermediate_group_lambdas:
            print(f"Running group GLMs with lambda = {lambda_candidate}")
            lambda_parameter.value = float(lambda_candidate)
            solve_cvxpy_problem(problem, self.solver_cfg)

            kept_variables = select_active_variables(np.asarray(beta.value, dtype=float), onehot_columns, self.input_variables)
            grouped_table.at[counter, "lambda"] = lambda_candidate
            grouped_table.at[counter, "var_nb"] = len(kept_variables)
            grouped_table.at[counter, "variables"] = kept_variables
            grouped_table = grouped_table.sort_values("lambda", ascending=True).reset_index(drop=True)
            counter += 1

        grouped_table["variables_tuple"] = grouped_table["variables"].apply(frozenset)
        unique_grouped_table = grouped_table.groupby("variables_tuple")["lambda"].mean().reset_index()
        unique_grouped_table["variables"] = unique_grouped_table["variables_tuple"].apply(lambda value: sorted(list(value)))
        unique_grouped_table = unique_grouped_table.drop(columns="variables_tuple")

        print("Group GLMs gives the following variable lists :")
        for variable_list in unique_grouped_table["variables"]:
            if variable_list:
                print(variable_list)

        print("\n----------------------------------")
        print("Step 2 : Iterate on each variable list\n")

        results = create_results_table(1)
        result_index = 0

        for row_index in range(len(unique_grouped_table)):
            input_list = unique_grouped_table.loc[row_index, "variables"]
            if not input_list:
                print("Skipping empty variable list.")
                continue

            group_lambda = float(unique_grouped_table.loc[row_index, "lambda"])
            print(f"Fused GLMs on this variable list : {input_list}")
            print("\n Step 2.1 : For each variable list, get range of Fused Lambdas\n")

            fused_problem, fused_beta, _, fused_lambda_parameter, fused_onehot_columns = self._get_fused_full_fit(input_list)
            lambda_candidate = 0.1
            max_lambda_reached = False
            fused_summary = pd.DataFrame(columns=["fused_lambda", "variable_number"], data=np.empty((1, 2), dtype=object))
            fused_row = 0

            while not max_lambda_reached:
                fused_lambda_parameter.value = float(lambda_candidate)
                solve_cvxpy_problem(fused_problem, self.solver_cfg)

                kept_variables = select_active_variables(
                    np.asarray(fused_beta.value, dtype=float),
                    fused_onehot_columns,
                    input_list,
                )
                fused_summary.at[fused_row, "fused_lambda"] = float(lambda_candidate)
                fused_summary.at[fused_row, "variable_number"] = len(kept_variables)

                if len(kept_variables) == 0:
                    max_lambda_reached = True
                    max_fused_lambda = float(lambda_candidate)

                lambda_candidate *= 10
                fused_row += 1

            lambda_min = float(fused_summary.loc[0, "fused_lambda"])
            step_size = (max_fused_lambda - lambda_min) / float(self.smoothness_step)
            candidate_fused_lambdas = [
                lambda_min + step_size * step for step in range(0, self.smoothness_step + 1)
            ]

            kept_per_fold = self._precompute_group_selection_per_fold(input_list, group_lambda, use_cache=True)
            for fused_lambda in candidate_fused_lambdas:
                print("\n Step 2.2 : For selected group and fused lambdas, get cross-val")
                results = self.crossval_pair_lambdas(
                    results,
                    input_list,
                    group_lambda,
                    float(fused_lambda),
                    result_index,
                    kept_per_fold=kept_per_fold,
                )
                result_index += 1

            print(f"End of curve for {input_list}")

        self.lambda_curve = results

    def plot_curve(self):
        categories = self.lambda_curve["group_lambda"].unique()
        colors = plt.cm.tab10(range(len(categories)))
        plt.figure(figsize=(8, 6))

        for category, color in zip(categories, colors):
            subset = self.lambda_curve[self.lambda_curve["group_lambda"] == category]
            plt.scatter(subset["variable_number"], subset["Deviance_cv_test"], color=color, label=str(np.around(category)))
            for _, row in subset.iterrows():
                label = f"{np.around(row['group_lambda'])} / {np.around(row['fused_lambda'])}"
                plt.text(row["variable_number"], row["Deviance_cv_test"], label, fontsize=8, ha="center", va="bottom")

        plt.xlabel("Final Variable Number")
        plt.ylabel("Deviance cv test")
        plt.title("Scatter plot with discrete colors for group lambda value")
        plt.legend(title="Group Lambda")
        plt.tight_layout()
        plt.show()
