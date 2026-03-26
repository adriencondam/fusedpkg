from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from fusedpkg.additional_functions.mod1 import choose_ref_modality, reorder_df_columns


def get_onehot_column_names(df: pd.DataFrame, prefixes: Sequence[str]) -> List[str]:
    """Return one-hot columns whose names start with any requested prefix."""
    prefix_tuple = tuple(prefixes)
    return [column for column in df.columns if column.startswith(prefix_tuple)]


def prepare_categorical_glm_data(
    data: pd.DataFrame,
    input_variables: Sequence[str],
    penalty_types: Dict[str, str],
    target: str,
    offset: str,
    reference_method: str,
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Build the one-hot design matrix used by the grid-search classes."""
    data_onehot, reference_modalities, input_var_onehot = choose_ref_modality(
        data,
        input_variables,
        reference_method,
    )
    data_onehot = reorder_df_columns(data_onehot, penalty_types, input_variables)
    data_onehot = pd.concat(
        [
            data_onehot,
            data[input_variables].rename(columns={column: f"non_dum_{column}" for column in input_variables}),
        ],
        axis=1,
    )
    data_onehot[target] = data[target]
    data_onehot[offset] = data[offset]
    return data_onehot, reference_modalities, input_var_onehot


def build_modality_lookup(data_onehot: pd.DataFrame, input_variables: Sequence[str]) -> pd.DataFrame:
    """Map each categorical variable to its one-hot modality labels."""
    modalities: List[str] = []
    variables: List[str] = []
    for variable in input_variables:
        for column in data_onehot.columns:
            if column.startswith(f"{variable}_"):
                variables.append(variable)
                modalities.append(column.replace(f"{variable}_", ""))
    return pd.DataFrame({"variable": variables, "modality": modalities})


def group_modalities_by_coefficient(
    data: pd.DataFrame,
    modality_lookup: pd.DataFrame,
    coefficients: Sequence[float],
    input_variables: Sequence[str],
    reference_modalities: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Group modalities with identical rounded coefficients."""
    coefficient_array = np.asarray(coefficients[1:], dtype=float)
    grouped_lookup = modality_lookup[modality_lookup["variable"].isin(input_variables)].copy()
    grouped_lookup["coefficient"] = [str(round(value, 6)) for value in coefficient_array]

    grouped_modalities = (
        grouped_lookup.groupby(["variable", "coefficient"])["modality"]
        .apply(lambda values: "_".join(values.astype(str)))
        .reset_index()
        .rename(columns={"modality": "modality_group"})
    )
    grouped_lookup = grouped_lookup.merge(grouped_modalities, how="left", on=["variable", "coefficient"])

    kept_variables: List[str] = []
    removed_variables: List[str] = []

    for variable in input_variables:
        original_column = f"non_dum_{variable}"
        variable_lookup = grouped_lookup[grouped_lookup["variable"] == variable].copy()

        if (np.abs(variable_lookup["coefficient"].astype(float)) < 1e-6).all():
            removed_variables.append(variable)
            continue

        data[original_column] = data[original_column].astype(str)
        variable_lookup["modality"] = variable_lookup["modality"].astype(str)

        data = data.merge(
            variable_lookup[["modality", "modality_group"]],
            how="left",
            left_on=original_column,
            right_on="modality",
        )
        reference_modality = str(reference_modalities[variable]).replace(f"{variable}_", "")
        data["modality_group"] = data["modality_group"].fillna(reference_modality)
        data.rename(columns={"modality_group": f"{variable}_grouped"}, inplace=True)
        data.drop(columns=["modality"], inplace=True)
        kept_variables.append(variable)

    kept_grouped_variables = [f"{variable}_grouped" for variable in kept_variables]
    return data, kept_grouped_variables, removed_variables


def select_active_variables(
    beta_vector: np.ndarray,
    input_var_onehot: Sequence[str],
    input_variables: Sequence[str],
    tolerance: float = 1e-6,
) -> List[str]:
    """Return variables whose coefficients are not all numerically zero."""
    beta_array = np.asarray(beta_vector, dtype=float)
    kept_variables: List[str] = []
    for variable in input_variables:
        indices = [index for index, column in enumerate(input_var_onehot) if column.startswith(f"{variable}_")]
        if indices and np.any(np.abs(beta_array[indices]) > tolerance):
            kept_variables.append(variable)
    return kept_variables
