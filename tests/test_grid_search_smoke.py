import pandas as pd
import pytest

from fusedpkg import GridSearch_Fused, GridSearch_Generalised, GridSearch_Group
from fusedpkg.glm.preprocessing import prepare_categorical_glm_data


def build_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": pd.Categorical(["red", "red", "blue", "blue", "green", "green", "red", "blue"]),
            "shape": pd.Categorical(["1", "2", "1", "2", "1", "2", "1", "2"]),
            "n_claims": [0, 1, 0, 2, 1, 0, 1, 0],
            "exposure": [0.0] * 8,
        }
    )


def test_prepare_categorical_glm_data_adds_original_columns() -> None:
    data = build_test_data()
    penalty_types = {"color": "g_fused", "shape": "fused"}

    data_onehot, reference_modalities, onehot_columns = prepare_categorical_glm_data(
        data,
        ["color", "shape"],
        penalty_types,
        "n_claims",
        "exposure",
        "First",
    )

    assert "non_dum_color" in data_onehot.columns
    assert "non_dum_shape" in data_onehot.columns
    assert "n_claims" in data_onehot.columns
    assert "exposure" in data_onehot.columns
    assert reference_modalities["color"] == "color_blue"
    assert len(onehot_columns) > 0


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (GridSearch_Group, {"family": "Poisson", "lbd_group": 0.1, "random_state": 0}),
        (GridSearch_Fused, {"family": "Poisson", "lbd_fused": 0.1, "random_state": 0}),
        (
            GridSearch_Generalised,
            {"family": "Poisson", "lbd_group": 0.1, "lbd_fused": 0.1, "random_state": 0},
        ),
    ],
)
def test_grid_search_smoke_paths(model_cls, kwargs) -> None:
    data = build_test_data()
    penalty_types = {"color": "g_fused", "shape": "fused"}

    model = model_cls(**kwargs)
    model.fit(
        data=data,
        penalty_types=penalty_types,
        input_variables=["color", "shape"],
        target="n_claims",
        offset="exposure",
        n_k_fold=2,
    )

    assert list(model.lambda_curve.columns) == [
        "group_lambda",
        "fused_lambda",
        "variable_number",
        "modalities_number",
        "var_mod_details",
        "Deviance_cv_train",
        "Deviance_cv_test",
        "variables",
        "betas",
    ]
    assert len(model.lambda_curve) == 1
