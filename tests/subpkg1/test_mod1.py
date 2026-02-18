import pandas as pd

from fusedpkg.additional_functions.mod1 import (
    choose_ref_modality,
    find_first_number,
    identify_biggest_modality,
    reorder_df_columns,
)


def test_identify_biggest_modality() -> None:
    df = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red", "Green", "Red"],
            "Shape": ["Circle", "Square", "Square", "Circle", "Circle"],
        }
    )
    results = identify_biggest_modality(df, ["Color", "Shape"])
    assert results == {"Color": "Red", "Shape": "Circle"}


def test_choose_ref_modality_biggest() -> None:
    df = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red", "Green"],
            "Shape": ["Circle", "Square", "Circle", "Circle"],
        }
    )
    onehot_df, ref_dict, cols = choose_ref_modality(df, ["Color", "Shape"], method="Biggest")
    assert ref_dict == {"Color": "Red", "Shape": "Circle"}
    assert "Color_Red" not in onehot_df.columns
    assert "Shape_Circle" not in onehot_df.columns
    assert list(cols) == list(onehot_df.columns)


def test_find_first_number() -> None:
    assert find_first_number("abc 123 def") == "123"
    assert find_first_number("alpha 4.5 beta 10") == "4.5"


def test_reorder_df_columns() -> None:
    data = pd.DataFrame(
        {
            "zip_1": [1, 0],
            "zip_10": [0, 1],
            "zip_2": [0, 1],
            "zip_20": [1, 1],
            "age_1": [-1, -1],
            "age_2": [0, 0],
        }
    )
    fused_types = {"zip": "fused", "age": "g_fused"}
    input_variables = ["zip", "age"]
    results = reorder_df_columns(data, fused_types, input_variables)
    assert list(results.columns) == [
        "zip_1",
        "zip_2",
        "zip_10",
        "zip_20",
        "age_1",
        "age_2",
    ]
   
