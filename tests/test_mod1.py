import pandas as pd

from fusedpkg.additional_functions.mod1 import choose_ref_modality, find_first_number


def test_choose_ref_modality_list_method() -> None:
    data = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red", "Green"],
            "Shape": ["Circle", "Square", "Circle", "Square"],
        }
    )
    expected_ref = {"Color": "Red", "Shape": "Circle"}

    onehot_df, ref_dict, cols = choose_ref_modality(
        data,
        ["Color", "Shape"],
        method="List",
        ref_modality_dict=expected_ref,
    )

    assert ref_dict == expected_ref
    assert "Color_Red" not in onehot_df.columns
    assert "Shape_Circle" not in onehot_df.columns
    assert list(cols) == list(onehot_df.columns)


def test_find_first_number_returns_none_when_absent() -> None:
    assert find_first_number("no numbers here") is None
