from mypkg.additional_functions.mod1 import identify_biggest_modality,choose_ref_modality,find_first_number,reorder_df_columns
import pandas as pd
def test_identify_biggest_modality() -> None:
    
    df = pd.DataFrame({
         'Color': ['Red', 'Blue', 'Red', 'Green', 'Red'],
         'Shape': ['Circle', 'Square', 'Square', 'Circle', 'Circle']
    })
    results = identify_biggest_modality(df, ['Color', 'Shape'])
    assert results == {'Color': 'Red', 'Shape': 'Circle'}



def test_choose_ref_modality() -> None:
    df = pd.DataFrame({
            "Color": ["Red", "Blue", "Red", "Green"],
            "Shape": ["Circle", "Square", "Circle", "Square"]
            })
    onehot_df, ref_dict, cols = choose_ref_modality(df, ["Color", "Shape"], method="Biggest")
    assert ref_dict == {'Color': 'Red', 'Shape': 'Circle'}  



def test_find_first_number() -> None:
    input_test = "abc 123 def"
    results = find_first_number(input_test)
    assert results == "123"



def test_find_first_number() -> None:
    data = pd.DataFrame({
                "zip_1": [1,0],
                "zip_10": [0,1],
                "zip_2": [0,1],
                "zip_20": [1,1],
                "age_1" : [-1,-1],
                "age_2" : [0,0],
                })
    fused_types = {"zip":"fused", "age":"g_fused"}
    input_variables = ["zip","age"]
    results = reorder_df_columns(data,fused_types,input_variables)
    assert results == ["zip_1",
                       "zip_2",
                       "zip_10",
                       "zip_20",
                       "age_1",
                       "age_2"]
   