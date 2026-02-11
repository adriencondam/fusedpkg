"""This is the docstring for the mod1 module in additional_functions."""

import pandas as pd
import re
import numpy as np

def identify_biggest_modality(data, input_variables):
    """
    Identify the most frequent (modal) value for each given variable 
    in a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing the variables of interest.
    input_variables : list of str
        A list of column names in `data` for which to compute the most frequent value.

    Returns
    -------
    dict
        A dictionary mapping each column name to its most frequent value (as a string).

    """
    temp_dict = {}

    # Iterate over the selected input variables
    for i in input_variables:
        # Find the most frequent value (mode) for the column `i`
        max_modality_temp = str(data[i].value_counts().idxmax())
        
        # Store result in dictionary, mapping column -> most frequent value
        temp_dict[i] = max_modality_temp

    # Return dictionary of results
    return temp_dict



def choose_ref_modality(data, input_var_list, method, ref_modality_dict=None):
    """
    Encode categorical variables into one-hot encoding while dropping a chosen
    "reference modality" for each variable to avoid collinearity.

    Depending on the chosen method, the reference modality is selected as:
        - "First"   : the first category encountered in one-hot encoding.
        - "Biggest" : the most frequent (modal) value in each column.
        - "List"    : user-specified dictionary of reference modalities.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset containing categorical variables.
    input_var_list : list of str
        List of column names to be one-hot encoded.
    method : str
        Method to determine the reference modality. One of:
        {"First", "Biggest", "List"}.
    ref_modality_dict : dict, optional
        Only required if method == "List". Dictionary mapping variable names
        to the modality that should be treated as the reference.

    Returns
    -------
    onehot_df : pandas.DataFrame
        One-hot encoded dataframe with reference modalities dropped.
    ref_modality_dict : dict
        Dictionary mapping variable names to their chosen reference modality.
    one_hot_columns : pandas.Index
        List of remaining one-hot encoded columns after dropping references.

    """
    
    if method == "First":
        # Generate one-hot encoding without dropping any category
        onehot_df = pd.get_dummies(data=data[input_var_list], drop_first=False)
        
        # Pick the first category for each variable as reference
        ref_modality_dict = {
            j: [col for i, col in enumerate(onehot_df.columns) if col.startswith(j)][0]
            for j in input_var_list
        }
        
        # Drop those reference columns from one-hot dataframe
        columns_to_drop = [
            [col for i, col in enumerate(onehot_df.columns) if col.startswith(j)][0]
            for j in input_var_list
        ]
        onehot_df = onehot_df.drop(columns=columns_to_drop)
    
    elif method == "Biggest":
        # Generate one-hot encoding
        onehot_df = pd.get_dummies(data=data[input_var_list], drop_first=False)
        
        # Select the most frequent modality in each variable
        ref_modality_dict = identify_biggest_modality(data, input_var_list)
        
        # Drop corresponding columns from one-hot encoding
        columns_to_drop = [f"{key}_{val}" for key, val in ref_modality_dict.items()]
        onehot_df = onehot_df.drop(columns=columns_to_drop)
        
    elif method == "List":
        if ref_modality_dict is None:
            print("Dictionary with list of modalities is needed")
            return
        else:
            # Use the provided reference modalities
            onehot_df = pd.get_dummies(data=data[input_var_list], drop_first=False)
            columns_to_drop = [f"{key}_{val}" for key, val in ref_modality_dict.items()]
            onehot_df = onehot_df.drop(columns=columns_to_drop)
            
    else:
        print("Wrong method given as input")
        return
    
    # Get list of one-hot columns after dropping reference modalities
    one_hot_columns = onehot_df.columns
    
    return onehot_df, ref_modality_dict, one_hot_columns


def find_first_number(string):
    """
    Find and return the first numeric value in a given string. It handles both decimal and integer numbers
    """
    match_digit = re.search(r'\d+\.\d+', string) #First check for numbers with decimals
    match_no_digit = re.search(r'\d+', string) #Then check for integers
    if match_digit:
        return match_digit.group()  # Return the first number found
    elif match_no_digit:
        return match_no_digit.group()
    else:
        return None



def reorder_df_columns(data,fused_types,input_variables):
    """
    Reorders the fused lasso columns by checking that they are displayed in the right numerical order
    """
    temp_variable_list=[]
    for col in input_variables:
        if fused_types.get(col)=="fused":
            one_hot_cols=[x for i,x in enumerate(data.columns) if x.startswith(col)]
            variable_modality=[float(find_first_number(j.split("_")[-1])) for j in one_hot_cols]
            reordered_one_hot_cols = [one_hot_cols[i] for i in np.argsort(variable_modality)]
            temp_variable_list=temp_variable_list+reordered_one_hot_cols
        else :
            temp_variable_list=temp_variable_list + [x for i,x in enumerate(data.columns) if x.startswith(col)]
    return data[temp_variable_list]


