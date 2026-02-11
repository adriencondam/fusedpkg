import numpy as np
import pandas as pd
# from sklearn.linear_model import Lasso
import cvxpy as cp
# from scipy.special import expit  # Sigmoid function for logistic regression
# import matplotlib
import matplotlib.pyplot as plt
# import re
# import random
from sklearn.model_selection import KFold
import math
# import time
from mypkg.additional_functions.mod1 import identify_biggest_modality,choose_ref_modality, reorder_df_columns

import time

def time_it(f):
    time_it.active = 0
    def tt(*args,**kwargs):
        time_it.active += 1
        t0 = time.time()
        tabs = '\t'* (time_it.active - 1)
        name = f.__name__
        print('{tabs}Executing <{name}>'.format(tabs=tabs,name=name))
        res = f(*args,**kwargs)
        elapsed = time.time()-t0
        mins = int(elapsed // 60)
        secs = elapsed % 60
        print('{tabs}Function <{name}> execution time : {mins:.0f} minutes and {secs:.0f} seconds'.format(tabs=tabs,name=name, mins=mins,secs=secs))
        time_it.active -= 1
        return res
    return tt



def get_onehot_columns(df, prefixes):
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]

def Compute_Poisson_Deviance(lambdahat, N, v):
    """
    Compute the Poisson deviance (per unit of exposure) for model evaluation.

    The Poisson deviance is a measure of model fit commonly used in generalized
    linear models (GLMs). It compares observed counts `N` to expected counts
    `lambdahat * v`.

    Parameters
    ----------
    lambdahat : array-like
        Predicted Poisson rates (λ̂).
    N : array-like
        Observed counts.
    v : array-like
        Exposure or weight values (same length as N).

    Returns
    -------
    float
        Scaled Poisson deviance: sum of contributions divided by total exposure.
    """
    # Initialize arrays with NaN placeholders
    nlogn = np.array([np.nan for _ in range(v.size)])
    dev = np.array([np.nan for _ in range(v.size)])

    # Compute deviance contributions for each observation
    for i in range(v.size):
        if N[i] == 0:
            # By convention: 0 * log(0/μ) = 0
            nlogn[i] = 0
        else:
            # N_i * log(N_i / (λ̂_i * v_i))
            nlogn[i] = N[i] * math.log(N[i] / (lambdahat[i] * v[i]))

        # Full deviance contribution: nlogn - (N - λ̂v)
        dev[i] = nlogn[i] - (N[i] - lambdahat[i] * v[i])

    # Return scaled deviance (average per unit of exposure)
    return dev.sum() / v.sum()


def compute_fused_pen(data, inputs, val, beta, ref_mod_length):
    """
    Compute fused penalty term.
    
    Parameters
    ----------
    data : pd.DataFrame or dict-like
        Input data containing variables.
    inputs : list
        List of input variable names corresponding to val.
    val : list
        List of indices (mapping to beta).
    beta : dict or list-like
        Coefficients (cvxpy variables).
    ref_mod_length : float
        Reference modality length.
    """
    penalty = 0
    # Reference modality
    weight = ((sum(data[inputs[val[0]]]) + ref_mod_length) / len(data)) ** 0.5
    penalty += cp.norm1(weight * (beta[val[0]] - 0))  # ref beta is 0

    # Sequential differences
    if len(val) > 1:
        for i in range(1, len(val)):
            weight = ((sum(data[inputs[val[i]]]) + sum(data[inputs[val[i - 1]]])) / len(data)) ** 0.5
            penalty += cp.norm1(weight * (beta[val[i]] - beta[val[i - 1]]))

    return penalty


def compute_g_fused_pen(data, inputs, val, beta, g_fused_graph_size, ref_mod_length):
    """
    Compute generalized fused penalty term.
    
    Parameters
    ----------
    data : pd.DataFrame or dict-like
        Input data containing variables.
    inputs : list
        List of input variable names corresponding to val.
    val : list
        List of indices (mapping to beta).
    beta : dict or list-like
        Coefficients (cvxpy variables).
    g_fused_graph_size : int
        Graph size scaling factor.
    ref_mod_length : float
        Reference modality length.
    """
    penalty = 0
    for k in range(len(val)):
        # Reference modality
        weight = (
            len(val) / g_fused_graph_size
        ) * ((sum(data[inputs[val[k]]]) + ref_mod_length) / len(data)) ** 0.5
        penalty += cp.norm1(weight * (beta[val[k]] - 0))  # ref beta is 0

        # Pairwise differences
        if len(val) > 1:
            for j in range(k, len(val)):
                weight = (
                    len(val) / g_fused_graph_size
                ) * ((sum(data[inputs[val[k]]]) + sum(data[inputs[val[j]]])) / len(data)) ** 0.5
                penalty += cp.norm1(weight * (beta[val[k]] - beta[val[j]]))

    return penalty

def compute_glm_no_pen(
    data,
    input_var,
    target_var,
    offset_var,
    family='Poisson'
):
    
    # Define optimization variables
    input_var_onehot = get_onehot_columns(data,input_var)
    intercept = cp.Variable()
    X = data[input_var_onehot].to_numpy()
    Y = data[target_var].to_numpy()
    n, p = X.shape
    beta = cp.Variable(p)  # model coefficients
    # Linear predictor: η = Xβ + intercept + offset
    eta = X @ beta + intercept + data[offset_var]


    # -------------------------------
    # Define log-likelihood per family
    # -------------------------------
    if family.lower() == "poisson":
        log_likelihood = cp.sum(cp.multiply(Y, eta) - cp.exp(eta))

    elif family.lower() == "gaussian":
        # Equivalent to least-squares loss
        log_likelihood = -0.5 * cp.sum_squares(Y - eta)

    elif family.lower() == "binomial":
        # Logistic regression likelihood
        log_likelihood = cp.sum(cp.multiply(Y, eta) - cp.logistic(eta))

    elif family.lower() == "gamma":
        # Canonical log-link
        log_likelihood = cp.sum(-Y / cp.exp(eta) - eta)

    elif family.lower() == "inverse_gaussian":
        mu = cp.exp(eta)
        log_likelihood = cp.sum(-0.5 * cp.square(Y - mu) / (mu**3))

    elif family.lower() == "linear":
        log_likelihood = cp.sum(cp.multiply(Y, eta) - eta)

    else:
        raise ValueError(f"Unsupported family: {family}")

            

    # -------------------------------
    # Define objective and solve
    # -------------------------------
    objective = cp.Minimize(-log_likelihood)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS)

    # Return coefficients (intercept first, then betas)
    return np.concatenate(([intercept.value], beta.value))


def compute_group_lasso_glm(
    data,
    input_var,
    target_var,
    offset_var,
    penalty_types,
    lambda_1=1.0,
    family='Poisson'
):
    """
    Fit a generalized linear model (GLM) with custom fused/group-fused penalties
    using CVXPY for convex optimization.

    This function supports Poisson, Gaussian, Binomial, Gamma, Inverse Gaussian,
    and Linear families, with the ability to impose specialized penalties for
    categorical variables encoded via one-hot groups.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing features, target, and offset variables.
    input_var : list of str
        Column names of predictor variables.
    target_var : str
        Column name of the target variable.
    offset_var : str
        Column name of the offset variable (e.g. log exposure in Poisson models).
    one_hot_groups : dict
        Mapping of categorical variable prefixes to lists of column indices
        (corresponding to one-hot encoded features).
    penalty_types : dict
        Mapping of variable prefixes to penalty types ("fused" or "g_fused").
    lambda_1 : float, default=1.0
        Regularization strength for group-fused penalties.
    lambda_2 : float, default=1.0
        Regularization strength for fused penalties.
    family : str, default='Poisson'
        Distribution family for the GLM. Supported values:
        {"Poisson", "Gaussian", "Binomial", "Gamma", "Inverse_Gaussian", "Linear"}.

    Returns
    -------
    numpy.ndarray
        Estimated coefficients, including intercept as the first element.
    """
    
    # Define optimization variables
    input_var_onehot = get_onehot_columns(data,input_var)
    intercept = cp.Variable()
    X = data[input_var_onehot].to_numpy()
    Y = data[target_var].to_numpy()
    n, p = X.shape
    beta = cp.Variable(p)  # model coefficients
    # Linear predictor: η = Xβ + intercept + offset
    eta = X @ beta + intercept + data[offset_var]

    penalty_types = {k: v for k, v in penalty_types.items() if any(k.startswith(prefix) for prefix in input_var)}
    one_hot_groups={key:[i for i,x in enumerate(data[input_var_onehot].columns) if (x.startswith(key+"_"))] for key,val in penalty_types.items()}
    one_hot_groups = {k: v for k, v in one_hot_groups.items() if v}

    # -------------------------------
    # Define log-likelihood per family
    # -------------------------------
    if family.lower() == "poisson":
        log_likelihood = cp.sum(cp.multiply(Y, eta) - cp.exp(eta))

    elif family.lower() == "gaussian":
        # Equivalent to least-squares loss
        log_likelihood = -0.5 * cp.sum_squares(Y - eta)

    elif family.lower() == "binomial":
        # Logistic regression likelihood
        log_likelihood = cp.sum(cp.multiply(Y, eta) - cp.logistic(eta))

    elif family.lower() == "gamma":
        # Canonical log-link
        log_likelihood = cp.sum(-Y / cp.exp(eta) - eta)

    elif family.lower() == "inverse_gaussian":
        mu = cp.exp(eta)
        log_likelihood = cp.sum(-0.5 * cp.square(Y - mu) / (mu**3))

    elif family.lower() == "linear":
        log_likelihood = cp.sum(cp.multiply(Y, eta) - eta)

    else:
        raise ValueError(f"Unsupported family: {family}")

    # -------------------------------
    # Construct custom penalties
    # -------------------------------
    group_penalty = 0

    # Second pass: build fused and group-fused penalties
    for key, val in one_hot_groups.items():
        w = np.sqrt(len(val))
        group_penalty += cp.norm1(w * beta[val])
        
        # for i in range(1, len(val)):
            
        #     weight = ((sum(data[input_var[val[i]]]) + sum(data[input_var[val[i-1]]])) / len(data))**0.5
            

    # -------------------------------
    # Define objective and solve
    # -------------------------------
    objective = cp.Minimize(-log_likelihood + lambda_1 * group_penalty)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.CLARABEL)

    # Return coefficients (intercept first, then betas)
    return np.concatenate(([intercept.value], beta.value))


def compute_group_lasso_glm(
    data,
    input_var,
    target_var,
    offset_var,
    penalty_types,
    lambda_1=1.0,
    family='Poisson'
):
    """
    Fit a generalized linear model (GLM) with custom fused/group-fused penalties
    using CVXPY for convex optimization.

    This function supports Poisson, Gaussian, Binomial, Gamma, Inverse Gaussian,
    and Linear families, with the ability to impose specialized penalties for
    categorical variables encoded via one-hot groups.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing features, target, and offset variables.
    input_var : list of str
        Column names of predictor variables.
    target_var : str
        Column name of the target variable.
    offset_var : str
        Column name of the offset variable (e.g. log exposure in Poisson models).
    one_hot_groups : dict
        Mapping of categorical variable prefixes to lists of column indices
        (corresponding to one-hot encoded features).
    penalty_types : dict
        Mapping of variable prefixes to penalty types ("fused" or "g_fused").
    lambda_1 : float, default=1.0
        Regularization strength for group-fused penalties.
    lambda_2 : float, default=1.0
        Regularization strength for fused penalties.
    family : str, default='Poisson'
        Distribution family for the GLM. Supported values:
        {"Poisson", "Gaussian", "Binomial", "Gamma", "Inverse_Gaussian", "Linear"}.

    Returns
    -------
    numpy.ndarray
        Estimated coefficients, including intercept as the first element.
    """
    
    # Define optimization variables
    input_var_onehot = get_onehot_columns(data,input_var)
    intercept = cp.Variable()
    X = data[input_var_onehot].to_numpy()
    Y = data[target_var].to_numpy()
    n, p = X.shape
    beta = cp.Variable(p)  # model coefficients
    # Linear predictor: η = Xβ + intercept + offset
    eta = X @ beta + intercept + data[offset_var]

    penalty_types = {k: v for k, v in penalty_types.items() if any(k.startswith(prefix) for prefix in input_var)}
    one_hot_groups={key:[i for i,x in enumerate(data[input_var_onehot].columns) if (x.startswith(key+"_"))] for key,val in penalty_types.items()}
    one_hot_groups = {k: v for k, v in one_hot_groups.items() if v}

    # -------------------------------
    # Define log-likelihood per family
    # -------------------------------
    if family.lower() == "poisson":
        log_likelihood = cp.sum(cp.multiply(Y, eta) - cp.exp(eta))

    elif family.lower() == "gaussian":
        # Equivalent to least-squares loss
        log_likelihood = -0.5 * cp.sum_squares(Y - eta)

    elif family.lower() == "binomial":
        # Logistic regression likelihood
        log_likelihood = cp.sum(cp.multiply(Y, eta) - cp.logistic(eta))

    elif family.lower() == "gamma":
        # Canonical log-link
        log_likelihood = cp.sum(-Y / cp.exp(eta) - eta)

    elif family.lower() == "inverse_gaussian":
        mu = cp.exp(eta)
        log_likelihood = cp.sum(-0.5 * cp.square(Y - mu) / (mu**3))

    elif family.lower() == "linear":
        log_likelihood = cp.sum(cp.multiply(Y, eta) - eta)

    else:
        raise ValueError(f"Unsupported family: {family}")

    # -------------------------------
    # Construct custom penalties
    # -------------------------------
    group_penalty = 0

    # Second pass: build fused and group-fused penalties
    for key, val in one_hot_groups.items():
        w = np.sqrt(len(val))
        group_penalty += cp.norm1(w * beta[val])
        
        # for i in range(1, len(val)):
            
        #     weight = ((sum(data[input_var[val[i]]]) + sum(data[input_var[val[i-1]]])) / len(data))**0.5
            

    # -------------------------------
    # Define objective and solve
    # -------------------------------
    objective = cp.Minimize(-log_likelihood + lambda_1 * group_penalty)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.CLARABEL)

    # Return coefficients (intercept first, then betas)
    return np.concatenate(([intercept.value], beta.value))


def group_modalities_based_on_betas(
    data,
    modality_var_df,
    beta_list,
    input_variables,
    ref_modalities
):
    """
    Group categorical modalities after applying fused penalization.

    This function inspects fitted model coefficients and merges categories
    (modalities) of a variable that share the same coefficient value (up to rounding).
    It creates new fused categorical variables in the dataset and removes variables
    with all-zero coefficients.

    Parameters
    ----------
    data : pandas.DataFrame
        Original dataset containing non-dummy-coded categorical variables.
    modality_var_df : pandas.DataFrame
        DataFrame mapping dummy-coded variables to their original variable and modality.
        Must contain columns: ["variable", "modality"].
    beta_list : list or array-like
        Estimated model coefficients (intercept first, then betas).
    input_variables : list of str
        Names of the categorical variables (dummy-coded in the model).
    ref_modalities : dict
        Mapping from variable name to its reference modality (dummy base category).

    Returns
    -------
    data : pandas.DataFrame
        Updated dataset with new fused categorical variables.
    input_variables_fused : list of str
        Names of variables that were kept, with "_grouped" suffix.
    removed_variables : list of str
        Variables removed because all their modalities had (near-)zero coefficients.
    """
    # Remove intercept and align betas with dummy variables
    chosen_model_temp = beta_list[1:]

    temp_modality_var_df = modality_var_df[modality_var_df['variable'].isin(input_variables)].copy()
    # Assign rounded coefficient values to modalities
    temp_modality_var_df["betas"] = [str(round(i, 6)) for i in chosen_model_temp]

    # Group modalities that share the same coefficient value
    modality_var_df_grouped = (
        temp_modality_var_df.groupby(['variable', 'betas'])['modality']
        .apply(lambda x: '_'.join(x))
        .reset_index()
    )
    modality_var_df_grouped.rename(columns={"modality": "modality_grouped"}, inplace=True)

    # Merge grouped modality info back into modality_var_df
    temp_modality_var_df = temp_modality_var_df.merge(
        modality_var_df_grouped, how="left", on=["variable", "betas"]
    )

    kept_variables = []
    removed_variables = []

    # -------------------------------
    # Process each categorical variable
    # -------------------------------
    for i in input_variables:
        original_var = "non_dum_" + i
        modality_var_df_one_var = temp_modality_var_df[temp_modality_var_df["variable"] == i].copy()
        # If all betas ≈ 0, remove this variable
        if (abs(modality_var_df_one_var["betas"].astype(float)) < 1e-6).all():
            removed_variables.append(i)
            
        else:
            # Keep variable: map original modalities → fused groups
            data[original_var] = data[original_var].astype(str)
            modality_var_df_one_var["modality"] = modality_var_df_one_var["modality"].astype(str)

            # Merge grouped modality info
            data = pd.merge(
                data,
                modality_var_df_one_var[["modality", "modality_grouped"]],
                how="left",
                left_on=original_var,
                right_on="modality"
            )

            # Fill missing values with reference modality
            ref_mod = ref_modalities[i].replace(i + "_", "")
            data["modality_grouped"] = data["modality_grouped"].fillna(ref_mod)

            # Rename new fused variable and drop merge helper column
            data.rename(columns={"modality_grouped": i + "_grouped"}, inplace=True)
            data.drop(columns=["modality"], inplace=True)

            kept_variables.append(i)

    # -------------------------------
    # Final outputs
    # -------------------------------
    input_variables_fused = [i + "_grouped" for i in kept_variables]
    return data, input_variables_fused, removed_variables


def keep_variables_after_group_lasso(
    data,
    modality_var_df,
    beta_list,
    input_variables
):

    # Remove intercept and align betas with dummy variables
    chosen_model_temp = beta_list[1:]

    temp_modality_var_df = modality_var_df[modality_var_df['variable'].isin(input_variables)].copy()
    # Assign rounded coefficient values to modalities
    temp_modality_var_df["betas"] = [str(round(i, 6)) for i in chosen_model_temp]

    # Group modalities that share the same coefficient value
    modality_var_df_grouped = (
        temp_modality_var_df.groupby(['variable', 'betas'])['modality']
        .apply(lambda x: '_'.join(x))
        .reset_index()
    )
    modality_var_df_grouped.rename(columns={"modality": "modality_grouped"}, inplace=True)

    # Merge grouped modality info back into modality_var_df
    temp_modality_var_df = temp_modality_var_df.merge(
        modality_var_df_grouped, how="left", on=["variable", "betas"]
    )

    kept_variables = []
    removed_variables = []

    for i in input_variables:
        modality_var_df_one_var = temp_modality_var_df[temp_modality_var_df["variable"] == i]
        # If all betas ≈ 0, remove this variable
        if (abs(modality_var_df_one_var["betas"].astype(float)) < 1e-6).all():
            removed_variables.append(i)
            
        else:
            kept_variables.append(i)

    return kept_variables, removed_variables



def custom_glm_with_fused_penalty(
    data,
    input_var,
    target_var,
    offset_var,
    penalty_types,
    lambda_1=1.0,
    family='Poisson'
):
    """
    Fit a generalized linear model (GLM) with custom fused/group-fused penalties
    using CVXPY for convex optimization.

    This function supports Poisson, Gaussian, Binomial, Gamma, Inverse Gaussian,
    and Linear families, with the ability to impose specialized penalties for
    categorical variables encoded via one-hot groups.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing features, target, and offset variables.
    input_var : list of str
        Column names of predictor variables.
    target_var : str
        Column name of the target variable.
    offset_var : str
        Column name of the offset variable (e.g. log exposure in Poisson models).
    one_hot_groups : dict
        Mapping of categorical variable prefixes to lists of column indices
        (corresponding to one-hot encoded features).
    penalty_types : dict
        Mapping of variable prefixes to penalty types ("fused" or "g_fused").
    lambda_1 : float, default=1.0
        Regularization strength for group-fused penalties.
    lambda_2 : float, default=1.0
        Regularization strength for fused penalties.
    family : str, default='Poisson'
        Distribution family for the GLM. Supported values:
        {"Poisson", "Gaussian", "Binomial", "Gamma", "Inverse_Gaussian", "Linear"}.

    Returns
    -------
    numpy.ndarray
        Estimated coefficients, including intercept as the first element.
    """

    input_var_one_hot = get_onehot_columns(data,input_var)
    # Define optimization variables
    intercept = cp.Variable()
    X = data[input_var_one_hot].to_numpy()
    Y = data[target_var].to_numpy()
    n, p = X.shape
    beta = cp.Variable(p)  # model coefficients
    penalty_types = {k: v for k, v in penalty_types.items() if any(k.startswith(prefix) for prefix in input_var)}
    one_hot_groups={key:[i for i,x in enumerate(data[input_var_one_hot].columns) if (x.startswith(key+"_"))] for key,val in penalty_types.items()}
    one_hot_groups = {k: v for k, v in one_hot_groups.items() if v}
    # Linear predictor: η = Xβ + intercept + offset
    eta = X @ beta + intercept + data[offset_var]

    # -------------------------------
    # Define log-likelihood per family
    # -------------------------------
    if family.lower() == "poisson":
        log_likelihood = cp.sum(cp.multiply(Y, eta) - cp.exp(eta))

    elif family.lower() == "gaussian":
        # Equivalent to least-squares loss
        log_likelihood = -0.5 * cp.sum_squares(Y - eta)

    elif family.lower() == "binomial":
        # Logistic regression likelihood
        log_likelihood = cp.sum(cp.multiply(Y, eta) - cp.logistic(eta))

    elif family.lower() == "gamma":
        # Canonical log-link
        log_likelihood = cp.sum(-Y / cp.exp(eta) - eta)

    elif family.lower() == "inverse_gaussian":
        mu = cp.exp(eta)
        log_likelihood = cp.sum(-0.5 * cp.square(Y - mu) / (mu**3))

    elif family.lower() == "linear":
        log_likelihood = cp.sum(cp.multiply(Y, eta) - eta)

    else:
        raise ValueError(f"Unsupported family: {family}")

    # -------------------------------
    # Construct custom penalties
    # -------------------------------
    g_fused_penalty = 0
    fused_penalty = 0
    g_fused_graph_size = 0

    # First pass: compute reference modality length for categorical groups
    for key, val in one_hot_groups.items():
        if penalty_types.get(key) == "g_fused":
            g_fused_graph_size += len(val) + 1

        ref_mod_length = len(data)
        for col in input_var_one_hot:
            if col.startswith(key):
                ref_mod_length -= sum(data[col])

    # Second pass: build fused and group-fused penalties
    for key, val in one_hot_groups.items():
        if penalty_types.get(key) == "fused":
            
            # Fused penalty: encourages adjacent categories to have similar coefficients
            weight = ((sum(data.iloc[:, val[0]]) + ref_mod_length) / len(data))**0.5
            fused_penalty += cp.norm1(weight * (beta[val[0]] - 0))  # reference modality = 0
            if len(val) > 1:
                for i in range(1, len(val)):
                    weight = ((sum(data.iloc[:, val[i]]) + sum(data.iloc[:, val[i-1]])) / len(data))**0.5
                    fused_penalty += cp.norm1(weight * (beta[val[i]] - beta[val[i-1]]))

        elif penalty_types.get(key) == "g_fused":
            # Group-fused penalty: all pairs within group are penalized
            for k in range(len(val)):
                weight = (len(val) / g_fused_graph_size) * ((sum(data.iloc[:, val[k]]) + ref_mod_length) / len(data))**0.5
                g_fused_penalty += cp.norm1(weight * (beta[val[k]] - 0))
                if len(val) > 1:
                    for j in range(k, len(val)):
                        weight = (len(val) / g_fused_graph_size) * ((sum(data.iloc[:, val[k]]) + sum(data.iloc[:, val[j]])) / len(data))**0.5
                        g_fused_penalty += cp.norm1(weight * (beta[val[k]] - beta[val[j]]))

        else:
            print(f"!!! Wrong penalisation type given for variable {key}")

    # -------------------------------
    # Define objective and solve
    # -------------------------------
    objective = cp.Minimize(-log_likelihood + lambda_1 * g_fused_penalty + lambda_1 * fused_penalty)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.CLARABEL)

    # Return coefficients (intercept first, then betas)
    return np.concatenate(([intercept.value], beta.value))


def get_data_ready_for_glm(data,input_variables,penalty_types,target,offset,method):
    data_onehot,ref_modality_dict,input_var_onehot = choose_ref_modality(data,input_variables,method)
    data_onehot=reorder_df_columns(data_onehot,penalty_types,input_variables) 
    data_onehot = pd.concat([data_onehot, 
                                          data[input_variables].rename(columns={col: "non_dum_" + col for col in input_variables})], axis=1)
    data_onehot[target]=data[target]
    data_onehot[offset]=data[offset]
    return(data_onehot,ref_modality_dict,input_var_onehot)




class GridSearch_Generalised:

    def __init__(self,family,var_nb_min=None,var_nb_max=None,parcimony_step=None,smoothness_step=None,lbd_fused=None,lbd_group=None):
        
        self.family = family
        self.var_nb_min = var_nb_min
        self.var_nb_max = var_nb_max
        self.parcimony_step = parcimony_step
        self.smoothness_step = smoothness_step
        self.lbd_group = lbd_group
        self.lbd_fused = lbd_fused

        group1_any = (parcimony_step is not None) or (smoothness_step is not None)
        group1_full = (parcimony_step is not None) and (smoothness_step is not None)
    
        # Group 2 checks
        group2_any = (lbd_group is not None) or (lbd_fused is not None)
        group2_full = (lbd_group is not None) and (lbd_fused is not None)
        
        group3 = (var_nb_min is not None) or (var_nb_max is not None)
        # Errors:

        # 1. Must provide exactly one full group
        if group1_full == group2_full:  # Both True or both False
            raise ValueError("Provide either (parcimony & smoothness) OR (group & fused) values, but not both.")
            
        # 2. Partial groups
        if group1_any and not group1_full:
            raise ValueError("Both parcimony and smoothness values must be provided together.")
        if group2_any and not group2_full:
            raise ValueError("Both group and fused values must be provided together.")
    
        # 2. Can't use limits on variables for a given lambda
        if group3 and group2_any:  # Both True or both False
            raise ValueError("Input var_nb_min and var_nb_max doesn't work with a given set of lambdas.")

    @time_it
    def fit(self,data,penalty_types,input_variables,target, offset, n_k_fold = 5):
        self.data = data
        self.penalty_types=penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.data_onehot,self.ref_modality_dict,self.input_var_onehot = get_data_ready_for_glm(self.data,
                                                                                               self.input_variables,
                                                                                               self.penalty_types,
                                                                                               target,
                                                                                               offset,
                                                                                               "First")
        modality_arr=[]
        variables_arr = []
        for val in self.input_variables:
            for i,x in enumerate(self.data_onehot.columns):
                if (x.startswith(val)):
                    modality_arr.append(x.replace(val + "_",""))
                    variables_arr.append(val)
        self.modality_var_df=pd.DataFrame(data={'variable': variables_arr, 'modality': modality_arr})
        self.fused_groups_dict={val:[i for i,x in enumerate(self.data_onehot.columns) if (x.startswith(val+"_"))] for val in self.input_variables}
        self.target = target
        self.offset = offset


        if (self.lbd_group == None) & (self.lbd_fused == None):
            self.get_lambda_curve()
        else:
            results = pd.DataFrame(columns = ['group_lambda',
                                              'fused_lambda',
                                              'variable_number',
                                              'modalities_number',
                                              'var_mod_details',
                                              'Deviance_cv_train',
                                              'Deviance_cv_test',
                                              'variables',
                                              'betas'],
                                  data = np.empty((1, 9)),dtype=object)

            if isinstance(self.lbd_group, (list, tuple)):
                if len(self.lbd_group) != len(self.lbd_fused) :
                    raise ValueError(f"the Fused and Group arrays must have the same size, got {len(self.lbd_group)} and {len(self.lbd_fused)}")
                for i in range(len(self.lbd_group)):
                    temp_lbd_fused = self.lbd_fused[i]
                    temp_lbd_group = self.lbd_group[i]
                    print(f"Running for fused value = {temp_lbd_fused} and group value = {temp_lbd_group}")
                    results = self.crossval_pair_lambdas(results,
                                                    input_variables,
                                                    temp_lbd_group,
                                                    temp_lbd_fused,
                                                    i
                                                    )
                
            else :
                results = self.crossval_pair_lambdas(results,
                                                    input_variables,
                                                    self.lbd_group,
                                                    self.lbd_fused,
                                                    0
                                                    )
                
            self.lambda_curve = results

    @time_it
    def fit_one_lambda(self,data,inputs,lbd_fused,lbd_group):
         
        grouped_model_temp = compute_group_lasso_glm(data,
                                                  inputs,
                                                  self.target,
                                                  self.offset,
                                                  self.penalty_types,
                                                  lbd_group, 
                                                  "Poisson"
                                                  )
    
        temp_var_kept, temp_var_removed = keep_variables_after_group_lasso(data,
            self.modality_var_df,   
            grouped_model_temp,
            inputs
        )

        temp_glm = custom_glm_with_fused_penalty(
                                                data,
                                                inputs,
                                                self.target,
                                                self.offset,
                                                self.penalty_types,
                                                lbd_fused, 
                                                "Poisson"
                                            )
        temp_data, temp_var_kept, temp_var_removed = group_modalities_based_on_betas(
                                                                                    data,
                                                                                    self.modality_var_df,   
                                                                                    temp_glm,
                                                                                    inputs,  
                                                                                    self.ref_modality_dict
                                                                                )
        if temp_var_kept != [] : 
            temp_data_onehot,temp_ref_modality_dict,temp_input_var_onehot = get_data_ready_for_glm(temp_data,
                                                                                                   temp_var_kept,
                                                                                                   self.penalty_types,
                                                                                                   self.target,
                                                                                                   self.offset,
                                                                                                   "First")
            betas_temp = compute_glm_no_pen(temp_data_onehot,
                                             temp_var_kept,
                                             self.target,
                                             self.offset)
    
            betas = betas_temp
            return(len(temp_ref_modality_dict),temp_ref_modality_dict,betas,temp_var_kept)
        else :
            print("NO VARIABLES")
            betas = [data[self.target].mean()]
            return(0,{},betas,[])

    
    @time_it
    def crossval_pair_lambdas(self,
                              results,
                                  inputs,
                                  group_lambda_val,
                                  fused_lambda_val,
                                  counter,
                                  ):

        print(f"Model with group lambda = {group_lambda_val} and fused lambda = {fused_lambda_val}")
        
        kf = KFold(n_splits=self.n_k_fold, shuffle=True)
        
        error_train_total = 0
        error_test_total = 0
        for j, (train_idx, test_idx) in enumerate(kf.split(self.data_onehot)):
            # Split data
            X_train_cv = self.data_onehot.iloc[train_idx]
            X_test_cv = self.data_onehot.iloc[test_idx]


            grouped_model_temp = compute_group_lasso_glm(X_train_cv,
                                                  inputs,
                                                  self.target,
                                                  self.offset,
                                                  self.penalty_types,
                                                  group_lambda_val, 
                                                  "Poisson"
                                                  )
    
            temp_var_kept, temp_var_removed = keep_variables_after_group_lasso(X_train_cv,
                self.modality_var_df,   
                grouped_model_temp,
                inputs
            )

            if temp_var_kept != [] : 
                # Fit GLM on training fold
                temp_betas = custom_glm_with_fused_penalty(
                    X_train_cv, temp_var_kept, self.target, self.offset,
                    # grouped_fused_groups_dict, 
                    self.penalty_types,
                    fused_lambda_val, "Poisson"
                )

                temp_var_kept_onehot = get_onehot_columns(X_train_cv,temp_var_kept)
                predict_train = np.exp(
                                        X_train_cv[temp_var_kept_onehot] @ temp_betas[1:]
                                        + temp_betas[0]
                                        + X_train_cv[self.offset]
                                        )
                
                # Test predictions + deviance
                predict_test = np.exp(
                                        X_test_cv[temp_var_kept_onehot] @ temp_betas[1:]
                                        + temp_betas[0]
                                        + X_test_cv[self.offset]
                                    )
                
            else:
                predict_train =  [X_train_cv[self.target].mean()] * len(X_train_cv)
                predict_test =  [X_train_cv[self.offset].mean()] * len(X_test_cv)
                # Training predictions + deviance

            error_train = Compute_Poisson_Deviance(
                                                    np.array(predict_train),
                                                    np.array(X_train_cv[self.target]),
                                                    np.array(X_train_cv[self.offset])
                                                )
            error_test = Compute_Poisson_Deviance(
                                                np.array(predict_test),
                                                np.array(X_test_cv[self.target]),
                                                np.array(X_test_cv[self.offset])
                                                )
            error_train_total = error_train_total + error_train
            error_test_total = error_test_total + error_test

        error_train_mean = error_train_total / self.n_k_fold
        error_test_mean = error_test_total / self.n_k_fold


        results.at[counter,'fused_lambda'] = fused_lambda_val
        results.at[counter,'group_lambda'] = group_lambda_val
        results.at[counter,"Deviance_cv_train"]=error_train_mean
        results.at[counter,"Deviance_cv_test"]=error_test_mean
        
        # FINAL MODEL
        nb_modalities,total_modality_dict,betas,kept_var = self.fit_one_lambda(self.data_onehot,inputs,fused_lambda_val,group_lambda_val)
        results.at[counter,"modalities_number"]=nb_modalities
        results.at[counter,"var_mod_details"]=total_modality_dict
        results.at[counter,'betas'] = betas
        results.at[counter,"variables"]= " ".join(map(str, kept_var))
        results.at[counter,"variable_number"]= len(kept_var)
        results = results.sort_values(by=['group_lambda','fused_lambda'], ascending=True).reset_index(drop=True)
        
        return results

    @time_it
    def test_grouping_variables(self,grouped_lasso_table,lambda_group_temp,counter):
        
        grouped_model_temp = compute_group_lasso_glm(self.data_onehot,
                                                      self.input_variables,
                                                      self.target,
                                                      self.offset,
                                                      self.penalty_types,
                                                      lambda_group_temp, 
                                                      "Poisson"
                                                      )
        
        temp_data, temp_var_kept, temp_var_removed = group_modalities_based_on_betas(self.data_onehot,
            self.modality_var_df,   
            grouped_model_temp,
            self.input_variables,   
            self.ref_modality_dict  
        )
        grouped_lasso_table.at[counter,'lambda'] = lambda_group_temp
        grouped_lasso_table.at[counter,'var_nb'] = len(temp_var_kept)
        grouped_lasso_table.at[counter,'variables'] = temp_var_kept
        
        
        grouped_lasso_table = grouped_lasso_table.sort_values('lambda', ascending=True).reset_index(drop=True)
        return(grouped_lasso_table)
    
    @time_it
    def get_deviance_one_model(self,results,inputs,fused_lambda_temp,group_lambda_temp,rownb):
        

        print(f"Based on {inputs} input variables (group lambda = {group_lambda_temp}), model with fused lambda = {fused_lambda_temp}")

        
        # -------------------------------
        # Step 1: Fit GLM at given λ
        # -------------------------------
        
        temp_glm = custom_glm_with_fused_penalty(
            self.data_onehot,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            fused_lambda_temp, 
            "Poisson"
        )
    
        # -------------------------------
        # Step 2: Group modalities after fusion
        # -------------------------------
        temp_data, temp_var_kept, temp_var_removed = group_modalities_based_on_betas(
            self.data_onehot,
            self.modality_var_df,   
            temp_glm,
            inputs,  
            self.ref_modality_dict
        )
        
        print(f"Based on {inputs} input variables , model with fused lambda = {fused_lambda_temp} returns {temp_var_kept} variables")
        
        results.at[rownb,'fused_lambda'] = fused_lambda_temp
        results.at[rownb,"variable_number"]=len(temp_var_kept)
        return(results)

    @time_it    
    def get_lambda_curve(self):
        # i=0

        max_lambda_group_reached=False
        lambda_group_temp=0.1
        grouped_lasso_table = pd.DataFrame(columns = ['lambda', 'var_nb','variables'],
                      data = np.empty((1, 3)),dtype=object)
        counter=0
        print('----------------------------------')
        print('Step 1 : Get range of Group Lambdas')
        print('----------------------------------\n')
        
        while not max_lambda_group_reached:
            print(f"Running group GLMs until reaching maximum lambda : {lambda_group_temp}")
            grouped_lasso_table = self.test_grouping_variables(grouped_lasso_table,
                                                         lambda_group_temp,
                                                         counter)
            counter +=1
            temp_var_kept = grouped_lasso_table.loc[len(grouped_lasso_table)-1,"variables"]
            if len(temp_var_kept) == 0 | len(temp_var_kept) <= self.var_nb_min:
                max_lambda_group_reached=True
                max_lambda_group = grouped_lasso_table['lambda'].max()
            
            lambda_group_temp=lambda_group_temp*10
                         
        jump = (max_lambda_group - grouped_lasso_table.loc[0,'lambda'])/self.parcimony_step
        tested_lambda_list = [grouped_lasso_table.loc[0,'lambda'] + jump * step for step in range(1,self.parcimony_step)]
        for lambda_group_temp in tested_lambda_list:
            print(f"Running group GLMs with lambda = {lambda_group_temp}")
            grouped_lasso_table = self.test_grouping_variables(grouped_lasso_table,
                                                     lambda_group_temp,
                                                     counter)
            counter +=1

        grouped_lasso_table['variables_tuple'] = grouped_lasso_table['variables'].apply(frozenset)
        # Group by the tuple version
        grouped_lasso_uniques = grouped_lasso_table.groupby('variables_tuple')['lambda'].mean().reset_index()
        grouped_lasso_uniques['variables'] = grouped_lasso_uniques['variables_tuple'].apply(lambda x: sorted(list(x)))
        grouped_lasso_uniques = grouped_lasso_uniques.drop(columns='variables_tuple')
        grouped_lasso_uniques["variables"] = grouped_lasso_uniques["variables"].apply(lambda lst: [x.replace('_grouped', '') for x in lst])
        
    
        # # remove duplicates by converting to set, then back to list of lists
        variable_list = [x for x in grouped_lasso_uniques['variables'] if x]
        
        # END OF GROUP  
        
        print(f"Group GLMs gives the following variable lists : ")
        for arr in variable_list:
            print(arr)

        print('\n----------------------------------')
        print('Step 2 : Iterate on each variable list \n')
        

        k=0
        results = pd.DataFrame(columns = ['group_lambda',
                                          'fused_lambda',
                                              'variable_number',
                                              'modalities_number',
                                              'var_mod_details',
                                              'Deviance_cv_train',
                                              'Deviance_cv_test',
                                              'variables',
                                              'betas'],
                                  data = np.empty((1, 9)),dtype=object)
        
        for m in range(len(grouped_lasso_uniques)):
            results_temp = pd.DataFrame(columns = ['fused_lambda',
                                              'variable_number'],
                                  data = np.empty((1, 2)),dtype=object)
            
            # print(grouped_lasso_uniques.loc[m,'variables'])
            input_list_temp = grouped_lasso_uniques.loc[m,'variables']
            
            if input_list_temp !=[]:
                group_lambda_temp = grouped_lasso_uniques.loc[m,'lambda']
                print(f"Fused GLMs on this variable list : {input_list_temp}")
                max_lambda_reached=False
                lambda_temp=0.1
                # i=0
                counter = 0
                print('\n Step 2.1 : For each variable list, get range of Fused Lambdas \n')
                while not max_lambda_reached:
                    results_temp = self.get_deviance_one_model(results_temp,input_list_temp,lambda_temp,group_lambda_temp,counter)
                    if results_temp.loc[counter,"variable_number"]==0:
                        max_lambda_reached=True
                        lambda_max = results_temp.loc[counter,"fused_lambda"]
                    # i+=1
                    lambda_temp=lambda_temp*10
                    counter+=1
                    
    
                lambda_temp=results_temp.loc[0,"fused_lambda"]
                jump = (lambda_max - results_temp.loc[0,'fused_lambda'])/self.smoothness_step
                tested_fused_lambda_list = [results_temp.loc[0,'fused_lambda'] + jump * step for step in range(0,self.smoothness_step+1)]
                
                for fused_lambda_temp in tested_fused_lambda_list:
                    print('\n Step 2.2 : For selected group and fused lambdas, get cross-val')
                    results = self.crossval_pair_lambdas(results,input_list_temp,group_lambda_temp,fused_lambda_temp,k)
                    k+=1
            print(f"End of curve for {input_list_temp}")
        self.lambda_curve = results


    def plot_curve(self):
        categories = self.lambda_curve['group_lambda'].unique()
        colors = plt.cm.tab10(range(len(categories)))  
        plt.figure(figsize=(8, 6))
    
        for cat, color in zip(categories, colors):
            subset = self.lambda_curve[self.lambda_curve['group_lambda'] == cat]
            plt.scatter(subset['variable_number'], 
                        subset['Deviance_cv_test'], 
                        color=color, 
                        label=str(np.around(cat)))
    
            # Add labels
            for _, row in subset.iterrows():
                label = f"{np.around(row['group_lambda'])} / {np.around(row['fused_lambda'])}"
                plt.text(
                    row['variable_number'],
                    row['Deviance_cv_test'],
                    label,
                    fontsize=8,
                    ha='center',
                    va='bottom'
                )
    
        plt.xlabel('Final Variable Number')
        plt.ylabel('Deviance cv test')
        plt.title('Scatter plot with discrete colors for group lambda value')
        plt.legend(title="Group Lambda")
        plt.tight_layout()
        plt.show()        
          
        
    
class GridSearch_Group:

    def __init__(self,family,var_nb_min=None,var_nb_max=None,parcimony_step=None,lbd_group=None):
        
        self.family = family
        self.var_nb_min = var_nb_min
        self.var_nb_max = var_nb_max
        self.parcimony_step = parcimony_step
        self.lbd_group = lbd_group

        group1 = (parcimony_step is not None)
        group2 = (lbd_group is not None)
        group3 = (var_nb_min is not None) or (var_nb_max is not None)

        # Errors:

        # 1. Must provide exactly one full group
        if group1 == group2:  # Both True or both False
            raise ValueError("Provide either parcimony OR group values, but not both or none.")
              
        # 2. Can't use limits on variables for a given lambda
        if group3 and group2:  # Both True or both False
            raise ValueError("Input var_nb_min and var_nb_max doesn't work with a given set of lambdas.")


    def fit(self,data,penalty_types,input_variables,target, offset,n_k_fold = 5):
        self.data = data
        self.penalty_types=penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.data_onehot,self.ref_modality_dict,self.input_var_onehot = get_data_ready_for_glm(self.data,
                                                                                               self.input_variables,
                                                                                               self.penalty_types,
                                                                                               target,
                                                                                               offset,
                                                                                               "First")
        modality_arr=[]
        variables_arr = []
        for val in self.input_variables:
            for i,x in enumerate(self.data_onehot.columns):
                if (x.startswith(val)):
                    modality_arr.append(x.replace(val + "_",""))
                    variables_arr.append(val)
        self.modality_var_df=pd.DataFrame(data={'variable': variables_arr, 'modality': modality_arr})
        self.fused_groups_dict={val:[i for i,x in enumerate(self.data_onehot.columns) if (x.startswith(val+"_"))] for val in self.input_variables}
        
        self.target = target
        self.offset = offset
        
        if self.lbd_group == None:
            self.get_lambda_curve()
        else:
            results = pd.DataFrame(columns = ['group_lambda',
                                          'variable_number',
                                          'Deviance_cv_train',
                                          'Deviance_cv_test',
                                          'variables',
                                          'betas'],
                                  data = np.empty((1, 6)),dtype=object)
                                  
            if isinstance(self.lbd_group, (list, tuple)):
                for i in range(len(self.lbd_group)):
                    temp_lbd_group = self.lbd_group[i]
                    print(f"Running for group value = {temp_lbd_group}")
                    results = self.crossval_lambda(results,
                        temp_lbd_group,
                        i)
                    
            else :
                results = self.crossval_lambda(results,
                        temp_lbd_group,
                        0)
            self.lambda_curve = results
            

    def fit_one_lambda(self,data,inputs,lbd_group):
         
        grouped_model_temp = compute_group_lasso_glm(data,
                                                  inputs,
                                                  self.target,
                                                  self.offset,
                                                  self.penalty_types,
                                                  lbd_group, 
                                                  "Poisson"
                                                  )
    
        temp_var_kept, temp_var_removed = keep_variables_after_group_lasso(data,
            self.modality_var_df,   
            grouped_model_temp,
            inputs
        )
        
        if temp_var_kept != [] : 
            betas_temp = compute_glm_no_pen(data,
                                             temp_var_kept,
                                             self.target,
                                             self.offset)
            betas = betas_temp
            return(betas,temp_var_kept)
        else :
            print("NO VARIABLES")
            betas = [data[self.target].mean()]
            return(betas,[])

    def test_grouping_variables(self,grouped_lasso_table,lambda_group_temp,counter):
        
        grouped_model_temp = compute_group_lasso_glm(self.data_onehot,
                                                      self.input_variables,
                                                      self.target,
                                                      self.offset,
                                                      self.penalty_types,
                                                      lambda_group_temp, 
                                                      "Poisson"
                                                      )
        
        temp_data, temp_var_kept, temp_var_removed = group_modalities_based_on_betas(self.data_onehot,
            self.modality_var_df,   
            grouped_model_temp,
            self.input_variables,   
            self.ref_modality_dict  # assumed global
        )
        grouped_lasso_table.at[counter,'lambda'] = lambda_group_temp
        grouped_lasso_table.at[counter,'var_nb'] = len(temp_var_kept)
        grouped_lasso_table.at[counter,'variables'] = temp_var_kept
        
        
        grouped_lasso_table = grouped_lasso_table.sort_values('lambda', ascending=True).reset_index(drop=True)
        return(grouped_lasso_table)

    def crossval_lambda(self,
                        results,
                        group_lambda_val,
                        counter
                        ):

        error_train_total = 0
        error_test_total = 0

        kf = KFold(n_splits=self.n_k_fold, shuffle=True)
        
    
        for j, (train_idx, test_idx) in enumerate(kf.split(self.data_onehot)):
            # Split data
            X_train_cv = self.data_onehot.iloc[train_idx]
            X_test_cv = self.data_onehot.iloc[test_idx]

            grouped_model_temp = compute_group_lasso_glm(X_train_cv,
                                                  self.input_variables,
                                                  self.target,
                                                  self.offset,
                                                  self.penalty_types,
                                                  group_lambda_val, 
                                                  "Poisson"
                                                  )
    
            temp_var_kept, temp_var_removed = keep_variables_after_group_lasso(X_train_cv,
                self.modality_var_df,   
                grouped_model_temp,
                self.input_variables
            )

            if temp_var_kept != [] : 
                # Fit GLM on training fold
                temp_betas = compute_glm_no_pen(X_train_cv,
                                             temp_var_kept,
                                             self.target,
                                             self.offset)

                temp_var_kept_onehot = get_onehot_columns(X_train_cv,temp_var_kept)
                predict_train = np.exp(
                                        X_train_cv[temp_var_kept_onehot] @ temp_betas[1:]
                                        + temp_betas[0]
                                        + X_train_cv[self.offset]
                                        )
                
                # Test predictions + deviance
                predict_test = np.exp(
                                        X_test_cv[temp_var_kept_onehot] @ temp_betas[1:]
                                        + temp_betas[0]
                                        + X_test_cv[self.offset]
                                    )
                
            else:
                predict_train =  [X_train_cv[self.target].mean()] * len(X_train_cv)
                predict_test =  [X_train_cv[self.offset].mean()] * len(X_test_cv)
                # Training predictions + deviance

            error_train = Compute_Poisson_Deviance(
                                                    np.array(predict_train),
                                                    np.array(X_train_cv[self.target]),
                                                    np.array(X_train_cv[self.offset])
                                                )
            error_test = Compute_Poisson_Deviance(
                                                np.array(predict_test),
                                                np.array(X_test_cv[self.target]),
                                                np.array(X_test_cv[self.offset])
                                                )
            # Store results
            error_train_total = error_train_total + error_train
            error_test_total = error_test_total + error_test

        error_train_mean = error_train_total / self.n_k_fold
        error_test_mean = error_test_total / self.n_k_fold
        results.at[counter,'group_lambda'] = group_lambda_val
        results.at[counter,"Deviance_cv_train"]=error_train_mean
        results.at[counter,"Deviance_cv_test"]=error_test_mean
        
        # FINAL MODEL
        if temp_var_kept !=[]:
            betas,kept_var = self.fit_one_lambda(self.data_onehot,temp_var_kept,group_lambda_val)
            results.at[counter,'betas'] = betas
            results.at[counter,"variables"]= " ".join(map(str, kept_var))
            results.at[counter,"variable_number"]= len(kept_var)
            results = results.sort_values(by=['group_lambda'], ascending=True).reset_index(drop=True)
        else : 
            results.at[counter,'betas'] = [self.data_onehot[self.target].mean()]
            results.at[counter,"variables"]= []
            results.at[counter,"variable_number"]= 0
        return results

        
    def get_lambda_curve(self):

        max_lambda_group_reached=False
        lambda_group_temp=0.1
        grouped_lasso_table = pd.DataFrame(columns = ['lambda', 'var_nb','variables'],
                      data = np.empty((1, 3)),dtype=object)
        counter=0
        print('----------------------------------')
        print('Step 1 : Get range of Group Lambdas')
        print('----------------------------------\n')
        
        while not max_lambda_group_reached:
            print(f"Running group GLMs until reaching maximum lambda : {lambda_group_temp}")
            grouped_lasso_table = self.test_grouping_variables(grouped_lasso_table,
                                                         lambda_group_temp,
                                                         counter)
            counter +=1
            temp_var_kept = grouped_lasso_table.loc[len(grouped_lasso_table)-1,"variables"]
            if len(temp_var_kept) == 0 | len(temp_var_kept) <= self.var_nb_min:
                max_lambda_group_reached=True
                max_lambda_group = grouped_lasso_table['lambda'].max()
            
            lambda_group_temp=lambda_group_temp*10  
  
        jump = (max_lambda_group - grouped_lasso_table.loc[0,'lambda'])/self.parcimony_step
        tested_lambda_list = [grouped_lasso_table.loc[0,'lambda'] + jump * step for step in range(0,self.parcimony_step+1)]
        results = pd.DataFrame(columns = ['group_lambda',
                                          'variable_number',
                                          'Deviance_cv_train',
                                          'Deviance_cv_test',
                                          'variables',
                                          'betas'],
                                  data = np.empty((1, 6)),dtype=object)
        counter = 0
        for lambda_group_temp in tested_lambda_list:
            print(f"Running group GLMs with group lambda = {lambda_group_temp}")
            results = self.crossval_lambda(results,
                        lambda_group_temp,
                        counter
                        )
            counter+=1
        self.lambda_curve = results

    

class GridSearch_Fused:

    def __init__(self,family,var_nb_min= None,var_nb_max = None,smoothness_step=None,lbd_fused=None):
        
        self.family = family
        self.var_nb_min = var_nb_min
        self.var_nb_max = var_nb_max
        self.smoothness_step = smoothness_step
        self.lbd_fused = lbd_fused

        group1 = (smoothness_step is not None)
        group2 = (lbd_fused is not None)
        group3 = (var_nb_min is not None) or (var_nb_max is not None)
        # Errors:

        # 1. Must provide exactly one full group
        if group1 == group2:  # Both True or both False
            raise ValueError("Provide either smoothness or fused values, but not both nor none.")

        # 2. Can't use limits on variables for a given lambda
        if group3 and group2:  # Both True or both False
            raise ValueError("Input var_nb_min and var_nb_max doesn't work with a given set of lambdas.")
            
    def fit(self,data,penalty_types,input_variables,target, offset,n_k_fold = 5):
        self.data = data
        self.penalty_types=penalty_types
        self.input_variables = input_variables
        self.n_k_fold = n_k_fold
        self.data_onehot,self.ref_modality_dict,self.input_var_onehot = get_data_ready_for_glm(self.data,
                                                                                               self.input_variables,
                                                                                               self.penalty_types,
                                                                                               target,
                                                                                               offset,
                                                                                               "First")
        modality_arr=[]
        variables_arr = []
        for val in self.input_variables:
            for i,x in enumerate(self.data_onehot.columns):
                if (x.startswith(val)):
                    modality_arr.append(x.replace(val + "_",""))
                    variables_arr.append(val)
        self.modality_var_df=pd.DataFrame(data={'variable': variables_arr, 'modality': modality_arr})
        self.fused_groups_dict={val:[i for i,x in enumerate(self.data_onehot.columns) if (x.startswith(val+"_"))] for val in self.input_variables}
        
        
        self.target = target
        self.offset = offset

        if self.lbd_fused == None:
            self.get_lambda_curve()
        else:
            results = pd.DataFrame(columns = ['fused_lambda',
                                          'variable_number',
                                          'modalities_number',
                                          'var_mod_details',
                                          'Deviance_cv_train',
                                          'Deviance_cv_test',
                                          'variables',
                                          'betas'],
                                  data = np.empty((1, 8)),dtype=object)
            
            if isinstance(self.lbd_fused, (list, tuple)):
                for i in range(len(self.lbd_fused)):
                    temp_lbd_fused = self.lbd_fused[i]
                    print(f"Running for fused value = {temp_lbd_fused}")
                    results = self.crossval_lambda(results,temp_lbd_fused,i)
                    
            else :
                print(f"Running for fused value = {self.lbd_fused}")
                results = self.crossval_lambda(results,self.lbd_fused,0)
            self.lambda_curve = results
                


    def fit_one_lambda(self,data,inputs,lbd_fused):
         
        temp_glm = custom_glm_with_fused_penalty(
            data,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            lbd_fused, 
            "Poisson"
        )
        temp_data, temp_var_kept, temp_var_removed = group_modalities_based_on_betas(
            data,
            self.modality_var_df,   
            temp_glm,
            inputs,  
            self.ref_modality_dict
        )
        if temp_var_kept != [] : 
            temp_data_onehot,temp_ref_modality_dict,temp_input_var_onehot = get_data_ready_for_glm(temp_data,
                                                                                                   temp_var_kept,
                                                                                                   self.penalty_types,
                                                                                                   self.target,
                                                                                                   self.offset,
                                                                                                   "First")
            betas_temp = compute_glm_no_pen(temp_data_onehot,
                                             temp_var_kept,
                                             self.target,
                                             self.offset)
    
            betas = betas_temp
            return(len(temp_ref_modality_dict),temp_ref_modality_dict,betas,temp_var_kept)
        else :
            print("NO VARIABLES")
            betas = [data[self.target].mean()]
            return(0,{},betas,[])
            

    def crossval_lambda(self,
                              results,
                                  fused_lambda_val,
                                  counter,
                                  ):
        
        kf = KFold(n_splits=self.n_k_fold, shuffle=True)
        
        
        error_train_total = 0
        error_test_total = 0
        for j, (train_idx, test_idx) in enumerate(kf.split(self.data_onehot)):
            # Split data
            X_train_cv = self.data_onehot.iloc[train_idx]
            X_test_cv = self.data_onehot.iloc[test_idx]


            # Fit GLM on training fold
            temp_betas = custom_glm_with_fused_penalty(
                X_train_cv, self.input_variables, self.target, self.offset,
                self.penalty_types,
                fused_lambda_val, "Poisson"
            )

            temp_var_kept_onehot = get_onehot_columns(X_train_cv,self.input_variables)
            predict_train = np.exp(
                                    X_train_cv[temp_var_kept_onehot] @ temp_betas[1:]
                                    + temp_betas[0]
                                    + X_train_cv[self.offset]
                                    )
            
            # Test predictions + deviance
            predict_test = np.exp(
                                    X_test_cv[temp_var_kept_onehot] @ temp_betas[1:]
                                    + temp_betas[0]
                                    + X_test_cv[self.offset]
                                )
            


            error_train = Compute_Poisson_Deviance(
                                                    np.array(predict_train),
                                                    np.array(X_train_cv[self.target]),
                                                    np.array(X_train_cv[self.offset])
                                                )
            error_test = Compute_Poisson_Deviance(
                                                np.array(predict_test),
                                                np.array(X_test_cv[self.target]),
                                                np.array(X_test_cv[self.offset])
                                                )
            error_train_total = error_train_total + error_train
            error_test_total = error_test_total + error_test

        error_train_mean = error_train_total / self.n_k_fold
        error_test_mean = error_test_total / self.n_k_fold


        results.at[counter,'fused_lambda'] = fused_lambda_val
        results.at[counter,"Deviance_cv_train"]=error_train_mean
        results.at[counter,"Deviance_cv_test"]=error_test_mean
        
        # FINAL MODEL
        nb_modalities,total_modality_dict,betas,kept_var = self.fit_one_lambda(self.data_onehot,self.input_variables,fused_lambda_val)
        results.at[counter,"modalities_number"]=nb_modalities
        results.at[counter,"var_mod_details"]=total_modality_dict
        results.at[counter,'betas'] = betas
        results.at[counter,"variables"]= " ".join(map(str, kept_var))
        results.at[counter,"variable_number"]= len(kept_var)
        results = results.sort_values(by=['fused_lambda'], ascending=True).reset_index(drop=True)
        
        return results

    
    def get_deviance_one_model(self,results,inputs,fused_lambda_temp,rownb):
        
        
        # -------------------------------
        # Step 1: Fit GLM at given λ
        # -------------------------------
        
        temp_glm = custom_glm_with_fused_penalty(
            self.data_onehot,
            inputs,
            self.target,
            self.offset,
            self.penalty_types,
            fused_lambda_temp, 
            "Poisson"
        )
    
        # -------------------------------
        # Step 2: Group modalities after fusion
        # -------------------------------
        temp_data, temp_var_kept, temp_var_removed = group_modalities_based_on_betas(
            self.data_onehot,
            self.modality_var_df,   
            temp_glm,
            inputs,  
            self.ref_modality_dict
        )
        
    
        results.at[rownb,'fused_lambda'] = fused_lambda_temp
        results.at[rownb,"variable_number"]=len(temp_var_kept)
        return(results)

        
    def get_lambda_curve(self):
        
        # i=0
        counter=0
        # k=0
        max_lambda_reached=False
        lambda_temp=0.1
        
        results = pd.DataFrame(columns = ['fused_lambda',
                                          'variable_number',
                                          'modalities_number',
                                          'var_mod_details',
                                          'Deviance_cv_train',
                                          'Deviance_cv_test',
                                          'variables',
                                          'betas'],
                                  data = np.empty((1, 8)),dtype=object)
        
        results_temp = pd.DataFrame(columns = ['fused_lambda',
                                            'variable_number'],
                                data = np.empty((1, 2)),dtype=object)
        

        print('\n Step 2.1 : For each variable list, get range of Fused Lambdas \n')
        while not max_lambda_reached:
            print(f"Running Fused Lambda until reaching maximum, fused lambda = {lambda_temp}")
        
            results_temp = self.get_deviance_one_model(results_temp,lambda_temp,counter)
            if results_temp.loc[counter,"variable_number"]==0:
                max_lambda_reached=True
                lambda_max = results_temp.loc[counter,"fused_lambda"]
            # i+=1
            lambda_temp=lambda_temp*10
            counter+=1
            

        lambda_temp=results_temp.loc[0,"fused_lambda"]
        jump = (lambda_max - results_temp.loc[0,'fused_lambda'])/self.smoothness_step
        tested_fused_lambda_list = [results_temp.loc[0,'fused_lambda'] + jump * step for step in range(0,self.smoothness_step+1)]
        counter=0        
        for fused_lambda_temp in tested_fused_lambda_list:
            print(f'Running fused lambda = {fused_lambda_temp}')
            results = self.crossval_lambda(results,fused_lambda_temp,counter)
            counter+=1
        self.lambda_curve = results

    