# from mypkg.mod2_refactored_parallel import GridSearch_Generalised
        # , GridSearch_Group, GridSearch_Fused

from mypkg.mod5 import GridSearch_Generalised, GridSearch_Group, GridSearch_Fused


import numpy as np
import pandas as pd



mtpl_python = pd.read_csv(r"C:\Users\AdrienCondamin\Documents\Modern Python Pricing\Motor vehicle insurance data.csv",sep=";")
mtpl_python.head()

mtpl_python["N_claims_year"] = mtpl_python["N_claims_year"].fillna(0)
mtpl_python = mtpl_python[mtpl_python["N_claims_year"]<5]


bins = np.linspace(mtpl_python['Weight'].min(),mtpl_python['Weight'].max(), 5)
mtpl_python["Weight2"] = pd.cut(mtpl_python['Weight'],bins=bins,include_lowest=True)

bins = np.linspace(mtpl_python['Value_vehicle'].min(),mtpl_python['Value_vehicle'].max(), 5)
mtpl_python["Value_vehicle2"] = pd.cut(mtpl_python['Value_vehicle'],bins=bins,include_lowest=True)

bins = np.linspace(mtpl_python['Power'].min(),mtpl_python['Power'].max(), 5)
mtpl_python["Power2"] = pd.cut(mtpl_python['Power'],bins=bins,include_lowest=True)

bins = np.linspace(mtpl_python['Cylinder_capacity'].min(),mtpl_python['Cylinder_capacity'].max(), 5)
mtpl_python["Cylinder_capacity2"] = pd.cut(mtpl_python['Cylinder_capacity'],bins=bins,include_lowest=True)

bins = np.linspace(mtpl_python['N_claims_history'].min(),mtpl_python['N_claims_history'].max(), 5)
mtpl_python["N_claims_history2"] = pd.cut(mtpl_python['N_claims_history'],bins=bins,include_lowest=True)

#Part that should be filled for any model 
target_variable = "N_claims_year" 
offset_variable = "EXP" 
input_variables=["Area","Second_driver","Power2","Cylinder_capacity2","Value_vehicle2","Type_fuel","Weight2"]
for c in input_variables:
    mtpl_python[c]=pd.Categorical(mtpl_python[c])  # Illustrates that Python requires indenting !!

for col in ["Date_last_renewal", "Date_lapse", "Date_next_renewal"]:
    mtpl_python[col] = pd.to_datetime(mtpl_python[col], errors="coerce")
    
mtpl_python["end_date"] = np.where(
    mtpl_python["Date_lapse"].isna(),
    mtpl_python["Date_next_renewal"],
    mtpl_python[["Date_lapse", "Date_next_renewal"]].min(axis=1)   # row-wise min
)
# Compute difference in years
mtpl_python["year_diff"] = (mtpl_python["end_date"] - mtpl_python["Date_last_renewal"]).dt.days / 365.25

mtpl_python["year_diff"] = np.where(mtpl_python["year_diff"]<=0,1,mtpl_python["year_diff"])
mtpl_python["EXP"] = mtpl_python["year_diff"].clip(upper=1)
mtpl_python["EXP"] = mtpl_python["EXP"].fillna(1)


fused_type={
    # "N_claims_history2":"fused",
            "Area":"g_fused",
            "Second_driver":"g_fused",
            "Power2":"fused",
            "Cylinder_capacity2":"fused",
            "Value_vehicle2":"fused",
            "Type_fuel":"g_fused",
            "Weight2":"fused"}





var_nb_min = 1
var_nb_max = 2
parcimony_step = 6
smoothness_step = 6



test2=GridSearch_Generalised('Poisson',
                var_nb_min,
                var_nb_max,
                5,
                5,
                # lbd_fused=[0.1,1],
                # lbd_group=[4,5]
                n_jobs=8)
test2.fit(mtpl_python,
          fused_type,
          input_variables,
          target_variable,
          offset_variable,
          n_k_fold = 5)
test2.lambda_curve
#44min
#29 min, 24 sec
#8 min, 49 secondes
#22min 31 secondes (parrallelised 1 job)
#13min 36 secondes (parrallelised 8 job)

test2.plot_curve()


gs = GridSearch_Group(
    family="Poisson",
    lbd_group=[0.1, 1.0, 10.0],   # try a few values (or a single float)
    n_jobs=1,                     # can set >1 for threaded CV
    random_state=0,
    solver="CLARABEL",            # or "ECOS", "SCS" if needed
)

# 5) Run CV + final fit
gs.fit(
    data=mtpl_python,
    penalty_types=fused_type,
    input_variables=input_variables,
    target=target_variable,
    offset=offset_variable,
    n_k_fold=5,
)


# test2=GridSearch_Fused('Poisson',
#                     #    var_nb_min,
#                     #    var_nb_max,
#                 # smoothness_step = 5,
#                 # lbd_fused=[0.1,1]
#                 lbd_fused=1
#                 )
# test2.fit(mtpl_python,
#           fused_type,
#           input_variables,
#           target_variable,
#           offset_variable,
#           n_k_fold = 5)
# test2.lambda_curve




# test2=GridSearch_Group('Poisson',
#                 # var_nb_min,
#                 # var_nb_max,
#                 # parcimony_step = 5,
#                 lbd_group=[0.1,1]
#                 )
# test2.fit(mtpl_python,
#           fused_type,
#           input_variables,
#           target_variable,
#           offset_variable,
#           n_k_fold = 5)
# test2.lambda_curve

