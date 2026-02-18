# from fusedpkg.mod5 import GridSearch_Generalised
# , GridSearch_Group, GridSearch_Fused

from fusedpkg.mod5 import GridSearch_Generalised, GridSearch_Group, GridSearch_Fused


import numpy as np
import pandas as pd
# import random



mtpl_python = pd.read_csv(r"C:\Users\AdrienCondamin\Documents\Stage\Vehiculier\Copie de db_glm1_om.csv", encoding='latin-1')

# mtpl_python.loc[1:10000,].to_csv(r"C:\Users\AdrienCondamin\Documents\Stage\Vehiculier\Copie de db_glm1_om_short.csv")

mtpl_python = mtpl_python[mtpl_python['garant']=='BG']

# 'cylindre','kwatt','exp_cnd','age_contrat','vehicule_age',
# 'places','Zone_NOA_Vol','nbre_sin_tort_tt','cdusage'
# 'marque','car_segment','libcarbu','classe_sociale'


bins = np.linspace(mtpl_python['cylindre'].min(),mtpl_python['cylindre'].max(), 10)
mtpl_python["cylindre2"] = pd.cut(mtpl_python['cylindre'],bins=bins,include_lowest=True)

mtpl_python=mtpl_python[mtpl_python['exp_cnd']!="M"]
mtpl_python['exp_cnd'] = mtpl_python['exp_cnd'].astype(float)
bins = np.linspace(mtpl_python['exp_cnd'].min(),mtpl_python['exp_cnd'].max(), 10)
mtpl_python["exp_cnd2"] = pd.cut(mtpl_python['exp_cnd'],bins=bins,include_lowest=True)

bins = np.linspace(mtpl_python['kwatt'].min(),mtpl_python['kwatt'].max(), 10)
mtpl_python["kwatt2"] = pd.cut(mtpl_python['kwatt'],bins=bins,include_lowest=True)

bins = np.linspace(mtpl_python['vehicule_age'].min(),mtpl_python['vehicule_age'].max(), 10)
mtpl_python["vehicule_age2"] = pd.cut(mtpl_python['vehicule_age'],bins=bins,include_lowest=True)

bins = np.linspace(mtpl_python['age_contrat'].min(),mtpl_python['age_contrat'].max(), 10)
mtpl_python["age_contrat2"] = pd.cut(mtpl_python['age_contrat'],bins=bins,include_lowest=True)

mtpl_python["places"] = mtpl_python["places"].astype(str)
mtpl_python["Zone_NOA_Vol"] = mtpl_python["Zone_NOA_Vol"].astype(str)
mtpl_python["nbre_sin_tort_tt"] = mtpl_python["nbre_sin_tort_tt"].astype(str)
mtpl_python["cdusage"] = mtpl_python["cdusage"].astype(str)

# Compute relative frequencies
freq = mtpl_python["marque"].value_counts(normalize=True)

# Categories below 1%
rare = freq[freq < 0.01].index

# Replace them with 'other'
mtpl_python["marque"] = mtpl_python["marque"].where(~mtpl_python["marque"].isin(rare), "other")


target_variable = "nb_sin" 
mtpl_python[target_variable]=mtpl_python[target_variable].fillna(0)

offset_variable = "offset" 
mtpl_python[offset_variable]=np.where(mtpl_python[offset_variable]>1,1,mtpl_python[offset_variable])

input_variables=['cylindre2','kwatt2','exp_cnd2','age_contrat2','vehicule_age2',
                 'places','nbre_sin_tort_tt',
                 'cdusage','Zone_NOA_Vol','marque','car_segment','libcarbu','classe_sociale']
for c in input_variables:
    mtpl_python[c]=pd.Categorical(mtpl_python[c])  # Illustrates that Python requires indenting !!

fused_type = {
            "cylindre2":"fused",
            "kwatt2":"fused",
            "exp_cnd2":"fused",
            "age_contr2":"fused",
            "vehicule_age2":"fused",
            "places":"fused",
            "nbre_sin_tort_tt":"fused",
            "cdusage":"g_fused",
            "Zone_NOA_Vol":"g_fused",
            "marque":"g_fused",
            "car_segment":"g_fused",
            "libcarbu":"g_fused",
            "classe_sociale":"g_fused"}

# mtpl_python[input_variables].isna().sum()
len(mtpl_python)

var_nb_min = 1
var_nb_max = 2
parcimony_step = 6
smoothness_step = 6

input_variables=[
                'cylindre2',
                 'kwatt2','exp_cnd2','age_contrat2','vehicule_age2'
                 ,'places','nbre_sin_tort_tt'
                 ,'cdusage','Zone_NOA_Vol',
                #  'marque',
                'car_segment','libcarbu','classe_sociale'
                 ]

test2=GridSearch_Generalised('Poisson',
                # var_nb_min,
                # var_nb_max,
                # parcimony_step = 5,
                # smoothness_step = 5,
                lbd_fused=0,
                lbd_group=0
                )

test2.fit(
    mtpl_python,
    # mtpl_python.loc[1:10000,],
          fused_type,
          input_variables,
          target_variable,
          offset_variable,
          n_k_fold = 5)
test2.lambda_curve
#174 minutes

mtpl_python[target_variable].value_counts()

test2=GridSearch_Fused('Poisson',
                    #    var_nb_min,
                    #    var_nb_max,
                # smoothness_step = 5,
                # lbd_fused=[0.1,1]
                lbd_fused=1
                )
test2.fit(mtpl_python,
          fused_type,
          input_variables,
          target_variable,
          offset_variable,
          n_k_fold = 5)
test2.lambda_curve


test2=GridSearch_Group('Poisson',
                # var_nb_min,
                # var_nb_max,
                # parcimony_step = 5,
                lbd_group=[0.1,1]
                )
test2.fit(mtpl_python,
          fused_type,
          input_variables,
          target_variable,
          offset_variable,
          n_k_fold = 5)
test2.lambda_curve




mtpl_python = pd.read_csv(r"C:\Users\AdrienCondamin\Documents\Modern Python Pricing\Reacfin trainings\car_frequency_sim1.csv")

bins = np.linspace(mtpl_python['power'].min(),mtpl_python['power'].max(), 5)
mtpl_python["power2"] = pd.cut(mtpl_python['power'],bins=bins,include_lowest=True)

bins = np.linspace(mtpl_python['age'].min(),mtpl_python['age'].max(), 5)
mtpl_python["age2"] = pd.cut(mtpl_python['age'],bins=bins,include_lowest=True)
mtpl_python["EXP"] = 1

target_variable = "N" 
offset_variable = "EXP" 
input_variables=["power2","age2"]
for c in input_variables:
    mtpl_python[c]=pd.Categorical(mtpl_python[c])  # Illustrates that Python requires indenting !!

print(mtpl_python.head())

fused_type={
            "power2":"fused",
            "age2":"fused",
            "brand":"g_fused"}



# TO DO : Test on bigger datasets
# TO DO : Parallel Running, memory footprint --> Ask
# TO DO : Documentation --> Tools (sphynx, )


## --> 2 objects : 1 group lasso / 1 fused lasso



#  OLD 
# test2=Generalised_Lasso('Poisson',
#                         # parcimony_step = 5,
#                         # smoothness_step = 5,
#                         # var_nb_min,
#                         # var_nb_max
#                         lbda=0.1,
#                         {"power2":"fused",
#                         "age2":"group",
#                         "var3":"g_fused"})

# likelhood - lmbda_1 * (pen_fused + pen_g_fused) - lmbda_2 * pen_group

# likelhood - lmbda_1 * (pen_fused + pen_g_fused + pen_group) --> FAIRE CELLE-CI


# TO DO
# Remove data, inputs...
# k fold as input





# test2=FusedLASSO(mtpl_python,penalty_type,'Poisson')
# test2.fit(input_variables,
#          target_variable,
#          offset_variable,
#          var_nb_min,
#          var_nb_max,
#          # parcimony_step = 5,
#          # smoothness_step = 5,
#          lbd_fused=[0.1,1],
#          lbd_group=[4,5])



# # test=FusedLASSO(mtpl_python,penalty_type,'Poisson')
# test=FusedLASSO(mtpl_python,penalty_type,'Poisson')
# test.fit(input_variables,target_variable,offset_variable,var_nb_min,var_nb_max,parcimony_step,smoothness_step)
# test.lambda_curve
# test.plot_curve()


# STILL PB OF HIGH LAMBDA ON CROSS VAL DATA --> OPTIMIZATION ALGO FAILS ---> HOW TO SOLVE IT?



# Changer l'objet FUSEDLASSO (generalised Lasso) --> Ou on met dans n'importe quelle penalité 
# + Un autre gridSearch --> 
# 
# Mettre les param de group lambda dans l'initialisation
