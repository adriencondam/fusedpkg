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

    
        for j, (train_idx, test_idx) in enumerate(self.cv_splits):
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
        
        # kf = KFold(n_splits=self.n_k_fold, shuffle=True)
        
        
        error_train_total = 0
        error_test_total = 0
        for j, (train_idx, test_idx) in enumerate(self.cv_splits):
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

    