import pandas as pd
import statsmodels.api as sm


def stepwise_linear_glm(logitdf, depvar, indepvars, maxstep, sle = 0.05, sls = 0.05):
    #Output the same results as sas proc logistic stepwise
    #Add fillna 0 for test - should remove for application - dat should be cleaned before modeling!
    logitdf['intercept'] = 1
    current_var_list =['intercept']
    temp_var_list = ['intercept']
    stop_ind = 0
    step = 0
    for s in range(maxstep): 
    # while step < maxstep:
        if stop_ind == 0:
            min_p = 1
            p_mod_count = 0
            prev_len = len(current_var_list)
            if step != 0:
                pvals = pd.DataFrame(sm.GLM(logitdf[depvar], logitdf[current_var_list].fillna(0), family=sm.families.Binomial()).fit(disp=False).pvalues).sort_values(by = 0, ascending = False)
                if pvals.index[0] != 'intercept':
                    p_curr, var_remove = pvals.values[0], pvals.index[0]
                else:
                    p_curr, var_remove = pvals.values[1], pvals.index[1]
           
            if step != 0 and p_curr > sls:
                temp_var_list.remove(var_remove)
                current_var_list =  temp_var_list
                print("step:",s+1,"  variable list:",current_var_list,"  var removed:", var_remove) 
                step -= 2
            else:
                for var in list(set(indepvars) - set(current_var_list)):           
                    if p_mod_count == 0:
                        test_var_list = current_var_list.copy()
                        test_var_list.append(var)
                    else:
                        test_var_list[step+1] = var

                    print('Try: {0}'.format(var))
                    p = sm.GLM(logitdf[depvar], logitdf[test_var_list].fillna(0), family=sm.families.Binomial()).fit(disp=False).pvalues[var]
                    if min_p >= p and p <= sle:
                        if p_mod_count == 0:
                            temp_var_list.append(var)
                        else:
                            temp_var_list[step+1] = var
                        min_p = p
                        p_mod_count = p_mod_count + 1
                        var_added = var
                if p_mod_count != 0:
                    current_var_list =  temp_var_list  
                    print("step:",s+1,"  variable list:",current_var_list, "  var added:",var_added) 
            post_len = len(current_var_list)
            if prev_len == post_len:
                stop_ind = 1    
        step += 1            
              
    final_logistic = sm.GLM(logitdf[depvar], logitdf[current_var_list].fillna(0), family=sm.families.Binomial()).fit(disp=False)
    return {'model': final_logistic, 'variables': current_var_list} 


def stepwise_linear_ols(logitdf, depvar, indepvars, maxstep, sle=0.05, sls=0.05):
    #Output the same results as sas proc logistic stepwise
    #Add fillna 0 for test - should remove for application - dat should be cleaned before modeling!
    logitdf['intercept'] = 1
    current_var_list = ['intercept']
    temp_var_list = ['intercept']
    stop_ind = 0
    step = 0

    for s in range(maxstep):
        if stop_ind == 0:
            min_p = 1
            p_mod_count = 0
            prev_len = len(current_var_list)
            if step != 0:
                pvals = pd.DataFrame(sm.OLS(logitdf[depvar], logitdf[current_var_list].fillna(0)).fit(disp=False).pvalues).sort_values(by=0, ascending=False)
                if pvals.index[0] != 'intercept':
                    p_curr, var_remove = pvals.values[0], pvals.index[0]
                else:
                    p_curr, var_remove = pvals.values[1], pvals.index[1]
           
            if step!=0 and p_curr>sls:
                temp_var_list.remove(var_remove)
                current_var_list =  temp_var_list
                print("step:", s+1, "  variable list:", current_var_list, "  var removed:", var_remove) 
                step -= 2
            else:
                for var in list(set(indepvars) - set(current_var_list)):           
                    if p_mod_count == 0:
                        test_var_list = current_var_list.copy()
                        test_var_list.append(var)
                    else:
                        test_var_list[step+1] = var

                    print('Try: {0}'.format(var))
                    p = sm.OLS(logitdf[depvar], logitdf[test_var_list].fillna(0)).fit(disp=False).pvalues[var]
                    # import pdb; pdb.set_trace()

                    if min_p >= p and p <= sle:
                        if p_mod_count == 0:
                            temp_var_list.append(var)
                        else:
                            temp_var_list[step+1] = var
                        min_p = p
                        p_mod_count = p_mod_count + 1
                        var_added = var
                if p_mod_count != 0:
                    current_var_list =  temp_var_list  
                    print("step:", s+1, "  variable list:", current_var_list, "  var added:", var_added) 
            post_len = len(current_var_list)
            if prev_len == post_len:
                stop_ind = 1    
        step += 1            
              
    final_logistic = sm.OLS(logitdf[depvar], logitdf[current_var_list].fillna(0)).fit(disp=False)
    return {'model': final_logistic, 'variables': current_var_list} 


def stepwise_logistic(logitdf, depvar, indepvars, maxstep, sle = 0.05, sls = 0.05):
    #Output the same results as sas proc logistic stepwise
    #Add fillna 0 for test - should remove for application - dat should be cleaned before modeling!
    logitdf['intercept'] = 1
    current_var_list =['intercept']
    temp_var_list = ['intercept']
    stop_ind = 0
    step = 0
    for s in range(maxstep): 
    # while step < maxstep:
        if stop_ind == 0:
            min_p = 1
            p_mod_count = 0
            prev_len = len(current_var_list)
            if step != 0:
                pvals = pd.DataFrame(sm.Logit(logitdf[depvar], logitdf[current_var_list].fillna(0)).fit(disp=False).pvalues).sort_values(by = 0, ascending = False)
                if pvals.index[0] != 'intercept':
                    p_curr, var_remove = pvals.values[0], pvals.index[0]
                else:
                    p_curr, var_remove = pvals.values[1], pvals.index[1]
            
            if step != 0 and p_curr > sls:
                temp_var_list.remove(var_remove)
                current_var_list =  temp_var_list
                print("step:",s+1,"  variable list:",current_var_list,"  var removed:", var_remove ) 
                step -= 2
            else:
                for var in list(set(indepvars) - set(current_var_list)):           
                    if p_mod_count == 0:
                        test_var_list = current_var_list.copy()
                        test_var_list.append(var)
                    else:
                        test_var_list[step+1] = var

                    print('Try: {0}'.format(var))
                    p = sm.Logit(logitdf[depvar], logitdf[test_var_list].fillna(0)).fit(disp=False).pvalues[var]
                    if min_p >= p and p <= sle:
                        if p_mod_count == 0:
                            temp_var_list.append(var)
                        else:
                            temp_var_list[step+1] = var
                        min_p = p
                        p_mod_count = p_mod_count + 1
                        var_added = var
                if p_mod_count != 0:
                    current_var_list =  temp_var_list  
                    print("step:",s+1,"  variable list:",current_var_list, "  var added:",var_added) 
            post_len = len(current_var_list)
            if prev_len == post_len:
                stop_ind = 1    
        step += 1            
      
    final_logistic = sm.Logit(logitdf[depvar], logitdf[current_var_list].fillna(0)).fit(disp=False)
    return {'model': final_logistic, 'variables': current_var_list} 


