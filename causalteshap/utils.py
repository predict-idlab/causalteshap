__author__ = "Jarne Verhaeghe"

import pandas as pd
import numpy as np

from scipy import stats
from statsmodels.stats.power import TTestPower

def p_values_arg_coef(coefficients, arg):
    return stats.percentileofscore(coefficients, arg)

import scipy.stats as st

def cumsum_kstest(A,B):
    A = np.sort(A)
    B = np.sort(B)
    na = A.shape[0]
    nb = B.shape[0]

    data_all = np.sort(np.concatenate([A,B]))

    cdf1 = np.searchsorted(A, data_all, side='right') / na
    cdf2 = np.searchsorted(B, data_all, side='right') / nb
    
    cddiffs = cdf1 - cdf2

    return data_all, cdf1, cdf2, cddiffs

def causalteshap_analysis(
    T1_matrix_list: pd.DataFrame,
    T0_matrix_list: pd.DataFrame,
    significance_factor: float,
    verbose:bool
):
    pred_matrix_list = np.abs(T1_matrix_list) - np.abs(T0_matrix_list)

    predictive_vars = np.array([])
    candidate_vars = np.array([])

    len_features = np.shape(T1_matrix_list)[1]

    analysis_df = pd.DataFrame(columns=["ks-statistic","ttest","predictive","candidate"])

    for i in range(len_features-2):

        _, cdf1, cdf2, _ = cumsum_kstest(
            np.abs(pred_matrix_list[:-1,i])-np.abs(pred_matrix_list[:-1,-1]),
            np.abs(pred_matrix_list[1:,-1])-np.abs(pred_matrix_list[:-1,-1]),
        )

        calculated_D_ks = np.max(np.abs((cdf1 - cdf2)[cdf1 - cdf2 >= 0 ]))
        n_samples = len(np.abs(pred_matrix_list[:-1,i]))
        comp_D_ks = np.sqrt(n_samples)*calculated_D_ks
        condition = np.sqrt(-0.5*np.log(1-significance_factor))

        p_value_ttest = np.round(
            st.ttest_ind(
                np.abs(T1_matrix_list[:,i]),
                np.abs(T0_matrix_list[:,i]),
                alternative='two-sided',
                equal_var = False
            )[1],3
            )

        if i == len_features-1:
            if comp_D_ks < condition:

                predictive_vars = np.append(predictive_vars,i)
                concat_df_results_series = pd.DataFrame(data=[[comp_D_ks,p_value_ttest,1,1]],columns=["ks-statistic","ttest","predictive","candidate"])
            else:
                concat_df_results_series = pd.DataFrame(data=[[comp_D_ks,p_value_ttest,0,0]],columns=["ks-statistic","ttest","predictive","candidate"])
                
        else:
            if (
                comp_D_ks < condition
                and p_value_ttest <= significance_factor
                ):
                predictive_vars = np.append(predictive_vars,i)

            elif p_value_ttest <= significance_factor:

                candidate_vars = np.append(candidate_vars,i)
                concat_df_results_series = pd.DataFrame(data=[[comp_D_ks,p_value_ttest,0,1]],columns=["ks-statistic","ttest","predictive","candidate"])
            
            else:
                concat_df_results_series = pd.DataFrame(data=[[comp_D_ks,p_value_ttest,0,0]],columns=["ks-statistic","ttest","predictive","candidate"])

        
        analysis_df = pd.concat([analysis_df,concat_df_results_series])

        if verbose:
            print("KS greater statistic (< condition): "+str(np.round(comp_D_ks,3))+ " > "+str(np.round(condition,3))+" ?")
            print("p-value (current ttest): "+str(p_value_ttest))
            print(10*"-")

    return analysis_df
