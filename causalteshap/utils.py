import scipy.stats as st
__author__ = "Jarne Verhaeghe"

import pandas as pd
import numpy as np

from scipy import stats
from statsmodels.stats.power import TTestPower


def p_values_arg_coef(coefficients, arg):
    return stats.percentileofscore(coefficients, arg)


def cumsum_kstest(A, B):
    A = np.sort(A)
    B = np.sort(B)
    na = A.shape[0]
    nb = B.shape[0]

    data_all = np.sort(np.concatenate([A, B]))

    cdf1 = np.searchsorted(A, data_all, side='right') / na
    cdf2 = np.searchsorted(B, data_all, side='right') / nb

    cddiffs = cdf1 - cdf2

    return data_all, cdf1, cdf2, cddiffs


def causalteshap_analysis(
    T1_matrix_list: pd.DataFrame,
    T0_matrix_list: pd.DataFrame,
    significance_factor: float,
    verbose: bool
):
    pred_matrix_list = np.abs(T1_matrix_list) - np.abs(T0_matrix_list)

    predictive_vars = np.array([])
    candidate_vars = np.array([])

    len_features = np.shape(T1_matrix_list)[1]

    analysis_df_columns = ["ks-statistic", "ttest",
                           "T1-T0", "predictive", "candidate"]

    analysis_df = pd.DataFrame(columns=analysis_df_columns)

    for i in range(len_features-2):

        data_all, cdf1, cdf2, _ = cumsum_kstest(
            np.abs(pred_matrix_list[:-1, i])-np.abs(pred_matrix_list[:-1, -1]),
            np.abs(pred_matrix_list[1:, -1])-np.abs(pred_matrix_list[:-1, -1]),
        )

        #######################################################################################################################################
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(data_all, cdf1,label="tested")
        # plt.plot(data_all, cdf2,label="random")
        # plt.legend()
        # plt.grid()
        #######################################################################################################################################

        calculated_D_ks = np.max(np.abs((cdf1 - cdf2)[cdf1 - cdf2 >= 0]))
        n_samples = len(np.abs(pred_matrix_list[:-1, i]))
        comp_D_ks = np.sqrt(n_samples)*calculated_D_ks
        condition = np.sqrt(-0.5*np.log(1-significance_factor))

        p_value_ttest = np.round(
            st.ttest_ind(
                np.abs(T1_matrix_list[:, i]),
                np.abs(T0_matrix_list[:, i]),
                alternative='two-sided',
                equal_var=False
            )[1], 3
        )

        #######################################################################################################################################
        # import matplotlib.pyplot as plt

        # plot_bins = 100#100

        # plot_1_max = np.max(np.abs(pred_matrix_list[:-1,i])-np.abs(pred_matrix_list[:-1,-1]))
        # plot_1_min = np.min(np.abs(pred_matrix_list[1:,-1])-np.abs(pred_matrix_list[:-1,-1]))

        # fig, axs = plt.subplots(1,2,figsize=(20,6))
        # axs[0].hist(np.abs(pred_matrix_list[:-1,i])-np.abs(pred_matrix_list[:-1,-1]),label="tested",bins=np.arange(plot_1_min,plot_1_max,(plot_1_max-plot_1_min)/plot_bins),density=True)
        # axs[0].hist(np.abs(pred_matrix_list[1:,-1])-np.abs(pred_matrix_list[:-1,-1]),label="random_subtr",alpha=0.5,bins=np.arange(plot_1_min,plot_1_max,(plot_1_max-plot_1_min)/plot_bins),density=True)
        # axs[0].legend()
        # axs[0].set_title("Random subtr vs tested subtr | condition = "+str(comp_D_ks))

        # # array_feature = np.append(np.append(array_tested[0],array_random[0]),[1 if feature_set_pred[i] in predictive_vars else 0])
        # # hist_dataset_df = pd.concat([hist_dataset_df,pd.DataFrame([array_feature],columns=colum_features)])

        # plot_2_max = np.max(np.abs(T1_matrix_list[:,i]))
        # plot_2_min = np.min(np.abs(T0_matrix_list[:,i]))

        # axs[1].hist(np.abs(T1_matrix_list[:,i]),label="T1",bins=np.arange(plot_2_min,plot_2_max,(plot_2_max-plot_2_min)/plot_bins))
        # axs[1].hist(np.abs(T0_matrix_list[:,i]),label="T0",alpha=0.5,bins=np.arange(plot_2_min,plot_2_max,(plot_2_max-plot_2_min)/plot_bins))
        # axs[1].legend()
        # axs[1].set_title("T1 vs T0 (abs) | p_value = "+str(p_value_ttest))
        #######################################################################################################################################

        #Calculate the percentage mean difference between the Shapley values of the T1 case and the T0 case
        mean_T1_minus_T0 = np.mean(
            (
                np.abs(T1_matrix_list[:, i]) - np.abs(T0_matrix_list[:, i])
            )
            /
            (
                np.max(
                    np.append(np.abs(T1_matrix_list[:, i]), np.abs(T0_matrix_list[:, i]))
                )
                -
                np.min(
                    np.append(np.abs(T1_matrix_list[:, i]), np.abs(T0_matrix_list[:, i]))
                )
            )
        )

        first_part_data_array = np.array([comp_D_ks, p_value_ttest, mean_T1_minus_T0])

        if i == len_features-1:
            if comp_D_ks < condition:

                predictive_vars = np.append(predictive_vars, i)
                concat_df_results_series = pd.DataFrame(
                    data=[np.append(first_part_data_array,np.array([1, 1]))], columns=analysis_df_columns)
            else:
                concat_df_results_series = pd.DataFrame(
                    data=[np.append(first_part_data_array,np.array([0, 0]))], columns=analysis_df_columns)

        else:
            if (
                comp_D_ks < condition
                and p_value_ttest <= significance_factor
            ):
                predictive_vars = np.append(predictive_vars, i)
                concat_df_results_series = pd.DataFrame(
                    data=[np.append(first_part_data_array,np.array([1, 1]))], columns=analysis_df_columns)

            elif p_value_ttest <= significance_factor:

                candidate_vars = np.append(candidate_vars, i)
                concat_df_results_series = pd.DataFrame(
                    data=[np.append(first_part_data_array,np.array([0, 1]))], columns=analysis_df_columns)

            else:
                concat_df_results_series = pd.DataFrame(
                    data=[np.append(first_part_data_array,np.array([0, 0]))], columns=analysis_df_columns)

        analysis_df = pd.concat([analysis_df, concat_df_results_series])

        if verbose:
            print("KS greater statistic (< condition): " +
                  str(np.round(comp_D_ks, 3)) + " > "+str(np.round(condition, 3))+" ?")
            print("p-value (current ttest): "+str(p_value_ttest))
            print(10*"-")

    return analysis_df.reset_index(drop=True)
