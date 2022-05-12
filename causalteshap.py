__author__ = "Jarne Verhaeghe"

import numpy as np
from numpy.random import RandomState
import shap
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats
from statsmodels.stats.power import TTestPower
from sklearn.model_selection import train_test_split
from scipy.special import logit
from sklearn.neighbors import NearestNeighbors

from catboost import Pool, CatBoostClassifier
from catboost.utils import get_roc_curve
from sklearn.metrics import auc


def p_values_arg_coef(coefficients, arg):
    return stats.percentileofscore(coefficients, arg)


def powerSHAP_statistical_analysis(
        shaps_df: pd.DataFrame,
        power_alpha: float,
        power_req_iterations: float,
        include_all: bool = False,
        two_sided: bool = False,
):
    p_values = []
    effect_size = []
    power_list = []
    required_iterations = []
    n_samples = len(shaps_df["random_uniform_feature"].values)
    mean_random_uniform = shaps_df["random_uniform_feature"].mean()
    for i in range(len(shaps_df.columns)):
        p_value = (
            p_values_arg_coef(
                np.array(shaps_df.values[:, i]), mean_random_uniform)
            / 100
        )

        if two_sided:
            if p_value > 0.5:
                p_value = 1-p_value

            p_vale = p_value*2  # to make it back to 0,1, also because it is two-sided

        p_values.append(p_value)

        if include_all or p_value < power_alpha:
            pooled_standard_deviation = np.sqrt(
                (
                    (shaps_df.std().values[i] ** 2)
                    + (shaps_df["random_uniform_feature"].values.std() ** 2)
                )
                / (2)
            )
            effect_size.append(
                (mean_random_uniform - shaps_df.mean().values[i])
                / pooled_standard_deviation
            )
            power_list.append(
                TTestPower().power(
                    effect_size=effect_size[-1],
                    nobs=n_samples,
                    alpha=power_alpha,
                    df=None,
                    alternative="two-sided" if two_sided else "smaller",
                )
            )
            if shaps_df.columns[i] == "random_uniform_feature":
                required_iterations.append(0)
            else:
                required_iterations.append(
                    TTestPower().solve_power(
                        effect_size=effect_size[-1],
                        nobs=None,
                        alpha=power_alpha,
                        power=power_req_iterations,
                        alternative="two-sided" if two_sided else "smaller",
                    )
                )

        else:
            required_iterations.append(0)
            effect_size.append(0)
            power_list.append(0)

    processed_shaps_df = pd.DataFrame(
        data=np.hstack(
            [
                np.reshape(shaps_df.mean().values, (-1, 1)),
                np.reshape(np.array(p_values), (len(p_values), 1)),
                np.reshape(np.array(effect_size), (len(effect_size), 1)),
                np.reshape(np.array(power_list), (len(power_list), 1)),
                np.reshape(
                    np.array(required_iterations), (len(
                        required_iterations), 1)
                ),
            ]
        ),
        columns=[
            "impact",
            "p_value",
            "effect_size",
            "power_" + str(power_alpha) + "_alpha",
            str(power_req_iterations) + "_power_its_req",
        ],
        index=shaps_df.mean().index,
    )
    processed_shaps_df = processed_shaps_df.reindex(
        processed_shaps_df.impact.abs().sort_values(ascending=False).index
    )

    return processed_shaps_df


class CausalTeShap():  # SelectorMixin, BaseEstimator):
    def __init__(
            self,
            model=None,
            iterations: int = 10,
            power_alpha: float = 0.01,
            power_req_iterations: float = 0.99,
            S_learner: bool = True,
            val_size: float = 0.2,
            verbose: bool = False,
            classification=True,
            **fit_kwargs,
    ):
        self.model = model
        self.iterations = iterations
        self.power_alpha = power_alpha
        self.S_learner = S_learner
        self.power_req_iterations, = power_req_iterations,
        self.val_size = val_size
        self.verbose = verbose
        self.fit_kwargs = fit_kwargs
        self._classification = classification

    def _print(
            self, *values
    ):
        """Helper method for printing if `verbose` is set to True."""
        if self.verbose:
            print(*values)

    def propensity_matching(
            self,
            X,
            y,
            T,
            index_ar=None,
            stratify=None,
            RS=42,
            one_to_one_matching=False,
            multiple_exact=1,
    ):

        X = X.copy(deep=True)

        if index_ar is None:
            index_ar = X.reset_index(drop=True).index.values

        train_idx, val_idx = train_test_split(
            index_ar,
            test_size=0.9,
            random_state=RS,
            stratify=stratify,
        )

        X_train = X[np.isin(index_ar, train_idx)].copy(deep=True)
        y_train = y[np.isin(index_ar, train_idx)]
        T_train = T[np.isin(index_ar, train_idx)]
        X_val = X[np.isin(index_ar, val_idx)].copy(deep=True)
        y_val = y[np.isin(index_ar, val_idx)]
        T_val = T[np.isin(index_ar, val_idx)]

        assert len(np.unique(T) ==
                   2), "only binary treatments are supported right now."

        propensity_class_balance = [
            T_train.sum()/len(T_train), 1-T_train.sum()/len(T_train)]
        propensity_model = CatBoostClassifier(
            verbose=0, n_estimators=250, class_weights=propensity_class_balance)
        propensity_model.fit(Pool(X_train, T_train))

        (fpr, tpr, thresholds) = get_roc_curve(
            propensity_model, Pool(data=X_val, label=T_val), plot=False)
        prev_propensity_auc = auc(fpr, tpr)

        X = X_val.copy(deep=True)
        y = y_val
        T = T_val
        index_ar = index_ar[np.isin(index_ar, val_idx)]

        X_matching = X.copy(deep=True).reset_index(drop=True)
        X_matching["target"] = y
        X_matching["index_col"] = index_ar
        X_matching["T"] = T
        X_matching["propensity_score_logit"] = 0

        T_train_proba = propensity_model.predict_proba(X)
        T_train_logit = np.array([logit(xi) for xi in T_train_proba[:, 1]])
        X_matching.loc[:, "propensity_score_logit"] = T_train_logit

        T0_X_matching = X_matching[T == 0].copy(deep=True)
        T0_X_matching = T0_X_matching.reset_index(drop=True)
        T1_X_matching = X_matching[T == 1].copy(deep=True)
        T1_X_matching = T1_X_matching.reset_index(drop=True)

        # if len(T0_X_matching)<len(T1_X_matching):
        Neighbours = len(
            T0_X_matching) if one_to_one_matching else multiple_exact
        # else:
        #Neighbours = len(T1_X_matching) if one_to_one_matching else multiple_exact

        # , p = 2, radius=np.std(y_train_logit) * 0.25)
        knn = NearestNeighbors(n_neighbors=Neighbours)
        knn.fit(T0_X_matching[['propensity_score_logit']].to_numpy())

        matched_element_arr = np.zeros((len(T1_X_matching), multiple_exact))
        j = 0
        if one_to_one_matching:
            for row in T1_X_matching.iterrows():
                distance, indexes = knn.kneighbors(np.reshape(
                    [row[1].propensity_score_logit], (-1, 1)), n_neighbors=Neighbours)

                for idx in indexes[0, :]:
                    if idx not in matched_element_arr[:, 0]:
                        matched_element_arr[j, 0] = idx
                        break
                j = j+1

        else:
            for row in T1_X_matching.iterrows():
                distance, indexes = knn.kneighbors(np.reshape(
                    [row[1].propensity_score_logit], (-1, 1)), n_neighbors=Neighbours)
                for i in range(multiple_exact):
                    matched_element_arr[j, i] = indexes[0, i]

                j = j+1

        if one_to_one_matching:
            X_all_matched = pd.concat(
                [T1_X_matching, T0_X_matching.iloc[matched_element_arr[:, 0]]])
        else:
            X_all_matched = pd.concat(
                [T1_X_matching, T0_X_matching.iloc[matched_element_arr[:, 0]]])
            for i in range(multiple_exact-1):
                X_all_matched = pd.concat(
                    [X_all_matched, T0_X_matching.iloc[matched_element_arr[:, i]]])

        X_all_matched = X_all_matched.drop_duplicates("index_col")

        (fpr, tpr, thresholds) = get_roc_curve(propensity_model, Pool(data=X_all_matched.drop(columns=[
            "index_col", "target", "T", "propensity_score_logit"]), label=X_all_matched["T"]), plot=False)
        after_propensity_auc = auc(fpr, tpr)

        return X_all_matched["index_col"].values, prev_propensity_auc, after_propensity_auc

    def fit(
            self,
            X,
            y,
            T,
            id_split_ar=None,
            id_ar=None,
            balance_T_val=False,
            propensity_matching=False,
            one_to_one_matching=False,
            multiple_exact=1,
            stratify=None,
            propensity_exclude_columns=[],
    ):

        CATE_shap_values_array = []
        T0_shap_values_array = []
        T1_shap_values_array = []
        train_CATE_array = []
        val_CATE_array = []

        #train_ATE_array = []
        #val_ATE_array = []

        self._propensity_matching = propensity_matching
        if propensity_matching:
            before_matching_auc_arr = []
            after_matching_auc_arr = []
            T_class_distribution = []

        X = X.copy(deep=True)
        X["random_uniform_feature"] = 0
        for i in tqdm(range(self.iterations), ascii=True):

            RS_loop = i
            npRandomState = RandomState(RS_loop)

            random_uniform_feature_array = npRandomState.uniform(
                -1, 1, (len(X)))
            X.loc[:, "random_uniform_feature"] = random_uniform_feature_array

            if id_split_ar is not None:
                train_idx, val_idx = train_test_split(
                    id_split_ar,
                    test_size=self.val_size,
                    random_state=RS_loop,
                    stratify=stratify,
                )
                if id_ar is None:
                    id_ar = id_split_ar

                X_train = X[np.isin(id_ar, train_idx)].copy(deep=True)
                y_train = y[np.isin(id_ar, train_idx)]
                T_train = T[np.isin(id_ar, train_idx)]
                X_val = X[~np.isin(id_ar, train_idx)].copy(deep=True)
                y_val = y[~np.isin(id_ar, train_idx)]
                T_val = T[~np.isin(id_ar, train_idx)]

                train_idx = id_ar[np.isin(id_ar, train_idx)]

            else:
                train_idx, val_idx = train_test_split(
                    np.arange(len(X)),
                    test_size=self.val_size,
                    random_state=RS_loop,
                    stratify=stratify,
                )
                X_train = X.iloc[np.sort(train_idx)].copy(deep=True)
                y_train = y[np.sort(train_idx)]
                T_train = T[np.sort(train_idx)]
                X_val = X.iloc[np.sort(val_idx)].copy(deep=True)
                y_val = y[np.sort(val_idx)]
                T_val = T[np.sort(val_idx)]

            if propensity_matching:

                propensity_idx, before_matching_auc, after_matching_auc = self.propensity_matching(
                    X_train.drop(columns=propensity_exclude_columns +
                                 ["random_uniform_feature"]),
                    y_train,
                    T_train,
                    stratify=stratify,
                    index_ar=train_idx,
                    one_to_one_matching=one_to_one_matching,
                    multiple_exact=multiple_exact,
                )
                X_train = X_train[np.isin(train_idx, propensity_idx)]
                y_train = y_train[np.isin(train_idx, propensity_idx)]
                T_train = T_train[np.isin(train_idx, propensity_idx)]

                before_matching_auc_arr += [before_matching_auc]
                after_matching_auc_arr += [after_matching_auc]
                T_class_distribution += [[y_train[T_train == 0].sum()/len(y_train[T_train == 0]), len(
                    T_train[T_train == 0]), y_train[T_train == 1].sum()/len(y_train[T_train == 1]), (len(T_train[T_train == 1]))]]

            X_train_T0 = X_train[T_train == 0].copy(deep=True)
            y_train_T0 = y_train[T_train == 0]

            X_train_T1 = X_train[T_train == 1].copy(deep=True)
            y_train_T1 = y_train[T_train == 1]

            X_val_T0 = X_val[T_val == 0].copy(deep=True)
            y_val_T0 = y_val[T_val == 0]

            X_val_T1 = X_val[T_val == 1].copy(deep=True)
            y_val_T1 = y_val[T_val == 1]

            # balances the labels in the validation set for each treatment group by undersampling
            if balance_T_val:
                assert len(np.unique(
                    y)) == 2, 'balancing on label is only supported for binary classification'
                # balance T=1
                if np.mean(y_val_T1) < 0.5:
                    T1_label_undersample = 0
                else:
                    T1_label_undersample = 1

                T1_undersample_idx = np.arange(len(y_val_T1))[
                    y_val_T1 == T1_label_undersample]
                chosen_T1_undersample_idx = np.random.choice(T1_undersample_idx, size=len(
                    y_val_T1[y_val_T1 == 1-T1_label_undersample]), replace=False)

                X_val_T1 = pd.concat(
                    [X_val_T1[y_val_T1 == 1-T1_label_undersample], X_val_T1.iloc[chosen_T1_undersample_idx]])
                y_val_T1 = np.append(
                    y_val_T1[y_val_T1 == 1-T1_label_undersample], y_val_T1[chosen_T1_undersample_idx])

                if np.mean(y_val_T0) < 0.5:
                    T0_label_undersample = 0
                else:
                    T0_label_undersample = 1

                T0_undersample_idx = np.arange(len(y_val_T0))[
                    y_val_T0 == T0_label_undersample]
                chosen_T0_undersample_idx = np.random.choice(T0_undersample_idx, size=len(
                    y_val_T0[y_val_T0 == 1-T0_label_undersample]), replace=False)

                X_val_T0 = pd.concat(
                    [X_val_T0[y_val_T0 == 1-T0_label_undersample], X_val_T0.iloc[chosen_T0_undersample_idx]])
                y_val_T0 = np.append(
                    y_val_T0[y_val_T0 == 1-T0_label_undersample], y_val_T0[chosen_T0_undersample_idx])

            y_val = np.append(y_val_T0, y_val_T1)
            X_val = pd.concat([X_val_T0, X_val_T1])

            y_train = np.append(y_train_T0, y_train_T1)

            if self.S_learner:
                X_train_T0["T"] = 0
                X_train_T1["T"] = 1

                X_train = pd.concat([X_train_T0, X_train_T1])

                X_train_S_T0 = X_train.copy(deep=True)
                X_train_S_T0["T"] = 0
                X_train_S_T1 = X_train.copy(deep=True)
                X_train_S_T1["T"] = 1

                X_val_S_T0 = X_val.copy(deep=True)
                X_val_S_T0["T"] = 0
                X_val_S_T1 = X_val.copy(deep=True)
                X_val_S_T1["T"] = 1

                X_val["T"] = 0.5

                if self._classification:
                    class_balance = [y_train.sum()/len(y_train),
                                     1-y_train.sum()/len(y_train)]
                    causal_model = self.model(
                        verbose=False, iterations=250, class_weights=class_balance)
                else:
                    causal_model = self.model(verbose=False, iterations=250)
                causal_model.fit(X_train, y_train)

                if self._classification:
                    train_CATE = causal_model.predict_proba(
                        X_train_S_T1)[:, 1]-causal_model.predict_proba(X_train_S_T0)[:, 1]
                    val_CATE = causal_model.predict_proba(
                        X_val_S_T1)[:, 1]-causal_model.predict_proba(X_val_S_T0)[:, 1]
                else:
                    train_CATE = causal_model.predict(
                        X_train_S_T1)-causal_model.predict(X_train_S_T0)
                    val_CATE = causal_model.predict(
                        X_val_S_T1)-causal_model.predict(X_val_S_T0)

                causal_explainer = shap.TreeExplainer(causal_model)
                T1_shap_values = causal_explainer.shap_values(X_val_S_T1)
                T0_shap_values = causal_explainer.shap_values(X_val_S_T0)

            else:
                X_train = pd.concat([X_train_T0, X_train_T1])

                if self._classification:
                    T0_class_balance = [
                        y_train_T0.sum()/len(y_train_T0), 1-y_train_T0.sum()/len(y_train_T0)]
                    T0_causal_model = self.model(
                        verbose=False, iterations=250, class_weights=T0_class_balance)
                else:
                    T0_causal_model = self.model(verbose=False, iterations=250)
                T0_causal_model.fit(X_train_T0, y_train_T0)

                if self._classification:
                    T1_class_balance = [
                        y_train_T1.sum()/len(y_train_T1), 1-y_train_T1.sum()/len(y_train_T1)]
                    T1_causal_model = self.model(
                        verbose=False, iterations=250, class_weights=T1_class_balance)
                else:
                    T1_causal_model = self.model(verbose=False, iterations=250)
                T1_causal_model.fit(X_train_T1, y_train_T1)

                if self._classification:
                    train_CATE = T1_causal_model.predict_proba(
                        X_train)[:, 1]-T0_causal_model.predict_proba(X_train)[:, 1]
                    val_CATE = T1_causal_model.predict_proba(
                        X_val)[:, 1]-T0_causal_model.predict_proba(X_val)[:, 1]
                else:
                    train_CATE = T1_causal_model.predict(
                        X_train)-T0_causal_model.predict(X_train)
                    val_CATE = T1_causal_model.predict(
                        X_val)-T0_causal_model.predict(X_val)

                T0_causal_explainer = shap.TreeExplainer(T0_causal_model)
                T1_causal_explainer = shap.TreeExplainer(T1_causal_model)

                T1_shap_values = T0_causal_explainer.shap_values(X_val)
                T0_shap_values = T1_causal_explainer.shap_values(X_val)

            CATE_shap_values = np.mean(
                T1_shap_values-T0_shap_values+np.abs(T1_shap_values)-np.abs(T0_shap_values), axis=0)
            #CATE_shap_values = np.mean(np.abs(T1_shap_values-T0_shap_values+np.abs(T1_shap_values)-np.abs(T0_shap_values)),axis=0)
            # CATE_shap_values = np.mean(np.abs(T1_shap_values-T0_shap_values),axis=0)#+np.abs(T1_shap_values)-np.abs(T0_shap_values),axis=0)
            # CATE_shap_values = np.mean(np.abs(T1_shap_values)-np.abs(T0_shap_values),axis=0)#+np.abs(T1_shap_values)-np.abs(T0_shap_values),axis=0)

            T0_shap_values = np.abs(T0_shap_values)
            T1_shap_values = np.abs(T1_shap_values)

            CATE_shap_values_array += [CATE_shap_values]
            T0_shap_values_array += [np.mean(T0_shap_values, axis=0)]
            T1_shap_values_array += [np.mean(T1_shap_values, axis=0)]
            train_CATE_array += [list(train_CATE)]
            val_CATE_array += [list(val_CATE)]

        CATE_shap_values_array = np.array(CATE_shap_values_array)
        T0_shap_values_array = np.array(T0_shap_values_array)
        T1_shap_values_array = np.array(T1_shap_values_array)

        # it does not matter which column array I pick, they all have the same
        CATE_shap_df = pd.DataFrame(
            data=CATE_shap_values_array, columns=X_val.columns.values)
        T0_shap_df = pd.DataFrame(
            data=T0_shap_values_array, columns=X_val.columns.values)
        T1_shap_df = pd.DataFrame(
            data=T1_shap_values_array, columns=X_val.columns.values)

        CATE_stats_shap_df = powerSHAP_statistical_analysis(
            CATE_shap_df,
            self.power_alpha,
            self.power_req_iterations,
            two_sided=True,
        )

        T0_stats_shap_df = powerSHAP_statistical_analysis(
            T0_shap_df,
            self.power_alpha,
            self.power_req_iterations,
        )

        T1_stats_shap_df = powerSHAP_statistical_analysis(
            T1_shap_df,
            self.power_alpha,
            self.power_req_iterations,
        )

        self._CATE_stats_shap_df = CATE_stats_shap_df
        self._T0_stats_shap_df = T0_stats_shap_df
        self._T1_stats_shap_df = T1_stats_shap_df
        self._val_CATE_array = val_CATE_array
        self._train_CATE_array = train_CATE_array

        if propensity_matching:
            self._T_class_distribution = np.array(T_class_distribution)
            self._before_matching_auc_arr = before_matching_auc_arr
            self._after_matching_auc_arr = after_matching_auc_arr

        self._print("Done")

    def show_all_causality_information(
            self, include_all=False
    ):
        if self._propensity_matching:
            print(
                "Propensity matching was included in fitting - Propensity information: ")
            print()
            print("Before matching auc \t | mean (std) = "+str(np.round(np.mean(self._before_matching_auc_arr), 4)
                                                               )+" ("+str(np.round(np.std(self._before_matching_auc_arr), 4))+")")
            print("After matching auc \t | mean (std) = "+str(np.round(np.mean(self._after_matching_auc_arr), 4)
                                                              )+" ("+str(np.round(np.std(self._after_matching_auc_arr), 4))+")")

            if self._classification:
                print("T0 amount of class 1 samples \t |  "+str(int(np.mean(self._T_class_distribution[:, 0]*self._T_class_distribution[:, 1])))+" samples = "+str(
                    np.round(np.mean(self._T_class_distribution[:, 0])*100, 2))+"% of all matched " + str(int(np.mean(self._T_class_distribution[:, 1]))) + " samples")
                print("T1 amount of class 1 samples \t | "+str(int(np.mean(self._T_class_distribution[:, 2]*self._T_class_distribution[:, 3])))+" samples = "+str(
                    np.round(np.mean(self._T_class_distribution[:, 2])*100, 2))+"% of all matched " + str(int(np.mean(self._T_class_distribution[:, 3]))) + " samples")
            else:
                print("T0 all matched " +
                      str(int(np.mean(self._T_class_distribution[:, 1]))) + " samples")
                print("T1 all matched " +
                      str(int(np.mean(self._T_class_distribution[:, 3]))) + " samples")

        print("")
        print("Causality information:")
        self.analyze_causality(include_all)

    def analyze_causality(
            self, include_all=False
    ):
        val_means = []
        for CATE_list in self._val_CATE_array:
            val_means += [np.mean(CATE_list)]

        train_means = []
        for CATE_list in self._train_CATE_array:
            train_means += [np.mean(CATE_list)]

        train_p_value = p_values_arg_coef(np.array(train_means), 0)/100
        if train_p_value > 0.5:
            train_p_value = 1-train_p_value

        val_p_value = p_values_arg_coef(np.array(val_means), 0)/100
        if val_p_value > 0.5:
            val_p_value = 1-val_p_value

        print("TRAIN: \t CATE = " + str(np.mean(train_means)) +
              " \t| p-value = "+str(train_p_value))
        print("VAL: \t CATE = " + str(np.mean(val_means)) +
              " \t| p-value = "+str(val_p_value))
        print(60*"=")

        if include_all:
            print("Include all mode")
            print(60*"=")
            print("CATE shap stats")
            print(70*"-")
            print(self._CATE_stats_shap_df.to_markdown())
            print(70*"=")
            print("T0 shap stats")
            print(70*"-")
            print(self._T0_stats_shap_df.to_markdown())
            print(70*"=")
            print("T1 shap stats")
            print(70*"-")
            print(self._T1_stats_shap_df.to_markdown())

        else:
            print("CATE shap stats")
            print(70*"-")
            print(self._CATE_stats_shap_df[self._CATE_stats_shap_df.p_value < self.power_alpha][[
                  "impact", "p_value"]].to_markdown())
            print(70*"=")
            print("T0 shap stats")
            print(70*"-")
            print(self._T0_stats_shap_df[self._T0_stats_shap_df.p_value < self.power_alpha][[
                  "impact", "p_value"]].to_markdown())
            print(70*"=")
            print("T1 shap stats")
            print(70*"-")
            print(self._T1_stats_shap_df[self._T1_stats_shap_df.p_value < self.power_alpha][[
                  "impact", "p_value"]].to_markdown())
            print(70*"=")

        print("")

        prognostic_variables = np.intersect1d(self._T0_stats_shap_df[self._T0_stats_shap_df.p_value < self.power_alpha].index.values,
                                              self._T1_stats_shap_df[self._T1_stats_shap_df.p_value < self.power_alpha].index.values)

        predictive_variables = self._CATE_stats_shap_df[self._CATE_stats_shap_df.p_value <
                                                        self.power_alpha].index.values

        print("All predictive variables = \n \t \t" +
              str(list(predictive_variables)))
        if self.S_learner:
            print("All prognostic variables = \n \t \t"+str(list(prognostic_variables))
                  + "\n (Beware, S-learner is True, look at the impact to really discern between prognostic and predictive or "
                    + "only prognostic for variables that are both in the predictive and prognostic set)")
        else:
            print("All prognostic variables = "+str(prognostic_variables))

    def export_results(
            self,
            data_path
    ):

        mean_before_auc = 0
        std_before_auc = 0
        mean_after_auc = 0
        std_after_auc = 0
        mean_T0_class_1_samples = 0
        mean_T0_class_0_samples = 0
        mean_T1_class_1_samples = 0
        mean_T1_class_0_samples = 0
        mean_T0_all_matched = 0
        mean_T0_all_matched = 0

        if self._propensity_matching:
            mean_before_auc = np.mean(self._before_matching_auc_arr)
            std_before_auc = np.std(self._before_matching_auc_arr)

            mean_after_auc = np.mean(self._after_matching_auc_arr)
            std_after_auc = np.std(self._after_matching_auc_arr)

            if self._classification:
                mean_T0_class_1_samples = int(
                    np.mean(self._T_class_distribution[:, 0]*self._T_class_distribution[:, 1]))
                mean_T0_class_0_samples = int(
                    np.mean((1-self._T_class_distribution[:, 0])*self._T_class_distribution[:, 1]))
                mean_T1_class_1_samples = int(
                    np.mean(self._T_class_distribution[:, 2]*self._T_class_distribution[:, 3]))
                mean_T1_class_0_samples = int(
                    np.mean((1-self._T_class_distribution[:, 0])*self._T_class_distribution[:, 1]))

            mean_T0_all_matched = int(
                np.mean(self._T_class_distribution[:, 1]))
            mean_T1_all_matched = int(
                np.mean(self._T_class_distribution[:, 3]))

        val_means = []
        for CATE_list in self._val_CATE_array:
            val_means += [np.mean(CATE_list)]

        train_means = []
        for CATE_list in self._train_CATE_array:
            train_means += [np.mean(CATE_list)]

        train_p_value = p_values_arg_coef(np.array(train_means), 0)/100
        if train_p_value > 0.5:
            train_p_value = 1-train_p_value

        val_p_value = p_values_arg_coef(np.array(val_means), 0)/100
        if val_p_value > 0.5:
            val_p_value = 1-val_p_value

        mean_train_CATE = np.mean(train_means)
        train_cate_p_value = train_p_value

        mean_val_CATE = np.mean(val_means)
        val_cate_p_value = val_p_value

        prognostic_variables = np.intersect1d(self._T0_stats_shap_df[self._T0_stats_shap_df.p_value < self.power_alpha].index.values,
                                              self._T1_stats_shap_df[self._T1_stats_shap_df.p_value < self.power_alpha].index.values)

        predictive_variables = self._CATE_stats_shap_df[self._CATE_stats_shap_df.p_value <
                                                        self.power_alpha].index.values

        export_df = pd.DataFrame(data=[[
            mean_train_CATE, train_cate_p_value,
            mean_val_CATE, val_cate_p_value,
            #list(prognostic_variables), list(predictive_variables),
            self._propensity_matching,
            mean_before_auc, std_before_auc,
            mean_after_auc, std_after_auc,
            mean_T0_class_1_samples, mean_T0_class_0_samples,
            mean_T1_class_1_samples, mean_T1_class_0_samples,
            mean_T0_all_matched, mean_T0_all_matched]],
            columns=[
                "train_CATE", "train_p_value",
                "val_CATE", "val_p_value",
                # "prognostic_variables","predictive_variables",
                "propensity_matching",
                "mean_before_auc", "std_before_auc",
                "mean_after_auc", "std_after_auc",
                "mean_T0_class_1_samples", "mean_T0_class_0_samples",
                "mean_T1_class_1_samples", "mean_T1_class_0_samples",
                "mean_T0_all_matched", "mean_T0_all_matched"
        ])

        export_df.to_csv(data_path, index=False)
