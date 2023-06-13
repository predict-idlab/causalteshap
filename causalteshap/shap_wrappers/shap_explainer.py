__author__ = "Jarne Verhaeghe, Jeroen Van Der Donckt"

import warnings
import shap
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from abc import ABC

from typing import Any, Callable


class ShapExplainer(ABC):
    """Interface class for a (Causalteshap explainer class."""

    def __init__(
        self,
        model: Any,
    ):
        """Create a Causalteshap explainer instance.

        Parameters
        ----------
        model: Any
            The  model from which causalteshap will use its shap values to perform feature
            selection.

        """
        assert self.supports_model(model)
        self.model = model

    # Should be implemented by subclass
    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, meta_learner, **kwargs
    ) -> np.array:
        raise NotImplementedError
    
    def _validate_data(self, validate_data: Callable, X, y, **kwargs):
        return validate_data(X, y, **kwargs)

    # Should be implemented by subclass
    @staticmethod
    def supports_model(model) -> bool:
        """Check whether the Causalteshap explainer supports the given model.

        Parameters
        ----------
        model: Any
            The model.

        Returns
        -------
        bool
            True if the Causalteshap explainer supports the given model, otherwise False.

        """
        raise NotImplementedError

    def explain(
        self,
        X: pd.DataFrame,
        T: np.array,
        y: np.array,
        val_size: float,
        stratify: np.array = None,
        groups: np.array = None,
        meta_learner: str = "S",
        random_seed_start: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """Get the shap values,

        Parameters
        ----------
        X: pd.DataFrame
            The features.
        y: np.array
            The treatments.
        y: np.array
            The labels.
        val_size: float
            The fractional size of the validation set. Should be a float between ]0,1[.
        stratify: np.array, optional
            The array used to create a stratified train_test_split. By default None.
        groups: np.array, optional
            The group labels for the samples used while splitting the dataset into
            train/test set. By default None.
        random_seed_start: int, optional
            The random seed to start the iterations with. By default 0.
        **kwargs: dict
            The keyword arguments for the fit method.
        """

        random_col_name = "random_uniform_feature"
        assert not random_col_name in X.columns

        assert meta_learner in ["S","T","X","R"]

        X = X.copy(deep=True)

        npRandomState = RandomState(random_seed_start)

        # Add uniform random feature to X
        random_uniform_feature = npRandomState.uniform(-1, 1, len(X))

        # The relative order of these assignments is important for the causalteanalysis later on
        X["T"] = T
        X["random_uniform_feature"] = random_uniform_feature

        # Perform train-test split
        if groups is None:
            # stratify may be None or not None
            train_idx, val_idx = train_test_split(
                np.arange(len(X)),
                test_size=val_size,
                random_state=random_seed_start,
                stratify=stratify,
            )
        elif stratify is None:
            # groups may be None or not None
            from sklearn.model_selection import GroupShuffleSplit

            train_idx, val_idx = next(
                GroupShuffleSplit(
                    random_state=random_seed_start,
                    n_splits=1,
                    test_size=val_size,
                ).split(X, y, groups=groups)
            )
        else:
            # stratify and groups are both not Noe
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                train_idx, val_idx = next(
                    StratifiedGroupKFold(
                        shuffle=True,
                        random_state=random_seed_start,
                        n_splits=int(1 / val_size),
                    ).split(X, y, groups=groups)
                )
            except:
                warnings.warn(
                    "Did not find StratifiedGroupKFold in sklearn install, "
                    + "this is only supported in sklearn 1.x.",
                    UserWarning,
                )
                train_idx, val_idx = train_test_split(
                    np.arange(len(X)),
                    test_size=val_size,
                    random_state=random_seed_start,
                    stratify=stratify,
                )

        X_train = X.iloc[np.sort(train_idx)]
        X_val = X.iloc[np.sort(val_idx)]
        Y_train = y[np.sort(train_idx)]
        Y_val = y[np.sort(val_idx)]

        shap_values_do1,shap_values_do0 = self._fit_get_shap(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            random_seed=random_seed_start,
            meta_learner=meta_learner,
            **kwargs,
        )

        if len(np.shape(shap_values_do1)) > 2:
            shap_values_do1 = np.max(shap_values_do1, axis=0)
            shap_values_do0 = np.max(shap_values_do0, axis=0)

        shaps_do1 = np.array(shap_values_do1)
        shaps_do0 = np.array(shap_values_do0)

        return shaps_do1, shaps_do0

    def _get_more_tags(self):
        return {}




### CATBOOST

from catboost import CatBoostRegressor, CatBoostClassifier

class CatboostExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        supported_models = [CatBoostRegressor, CatBoostClassifier]
        return isinstance(model, tuple(supported_models))

    def _validate_data(self, validate_data: Callable, X, y, **kwargs):
        kwargs["force_all_finite"] = False  # catboost allows NaNs and infs in X
        kwargs["dtype"] = None  # allow non-numeric data
        return super()._validate_data(validate_data, X, y, **kwargs)

    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, meta_learner, **kwargs
    ) -> np.array:
        
        # Fit the model
        if meta_learner == "S":
            S_model = self.model.copy().set_params(random_seed=random_seed)
            S_model.fit(X_train, Y_train, eval_set=(X_val, Y_val))

            npRandomState_ran_feature = RandomState(random_seed)

            X_do1 = X_val.copy(deep=True)
            X_do1.loc[:,"T"] = 1
            X_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            X_do0 = X_val.copy(deep=True)
            X_do0.loc[:,"T"] = 0
            X_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            # Calculate the shap values
            C_explainer = shap.TreeExplainer(S_model)

            return C_explainer.shap_values(X_do1), C_explainer.shap_values(X_do0)
        
        elif meta_learner == "T":
            T1_model = self.model.copy().set_params(random_seed=random_seed)
            T0_model = self.model.copy().set_params(random_seed=random_seed)

            T1_train_mask = X_train["T"]==1
            T1_val_mask = X_val["T"]==1

            T0_train_mask = X_train["T"]==0
            T0_val_mask = X_val["T"]==0

            T1_model.fit(X_train[T1_train_mask].drop(columns=["T"]), 
                         Y_train[T1_train_mask], 
                         eval_set=(X_val[T1_val_mask].drop(columns=["T"]), Y_val[T1_val_mask]))
            T0_model.fit(X_train[T0_train_mask].drop(columns=["T"]), 
                         Y_train[T0_train_mask], 
                         eval_set=(X_val[T0_val_mask].drop(columns=["T"]), Y_val[T0_val_mask]))

            npRandomState_ran_feature = RandomState(random_seed+1)

            X_do1 = X_val.drop(columns=["T"]).copy(deep=True)
            X_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            X_do0 = X_val.drop(columns=["T"]).copy(deep=True)
            X_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            # Calculate the shap values
            T1_explainer = shap.TreeExplainer(T1_model)
            T0_explainer = shap.TreeExplainer(T0_model)

            return T1_explainer.shap_values(X_do1), T0_explainer.shap_values(X_do0)
        
        else:
            raise NotImplementedError 
        


    def _get_more_tags(self):
        return {"allow_nan": True}


class LGBMExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from lightgbm import LGBMClassifier, LGBMRegressor
        supported_models = [LGBMClassifier, LGBMRegressor]
        return isinstance(model, tuple(supported_models))

    def _validate_data(self, validate_data: Callable, X, y, **kwargs):
        kwargs["force_all_finite"] = False  # lgbm allows NaNs and infs in X
        return super()._validate_data(validate_data, X, y, **kwargs)

    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, meta_learner,  **kwargs
    ) -> np.array:
        # Fit the model

        # Why we need to use deepcopy and delete LGBM __deepcopy__
        # https://github.com/microsoft/LightGBM/issues/4085
        from copy import copy

        if meta_learner == "S":
            S_model = copy(self.model).set_params(random_seed=random_seed)
            S_model.fit(X_train, Y_train, eval_set=(X_val, Y_val))

            npRandomState_ran_feature = RandomState(random_seed+1)

            X_do1 = X_val.copy(deep=True)
            X_do1.loc[:,"T"] = 1
            X_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            X_do0 = X_val.copy(deep=True)
            X_do0.loc[:,"T"] = 0
            X_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            # Calculate the shap values
            C_explainer = shap.TreeExplainer(S_model)

            return C_explainer.shap_values(X_do1), C_explainer.shap_values(X_do0)
        
        elif meta_learner == "T":
            T1_model = copy(self.model).set_params(random_seed=random_seed)
            T0_model = copy(self.model).set_params(random_seed=random_seed)

            T1_train_mask = X_train["T"]==1
            T1_val_mask = X_val["T"]==1

            T0_train_mask = X_train["T"]==0
            T0_val_mask = X_val["T"]==0

            T1_model.fit(X_train[T1_train_mask].drop(columns=["T"]), 
                         Y_train[T1_train_mask], 
                         eval_set=(X_val[T1_val_mask].drop(columns=["T"]), Y_val[T1_val_mask]))
            T0_model.fit(X_train[T0_train_mask].drop(columns=["T"]), 
                         Y_train[T0_train_mask], 
                         eval_set=(X_val[T0_val_mask].drop(columns=["T"]), Y_val[T0_val_mask]))

            npRandomState_ran_feature = RandomState(random_seed+1)

            X_do1 = X_val.drop(columns=["T"]).copy(deep=True)
            X_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            X_do0 = X_val.drop(columns=["T"]).copy(deep=True)
            X_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            # Calculate the shap values
            T1_explainer = shap.TreeExplainer(T1_model)
            T0_explainer = shap.TreeExplainer(T0_model)

            return T1_explainer.shap_values(X_do1), T0_explainer.shap_values(X_do0)
        
        else:
            raise NotImplementedError 
        

    def _get_more_tags(self):
        return {"allow_nan": True}



### RANDOMFOREST


class EnsembleExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        # TODO: these first 2 require extra checks on the base_estimator
        # from sklearn.ensemble._weight_boosting import BaseWeightBoosting
        # from sklearn.ensemble._bagging import BaseBagging
        from sklearn.ensemble._forest import ForestRegressor, ForestClassifier
        from sklearn.ensemble._gb import BaseGradientBoosting
        # from sklearn.ensemble._hist_gradient_boosting import BaseHistGradientBoosting

        supported_models = [ForestRegressor, ForestClassifier, BaseGradientBoosting]
        return issubclass(type(model), tuple(supported_models))

    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, meta_learner,  **kwargs
    ) -> np.array:
        from sklearn.base import clone

        if meta_learner == "S":
            S_model = clone(self.model).set_params(random_state=random_seed)
            S_model.fit(X_train, Y_train)

            npRandomState_ran_feature = RandomState(random_seed+1)

            X_do1 = X_val.copy(deep=True)
            X_do1.loc[:,"T"] = 1
            X_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            X_do0 = X_val.copy(deep=True)
            X_do0.loc[:,"T"] = 0
            X_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            # Calculate the shap values
            C_explainer = shap.TreeExplainer(S_model)

            return C_explainer.shap_values(X_do1), C_explainer.shap_values(X_do0)
        
        elif meta_learner == "T":
            T1_model = clone(self.model).set_params(random_state=random_seed)
            T0_model = clone(self.model).set_params(random_state=random_seed)

            T1_train_mask = X_train["T"]==1

            T0_train_mask = X_train["T"]==0

            T1_model.fit(X_train[T1_train_mask].drop(columns=["T"]), 
                         Y_train[T1_train_mask])
            T0_model.fit(X_train[T0_train_mask].drop(columns=["T"]), 
                         Y_train[T0_train_mask])

            npRandomState_ran_feature = RandomState(random_seed+1)

            X_do1 = X_val.drop(columns=["T"]).copy(deep=True)
            X_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            X_do0 = X_val.drop(columns=["T"]).copy(deep=True)
            X_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            # Calculate the shap values
            T1_explainer = shap.TreeExplainer(T1_model)
            T0_explainer = shap.TreeExplainer(T0_model)

            return T1_explainer.shap_values(X_do1), T0_explainer.shap_values(X_do0)
        
        else:
            raise NotImplementedError 

### LINEAR


class LinearExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from sklearn.linear_model._base import LinearClassifierMixin, LinearModel
        from sklearn.linear_model._stochastic_gradient import BaseSGD

        supported_models = [LinearClassifierMixin, LinearModel, BaseSGD]
        return issubclass(type(model), tuple(supported_models))

    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, meta_learner,  **kwargs
    ) -> np.array:
        from sklearn.base import clone

        # Fit the model
        try:
            PowerShap_model = clone(self.model).set_params(random_state=random_seed)
        except:
            PowerShap_model = clone(self.model)
        PowerShap_model.fit(X_train, Y_train)

        if meta_learner == "S":
            try:
                S_model = clone(self.model).set_params(random_state=random_seed)
            except:
                S_model = clone(self.model)

            S_model.fit(X_train, Y_train)

            npRandomState_ran_feature = RandomState(random_seed+1)

            X_do1 = X_val.copy(deep=True)
            X_do1.loc[:,"T"] = 1
            X_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            X_do0 = X_val.copy(deep=True)
            X_do0.loc[:,"T"] = 0
            X_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            # Calculate the shap values
            C_explainer = shap.TreeExplainer(S_model)

            return C_explainer.shap_values(X_do1), C_explainer.shap_values(X_do0)
        
        elif meta_learner == "T":
            try:
                T1_model = clone(self.model).set_params(random_state=random_seed)
            except:
                T1_model = clone(self.model)

            try:
                T0_model = clone(self.model).set_params(random_state=random_seed)
            except:
                T0_model = clone(self.model)

            T1_train_mask = X_train["T"]==1
            T0_train_mask = X_train["T"]==0

            T1_model.fit(X_train[T1_train_mask].drop(columns=["T"]), 
                         Y_train[T1_train_mask])
            T0_model.fit(X_train[T0_train_mask].drop(columns=["T"]), 
                         Y_train[T0_train_mask])

            npRandomState_ran_feature = RandomState(random_seed+1)

            X_train_do1 = X_train.drop(columns=["T"]).copy(deep=True)
            X_train_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_train))
            
            X_do1 = X_val.drop(columns=["T"]).copy(deep=True)
            X_do1.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            X_train_do0 = X_train.drop(columns=["T"]).copy(deep=True)
            X_train_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_train))

            X_do0 = X_val.drop(columns=["T"]).copy(deep=True)
            X_do0.loc[:,"random_uniform_feature"] = npRandomState_ran_feature.uniform(-1,1,len(X_val))

            # Calculate the shap values
            T1_explainer = shap.explainers.Linear(T1_model, X_train_do1)
            T0_explainer = shap.explainers.Linear(T0_model, X_train_do0)

            return T1_explainer.shap_values(X_do1), T0_explainer.shap_values(X_do0)
        
        else:
            raise NotImplementedError 


### DEEP LEARNING


# class DeepLearningExplainer(ShapExplainer):
#     @staticmethod
#     def supports_model(model) -> bool:
#         import tensorflow as tf  # ; import torch

#         # import torch  ## TODO: do we support pytorch??

#         supported_models = [tf.keras.Model]  # , torch.nn.Module]
#         return isinstance(model, tuple(supported_models))

#     def _fit_get_shap(
#         self, X_train, Y_train, X_val, Y_val, random_seed, meta_learner,  **kwargs
#     ) -> np.array:
#         import tensorflow as tf

#         tf.compat.v1.disable_v2_behavior()  # https://github.com/slundberg/shap/issues/2189

#         # Fit the model
#         PowerShap_model = tf.keras.models.clone_model(self.model)
#         metrics = kwargs.get("nn_metric")
#         PowerShap_model.compile(
#             loss=kwargs["loss"],
#             optimizer=kwargs["optimizer"],
#             metrics=metrics if metrics is None else [metrics],
#         )
#         _ = PowerShap_model.fit(
#             X_train,
#             Y_train,
#             batch_size=kwargs["batch_size"],
#             epochs=kwargs["epochs"],
#             validation_data=(X_val, Y_val),
#             verbose=False,
#         )
#         # Calculate the shap values
#         C_explainer = shap.DeepExplainer(PowerShap_model, X_train)
#         return C_explainer.shap_values(X_val)
