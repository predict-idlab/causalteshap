__author__ = "Jarne Verhaeghe, Jeroen Van der Donckt"

import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted

from .shap_wrappers import ShapExplainerFactory
from .utils import causalteshap_analysis


class CausalteShap(SelectorMixin, BaseEstimator):
    """
    Feature selection based on significance of shap values.

    """

    def __init__(
        self,
        model=None,
        significance_factor: float = 0.02,
        val_size: float = 0.3,
        stratify: bool = False,
        cv: BaseCrossValidator = None,
        show_progress: bool = True,
        verbose: bool = False,
        meta_learner: str = "S",
        **fit_kwargs,
    ):
        """
        Create a powershap object.

        Parameters
        ----------
        model: Any, optional
            The model used for for the powershap calculation. The currently supported
            models are; catboost, sklearn tree-based, sklearn linear, and tensorflow
            deep learning models.
            If no model is passed, by default, a catboost model will be used. If the
            data type is of type float, a CatBoostRegressor will be selected, for all
            the other cases a CatBoostClassifier is selected.
            ..note::
                The deep learning model should take |features| + 1 as input size.
                It is thus the user his/her responsibility to account for the added
                random feature, when using deep learning models.
        significance_factor: float, optional
            The alpha value used for the power-calculation of the used statistical test
            and significance threshold. Should be a float between ]0,1[. By default
            0.01.
        val_size: float, optional
            The fractional size of the validation set. Should be a float between ]0,1[.
            By default 0.2.
        stratify: bool, optional
            Whether to create a stratified train_test_split (based on the `y` that is
            given to the `.fit` method). By default False.
            ..note::
                If you want to pass a specific array as stratify (that is not `y`), you
                can pass it as `stratify` argument to the `.fit` method.
        cv: BaseCrossValidator, optional
            The cross-validator to use. By default None.
            This cross-validator should have a `.split` method which yields
            (train_idx, test_idx) tuples. The arguments of the `.split` method should be
            X, y, groups. This splitter will be wrapped to yield infinite splits.
            ..note::
                If the given coss validator has no `random_state` argument, the same
                splits will be used multiple times in the powershap iterations. This
                may lead to overfitting on the cross-validation splits (and thus
                selection of non-informative variables).
        show_progress: bool, optional
            Flag indicating whether progress of the powershap iterations should be
            shown. By default True.
        verbose: bool, optional
            Flag indicating whether verbose console output should be shown. By default
            False.
        **fit_kwargs: dict
            Keyword arguments for fitting the model.
            ..note::
                For a deep learning model, the following keyword arguments are required:
                "epochs", "optimizer", "batch_size", "nn_metric", "loss"

        """
        self.model = model
        self.significance_factor = significance_factor
        self.val_size = val_size
        self.stratify = stratify
        self.show_progress = show_progress
        self.verbose = verbose
        self.fit_kwargs = fit_kwargs
        self.meta_learner = meta_learner

        def _infinite_splitter(cv):
            """Infinite yields for the given splitter.
            If the splitter is exhausted, it will be reset and restarted.
            """
            from copy import deepcopy

            cv = deepcopy(cv)
            splitter = None
            random_state = 0

            def split(X, y=None, groups=None):
                nonlocal splitter, random_state
                if splitter is None:
                    if hasattr(cv, "random_state"):  # Update random state
                        cv.__setattr__("random_state", random_state)
                        random_state += 1
                    splitter = cv.split(X, y=y, groups=groups)
                while True:
                    try:
                        yield next(splitter)
                    except StopIteration:
                        if hasattr(cv, "random_state"):  # Update random state
                            cv.__setattr__("random_state", random_state)
                            random_state += 1
                        splitter = cv.split(X, y=y, groups=groups)
                        yield next(splitter)

            return split

        if cv is not None:
            self.cv = _infinite_splitter(cv)
        else:
            self.cv = None

        if model is not None:
            self._explainer = ShapExplainerFactory.get_explainer(model=model)

    @staticmethod
    def _get_default_model(y: np.ndarray):
        from catboost import CatBoostClassifier, CatBoostRegressor

        assert isinstance(y, np.ndarray)
        dtype = y.dtype
        if np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.integer):
            return CatBoostRegressor(
                n_estimators=500, od_type="Iter", od_wait=25, use_best_model=True, verbose=0
            )
        if np.issubdtype(dtype, np.integer) and len(np.unique(y.ravel())) >= 5:
            warnings.warn(
                "Classifying although there are >= 5 integers in the labels.", UserWarning
            )
        return CatBoostClassifier(
            n_estimators=500, od_type="Iter", od_wait=25, use_best_model=True, verbose=0
        )

    def _log_feature_names_sklean_v0(self, X):
        """Log the feature names if we have sklearn 0.x"""
        assert sklearn.__version__.startswith("0.")
        feature_names = np.asarray(X.columns) if hasattr(X, "columns") else None
        if feature_names is not None and len(feature_names) > 0:
            # Check if all feature names of type string
            types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))
            if len(types) > 1 or types[0] != "str":
                feature_names = None
                warnings.warn("Feature names only support names that are all strings.", UserWarning)

        if feature_names is not None and len(feature_names) > 0:
            self.feature_names_in_ = feature_names
        elif hasattr(self, "feature_names_in_"):
            # Delete the attribute when the estimator is fitted on a new dataset that
            # has no feature names.
            delattr(self, "feature_names_in_")

    def _print(self, *values):
        """Helper method for printing if `verbose` is set to True."""
        if self.verbose:
            print(*values)

    def fit(self, X, T, y, stratify=None, groups=None, **kwargs):
        """Fit the powershap feature selector.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y: array-like of shape (n_samples,)
            The treatment variable for treatment effect estimation.
        y: array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        stratify: array-like of shape (n_samples,), optional
            Array that will be used to perform stratified train-test splits. By default
            None.
            Note: if None, than `y` will be used as `stratify` if the stratify flag of
            the object is True.
        groups: array-like of shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. By default None.

        """
        if stratify is None and self.stratify:
            # Set stratify to y, if no stratify is given and self.stratify is True
            stratify = y

        # kwargs take precedence over fit_kwargs
        kwargs = {**self.fit_kwargs, **kwargs}

        if self.model is None:
            # If no model is passed to the constructor -> select the default catboost
            # model
            self.model = CausalteShap._get_default_model(np.asarray(y))
            self._explainer = ShapExplainerFactory.get_explainer(self.model)

        if sklearn.__version__.startswith("0."):
            # Log the feature names if we have sklearn 0.x
            self._log_feature_names_sklean_v0(X)

        # Perform the necessary sklearn checks -> X and y are both ndarray.
        # Logs the feature names as well (in self.feature_names_in_ in sklearn 1.x).
        #
        # These two operations (_validate_data and pd.DataFrame) will also copy
        # the data into a new place in memory, avoiding data mutation. How this
        # happens may not be obvious at first glance:
        #
        # 1. _validate_data ensures that the data is a numpy array, copying it
        # upon conversion if necessary.
        #
        # 2. pd.DataFrame then copies X, which is now an numpy array, into a
        # new pandas dataframe.
        #
        # If this is changed in some way which would allow explain() to mutate
        # the original data, it should cause the data mutation tests to fail.
        X, y = self._explainer._validate_data(self._validate_data, X, y, multi_output=True)
        X = pd.DataFrame(data=X, columns=list(range(X.shape[1])))

        self._print("Starting causalteshap")

        shaps_do1, shaps_do0 = self._explainer.explain(
            X=X,
            T=T,
            y=y,
            val_size=self.val_size,
            stratify=stratify,
            groups=groups,
            cv_split=self.cv,  # pass the wrapped cv split function
            show_progress=self.show_progress,
            meta_learner=self.meta_learner,
            **kwargs,
        )

        analysis_df = causalteshap_analysis(
            T1_matrix_list=shaps_do1, T0_matrix_list=shaps_do0, significance_factor=self.significance_factor,verbose=self.verbose
        )

        self._print("Done!")

        ## Store the processed_shaps_df in the object
        self._analysis_df = analysis_df

        ## Store the processed_shaps_df in the object
        if hasattr(self, "feature_names_in_"):
            self._analysis_df.index = [
                self.feature_names_in_[i] if isinstance(i, np.int64) else i
                for i in analysis_df.index.values
            ]
        
        # It is convention to return self
        return self

    # This is the only method that needs to be implemented to serve the transform
    # functionality
    def _get_support_mask(self):
        # Select the significant features
        return self._analysis_df[self._analysis_df.predictive==1].index.values

    def get_predictive(self):
        return self._analysis_df[self._analysis_df.predictive==1].index.values
    
    def get_candidate_predictive(self):
        return self._analysis_df[self._analysis_df.candidate==1].index.values

    def get_analysis_df(self):
        return self._analysis_df.sort_values(["ttest","ks-statistic"])

    def transform(self, X):
        check_is_fitted(self, ["_analysis_df", "_explainer"])
        if hasattr(self, "feature_names_in_") and isinstance(X, pd.DataFrame):
            assert np.all(X.columns.values == self.feature_names_in_)
            return pd.DataFrame(
                super().transform(X), columns=self._get_support_mask()
            )
        return super().transform(X)

    def _more_tags(self):
        return self._explainer._get_more_tags()