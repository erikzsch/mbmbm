import logging
from pathlib import Path

from sklearn.feature_selection import (
    RFE,
    GenericUnivariateSelect,
    SelectFromModel,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)

from mbmbm.featureengine.featureenginebase import FeatureEngineResult, FeatureSelectorBase
from mbmbm.featureengine.utils.fcbf import FastCorrelationBasedFilter
from mbmbm.featureengine.utils.indval import IndValSelector

# Configure logger
logger = logging.getLogger(__name__)


class FeatureSelectorFromModel(FeatureSelectorBase):
    """
    Feature selector using an existing Scikit-Learn model.

    This feature selector uses an estimator model to determine feature importance and select features.

    :param model: The model used for selecting features based on its internal feature importance.
    :type model: sklearn.base.BaseEstimator
    """

    def __init__(self, model=None):
        super().__init__()
        if model is None:
            logger.error("Model cannot be None for FeatureSelectorFromModel.")
            raise ValueError("Model cannot be None")
        self._model = model
        logger.info("FeatureSelectorFromModel initialized with model: %s", model)

    def _set_method(self):
        return SelectFromModel(estimator=self._model)


ALLOWED_SCORE_FCN_MAP = {
    "chi2": chi2,
    "f_classif": f_classif,
    "mutual_info_classif": mutual_info_classif,
    "f_regression": f_regression,
    "r_regression": r_regression,
    "mutual_info_regression": mutual_info_regression,
}


class FeatureSelectorGUS(FeatureSelectorBase):
    """
    Feature Selector using Generic Univariate Selection (GUS).

    :param score_func: The scoring function to use for feature selection.
    Must be one of 'chi2', 'f_classif', 'mutual_info_classif',
    'f_regression', 'r_regression', 'mutual_info_regression'.
    :type score_func: str
    :param mode: The mode of feature selection ('k_best', 'percentile', etc.).
    :type mode: str
    :param param: Parameter value to control the number of features selected.
    :type param: int
    """

    def __init__(self, score_func=None, mode: str = "k_best", param: int = 100):
        super().__init__()
        if score_func not in ALLOWED_SCORE_FCN_MAP:
            logger.error("Invalid score function provided: %s", score_func)
            raise KeyError(f"Invalid score function: {score_func}")
        self._score_func_str = score_func
        self._score_func = ALLOWED_SCORE_FCN_MAP[score_func]
        self._mode = mode
        self._param = param
        logger.info("FeatureSelectorGUS initialized with score_func: %s, mode: %s, param: %d", score_func, mode, param)

    def _set_method(self):
        return GenericUnivariateSelect(score_func=self._score_func, mode=self._mode, param=self._param)

    def get_additional_infos(self):
        return {"score_func": str(self._score_func_str)}

    def get_selected_indices(self):
        return self._method.get_support(indices=True)


class FeatureSelectorVarianceThreshold(FeatureSelectorBase):
    """
    Feature Selector using Variance Threshold.

    Removes features with a variance lower than the specified threshold.

    :param threshold: Features with a training-set variance lower than this threshold will be removed.
    :type threshold: float
    """

    def __init__(self, threshold: float = 0.8):
        super().__init__()
        self._threshold = threshold
        logger.info("FeatureSelectorVarianceThreshold initialized with threshold: %f", threshold)

    def _set_method(self):
        return VarianceThreshold(threshold=self._threshold)

    def get_feature_scores(self):
        return self._method.variances_


class FeatureSelectorRFE(FeatureSelectorBase):
    """
    Feature Selector using Recursive Feature Elimination (RFE).

    :param n_features_to_select: The number of features to select.
    :type n_features_to_select: int
    :param step: The number of features to remove at each iteration.
    :type step: int
    :param estimator: The estimator to use for RFE.
    :type estimator: sklearn.base.BaseEstimator
    """

    def __init__(self, n_features_to_select: int = 100, step: int = 1, estimator=None):
        super().__init__()
        if estimator is None:
            logger.error("No estimator provided for FeatureSelectorRFE.")
            raise ValueError("No estimator provided")
        self._n_features_to_select = n_features_to_select
        self.step = step
        self.estimator = estimator
        logger.info(
            "FeatureSelectorRFE initialized with estimator: %s, n_features_to_select: %d, step: %d",
            estimator,
            n_features_to_select,
            step,
        )

    def _set_method(self):
        return RFE(
            estimator=self.estimator, n_features_to_select=self._n_features_to_select, step=self.step, verbose=300
        )

    def fit(self, arr_input, arr_target, dataset_id="unset", force_fit=False, plot_dir=None) -> FeatureEngineResult:
        if not force_fit and self._is_fitted:
            logger.warning("FeatureEngine already fitted. Use force_fit=True to refit.")
            raise AssertionError("FeatureEngine already fitted")
        self.check_all_initialized()

        logger.debug("Starting fit method for FeatureSelectorRFE.")
        self._method.fit(arr_input, arr_target)
        logger.info("FeatureSelectorRFE fitting completed.")

        selected_indices = self.get_selected_indices()
        feature_scores = self._method.ranking_

        self._is_fitted = True

        if plot_dir is not None:
            plot_file_path = Path(plot_dir, "feature_selection.png")
            logger.debug("Plotting feature selection results to %s", plot_file_path)
            self.plot(
                arr_input,
                arr_target,
                selected_indices,
                feature_scores=feature_scores,
                dataset_id=dataset_id,
                plot_file_path=plot_file_path,
                y_label="Rank",
            )

        return FeatureEngineResult(
            source_engine=self.__class__.__name__,
            dataset_id=dataset_id,
            selection=selected_indices,
        )

    def get_selected_indices(self):
        return [i for i, v in enumerate(self._method.support_) if v]


class FeatureSelectorFCBF(FeatureSelectorBase):
    """
    Feature Selector using Fast Correlation-Based Filter (FCBF).

    This method uses Fast Correlation-Based Filtering for feature selection.
    """

    def __init__(self):
        super().__init__()
        logger.info("FeatureSelectorFCBF initialized.")

    def _set_method(self):
        return FastCorrelationBasedFilter()

    def fit(self, arr_input, arr_target, dataset_id="unset", force_fit=False, plot_dir=None) -> FeatureEngineResult:
        if not force_fit and self._is_fitted:
            logger.warning("FeatureEngine already fitted. Use force_fit=True to refit.")
            raise AssertionError("FeatureEngine already fitted")
        self.check_all_initialized()

        logger.debug("Starting fit method for FeatureSelectorFCBF.")
        self._method.fit(arr_input, arr_target)
        logger.info("FeatureSelectorFCBF fitting completed.")

        selected_indices = self.get_selected_indices()
        feature_scores = self._method.scores_

        self._is_fitted = True

        if plot_dir is not None:
            plot_file_path = Path(plot_dir, "feature_selection.png")
            logger.debug("Plotting feature selection results to %s", plot_file_path)
            self.plot(
                arr_input,
                arr_target,
                selected_indices,
                feature_scores=feature_scores,
                dataset_id=dataset_id,
                plot_file_path=plot_file_path,
                y_label="Rank",
            )

        return FeatureEngineResult(
            source_engine=self.__class__.__name__,
            dataset_id=dataset_id,
            selection=selected_indices,
        )


class FeatureSelectorIndVal(FeatureSelectorBase):
    """
    Feature Selector using Indicator Value (IndVal).

    :param percentile: Percentile of features to keep.
    :type percentile: int
    :param indval_func: The indicator value function to use.
    :type indval_func: str
    :param num_permutations: Number of permutations for statistical testing.
    :type num_permutations: int
    :param min_order: Minimum order for selection.
    :type min_order: int
    :param max_order: Maximum order for selection.
    :type max_order: int, optional
    """

    def __init__(self, percentile=50, indval_func="indval.g", num_permutations=20, min_order=1, max_order=None):
        super().__init__()
        self.percentile = percentile
        self.indval_func = indval_func
        self.num_permutations = num_permutations
        self.min_order = min_order
        self.max_order = max_order
        logger.info("FeatureSelectorIndVal initialized with percentile: %d, indval_func: %s", percentile, indval_func)

    def _set_method(self):
        return IndValSelector(
            percentile=self.percentile,
            indval_func=self.indval_func,
            num_permutations=self.num_permutations,
            min_order=self.min_order,
            max_order=self.max_order,
        )

    def fit(self, arr_input, arr_target, dataset_id="unset", force_fit=False, plot_dir=None) -> FeatureEngineResult:
        if not force_fit and self._is_fitted:
            logger.warning("FeatureEngine already fitted. Use force_fit=True to refit.")
            raise AssertionError("FeatureEngine already fitted")
        self.check_all_initialized()

        logger.debug("Starting fit method for FeatureSelectorIndVal.")
        self._method.fit(arr_input, arr_target)
        logger.info("FeatureSelectorIndVal fitting completed.")

        selected_indices = self.get_selected_indices()
        feature_scores = self._method.indicator_values

        self._is_fitted = True

        if plot_dir is not None:
            plot_file_path = Path(plot_dir, "feature_selection.png")
            logger.debug("Plotting feature selection results to %s", plot_file_path)
            self.plot(
                arr_input,
                arr_target,
                selected_indices,
                feature_scores=feature_scores,
                dataset_id=dataset_id,
                plot_file_path=plot_file_path,
                y_label="Indicator Value",
            )

        return FeatureEngineResult(
            source_engine=self.__class__.__name__,
            dataset_id=dataset_id,
            selection=selected_indices,
        )

    def get_selected_indices(self):
        return self._method.get_support(indices=True)

    def get_additional_infos(self):
        return {
            "percentile": self.percentile,
            "indval_func": self.indval_func,
            "num_permutations": self.num_permutations,
            "min_order": self.min_order,
            "max_order": self.max_order,
        }
