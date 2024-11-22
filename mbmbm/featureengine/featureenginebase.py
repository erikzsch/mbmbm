import csv
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from pickle import dump, load
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mbmbm import FEATURE_ENGINE_SAVE_NAME, logger
from mbmbm.core.dataset_provider import DatasetProvider


@dataclass
class FeatureEngineResult:
    """
    Data class to store the result of feature selection or reduction.

    :param source_engine: The source feature engine used.
    :type source_engine: str
    :param dataset_id: The ID of the dataset used for feature engineering.
    :type dataset_id: str
    :param selection: List of selected feature indices.
    :type selection: List[int]
    :param reduction: Information regarding feature reduction.
    :type reduction: Dict
    """

    source_engine: str
    dataset_id: str
    selection: List[int] = None
    reduction: Dict = None


class FeatureEngineBase:
    """
    Base interface for Feature Engine classes, providing core methods for fitting, transforming, and saving.
    """

    def __init__(self):
        self._is_fitted = False
        self._method = None

    def init_method(self):
        """
        Initialize the feature engineering method.
        """
        logger.info("Initializing method for feature engineering.")
        self._method = self._set_method()

    def check_all_initialized(self):
        """
        Check if all necessary components are initialized.

        :raises AssertionError: If the method is not initialized.
        """
        assert self._method is not None, "Feature engineering method is not defined. Please initialize the method."
        logger.debug("All components are initialized.")

    def save(self, out_dir: PathLike):
        """
        Save the fitted feature engineering method to a file.

        :param out_dir: Output directory where the method will be saved.
        :type out_dir: PathLike
        :raises AssertionError: If the method is not fitted.
        """
        assert self._is_fitted, "Only fitted methods are savable."
        with (Path(out_dir) / FEATURE_ENGINE_SAVE_NAME).open(mode="wb") as f:
            dump(self._method, f)
        logger.info(f"Feature engineering method saved to {out_dir}.")

    def load(self, load_dir: PathLike, skip_checks=False):
        """
        Load a fitted feature engineering method from a file.

        :param load_dir: Directory from which the method will be loaded.
        :type load_dir: PathLike
        :param skip_checks: Flag to skip validation checks.
        :type skip_checks: bool
        :raises AssertionError: If the file does not exist or method mismatch occurs.
        """
        if not skip_checks:
            current_method_cls = self._method.__class__.__name__
        filepath = Path(load_dir) / FEATURE_ENGINE_SAVE_NAME
        assert filepath.exists(), f"The specified path {filepath} does not exist."
        with filepath.open(mode="rb") as f:
            self._method = load(f)
            self._is_fitted = True
        logger.info(f"Feature engineering method loaded from {load_dir}.")

        if not skip_checks:
            new_method_cls = self._method.__class__.__name__
            assert current_method_cls == new_method_cls, (
                f"Loaded method and class-defined method mismatch.\n"
                f"Expected: {current_method_cls}, but got: {new_method_cls}"
            )
            logger.error(f"Method mismatch: {current_method_cls} != {new_method_cls}")

    def _set_method(self):
        """
        Set the specific feature engineering method. This should be implemented by subclasses.

        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement _set_method.")

    def fit(self, arr_input, arr_target, dataset_id="unset", force_fit=False, plot_dir=None) -> FeatureEngineResult:
        """
        Fit the feature engineering method on the given data.

        :param arr_input: Input feature array.
        :type arr_input: np.ndarray
        :param arr_target: Target array.
        :type arr_target: np.ndarray
        :param dataset_id: Identifier for the dataset.
        :type dataset_id: str
        :param force_fit: Whether to force refitting.
        :type force_fit: bool
        :param plot_dir: Directory to save plots, if applicable.
        :type plot_dir: PathLike
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement fit.")

    def transform(self, arr_input, plot_dir=None) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineering method.

        :param arr_input: Input feature array to be transformed.
        :type arr_input: np.ndarray
        :param plot_dir: Directory to save plots, if applicable.
        :type plot_dir: PathLike, optional
        :return: Transformed input as a DataFrame.
        :rtype: pd.DataFrame
        :raises AssertionError: If the method is not fitted.
        """
        assert self._is_fitted, "FeatureEngine not fitted. Please fit the method first."
        self.check_all_initialized()

        np_res = self._method.transform(arr_input)
        logger.debug("Input data transformed successfully.")

        if plot_dir is not None:
            logger.info(f"Plotting transformation results to {plot_dir}.")
            # Placeholder for plot logic

        return np_res

    def get_infos(self):
        """
        Get information about the current method and its parameters.

        :return: Dictionary with class name and method parameters.
        :rtype: dict
        """
        params = {k: v for k, v in self._method.get_params().items() if isinstance(v, (int, float, str, bool))}
        params.update(self.get_additional_infos())
        logger.debug("Retrieved method information.")
        return {"cls": self._method.__class__.__name__, "params": params}

    def get_additional_infos(self):
        """
        Get additional information about the feature engineering method.

        :return: Additional information as a dictionary.
        :rtype: dict
        """
        return {}

    def plot(self, arr_input, arr_target, **kwargs):
        """
        Plot method for visualizing feature engineering effects.

        :param arr_input: Input feature array.
        :type arr_input: np.ndarray
        :param arr_target: Target array.
        :type arr_target: np.ndarray
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement plot.")


class FeatureSelectorBase(FeatureEngineBase):
    """
    Base class for feature selection, inheriting from FeatureEngineBase.
    """

    def fit(self, arr_input, arr_target, dataset_id="unset", force_fit=False, plot_dir=None) -> FeatureEngineResult:
        """
        Fit the feature selection method to the given data.

        :param arr_input: Input feature array.
        :type arr_input: np.ndarray
        :param arr_target: Target array.
        :type arr_target: np.ndarray
        :param dataset_id: Identifier for the dataset.
        :type dataset_id: str
        :param force_fit: Whether to force refitting.
        :type force_fit: bool
        :param plot_dir: Directory to save plots, if applicable.
        :type plot_dir: PathLike
        :return: Result of the feature selection.
        :rtype: FeatureEngineResult
        :raises AssertionError: If already fitted and force_fit is False.
        """
        if not force_fit:
            assert not self._is_fitted, "FeatureEngine already fitted. Set force_fit=True to refit."
        self.check_all_initialized()

        self._method.fit(arr_input, arr_target)
        logger.info("Feature selection method fitted successfully.")

        selected_indices = self.get_selected_indices()
        logger.debug(f"Selected indices: {selected_indices}")

        feature_scores = self.get_feature_scores()
        logger.debug(f"Feature scores: {feature_scores}")

        self._is_fitted = True

        if plot_dir is not None:
            plot_file_path = Path(plot_dir, f"feature_selection_{dataset_id}.png")
            output_file = Path(plot_dir, f"selected_indices_gus_{dataset_id}.csv")
            logger.info(f"Saving plot to: {plot_file_path}")
            logger.info(f"Saving selected indices to: {output_file}")

            # Save the plot
            self.plot(
                arr_input,
                arr_target,
                selected_indices,
                feature_scores=feature_scores,
                dataset_id=dataset_id,
                plot_file_path=plot_file_path,
            )

            # Save the selected indices to CSV
            with output_file.open(mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Selected Indices"])
                for index in selected_indices:
                    writer.writerow([index])
            logger.info("Selected indices saved to CSV.")

        return FeatureEngineResult(
            source_engine=self.__class__.__name__,
            dataset_id=dataset_id,
            selection=selected_indices,
        )

    def get_feature_scores(self):
        """
        Get the scores for each feature based on the selection method.

        :return: Feature scores.
        :rtype: list
        """
        return self._method.scores_

    def get_selected_indices(self):
        """
        Get the indices of the selected features.

        :return: Selected feature indices.
        :rtype: list
        """
        return self._method.get_support(indices=True)

    def plot(
        self,
        arr_input,
        arr_target,
        selected_indices=None,
        feature_scores=None,
        dataset_id=None,
        plot_file_path=None,
        **kwargs,
    ):
        """
        Plot the selected features and their scores.

        :param arr_input: Input feature array.
        :type arr_input: np.ndarray
        :param arr_target: Target array.
        :type arr_target: np.ndarray
        :param selected_indices: Selected feature indices.
        :type selected_indices: list
        :param feature_scores: Scores of the selected features.
        :type feature_scores: list
        :param dataset_id: Dataset identifier.
        :type dataset_id: str
        :param plot_file_path: File path to save the plot.
        :type plot_file_path: PathLike
        :param kwargs: Additional arguments for plotting.
        :type kwargs: dict
        """
        selected_scores = [feature_scores[idx] for idx in selected_indices]
        logger.debug(f"Number of selected scores: {len(selected_scores)}")

        # Get the dataset provider
        dp = DatasetProvider()
        config_data, csv_path_target, csv_file_path, input_ra, label_map = dp._get_dataset_info(
            config="default", stage="train", id=dataset_id, include_targets=True
        )

        # Read the dataset and extract the feature names
        feature_names = list(pd.read_csv(csv_file_path).columns)[1:]
        selected_feature_names = [feature_names[idx] for idx in selected_indices]
        logger.debug(f"Selected feature names: {selected_feature_names}")

        plot_data = {"x": selected_feature_names, "y": selected_scores}
        y_label = kwargs.get("y_label", "Scores")

        # Apply the default theme
        sns.set_theme()
        ax = sns.barplot(data=plot_data, x="x", y="y")
        ax.set(xlabel="Feature Names", ylabel=y_label)
        plt.xticks(rotation=90)
        plt.savefig(str(plot_file_path))
        plt.close()
        logger.info(f"Feature selection plot saved to {plot_file_path}")


class FeatureReducerBase(FeatureEngineBase):
    """
    Base class for feature reduction, inheriting from FeatureEngineBase.
    """

    def fit(self, arr_input, arr_target, dataset_id="unset", force_fit=False, plot_dir=None) -> FeatureEngineResult:
        """
        Fit the feature reduction method to the given data.

        :param arr_input: Input feature array.
        :type arr_input: np.ndarray
        :param arr_target: Target array.
        :type arr_target: np.ndarray
        :param dataset_id: Identifier for the dataset.
        :type dataset_id: str
        :param force_fit: Whether to force refitting.
        :type force_fit: bool
        :param plot_dir: Directory to save plots, if applicable.
        :type plot_dir: PathLike
        :return: Result of the feature reduction.
        :rtype: FeatureEngineResult
        :raises AssertionError: If already fitted and force_fit is False.
        """
        if not force_fit:
            assert not self._is_fitted, "FeatureEngine already fitted. Set force_fit=True to refit."
        self.check_all_initialized()

        self._fit(arr_input, arr_target)
        reduction_info = self.get_reduction_info()
        logger.info("Feature reduction method fitted successfully.")

        self._is_fitted = True
        if plot_dir is not None:
            logger.info(f"Plotting reduction results to {plot_dir}")
            self.plot(arr_input, arr_target, reduction_info)

        return FeatureEngineResult(
            source_engine=self.__class__.__name__,
            dataset_id=dataset_id,
            reduction=reduction_info,
        )

    def _fit(self, arr_input, arr_target):
        """
        Internal method to fit the reduction technique.

        :param arr_input: Input feature array.
        :type arr_input: np.ndarray
        :param arr_target: Target array.
        :type arr_target: np.ndarray
        """
        return self._method.fit(arr_input, arr_target)

    def transform(self, arr_input, plot_dir=None):
        """
        Transform the input data using the fitted feature reduction method.

        :param arr_input: Input feature array to be transformed.
        :type arr_input: np.ndarray
        :param plot_dir: Directory to save plots, if applicable.
        :type plot_dir: PathLike, optional
        :return: Transformed input.
        :rtype: np.ndarray
        :raises AssertionError: If the method is not fitted.
        """
        assert self._is_fitted, "FeatureEngine not fitted. Please fit the method first."
        self.check_all_initialized()
        np_res = self._transform(arr_input)

        logger.debug("Input data transformed successfully for feature reduction.")

        if plot_dir is not None:
            logger.info(f"Plotting transformation results to {plot_dir}")
            # Placeholder for plot logic

        return np_res

    def _transform(self, arr_input):
        """
        Internal method to transform the input array.

        :param arr_input: Input feature array to be transformed.
        :type arr_input: np.ndarray
        :return: Transformed input.
        :rtype: np.ndarray
        """
        return self._method.transform(arr_input)

    def plot(self, arr_input, arr_target, reduction_info=None, **kwargs):
        """
        Plot method for visualizing feature reduction effects.

        :param arr_input: Input feature array.
        :type arr_input: np.ndarray
        :param arr_target: Target array.
        :type arr_target: np.ndarray
        :param reduction_info: Information about feature reduction.
        :type reduction_info: dict
        :param kwargs: Additional arguments for plotting.
        :type kwargs: dict
        """
        # Placeholder for plot logic
        logger.debug("Plotting feature reduction results.")

    def get_reduction_info(self):
        """
        Get information regarding the feature reduction.

        :return: Reduction information.
        :rtype: dict
        """
        return {}
