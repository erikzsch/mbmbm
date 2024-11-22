from os import PathLike
from pathlib import Path
from pickle import dump, load
from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression

from mbmbm import SK_MODEL_SAVE_NAME, logger


class ModelBase:
    """
    Model base class that implements an interface and basic actions for fitting, predicting,
    saving, and loading machine learning models.

    :param model: Machine learning model, defaults to LinearRegression if None.
    :type model: scikit-learn model, optional
    :param load_dir: Directory path from which to load a pre-trained model.
    :type load_dir: PathLike, optional
    :param skl_model: Indicates if the model is a scikit-learn model, defaults to True.
    :type skl_model: bool, optional
    :param is_classif: Specifies if the model is a classifier, defaults to False.
    :type is_classif: bool, optional
    """

    def __init__(self, model=None, load_dir: PathLike = None, skl_model=True, is_classif=False):
        self.skl_model = skl_model
        self.is_classif = is_classif
        if load_dir is not None:
            self._model = None
            load_dir = Path(load_dir)
            self.load(load_dir / SK_MODEL_SAVE_NAME, skip_checks=True)
            self._is_fitted = True
            logger.info(f"Model loaded successfully from {load_dir}.")
        else:
            self._model = model if model is not None else LinearRegression()
            self._is_fitted = False
            logger.debug("Initialized a new model instance.")

    def fit(self, arr_input, arr_target, force_fit=False, plot_dir=None):
        """
        Train the model based on input and target pairs.

        :param arr_input: Input features for training the model.
        :type arr_input: array-like
        :param arr_target: Target values for training the model.
        :type arr_target: array-like
        :param force_fit: If True, forces refitting even if already fitted, defaults to False.
        :type force_fit: bool, optional
        :param plot_dir: Directory path for saving plots, defaults to None.
        :type plot_dir: PathLike, optional
        :return: The fitted model.
        :rtype: scikit-learn model instance
        """
        if not force_fit:
            assert not self._is_fitted, "Model already fitted"
        res = self._model.fit(np.float32(arr_input), arr_target)
        self._is_fitted = True
        logger.info("Model training complete.")
        if plot_dir is not None:
            logger.debug("Plotting training results...")
            # Call plotting methods here, located under mbmbm.utils.plotting
            ...
        return res

    def predict(self, arr_input, plot_dir=None):
        """
        Use the trained model to make predictions for input data.

        :param arr_input: Input features for prediction.
        :type arr_input: array-like
        :param plot_dir: Directory path for saving plots, defaults to None.
        :type plot_dir: PathLike, optional
        :return: Predicted values.
        :rtype: array-like
        """
        assert self._is_fitted, "Model not fitted. Fit the model first."
        arr_output = (
            self._model.predict_proba(np.float32(arr_input))
            if self.is_classif
            else self._model.predict(np.float32(arr_input))
        )
        logger.info("Prediction complete.")
        if plot_dir is not None:
            logger.debug("Plotting prediction results...")
            # Call plotting methods here, located under mbmbm.utils.plotting
            ...
        return arr_output

    def get_infos(self) -> Dict[str, str]:
        """
        Returns general model information for saving.

        :return: Dictionary containing model information.
        :rtype: Dict[str, str]
        """
        info = {"cls": self._model.__class__.__name__}
        if hasattr(self._model, "get_infos"):
            prefix = "model"
            info.update({f"{prefix}.{k}": v for k, v in self._model.get_infos().items()})
        logger.debug("Model information retrieved.")
        return info

    def save(self, out_dir: PathLike):
        """
        Save the model to a local file.

        :param out_dir: Directory path to save the model.
        :type out_dir: PathLike
        """
        assert self._is_fitted, "Only fitted models can be saved."
        out_path = Path(out_dir) / SK_MODEL_SAVE_NAME
        with out_path.open(mode="wb") as f:
            dump(self._model, f)
        logger.info(f"Model saved successfully to {out_path}.")

    def load(self, filepath: PathLike, skip_checks=False):
        """
        Load the model from a local file.

        :param filepath: File path from which to load the model.
        :type filepath: PathLike
        :param skip_checks: If True, skips model compatibility checks, defaults to False.
        :type skip_checks: bool, optional
        """
        filepath = Path(filepath)
        assert filepath.exists(), f"File {filepath} does not exist."
        with filepath.open(mode="rb") as f:
            self._model = load(f)
        new_model_cls = self._model.__class__.__name__

        if not skip_checks:
            current_model_cls = self._model.__class__.__name__
            assert current_model_cls == new_model_cls, (
                f"Loaded model and class-defined model mismatch.\n" f"{current_model_cls} != {new_model_cls}"
            )
        logger.info(f"Model loaded successfully from {filepath}.")
