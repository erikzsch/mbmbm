import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR

from mbmbm import logger
from mbmbm.core.dataset_provider import DatasetProvider

PREDEFINED_MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "Lasso": Lasso(alpha=0.1),
}


def init_data_provider(dataset_id=None):
    logger.info("Init DatasetProvider")
    dataset_provider = DatasetProvider()
    available_ds_ids = dataset_provider.get_current_ids()
    if dataset_id is not None:
        assert dataset_id in available_ds_ids, f"{dataset_id} not found in available dataset ids: {available_ds_ids}"
    return dataset_provider


def save_prediction(arr_predict, predict_res_path):
    logger.info(f"Save predictions to file: {predict_res_path}")
    predict_res_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr_predict, list):
        if isinstance(arr_predict[0], torch.Tensor):
            arr_predict = torch.concat(arr_predict)
    arr_predict_np = np.array(arr_predict)
    np.save(predict_res_path, arr_predict_np)


def save_col_crop(col_crop_arr, save_path):
    logger.info(f"Save column cropping info to file: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, col_crop_arr=col_crop_arr)


def load_col_crop(load_path):
    logger.info(f"Load column cropping info from file: {load_path}")
    data = np.load(load_path)
    return data["col_crop_arr"]


def save_arr(arr_np, res_path):
    np.save(res_path, arr_np)
