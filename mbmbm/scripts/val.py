import json
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score

from mbmbm import (
    CROP_SAVE_NAME,
    INPUT_RESULT_DIR_NAME,
    PREDICTION_RESULT_DIR_NAME,
    TARGET_RESULT_DIR_NAME,
    VALIDATION_RESULT_DIR_NAME,
    logger,
)
from mbmbm.core.modelbase import ModelBase
from mbmbm.featureengine.util import init_feature_engine
from mbmbm.scripts.utils import init_data_provider, load_col_crop, save_arr, save_prediction
from mbmbm.utils.logging import LogTime
from mbmbm.utils.plotting import plot_logged_times


def run_eval(cfg: DictConfig, save_inputs=False, save_predictions=False, save_targets=False, insert_result_dirs=True, plot_dir: Path = None):
    """Eval script"""

    logger.info(f"Start eval script")

    # timing
    time_dict = {}

    with LogTime("Overall calculations:", time_dict=time_dict, time_id="global"):
        _config_check(cfg)

        _out_dir = Path(cfg.out_dir)
        _out_name = cfg.out_name
        _val_load_dir = Path(cfg.val_load_dir)
        _dataset_id = cfg.dataset_id
        _dataset_config = cfg.dataset_config
        _dataset_stage = cfg.dataset_stage if "dataset_stage" in cfg else "test"
        _is_plotting = cfg.is_plotting if "is_plotting" in cfg else False
        _plot_dir = plot_dir if plot_dir else (_out_dir / _out_name / cfg.plot_subdir if "plot_subdir" in cfg and _is_plotting else None)

        logger.info(f"Plot directory set to: {_plot_dir}")

        if _plot_dir is not None:
            _plot_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Start Initializing")

        # data
        with LogTime("Data Loading:", time_dict=time_dict, time_id="data_load"):
            dataset_provider = init_data_provider(_dataset_id)
            arr_input, arr_target = dataset_provider.get_arrays(
                id=_dataset_id, config=_dataset_config, stage=_dataset_stage
            )

        crop_res_path = _val_load_dir / CROP_SAVE_NAME
        final_remaining_train_test_col = load_col_crop(crop_res_path)
        rm_col = [idx for idx in range(np.shape(arr_input)[1]) if idx not in final_remaining_train_test_col]
        arr_input = np.delete(arr_input, rm_col, axis=1)

        if "metrics" in cfg:
            logger.info(f"Initialize Metrics from config: {cfg.metrics}")
            metrics = MetricCollection(list(instantiate(cfg.metrics)))
        else:
            logger.info(f"Initialize default Metrics: MSE, MAE, R2")
            metrics = MetricCollection([MeanSquaredError(), MeanAbsoluteError(), R2Score()])

        model = ModelBase(load_dir=_val_load_dir)

        # init and apply feature engineerer
        feature_engine = None
        if "feature_engine" in cfg:
            feature_engine = init_feature_engine(cfg, model)

            # load state of feature engine from file
            logger.info(f"Load feature engine from: {_val_load_dir}")
            feature_engine.load(_val_load_dir, skip_checks=True)

            with LogTime("Apply Featureengine:", time_dict=time_dict, time_id="feature_engine_transform"):
                arr_input = feature_engine.transform(arr_input, plot_dir=_plot_dir)

        with LogTime("Predict:", time_dict=time_dict, time_id="model_predict"):
            arr_predict = model.predict(arr_input, plot_dir=_plot_dir)

        if save_predictions:
            logger.info(f"Saving predictions")
            predict_res_file_name = "prediction_" + datetime.now().strftime("%Y_%m_%e-%H_%M_%S").replace(" ", "0")
            res_dir = PREDICTION_RESULT_DIR_NAME if insert_result_dirs else "."
            predict_res_path = _out_dir / _out_name / res_dir / f"{predict_res_file_name}"
            save_prediction(arr_predict, predict_res_path)

        if save_targets:
            logger.info(f"Saving targets")
            target_res_file_name = "targets_" + datetime.now().strftime("%Y_%m_%e-%H_%M_%S").replace(" ", "0")
            res_dir = TARGET_RESULT_DIR_NAME if insert_result_dirs else "."
            target_res_path = _out_dir / _out_name / res_dir / f"{target_res_file_name}"
            save_arr(arr_target, target_res_path)

        if save_inputs:
            logger.info(f"Saving inputs")
            inputs_res_file_name = "inputs_" + datetime.now().strftime("%Y_%m_%e-%H_%M_%S").replace(" ", "0")
            res_dir = INPUT_RESULT_DIR_NAME if insert_result_dirs else "."
            inputs_res_path = _out_dir / _out_name / res_dir / f"{inputs_res_file_name}"
            save_arr(arr_input, inputs_res_path)

        with LogTime("Calculate metrics:", time_dict=time_dict, time_id="metrics_apply"):
            metrics_dict = metrics(torch.tensor(arr_predict).squeeze(), torch.tensor(arr_target).squeeze())
            metrics_dict = {k: float(v) for k, v in metrics_dict.items()}

    with LogTime("Accumulate results:"):
        for metric_name, metric_score in metrics_dict.items():
            logger.info(f"{metric_name}: {metric_score}")

        results_dict = {"metrics": metrics_dict, "model": model.get_infos(), "time": time_dict}
        if feature_engine is not None:
            results_dict.update({"selector": feature_engine.get_infos()})

    if _plot_dir is not None:
        plot_logged_times(time_dict=time_dict, plot_dir=_plot_dir)

    logger.info(f"Results:\n{json.dumps(results_dict, sort_keys=True, indent=4)}")

    val_res_file_name = datetime.now().strftime("%Y_%m_%e-%H_%M_%S").replace(" ", "0")
    val_res_path = _out_dir / _out_name / VALIDATION_RESULT_DIR_NAME / (val_res_file_name + ".json")
    val_res_path.parent.mkdir(parents=True, exist_ok=False)
    with val_res_path.open(mode="w") as f:
        json.dump(results_dict, f, sort_keys=True, indent=4)

    logger.info(f"Finished eval.")

    return results_dict


def _config_check(cfg):
    assert "out_dir" in cfg
    assert "out_name" in cfg
    assert "val_load_dir" in cfg
    assert "dataset_id" in cfg
    assert "dataset_config" in cfg


@hydra.main(config_path=str((Path.cwd() / "config").absolute()), config_name="default", version_base=None)
def main(cfg: DictConfig):
    """main

    Important: Use this for testing and building the config with hydra.initialize. Use hydra.compose to enable debugging
    """
    run_eval(cfg)


if __name__ == "__main__":
    main()
