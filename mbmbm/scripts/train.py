import csv
import json
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score

from mbmbm import CROP_SAVE_NAME, METRICS_SAVE_NAME, RESULTS_SAVE_NAME, logger
from mbmbm.core.modelbase import ModelBase
from mbmbm.featureengine.featureenginebase import FeatureEngineResult
from mbmbm.featureengine.util import init_feature_engine
from mbmbm.scripts.utils import PREDEFINED_MODELS, init_data_provider, save_col_crop
from mbmbm.transform.transformbase import TransformBase
from mbmbm.utils.logging import LogTime, close_file_logger, setup_file_logger
from mbmbm.utils.plotting import plot_logged_times


def run_train(cfg: DictConfig, dryrun=False, plot_dir=None):
    """train script"""

    logger.info(f"Start train script")

    # timing
    time_dict = {}

    with LogTime("Overall calculations:", time_dict=time_dict, time_id="total"):
        _config_check(cfg)

        # set optional defaults
        _out_dir = Path(cfg.out_dir) if "out_dir" in cfg else Path("models")
        _out_name = cfg.out_name if "out_name" in cfg else "tmp"
        _log_level = cfg.log_level if "log_level" in cfg else "INFO"
        _dataset_id = cfg.dataset_id if "dataset_id" in cfg else None
        _dataset_config = cfg.dataset_config if "dataset_config" in cfg else "default"
        _feature_engine_dir = Path(cfg.feature_engine_dir) if "feature_engine_dir" in cfg else None
        _is_plotting = cfg.is_train_plotting if "is_train_plotting" in cfg else False
        _plot_dir = plot_dir or (
            _out_dir / _out_name / cfg.plot_subdir if "plot_subdir" in cfg and _is_plotting else None
        )
        if _plot_dir is not None:
            _plot_dir.mkdir(exist_ok=True, parents=True)

        log_handler_id = setup_file_logger(file_path=Path.cwd() / _out_dir / _out_name / "train.log", level=_log_level)
        specific_logger = logger.bind(handler_id=log_handler_id)

        specific_logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

        # check that model save infos are set with something
        assert _out_dir is not None
        assert _out_name is not None
        assert _dataset_id is not None

        specific_logger.info("Start Initializing")

        if dryrun:
            if "target_preproc" in cfg:
                instantiate(cfg.target_preproc)
            if "preproc" in cfg:
                instantiate(cfg.preproc)
            instantiate(cfg.model)
            instantiate(cfg.metrics)
            close_file_logger(log_handler_id)
            return

        with LogTime("Data Loading (arr):", time_dict=time_dict, time_id="data_load"):
            dataset_provider = init_data_provider(_dataset_id)
            arr_train, arr_train_target = dataset_provider.get_arrays(
                id=_dataset_id, config=_dataset_config, stage="train"
            )
            arr_test, arr_test_target = dataset_provider.get_arrays(
                id=_dataset_id, config=_dataset_config, stage="test"
            )

        ori_row_nr = np.shape(arr_train)[0]
        ori_col_nr = np.shape(arr_train)[1]
        final_remaining_train_test_col = np.array(list(range(np.shape(arr_train)[1])))
        final_remaining_train_row = np.array(list(range(np.shape(arr_train)[0])))

        # target transforms
        with LogTime("Target transforms:", time_dict=time_dict, time_id="transform_target"):
            if "target_preproc" in cfg:
                specific_logger.info("cfg.target_preproc found. Instantiate transforms...")
                target_transform_list: List[TransformBase] = instantiate(cfg.target_preproc)
                specific_logger.info(f"{len(target_transform_list)} transforms instantiated.")
            else:
                target_transform_list: List[TransformBase] = []

            specific_logger.info(f"Run {len(target_transform_list)} target transforms")
            for tr in target_transform_list:
                specific_logger.info(f"Apply {tr.__class__.__name__}")
                tr.fit(arr_train_target)
                arr_train_target, rm_train_target_row, rm_train_target_col = tr.transform(arr_train_target)
                arr_test_target, rm_test_target_row, rm_test_target_col = tr.transform(arr_test_target)

                if any([e is not None for e in [rm_train_target_col, rm_test_target_col]]):
                    raise ValueError("Col removal is not allowed for target transforms.")
                if any([e is not None for e in [rm_test_target_row]]):
                    raise ValueError("Row removal is not allowed for test target transforms.")
                if rm_train_target_row is not None:
                    # rm rows from target if needed
                    arr_train = np.delete(arr_train, rm_train_target_row, axis=0)
                    final_remaining_train_row = np.delete(final_remaining_train_row, rm_train_target_row, axis=0)
                    specific_logger.warning(
                        f"Removed {len(rm_train_target_row)} rows from train input and target "
                        f"due to preproc transforms."
                    )

        # data transforms
        with LogTime("Input transforms:", time_dict=time_dict, time_id="transform_input"):
            if "preproc" in cfg:
                specific_logger.info("cfg.preproc found. Instantiate transforms...")
                transform_list: List[TransformBase] = instantiate(cfg.preproc)
                specific_logger.info(f"{len(transform_list)} transforms instantiated.")
            else:
                transform_list: List[TransformBase] = []

            for tr in transform_list:
                specific_logger.info(f"Apply {tr.__class__.__name__}")
                tr.fit(arr_train)
                arr_train, rm_train_row, rm_train_col = tr.transform(arr_train)
                if rm_train_row is not None:
                    # rm rows from target if needed
                    arr_train_target = np.delete(arr_train_target, rm_train_row, axis=0)
                    final_remaining_train_row = np.delete(final_remaining_train_row, rm_train_row, axis=0)
                    specific_logger.warning(
                        f"Removed {len(rm_train_row)} rows from train input and target due to preproc transforms."
                    )
                if rm_train_col is not None:
                    # rm cols from test target if needed
                    arr_test = np.delete(arr_test, rm_train_col, axis=1)
                    final_remaining_train_test_col = np.delete(final_remaining_train_test_col, rm_train_col, axis=0)
                    specific_logger.debug(
                        f"Removed {len(rm_train_col)} cols from train and test input due to preproc transforms."
                    )

            assert (
                len(final_remaining_train_row) > 0
            ), f"Row count after transformations is 0. Checks transform settings."
            assert (
                len(final_remaining_train_test_col) > 0
            ), f"Column count after transformations is 0. Checks transform settings."
            if len(final_remaining_train_row) / ori_row_nr < 0.1:
                specific_logger.warning(f"Row count reduced to 10% after transformation")
            if len(final_remaining_train_test_col) / ori_col_nr < 0.1:
                specific_logger.warning(f"Column count reduced to 10% after transformation")
            if len(final_remaining_train_row) < 20:
                specific_logger.warning(f"Row count reduced to below 20 after transformation")
            if len(final_remaining_train_test_col) < 20:
                specific_logger.warning(f"Column count reduced to below 20 after transformation")

        crop_res_path = _out_dir / _out_name / CROP_SAVE_NAME
        save_col_crop(final_remaining_train_test_col, crop_res_path)

        is_classif_model = cfg.is_classif_model if "is_classif_model" in cfg else False
        assert "model" in cfg
        if isinstance(cfg.model, str):
            specific_logger.info("cfg.model is a string. Load from PREDEFINED_MODELS dict")
            assert cfg.model in PREDEFINED_MODELS, f"cannot find cfg.model in PREDEFINED_MODELS"
            model = ModelBase(model=PREDEFINED_MODELS[cfg.model], is_classif=is_classif_model)
        else:
            specific_logger.info("cfg.model is not a string. Initialize from DictConfig")
            model = ModelBase(model=instantiate(cfg.model), is_classif=is_classif_model)

        # init and apply feature engineerer
        feature_engine = None
        if "feature_engine" in cfg:
            feature_engine = init_feature_engine(cfg, model)
            if _feature_engine_dir is not None:
                # load state of feature engine from file
                specific_logger.info(f"Load feature engine from: {_feature_engine_dir}")
                feature_engine.load(_feature_engine_dir)
            else:
                # fit feature engine
                feature_engine.init_method()

                specific_logger.info("Start feature engine fit")
                with LogTime(caption="Feature engine fit:", time_dict=time_dict, time_id="feature_engine_fit"):
                    feature_engine_result: FeatureEngineResult = feature_engine.fit(
                        arr_train, arr_train_target, dataset_id=_dataset_id, plot_dir=_plot_dir
                    )
                specific_logger.info(f"{feature_engine_result=}")
            feature_engine.save(out_dir=_out_dir / _out_name)

            with LogTime(
                caption="Transforming train and test input data:",
                time_dict=time_dict,
                time_id="feature_engine_transform",
            ):
                arr_train = feature_engine.transform(arr_train, plot_dir=_plot_dir)
                arr_test = feature_engine.transform(arr_test, plot_dir=_plot_dir)
                specific_logger.info(f"{arr_train.shape=}")

        if "metrics" in cfg:
            specific_logger.info(f"Initialize Metrics from config: {cfg.metrics}")
            metrics = MetricCollection(list(instantiate(cfg.metrics)))
        else:
            specific_logger.info(f"Initialize default Metrics: MSE, MAE, R2")
            metrics = MetricCollection([MeanSquaredError(), MeanAbsoluteError(), R2Score()])

        with LogTime("Fit model:", time_dict=time_dict, time_id="model_fit"):
            model.fit(arr_train, arr_train_target, plot_dir=_plot_dir)
            model.save(out_dir=_out_dir / _out_name)

        with LogTime("Predict test:", time_dict=time_dict, time_id="model_predict_test"):
            arr_test_predict = model.predict(arr_test, plot_dir=_plot_dir)

        with LogTime("Calculate metrics:", time_dict=time_dict, time_id="metrics_apply"):
            if isinstance(arr_test_predict, list):
                arr_test_predict = torch.concat(arr_test_predict)
            metrics_dict = metrics(torch.tensor(arr_test_predict).squeeze(), torch.tensor(arr_test_target.squeeze()))
            metrics_dict = {k: float(v) for k, v in metrics_dict.items()}

    with LogTime("Accumulate results:", time_dict=time_dict, time_id="metrics_save"):
        for metric_name, metric_score in metrics_dict.items():
            specific_logger.info(f"{metric_name}: {metric_score}")
        with (_out_dir / _out_name / METRICS_SAVE_NAME).open(mode="w") as f_csv:
            w = csv.writer(f_csv)
            w.writerows(metrics_dict.items())

        results_dict = {"metrics": metrics_dict, "model": model.get_infos(), "time": time_dict}
        if feature_engine is not None:
            results_dict.update({"selector": feature_engine.get_infos()})

    result_dump = json.dumps(results_dict, sort_keys=True, indent=4)
    specific_logger.info(f"Results:\n{result_dump}")
    with (_out_dir / _out_name / (RESULTS_SAVE_NAME + ".json")).open(mode="w") as f_res:
        f_res.write(result_dump)

    if _plot_dir is not None:
        plot_logged_times(time_dict=time_dict, plot_dir=_plot_dir)

    specific_logger.info(f"Finished training.")
    close_file_logger(log_handler_id)
    return results_dict


def _config_check(cfg):
    ...


@hydra.main(config_path=str((Path.cwd() / "config").absolute()), config_name="default", version_base=None)
def main(cfg: DictConfig):
    """main

    Important: Use this for testing and building the config with hydra.initialize. Use hydra.compose to enable debugging
    """
    run_train(cfg)


if __name__ == "__main__":
    main()
