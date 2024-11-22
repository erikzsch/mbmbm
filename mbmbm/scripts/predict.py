from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from mbmbm import CROP_SAVE_NAME, PREDICTION_RESULT_DIR_NAME, logger
from mbmbm.core.modelbase import ModelBase
from mbmbm.featureengine.util import init_feature_engine
from mbmbm.scripts.utils import init_data_provider, load_col_crop, save_prediction
from mbmbm.utils.logging import LogTime


def run_predict(cfg: DictConfig):
    """Model predict script"""

    logger.info(f"Start model predict script")

    # timing
    time_dict = {}

    with LogTime("Overall:", time_dict=time_dict, time_id="global"):
        _config_check(cfg)

        _out_dir = Path(cfg.out_dir)
        _out_name = cfg.out_name
        _predict_load_dir = Path(cfg.predict_load_dir)

        _dataset_id = cfg.dataset_id
        _dataset_config = cfg.dataset_config
        _dataset_stage = cfg.dataset_stage if "dataset_stage" in cfg else "test"

        logger.info("Start Initializing")

        # data
        with LogTime("Data Loading:", time_dict=time_dict, time_id="data_load"):
            dataset_provider = init_data_provider(_dataset_id)
            arr_input, _ = dataset_provider.get_arrays(
                id=_dataset_id, config=_dataset_config, stage=_dataset_stage, include_targets=False
            )

        crop_res_path = _predict_load_dir / CROP_SAVE_NAME
        final_remaining_train_test_col = load_col_crop(crop_res_path)
        rm_col = [idx for idx in range(np.shape(arr_input)[1]) if idx not in final_remaining_train_test_col]
        arr_input = np.delete(arr_input, rm_col, axis=1)

        logger.info(f"Init and load model")
        model = ModelBase(load_dir=_predict_load_dir)

        if "feature_engine" in cfg:
            logger.info(f"Init and load FeatureEngine")
            feature_engine = init_feature_engine(cfg, model)
            logger.info(f"Load FeatureEngine from: {_predict_load_dir}")
            feature_engine.load(_predict_load_dir, skip_checks=True)

            logger.info(f"Apply FeatureEngine")
            with LogTime(
                "Transforming train and test input data:", time_dict=time_dict, time_id="feature_engine_transform"
            ):
                arr_input = feature_engine.transform(arr_input)
                # logger.info(f"{df_input.size=}")

        with LogTime("Apply Model:", time_dict=time_dict, time_id="model_predict"):
            arr_predict = model.predict(arr_input)

        logger.info(f"Saving predictions")
        predict_res_file_name = "prediction_" + datetime.now().strftime("%Y_%m_%e-%H_%M_%S").replace(" ", "0")
        predict_res_path = _out_dir / _out_name / PREDICTION_RESULT_DIR_NAME / f"{predict_res_file_name}"
        save_prediction(arr_predict, predict_res_path)

    logger.info(f"Finished prediction.")


def _config_check(cfg):
    assert "predict_load_dir" in cfg
    assert len(cfg.predict_load_dir) > 0


@hydra.main(config_path=str((Path.cwd() / "config").absolute()), config_name="reducer_default", version_base=None)
def main(cfg: DictConfig):
    """main

    Important: Use this for testing and building the config with hydra.initialize. Use hydra.compose to enable debugging
    """
    run_predict(cfg)


if __name__ == "__main__":
    main()
