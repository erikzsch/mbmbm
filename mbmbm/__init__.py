import os
import sys
from pathlib import Path

from loguru import logger

logger.remove()


logger.add(sys.stdout, level=os.environ.get("MBMBM_LOG_LEVEL", "INFO"))
IS_TESTING = os.environ.get("MBMBM_IS_TESTING", "False") == "True"
if not IS_TESTING:
    log_dir = Path(os.environ.get("MBMBM_LOG_DIR", "."))
    if not log_dir.exists():
        raise NotADirectoryError(f"{log_dir=} does not exist!")
    logger.add(log_dir / "mbmbm_error.log", mode="w", level="ERROR")


default_data_key = "MBMBM_DATASET_DIR"

CROP_SAVE_NAME = "crop_info.npz"
FEATURE_ENGINE_SAVE_NAME = "feature_engine.pkl"
SK_MODEL_SAVE_NAME = "sk_model.pkl"
METRICS_SAVE_NAME = "metric_scores.csv"
RESULTS_SAVE_NAME = "results"
VALIDATION_RESULT_DIR_NAME = "validation_results"
PREDICTION_RESULT_DIR_NAME = "prediction_results"
TARGET_RESULT_DIR_NAME = "target_results"
INPUT_RESULT_DIR_NAME = "input_results"
BENCHMARK_RESULT_DIR_NAME = "benchmark_results"
BENCHMARK_RESULT_OVERVIEW_NAME = "benchmark_overview"
TRAIN_RESULT_DIR_NAME = "train_results"

LOGGED_TIMES_SAVE_NAME = "times.png"
