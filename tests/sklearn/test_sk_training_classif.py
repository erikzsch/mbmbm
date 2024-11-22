import os

from hydra import compose, initialize_config_dir

from mbmbm.scripts.train import run_train
from mbmbm.utils.chdir import ChDir


def test_train_skl_classif(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_basic_classif"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            run_train(cfg)


def test_train_skl_classif_from_labels(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_label_classif"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            run_train(cfg)


def test_train_neural_classif(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_basic_neural_classif"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            run_train(cfg)
