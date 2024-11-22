import os

from hydra import compose, initialize_config_dir

from mbmbm.scripts.train import run_train
from mbmbm.utils.chdir import ChDir


def test_dimred_nmf(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_dimred_nmf"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            run_train(cfg)


def test_dimred_lda(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_dimred_lda"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            run_train(cfg)


def test_dimred_pca(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_dimred_pca"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            run_train(cfg)


def test_dimred_nmds(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_dimred_nmds"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            run_train(cfg)
