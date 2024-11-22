import os

from hydra import compose, initialize_config_dir

from mbmbm.scripts.train import run_train
from mbmbm.scripts.val import run_eval
from mbmbm.utils.chdir import ChDir


def test_eval(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_basic"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            res_train = run_train(cfg)

        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            res_eval = run_eval(cfg)
    for k, v in res_eval["metrics"].items():
        assert v == res_train["metrics"][k]


def test_eval_with_transforms(test_resources, config_dir, tmp_path):
    config_dir = str(config_dir / "sklearn")
    config_name = "sklearn_basic_w_transforms"

    # initial model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            res_train = run_train(cfg)

        cfg = compose(
            config_name=config_name,
            overrides=[],
        )
        os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
        with ChDir(tmp_path):
            res_eval = run_eval(cfg)
    for k, v in res_eval["metrics"].items():
        assert v == res_train["metrics"][k]
