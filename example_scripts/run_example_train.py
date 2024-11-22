"""
This script runs a training for a given config.
"""

import os
from pathlib import Path

from hydra import compose, initialize_config_dir

from mbmbm.scripts.train import run_train
from mbmbm.utils.chdir import ChDir

# folder that contains configs
config_dir = Path(__file__).parent / "config"
# config name without extension
config_name = "default"
data_path = Path.home() / "devel" / "projects" / "otcg-paper" / "data"
# workdir, which will contain the results
# model results will be saved here
workdir_path = Path.home() / "devel" / "projects" / "otcg-paper"

###

# set data path environment variable. Not needed, if it is set system-wide
os.environ["MBMBM_DATASET_DIR"] = str(data_path)

# init hydra config management
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(
        config_name=config_name,
        overrides=[],
    )
    # change current directory to workdir
    with ChDir(workdir_path):
        # run train script
        run_train(cfg)

if __name__ == "__main__":
    ...
