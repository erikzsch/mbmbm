import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from mbmbm import BENCHMARK_RESULT_DIR_NAME, BENCHMARK_RESULT_OVERVIEW_NAME, logger
from mbmbm.scripts.predict import run_predict
from mbmbm.scripts.train import run_train
from mbmbm.scripts.val import run_eval
from mbmbm.utils.chdir import ChDir


class Benchmark:
    def __init__(self, configs_dir=None, workdir=None, out_dir=None, plot_dir=None):
        assert configs_dir is not None
        self.configs_dir = Path(configs_dir)
        self.configs_name_list = [e.stem for e in self.configs_dir.rglob("*.yaml")]
        self.workdir = Path(workdir) if workdir is not None else Path.cwd()
        self.out_dir = (Path(out_dir) if out_dir is not None else self.workdir) / BENCHMARK_RESULT_DIR_NAME
        self.plot_dir = Path(plot_dir) if plot_dir is not None else None
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run_trains(self):
        logger.info(f"Benchmarking configs in dir: {self.configs_dir}")
        with ChDir(self.workdir):
            config_dir = str(self.configs_dir)
            final_dict = {}
            for config_name in self.configs_name_list:
                logger.info(f"Run config: {config_name}")
                with initialize_config_dir(version_base=None, config_dir=config_dir):
                    cfg = compose(
                        config_name=config_name,
                        overrides=[],
                    )
                    with ChDir(self.workdir):
                        final_dict.update({config_name: run_train(cfg, plot_dir=self.plot_dir)})
            final_dict_path = self.out_dir / (BENCHMARK_RESULT_OVERVIEW_NAME + ".json")
            logger.info(f"Writing final results to: {final_dict_path}")
            final_dict_str = json.dumps(final_dict, sort_keys=True, indent=4)
            with final_dict_path.open("w") as f:
                f.write(final_dict_str)

    def run_predicts(self):
        logger.info(f"Benchmarking configs in dir: {self.configs_dir}")
        with ChDir(self.workdir):
            config_dir = str(self.configs_dir)
            for config_name in self.configs_name_list:
                logger.info(f"Run config: {config_name}")
                with initialize_config_dir(version_base=None, config_dir=config_dir):
                    res_dir = self.out_dir / config_name
                    res_dir.mkdir(exist_ok=True, parents=True)
                    cfg = compose(
                        config_name=config_name,
                        overrides=[f"++out_dir={self.out_dir}", f"++out_name={config_name}"],
                    )
                    with ChDir(self.workdir):
                        run_predict(cfg)

    def run_vals(self):
        logger.info(f"Benchmarking configs in dir: {self.configs_dir}")
        with ChDir(self.workdir):
            config_dir = str(self.configs_dir)
            for config_name in self.configs_name_list:
                logger.info(f"Run config: {config_name}")
                with initialize_config_dir(version_base=None, config_dir=config_dir):
                    res_dir = self.out_dir / config_name
                    res_dir.mkdir(exist_ok=True, parents=True)
                    cfg = compose(
                        config_name=config_name,
                        overrides=[f"++out_dir={self.out_dir}", f"++out_name={config_name}"],
                    )
                    with ChDir(self.workdir):
                        run_eval(
                            cfg,
                            plot_dir=self.plot_dir,
                            save_targets=True,
                            save_predictions=True,
                            insert_result_dirs=False,
                        )
