import argparse
import multiprocessing
import os
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

from hydra import compose, initialize_config_dir

from mbmbm import logger
from mbmbm.scripts.train import run_train


def run():
    """
    Entry point for the script. Parses arguments and calls the main function.
    """
    main(parse_args())


def run_task(task_args: Tuple[int, Path, Path, Path, bool]) -> None:
    """
    Executes a single training task based on the provided configuration.

    Args:
        task_args (tuple): A tuple containing the task index, config path, output root path,
                           configuration root path, and dryrun flag.
    """
    task_idx, config, out_root, config_root, dryrun = task_args
    logger.success(f"Processing id={task_idx}: {config}")
    try:
        with initialize_config_dir(version_base=None, config_dir=str(config.parent)):
            out_dir = out_root / Path(config).relative_to(config_root).parent / str(Path(config).stem).replace(".", "_")
            if (out_dir / "results" / "results.json").exists():
                logger.warning(f"Result already exists. Skipping id={task_idx}: {config}")
                return

            cfg = compose(
                config_name=config.stem,
                overrides=[
                    f"++out_dir={out_dir}",
                    f"++val_load_dir={out_dir}/results",
                    f"++out_name=results",
                ],
            )
            run_train(cfg, dryrun)
    except Exception:
        logger.error(f"\n\n# Something went wrong ####################################################################")
        logger.error(f"{config=}")
        logger.error(f"{cfg.dataset_id=}")
        logger.error(traceback.format_exc())
        logger.error(f"###########################################################################################\n\n")


def main(args):
    """
    Main function to orchestrate the processing of configurations.

    Args:
        args: Parsed command-line arguments.
    """

    root = Path(args.root).absolute()
    data_root = root / args.data
    os.environ["MBMBM_DATASET_DIR"] = str(data_root)

    out_root = root / "results"
    config_root = root / args.configs
    configs_list = list(config_root.rglob("*.yaml"))

    # dataset_id_list = [e.stem for e in data_root.glob("*")]

    # all_combinations_list = list(product(dataset_id_list, configs_list))
    # task_args_list = [
    #     (dataset_id, config, out_root, config_root, args.dryrun) for dataset_id, config in all_combinations_list
    # ]

    all_combinations_list = configs_list
    task_args_list = [(config, out_root, config_root, args.dryrun) for config in all_combinations_list]

    logger.info(f"Found {len(all_combinations_list)} tasks based on dataset and config combinations")

    logger.info(all_combinations_list)
    if args.worker > 1:
        pool = multiprocessing.Pool(args.worker)
        pool.map(
            run_task,
            [
                (task_idx, config, out_root, config_root, dryrun)
                for task_idx, (config, out_root, config_root, dryrun) in enumerate(task_args_list)
            ],
        )

    else:
        for task_idx, (config, out_root, config_root, dryrun) in enumerate(task_args_list):
            run_task((task_idx, config, out_root, config_root, dryrun))

    logger.info("done")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses command-line arguments.

    Args:
        args (list, optional): List of arguments for testing purposes. Defaults to None.

    Returns:
        Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", "-r", type=str, required=True, help="Root directory for the project.")
    parser.add_argument("--worker", "-w", default=1, type=int, help="Number of parallel workers to process tasks.")
    parser.add_argument(
        "--data", "-d", default="data", type=str, help="Relative path to the data directory from the root."
    )
    parser.add_argument(
        "--configs",
        "-c",
        default="configs",
        type=str,
        help="Relative path to the configuration directory from the root.",
    )
    parser.add_argument(
        "--dryrun", action="store_true", help="If set, perform a dry run without executing the actual training."
    )

    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    run()
