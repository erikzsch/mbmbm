import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mbmbm import default_data_key, logger

ALLOWED_STAGES = ["train", "test"]

DATA_SET_ERR_MSG = (
    f"Set the environment variable {default_data_key}. \n"
    f"Expected folder structure: \n"
    f"path-from-envvar/\n"
    f"\tdataset_id1/\n"
    f"\t\tconfig.yaml\n"
    f"\t\ttrain/\n"
    f"\t\t\tmetadata_train.csv\n"
    f"\t\t\totus_train.csv\n"
    f"\t\ttest/\n"
    f"\t\t\tmetadata_test.csv\n"
    f"\t\t\totus_test.csv\n"
    f"\tdataset_id2/\n"
    f"\t..."
)


class DatasetProvider:
    """
    Handles loading of local data for training and evaluation.

    Attributes
    ----------
    root : Path
        Path to the root directory containing dataset folders.
    id_path_map : dict
        A mapping from dataset ids to their respective directory paths.
    """

    def __init__(self, root=None):
        """
        Initialize DatasetProvider.

        Parameters
        ----------
        root : str, optional
            Path to the root directory of datasets. If None, uses the dataset root path from the
            environment variable defined by `mbmbm.default_data_key`.
        """
        logger.info("Initializing DatasetProvider.")
        if root:
            self.root = Path(root)
        else:
            default_datasets_path = os.environ.get(default_data_key, None)
            if default_datasets_path is None:
                logger.error("Dataset root path environment variable is not set.")
                raise EnvironmentError(DATA_SET_ERR_MSG)
            self.root = Path(default_datasets_path)

        self.id_path_map = {e.name: e for e in self.root.glob("*") if e.is_dir()}

        # Check dataset structure
        for id, path in self.id_path_map.items():
            subdir_dict = {"train": path / "train", "test": path / "test"}
            for subdir_name, subdir_path in subdir_dict.items():
                if not subdir_path.exists():
                    logger.error(f"{subdir_name} folder does not exist in {path}")
                    raise FileNotFoundError(f"{subdir_name} folder does not exist in {path}")

                for csv_fn in ["otus", "metadata"]:
                    current_file = f"{csv_fn}_{subdir_name}.csv"
                    if not (subdir_path / current_file).exists():
                        logger.error(f"{current_file} does not exist in {subdir_path}")
                        raise FileNotFoundError(f"{current_file} does not exist in {subdir_path}")
        logger.info("DatasetProvider initialized successfully.")

    def __len__(self):
        """
        Get the number of available dataset IDs.

        Returns
        -------
        int
            Number of available dataset IDs.
        """
        return len(self.id_path_map)

    def get_current_ids(self):
        """
        Get the currently available dataset IDs.

        Returns
        -------
        list of str
            List of dataset IDs available.
        """
        logger.debug("Fetching current dataset IDs.")
        return list(self.id_path_map.keys())

    def get_configs(self, id):
        """
        Get the available configurations for a given dataset ID.

        Parameters
        ----------
        id : str
            The dataset ID for which to get the configuration.

        Returns
        -------
        dict
            Dictionary containing configuration parameters from `config.yaml`.

        Raises
        ------
        AssertionError
            If the dataset ID is unknown or the configuration file is missing.
        """
        logger.info(f"Getting configurations for dataset ID: {id}")
        if id not in self.id_path_map:
            logger.error(f"Unknown dataset ID: {id}")
            raise KeyError(f"Unknown dataset ID. Known IDs: {self.get_current_ids()}")
        ds_path = Path(self.id_path_map[id])
        config_path = ds_path / "config.yaml"
        if not config_path.exists():
            logger.error(f"Configuration file `config.yaml` does not exist in {ds_path}")
            raise FileNotFoundError(f"Configuration file `config.yaml` does not exist in {ds_path}")
        with config_path.open(mode="r") as f:
            json_dict = yaml.safe_load(f)
        logger.debug(f"Configurations for dataset ID {id} loaded successfully.")
        return json_dict

    def _get_dataset_info(self, config, id, include_targets, stage):
        """
        Get detailed dataset information for a given dataset ID and stage.

        Parameters
        ----------
        config : str
            Configuration name to use.
        id : str
            The dataset ID.
        include_targets : bool
            Whether to include target data.
        stage : str
            The stage of the dataset ('train' or 'test').

        Returns
        -------
        tuple
            A tuple containing configuration data, target CSV path, training CSV path, input range, and label map.

        Raises
        ------
        KeyError
            If the dataset ID or configuration is not found.
        Exception
            If an unknown `input_col_idxs` type is encountered.
        """
        logger.info(f"Reading dataset (ID={id}) for config='{config}' and stage='{stage}'")
        assert stage in ALLOWED_STAGES, f"Invalid stage: {stage}. Must be one of {ALLOWED_STAGES}"
        ds_path = self.id_path_map.get(id)
        if not ds_path:
            logger.error(f"Dataset ID '{id}' not found.")
            raise KeyError(f"Dataset ID '{id}' not found.")
        configs = self.get_configs(id)
        if config not in configs:
            logger.error(f"Configuration '{config}' not found for dataset ID '{id}'")
            raise KeyError(f"Configuration '{config}' not found for dataset ID '{id}'")
        config_data = configs[config]

        input_col_type = config_data["input_col_idxs"]["type"]
        if input_col_type == "range":
            input_ra = list(range(config_data["input_col_idxs"]["min"], config_data["input_col_idxs"]["max"]))
        elif input_col_type == "idx_list":
            input_ra = [int(e) for e in config_data["input_col_idxs"]["data"]]
        elif input_col_type == "full":
            input_ra = None
        else:
            logger.error(f"Unknown input_col_idxs type: {input_col_type}")
            raise Exception(f"Unknown input_col_idxs type: {input_col_type}")

        # Ensure the leading index column is included
        if input_ra is not None and 0 not in input_ra:
            logger.warning("Inserting the leading column for train DataFrame column selection.")
            input_ra.insert(0, 0)

        label_map = config_data.get("label_map", None)

        csv_path_train = str(ds_path / stage / f"otus_{stage}.csv")
        csv_path_target = str(ds_path / stage / f"metadata_{stage}.csv") if include_targets else None
        return config_data, csv_path_target, csv_path_train, input_ra, label_map

    def get_dataframes(self, id, config="default", stage="train", include_targets=True):
        """
        Get dataframes for a given dataset ID, configuration, and stage.

        Parameters
        ----------
        id : str
            The dataset ID.
        config : str, optional
            The configuration name to use (default is 'default').
        stage : str, optional
            The stage of the dataset ('train' or 'test', default is 'train').
        include_targets : bool, optional
            Whether to include target data (default is True).

        Returns
        -------
        tuple of pandas.DataFrame
            A tuple containing the input dataframe and target dataframe (if `include_targets` is True).
        """
        logger.info(f"Getting dataframes for dataset ID='{id}', config='{config}', stage='{stage}'")
        config_data, csv_path_target, csv_path_train, input_ra, label_map = self._get_dataset_info(
            config, id, include_targets, stage
        )

        df_train = pd.read_csv(csv_path_train, index_col=0, usecols=input_ra)
        logger.info(f"Input dataframe loaded with {len(df_train)} rows.")

        if include_targets:
            df_target_onerow = pd.read_csv(csv_path_target, index_col=0, nrows=1)
            ra = [0, df_target_onerow.columns.get_loc(config_data["target_col_name"]) + 1]
            df_target = pd.read_csv(csv_path_target, index_col=0, usecols=ra)
            logger.info(f"Target dataframe loaded with {len(df_target)} rows.")
            return df_train, df_target
        else:
            return df_train, None

    def get_arrays(self, id, config="default", stage="train", include_targets=True):
        """
        Get numpy arrays for a given dataset ID, configuration, and stage.

        Parameters
        ----------
        id : str
            The dataset ID.
        config : str, optional
            The configuration name to use (default is 'default').
        stage : str, optional
            The stage of the dataset ('train' or 'test', default is 'train').
        include_targets : bool, optional
            Whether to include target data (default is True).

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing the input array and target array (if `include_targets` is True).
        """
        logger.info(f"Getting arrays for dataset ID='{id}', config='{config}', stage='{stage}'")
        config_data, csv_path_target, csv_path_train, input_ra, label_map = self._get_dataset_info(
            config, id, include_targets, stage
        )

        arr_train = np.genfromtxt(csv_path_train, delimiter=",", skip_header=1, usecols=input_ra)[:, 1:]
        logger.info(f"Input array loaded with shape: {arr_train.shape}")

        if include_targets:
            df_target_onerow = pd.read_csv(csv_path_target, index_col=0, nrows=1)
            target_ra = [df_target_onerow.columns.get_loc(config_data["target_col_name"]) + 1]
            if len(target_ra) != 1:
                logger.error("Only one target column is allowed.")
                raise ValueError("Only one target column is allowed.")

            if label_map is not None:
                target_col_list = []
                with Path(csv_path_target).open("r") as f:
                    reader = csv.reader(f, delimiter=",")
                    for row_idx, row in enumerate(reader):
                        if row_idx == 0:
                            continue
                        target_col_list.append(row[target_ra[0]])
                target_col_list_replaced = []
                for e in target_col_list:
                    if e not in label_map.keys():
                        logger.error(
                            f"Target entry '{e}' not found in label map {label_map} "
                            f"of dataset config '{config}' in dataset '{id}'"
                        )
                        raise KeyError(
                            f"Target entry '{e}' not found in label map {label_map} "
                            f"of dataset config '{config}' in dataset '{id}'"
                        )
                    target_col_list_replaced.append(label_map[e])
                arr_target = np.array(target_col_list_replaced)
            else:
                arr_target = np.genfromtxt(csv_path_target, delimiter=",", skip_header=1, usecols=target_ra)

            logger.info(f"Target array loaded with shape: {arr_target.shape}")
            return arr_train, arr_target
        else:
            return arr_train, None
