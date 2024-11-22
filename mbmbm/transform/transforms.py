import bisect
from typing import List

import numpy as np

from mbmbm import logger
from mbmbm.transform.transformbase import TransformBase


class RemoveOutOfRangeColumnsTransform(TransformBase):
    """
    RemoveOutOfRangeColumnsTransform removes columns from a NumPy array that contain values
    outside a specified range.

    :param min_val: Minimum value allowed for a column
    :type min_val: float
    :param max_val: Maximum value allowed for a column
    :type max_val: float
    """

    def __init__(self, min_val, max_val):
        super().__init__()
        if min_val > max_val:
            raise ValueError("min_val must not be greater than max_val.")
        self.min_val = min_val
        self.max_val = max_val

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        # Find columns with values outside the given range
        cols_out_of_range = [
            col
            for col in range(arr.shape[1])
            if not np.all((arr[:, col] >= self.min_val) & (arr[:, col] <= self.max_val))
        ]
        # Remove columns with values outside the given range
        transformed_arr = np.delete(arr, cols_out_of_range, axis=1)
        return transformed_arr, None, cols_out_of_range


class RemoveOutOfRangeRowsTransform(TransformBase):
    """
    RemoveOutOfRangeRowsTransform removes rows from a NumPy array that contain values outside a specified range.

    :param min_val: Minimum value allowed for a row
    :type min_val: float
    :param max_val: Maximum value allowed for a row
    :type max_val: float
    """

    def __init__(self, min_val, max_val):
        super().__init__()
        if min_val > max_val:
            raise ValueError("min_val must not be greater than max_val.")
        self.min_val = min_val
        self.max_val = max_val

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        # Find rows with values outside the given range
        rows_out_of_range = [
            row
            for row in range(arr.shape[0])
            if not np.all((arr[row, :] >= self.min_val) & (arr[row, :] <= self.max_val))
        ]
        # Remove rows with values outside the given range
        transformed_arr = np.delete(arr, rows_out_of_range, axis=0)
        return transformed_arr, rows_out_of_range, None


class RemoveNaNColumnsTransform(TransformBase):
    """
    RemoveNaNColumnsTransform removes columns from a NumPy array that contain np.nan values.

    :param arr: Input array
    :type arr: np.ndarray
    :return: Transformed array, None, and list of removed column indices
    :rtype: (np.ndarray, None, List[int])
    """

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        # Detect columns with np.nan values
        cols_with_nan = np.any(np.isnan(arr), axis=0)
        # Remove columns with np.nan values
        transformed_arr = arr[:, ~cols_with_nan]
        return transformed_arr, None, list(np.where(cols_with_nan)[0])


class RemoveNaNRowsTransform(TransformBase):
    """
    RemoveNaNRowsTransform removes rows from a NumPy array that contain np.nan values.

    :param arr: Input array
    :type arr: np.ndarray
    :return: Transformed array, list of removed row indices, and None
    :rtype: (np.ndarray, List[int], None)
    """

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        # Detect rows with np.nan values
        if len(arr.shape) == 2:
            rows_with_nan = np.any(np.isnan(arr), axis=1)
            transformed_arr = arr[~rows_with_nan, :]
        elif len(arr.shape) == 1:
            rows_with_nan = np.isnan(arr)
            transformed_arr = arr[~rows_with_nan]
        else:
            raise Exception("Strange array shape")
        res_row_idxs = list(np.where(rows_with_nan)[0])
        res_row_idxs = res_row_idxs if len(res_row_idxs) > 0 else None
        return transformed_arr, res_row_idxs, None


class ReplaceNaNTransform(TransformBase):
    """
    ReplaceNaNTransform replaces np.nan values in the entire NumPy array with a specified value.

    :param value_to_replace_with: Value used to replace np.nan
    :type value_to_replace_with: float
    """

    def __init__(self, value_to_replace_with):
        super().__init__()
        self.value_to_replace_with = value_to_replace_with

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        # Replace None or np.nan with the given value
        where_nan = np.isnan(arr)
        transformed_arr = np.copy(arr)  # Make a copy to avoid changing the input array
        transformed_arr[where_nan] = self.value_to_replace_with
        return transformed_arr, None, None


class Binning2ClsIdxTransform(TransformBase):
    """
    Binning2ClsIdxTransform bins continuous float values into discrete categories.

    For bin_borders = [b_0, b_1, b_2, ..., b_n-1], the bins are defined as:
    bin_0 = {x | x < b_0}
    bin_1 = {x | b_0 <= x < b_1}
    ...
    bin_n = {x | b_n-1 <= x }

    Returns the bin index array in a zero or one based format.

    :param bin_borders: List of bin borders
    :type bin_borders: List[float]
    :param zero_base: If True, indices start from 0, otherwise from 1
    :type zero_base: bool
    """

    def __init__(self, bin_borders: List[float], zero_base=True):
        super().__init__()
        self.bin_borders = bin_borders
        self.zero_base = zero_base

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        binning = np.zeros(shape=arr.shape, dtype=int)
        if len(arr.shape) != 1:
            raise Exception("Dimension len has to be 1")
        for a_idx, a in enumerate(arr):
            binning[a_idx] = bisect.bisect_left(self.bin_borders, a)

        if len(set(list(binning))) != len(self.bin_borders) + 1:
            raise ValueError(
                f"Your target binning {self.bin_borders} is invalid for the given data. "
                f"Not all bins are covered by the data. "
                f"A model will not fit correctly. Adapt the bins."
                f"Current bin coverage (zero-based): {set(list(binning))}"
            )

        return binning, None, None


class RelativeBinning2ClsIdxTransform(TransformBase):
    """
    This transform bins continuous float values into discrete categories with index starting at 0 or 1.

    Relative data-dependent bin borders are used, specified as percentages or values in the range (0..1).

    :param rel_bin_borders: A list of relative bin borders as float values, either in percentage or in range (0..1).
    :type rel_bin_borders: List[float]
    :param zero_base: If True, the bin indices start at 0; otherwise, they start at 1.
    :type zero_base: bool

    :raises ValueError: If the input array is not one-dimensional.
    :raises ValueError: If the binning does not cover the entire data range.

    :return: The bin index array in zero or one-based format.
    """

    def __init__(self, rel_bin_borders: List[float], zero_base=True):
        super().__init__()
        self.rel_bin_borders = rel_bin_borders
        self.bin_borders = None
        self.zero_base = zero_base

    def fit(self, arr: np.ndarray):
        rbb = self.rel_bin_borders
        rel_bin_borders = rbb if np.max(rbb) <= 1.0 else [r / 100.0 for r in rbb]
        mi, ma = np.min(arr), np.max(arr)
        ra = ma - mi
        self.bin_borders = [mi + rb * ra for rb in rel_bin_borders]
        logger.info(f"Generated bin borders: {self.bin_borders}")

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        binning = np.zeros(shape=arr.shape, dtype=int)
        if len(arr.shape) != 1:
            raise Exception("Dimension len has to be 1")

        for a_idx, a in enumerate(arr):
            binning[a_idx] = bisect.bisect_left(self.bin_borders, a)

        if len(set(list(binning))) != len(self.bin_borders) + 1:
            raise ValueError(
                f"Your target binning {self.bin_borders} is invalid for the given data. "
                f"Not all bins are covered by the data. "
                f"A model will not fit correctly. Adapt the bins."
                f"Current bin coverage (zero-based): {set(list(binning))}"
            )

        return binning, None, None


class Label2ClsIdxTransform(TransformBase):
    """
    This transform bins continuous float values into discrete categories with index starting at 0 or 1.

    :param bin_borders: A list of bin borders defining the boundaries of the bins.
    :type bin_borders: List[float]
    :param zero_base: If True, the bin indices start at 0; otherwise, they start at 1.
    :type zero_base: bool

    :return: The bin index array in zero or one-based format.
    """

    def __init__(self, bin_borders: List[float], zero_base=True):
        super().__init__()
        self.bin_borders = bin_borders
        self.zero_base = zero_base

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        binning = np.zeros(shape=arr.shape, dtype=int)
        if len(arr.shape) != 1:
            raise Exception("Dimension len has to be 1")
        for a_idx, a in enumerate(arr):
            binning[a_idx] = bisect.bisect_left(self.bin_borders, a)

        return binning, None, None


class RarefyOTUFilter(TransformBase):
    """
    Filters out features (columns) from a NumPy array that are not present in at least a specified percentage
    of samples with a minimum relative abundance.

    :param min_samples_percent: The minimum percentage of samples in which a feature must appear.
    :type min_samples_percent: float
    :param min_abundance_percent: The minimum relative abundance percentage for the feature to be retained.
    :type min_abundance_percent: float

    :return: Filtered NumPy array, None, and a list of removed columns.
    """

    def __init__(self, min_samples_percent=1.0, min_abundance_percent=0.1):
        super().__init__()
        self.min_samples_percent = min_samples_percent
        self.min_abundance_percent = min_abundance_percent

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        num_samples = arr.shape[0]
        min_samples = np.ceil((self.min_samples_percent / 100) * num_samples).astype(int)
        min_abundance = self.min_abundance_percent / 100

        # Find columns that meet the criteria
        sums = np.sum(arr, axis=0)
        abundance_arr = np.divide(arr, sums, where=sums > 0)
        count_valid_samples = np.sum(abundance_arr >= min_abundance, axis=0)
        valid_columns = np.where(count_valid_samples >= min_samples)[0].tolist()

        # valid_columns = []
        # for col in range(arr.shape[1]):
        #     if np.max(arr[:, col])>0:
        #         abundance = arr[:, col] / np.sum(arr[:, col])
        #     else:
        #         abundance = arr[:, col]
        #     count_valid_samples = np.sum(abundance >= min_abundance)
        #     if count_valid_samples >= min_samples:
        #         valid_columns.append(col)

        # Filter out the invalid columns
        transformed_arr = arr[:, valid_columns]
        removed_columns = [col for col in range(arr.shape[1]) if col not in valid_columns]

        return transformed_arr, None, removed_columns


class MakeRelativeAbundance(TransformBase):
    """
    Turns count data into relative abundances, especially relevant for compositional OTU/ASV tables.

    :return: Array with counts converted to relative abundances.
    """

    def __init__(self):
        super().__init__()

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        arr = arr / arr.sum(axis=1)[:, None]
        return arr, None, None


class CenterLogRatio(TransformBase):
    """
    Performs centered log-ratio transformation, especially relevant for compositional OTU/ASV tables.

    :return: The transformed array with the centered log ratio.
    """

    def __init__(self):
        super().__init__()

    def transform(self, arr: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        arr = np.log(arr + np.finfo(arr.dtype).eps)
        arr -= arr.mean(1, keepdims=True)
        return arr, None, None
