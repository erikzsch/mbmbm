from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class FastCorrelationBasedFilter(SelectorMixin, BaseEstimator):
    def __init__(self, threshold: float = 0.01):
        """
        Initialize the FastCorrelationBasedFilter.

        Parameters:
        - threshold (float): The threshold for feature selection.
        """
        self.threshold = threshold
        self.scores_: List[float] = []
        self.len_input = None

    def transform(self, arr_input: np.ndarray):
        return arr_input[:, self.idx_sel_]

    def fit(self, arr_input: np.ndarray, arr_target: np.ndarray) -> Any:
        """
        Fit the FastCorrelationBasedFilter to the data and select relevant features.

        Parameters:
        - X (pd.DataFrame): The input data with features.
        - y (pd.Series): The target variable.

        Returns:
        - self
        """

        self.idx_sel_ = []  # Initialize a list to store selected feature indices
        # self.labels_ = X.columns  # Store the feature labels
        self.len_input = arr_input.shape[1]

        # First Stage: Compute the SU (Symmetrical Uncertainty) for each feature with the response.
        SU_vec = np.apply_along_axis(_symmetricalUncertain, 0, arr_input, arr_target)
        print("Stage 1: SU computation finished")

        SU_list = SU_vec[SU_vec > self.threshold]  # Filter features based on threshold
        SU_list[::-1].sort()
        m = arr_input[:, SU_vec > self.threshold].shape
        x_sorted = np.zeros(shape=m)

        i = 0

        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = 0
            x_sorted[:, i] = arr_input[:, ind].copy()
            self.idx_sel_.append(ind)

            i = i + 1
            print(f"Stage 1, iteration: {i} finished")

        print("Stage 1 finished")

        # Second Stage: Identify relationships between features to remove redundancy.
        j = 0

        i = 0

        while True:
            if j >= x_sorted.shape[1]:
                break

            y = x_sorted[:, j].copy()
            x_list = x_sorted[:, j + 1 :].copy()

            if x_list.shape[1] == 0:
                break

            SU_list_2 = SU_list[j + 1 :]
            SU_x = np.apply_along_axis(_symmetricalUncertain, 0, x_list, y)

            comp_SU = SU_x >= SU_list_2
            to_remove = np.where(comp_SU)[0] + j + 1

            if to_remove.size > 0:
                x_sorted = np.delete(x_sorted, to_remove, axis=1)
                SU_list = np.delete(SU_list, to_remove, axis=0)
                to_remove.sort()

                for r in reversed(to_remove):
                    self.idx_sel_.remove(self.idx_sel_[r])

            j = j + 1

            i = i + 1
            print(f"Stage 2, iteration: {i} finished")

        print("Stage 2 finished")

        self.scores_ = list(comp_SU)

        return self

    def _suGroup(self, x: np.ndarray, n: int):
        """
        Calculate SU (Symmetrical Uncertainty) between groups of features.

        Parameters:
        - x: Input data.
        - n: Number of groups.

        Returns:
        - SU_matrix: Symmetrical Uncertainty matrix.
        """
        m = x.shape[0]
        x = np.reshape(x, (n, m // n)).T
        m = x.shape[1]
        SU_matrix = np.zeros(shape=(m, m))

        for j in range(m - 1):
            x2 = x[:, j + 1 : :]
            y = x[:, j]
            temp = np.apply_along_axis(_symmetricalUncertain, 0, x2, y)

            for k in range(temp.shape[0]):
                SU_matrix[j, j + 1 : :] = temp
                SU_matrix[j + 1 : :, j] = temp

        return 1 / float(m - 1) * np.sum(SU_matrix, axis=1)

    def _isprime(self, a: int):
        """
        Check if a number is prime.

        Parameters:
        - a: The number to check.

        Returns:
        - True if a is prime, False otherwise.
        """
        return all(a % i for i in range(2, a))

    def _get_support_mask(self) -> pd.Series:
        """
        Get the mask of selected features.

        Returns:
        - support_mask: A mask indicating selected features.
        """
        check_is_fitted(
            self
        )  # Checks whether the model has attributes that end with an underscore as a sign that the estimator is fitted

        return [i in self.idx_sel_ for i in range(self.len_input)]
        # return pd.Series(
        #     [True if i in self.idx_sel_ else False for i, s in enumerate(self.labels_)]
        # )


def _count_vals(x: np.ndarray) -> np.ndarray:
    """
    Count the occurrences of unique values in an array.

    Parameters:
    - x: Input array.

    Returns:
    - occ: Array of occurrences for each unique value.
    """
    vals = np.unique(x)
    occ = np.zeros(shape=vals.shape)

    for i in range(vals.size):
        occ[i] = np.sum(x == vals[i])

    return occ


def _entropy(x: np.ndarray) -> float:
    """
    Calculate the entropy of an array.

    Parameters:
    - x: Input array.

    Returns:
    - Entropy value.
    """
    n = float(x.shape[0])
    ocurrence = _count_vals(x)
    px = ocurrence / n

    return -1 * np.sum(px * np.log2(px))


def _symmetricalUncertain(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Symmetrical Uncertainty (SU) between two arrays.

    Parameters:
    - x: First array.
    - y: Second array.

    Returns:
    - SU value.
    """
    n = float(y.shape[0])
    vals = np.unique(y)

    Hx = _entropy(x)
    Hy = _entropy(y)
    partial = np.zeros(shape=(vals.shape[0]))

    for i in range(vals.shape[0]):
        partial[i] = _entropy(x[y == vals[i]])

    partial[np.isnan(partial) == 1] = 0
    py = _count_vals(y).astype(dtype="float64") / n
    Hxy = np.sum(py[py > 0] * partial)
    IG = Hx - Hxy

    return 2 * IG / (Hx + Hy)
