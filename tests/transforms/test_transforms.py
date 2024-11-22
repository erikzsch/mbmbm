import numpy as np

from mbmbm.transform.transforms import (
    RemoveNaNColumnsTransform,
    RemoveNaNRowsTransform,
    RemoveOutOfRangeColumnsTransform,
    RemoveOutOfRangeRowsTransform,
    ReplaceNaNTransform,
)


def test_col_oor(np_array: np.array):
    t = RemoveOutOfRangeColumnsTransform(0, 9)
    res, rm_rows, rm_cols = t.transform(np_array)
    assert np.all(np.equal(res, np.array([[1], [5], [9]])))


def test_row_oor(np_array: np.array):
    t = RemoveOutOfRangeRowsTransform(0, 4)
    res, rm_rows, rm_cols = t.transform(np_array)
    assert rm_rows == [1, 2]
    assert np.all(np.equal(res, np.array([[1, 2, 3, 4]])))


def test_col_nan(np_array_with_nan: np.array):
    t = RemoveNaNColumnsTransform()
    res, rm_rows, rm_cols = t.transform(np_array_with_nan)
    assert np.all(np.equal(res, np_array_with_nan[:, [0, 1, 3]]))


def test_row_nan(np_array_with_nan: np.array):
    t = RemoveNaNRowsTransform()
    res, rm_rows, rm_cols = t.transform(np_array_with_nan)
    assert np.all(np.equal(res, np_array_with_nan[[0, 2], :]))


def test_relace_nan(np_array_with_nan: np.array):
    t = ReplaceNaNTransform(4)
    res, rm_rows, rm_cols = t.transform(np_array_with_nan)
    ref = np.array(np_array_with_nan)
    ref[1, 2] = 4
    assert np.all(np.equal(res, ref))
