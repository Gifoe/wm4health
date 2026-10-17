"""
Module for creating dataframe with two-level indexing:
    - Can be used to assign dict of vectors.
    - Can be used to assign matrices.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import pandas as pd
import numpy as np


def get_df_empty():
    """
    Create a return an empty DataFrame.
    """

    df = pd.DataFrame()

    return df


def set_df_idx(df_old, tag, col, mat):
    """
    Store a two-dimensional array into a Pandas DataFrame with a two-level index.
        - The first column index is global for the matrix.
        - The second column index is specified for the rows.
    """

    # get the shape
    n_row = mat.shape[0]
    n_col = mat.shape[1]

    # check the data
    assert (len(df_old) == 0) or (len(df_old) == n_row), "invalid dimension"
    assert len(tag) == n_col, "invalid dimension"
    assert len(col) == n_col, "invalid dimension"
    assert n_row > 0, "invalid dimension"
    assert n_col > 0, "invalid dimension"

    # create the DataFrame
    idx = pd.MultiIndex.from_arrays([tag, col], names=("var", "sub"))
    df_tmp = pd.DataFrame(data=mat, columns=idx)

    # merge the created DataFrame to the original DataFrame
    df_new = pd.concat([df_old, df_tmp], axis=1, ignore_index=False)

    return df_new


def set_df_mat(df_old, tag, mat):
    """
    Add a dict of arrays into a Pandas DataFrame.
    """

    tag = [tag for _ in range(mat.shape[1])]
    col = [idx for idx in range(mat.shape[1])]
    df_new = set_df_idx(df_old, tag, col, mat)

    return df_new


def set_df_dict(df_old, tag, var):
    """
    Add a NumPy matrix (two-dimensional array) into a Pandas DataFrame.
    """

    tag = [tag for _ in var.keys()]
    col = [key for key in var.keys()]
    mat = np.stack(list(var.values())).transpose()
    df_new = set_df_idx(df_old, tag, col, mat)

    return df_new
