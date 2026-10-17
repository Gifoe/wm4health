"""
Load and parse the CSV files containing the measurement results:
    - Load the dataset and select a temperature.
    - Compute the gradient of the magnetic flux.
    - Remove offset and align the phase of the signals.
    - Select the training and validation sets.
    - Assemble the generated data in a DataFrame.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
from odesolver.utils import dataframe_index


def _load_data(tag_input, T):
    """
    Load the CSV files and select a temperature.
    """

    # load the datasets
    B_mat = np.loadtxt(f"{tag_input}_B.csv.gz", delimiter=",")
    H_mat = np.loadtxt(f"{tag_input}_H.csv.gz", delimiter=",")
    T_vec = np.loadtxt(f"{tag_input}_T.csv.gz", delimiter=",")

    # check the data
    assert B_mat.shape[0] == T_vec.size, "invalid data size"
    assert H_mat.shape[0] == T_vec.size, "invalid data size"
    assert H_mat.shape[1] == B_mat.shape[1], "invalid data size"

    # select the temperature
    idx_select = np.isclose(T_vec, T)
    n_data = np.count_nonzero(idx_select)
    B_mat = B_mat[idx_select]
    H_mat = H_mat[idx_select]

    return B_mat, H_mat, n_data


def _compute_gradient(B_mat, dt):
    """
    Compute the flux gradient and get the timebase.
    """

    # compute the gradient
    dBdt_mat = np.gradient(B_mat, dt, axis=1)
    n_data = B_mat.shape[0]
    n_time = B_mat.shape[1]

    # create the time vector
    t_vec = dt * np.arange(n_time)
    t_mat = np.tile(t_vec, (n_data, 1))

    return t_mat, dBdt_mat


def _clean_dataset(dBdt_mat, B_mat, H_mat):
    """
    Remove any offset and align the phase with respect to the magnetic field.
    """

    # remove any offset
    dBdt_mat -= np.mean(dBdt_mat, axis=1, keepdims=True)
    B_mat -= np.mean(B_mat, axis=1, keepdims=True)
    H_mat -= np.mean(H_mat, axis=1, keepdims=True)

    # align the phase with respect to the magnetic field
    idx = np.argmin(np.abs(H_mat), axis=1)
    dBdt_mat = np.array([np.roll(row, -pos) for row, pos in zip(dBdt_mat, idx, strict=True)])
    B_mat = np.array([np.roll(row, -pos) for row, pos in zip(B_mat, idx, strict=True)])
    H_mat = np.array([np.roll(row, -pos) for row, pos in zip(H_mat, idx, strict=True)])

    return dBdt_mat, B_mat, H_mat


def _split_indices(n_data, n_train, n_test, rng):
    """
    Select the training and validation sets.
    """

    # random selection of the samples
    idx_all = rng.choice(np.arange(n_data), n_train + n_test, replace=False)

    # split the train and test sets
    idx_perm = rng.permutation(np.arange(n_train + n_test))
    idx_train = idx_perm[:n_train]
    idx_test = idx_perm[n_train:]

    # assign boolean for the train and test sets
    is_test = np.full(len(idx_perm), False)
    is_train = np.full(len(idx_perm), False)
    is_test[idx_test] = True
    is_train[idx_train] = True

    return idx_all, is_test, is_train


def _assemble_data(t_mat, dBdt_mat, B_mat, H_mat, idx_all, is_test, is_train):
    """
    Assemble the DataFrame.
    """

    # random selection of the samples
    t_mat = t_mat[idx_all]
    dBdt_mat = dBdt_mat[idx_all]
    B_mat = B_mat[idx_all]
    H_mat = H_mat[idx_all]

    # create a local index
    idx_tmp = np.arange(len(idx_all))

    # create a DataFrame
    raw = dataframe_index.get_df_empty()

    # assign the vectors
    var = {"idx_global": idx_all, "idx_local": idx_tmp}
    raw = dataframe_index.set_df_dict(raw, "idx_var", var)

    var = {"is_test": is_test, "is_train": is_train}
    raw = dataframe_index.set_df_dict(raw, "stat_var", var)

    # assign the matrices
    raw = dataframe_index.set_df_mat(raw, "t_mat", np.array(t_mat))
    raw = dataframe_index.set_df_mat(raw, "dBdt_mat", np.array(dBdt_mat))
    raw = dataframe_index.set_df_mat(raw, "B_mat", np.array(B_mat))
    raw = dataframe_index.set_df_mat(raw, "H_mat", np.array(H_mat))

    return raw


def get_data(tag_input, dt, T, n_train, n_test, rng):
    """
    Load and parse the CSV files containing the measurement results.
    """

    # load the CSV files and select a temperature
    (B_mat, H_mat, n_data) = _load_data(tag_input, T)

    # compute the flux gradient and get the timebase
    (t_mat, dBdt_mat) = _compute_gradient(B_mat, dt)

    # remove any offset and align the phase with respect to the magnetic field
    (dBdt_mat, B_mat, H_mat) = _clean_dataset(dBdt_mat, B_mat, H_mat)

    # select the training and validation sets
    (idx_all, is_test, is_train) = _split_indices(n_data, n_train, n_test, rng)

    # assemble the DataFrame.
    raw = _assemble_data(t_mat, dBdt_mat, B_mat, H_mat, idx_all, is_test, is_train)

    return raw
