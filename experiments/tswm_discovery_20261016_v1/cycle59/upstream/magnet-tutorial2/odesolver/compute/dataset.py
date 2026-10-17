"""
Module for creating and managing datasets:
    - Store all the data into a single Pandas DataFrame.
    - Parse and prepare the raw data for the ODE models.
    - Set the model prediction and compute error metrics.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
from odesolver.utils import dataframe_index


def _get_time(t_vec, v_vec, sig):
    """
    Get the time vectors for ODE integration and evaluation:
        - A time vector where the solution will be evaluated (t_out_vec).
        - A time vector where the excitation will be interpolated (t_int_vec).
        - A vector containing the excitation applied to the system (v_int_vec).
    """

    # extract the data
    r_int = sig["r_int"]
    r_out = sig["r_out"]
    n_out = sig["n_out"]
    n_wait = sig["n_wait"]

    # get the original timebase
    n_base = len(t_vec)
    dt_base = np.mean(np.diff(t_vec))
    t_period = n_base * dt_base

    # compute the number of samples for the evaluation vectors
    ns_out = int(np.round(r_out * n_out * n_base))

    # compute the number of samples for the interpolation vectors
    ns_int = int(np.round(r_int * n_base))

    # get the new timebase for the evaluation vectors
    dt_out = (n_out * t_period) / ns_out
    t_out_vec = (n_wait * t_period) + dt_out * np.arange(ns_out)

    # get the new interpolation time and value vectors
    t_int_vec = np.linspace(0.0 * t_period, 1.0 * t_period, ns_int)
    v_int_vec = np.interp(t_int_vec, t_vec, v_vec, period=t_period)

    return t_out_vec, t_int_vec, v_int_vec


def _get_interp(t_vec, v_vec, t_interp_vec):
    """
    Interpolate a signal into a new timebase.
    """

    # get the old timebase
    ns = len(t_vec)
    dt = np.mean(np.diff(t_vec))

    # interpolate into the new timebase (periodic signal)
    v_interp_vec = np.interp(t_interp_vec, t_vec, v_vec, period=dt * ns)

    return v_interp_vec


def _get_var_metrics(var_mat):
    """
    Compute base metrics (RMS, average, and peak-peak) for a signal.
    """

    var = {
        "rms": np.sqrt(np.mean(var_mat**2, axis=1)),
        "pkpk": np.max(var_mat, axis=1) - np.min(var_mat, axis=1),
        "avg": np.mean(var_mat, axis=1),
    }

    return var


def parse_dataset(raw, sig):
    """
    Parse and prepare the raw data for the ODE models:
        - Get the integration and evaluation timebase.
        - Set the training and testing samples.
        - Return a new Pandas DataFrame.
    """

    # cast the raw data into arrays
    t_mat = np.array(raw["t_mat"])
    dBdt_mat = np.array(raw["dBdt_mat"])
    B_mat = np.array(raw["B_mat"])
    H_mat = np.array(raw["H_mat"])

    # copy the invariant information
    dset = raw[["idx_var", "stat_var"]]

    # lists for the parsed data
    t_out_mat = []
    t_int_mat = []
    dBdt_int_mat = []
    dBdt_ref_mat = []
    B_ref_mat = []
    H_ref_mat = []

    # get the integration and evaluation timebase for each sample
    for t_vec, dBdt_vec, B_vec, H_vec in zip(t_mat, dBdt_mat, B_mat, H_mat, strict=True):
        # get the timebase and excitation
        (t_out_vec, t_int_vec, dBdt_int_vec) = _get_time(t_vec, dBdt_vec, sig)

        # interpolate to the solution period
        dBdt_ref_vec = _get_interp(t_vec, dBdt_vec, t_out_vec)
        B_ref_vec = _get_interp(t_vec, B_vec, t_out_vec)
        H_ref_vec = _get_interp(t_vec, H_vec, t_out_vec)

        # add the data
        t_out_mat.append(t_out_vec)
        t_int_mat.append(t_int_vec)
        dBdt_int_mat.append(dBdt_int_vec)
        dBdt_ref_mat.append(dBdt_ref_vec)
        B_ref_mat.append(B_ref_vec)
        H_ref_mat.append(H_ref_vec)

    # assign the resampled matrices
    dset = dataframe_index.set_df_mat(dset, "t_int_mat", np.array(t_int_mat))
    dset = dataframe_index.set_df_mat(dset, "dBdt_int_mat", np.array(dBdt_int_mat))
    dset = dataframe_index.set_df_mat(dset, "dBdt_ref_mat", np.array(dBdt_ref_mat))
    dset = dataframe_index.set_df_mat(dset, "t_out_mat", np.array(t_out_mat))
    dset = dataframe_index.set_df_mat(dset, "B_ref_mat", np.array(B_ref_mat))
    dset = dataframe_index.set_df_mat(dset, "H_ref_mat", np.array(H_ref_mat))

    return dset


def set_solution(dset, H_cmp_mat):
    """
    Add the magnetic field predictions into the Pandas DataFrame.
    Compute the error between the prediction and the reference values.
    """

    # cast the solution to array
    H_cmp_mat = np.array(H_cmp_mat)

    # get the data
    H_ref_mat = np.array(dset["H_ref_mat"])
    dBdt_ref_mat = np.array(dset["dBdt_ref_mat"])

    # compute the powers and errors
    P_ref_mat = dBdt_ref_mat * H_ref_mat
    P_cmp_mat = dBdt_ref_mat * H_cmp_mat
    H_err_mat = H_cmp_mat - H_ref_mat
    P_err_mat = P_cmp_mat - P_ref_mat

    # set the solution
    dset = dataframe_index.set_df_mat(dset, "H_cmp_mat", H_cmp_mat)
    dset = dataframe_index.set_df_mat(dset, "H_err_mat", H_err_mat)
    dset = dataframe_index.set_df_mat(dset, "P_ref_mat", P_ref_mat)
    dset = dataframe_index.set_df_mat(dset, "P_cmp_mat", P_cmp_mat)
    dset = dataframe_index.set_df_mat(dset, "P_err_mat", P_err_mat)

    return dset


def set_metrics(dset):
    """
    Compute and add various metrics to the Pandas DataFrame.
    """

    # extract the data
    t_int_mat = np.array(dset["t_int_mat"])
    t_out_mat = np.array(dset["t_out_mat"])
    dBdt_ref_mat = np.array(dset["dBdt_ref_mat"])
    B_ref_mat = np.array(dset["B_ref_mat"])
    H_ref_mat = np.array(dset["H_ref_mat"])
    H_cmp_mat = np.array(dset["H_cmp_mat"])
    H_err_mat = np.array(dset["H_err_mat"])
    P_ref_mat = np.array(dset["P_ref_mat"])
    P_cmp_mat = np.array(dset["P_cmp_mat"])
    P_err_mat = np.array(dset["P_err_mat"])

    # compute the metrics for the variables
    dset = dataframe_index.set_df_dict(dset, "dBdt_ref_var", _get_var_metrics(dBdt_ref_mat))
    dset = dataframe_index.set_df_dict(dset, "B_ref_var", _get_var_metrics(B_ref_mat))
    dset = dataframe_index.set_df_dict(dset, "H_err_var", _get_var_metrics(H_err_mat))
    dset = dataframe_index.set_df_dict(dset, "H_ref_var", _get_var_metrics(H_ref_mat))
    dset = dataframe_index.set_df_dict(dset, "H_cmp_var", _get_var_metrics(H_cmp_mat))
    dset = dataframe_index.set_df_dict(dset, "P_err_var", _get_var_metrics(P_err_mat))
    dset = dataframe_index.set_df_dict(dset, "P_ref_var", _get_var_metrics(P_ref_mat))
    dset = dataframe_index.set_df_dict(dset, "P_cmp_var", _get_var_metrics(P_cmp_mat))

    # compute the timing metrics
    dt_int = np.mean(np.diff(t_int_mat), axis=1)
    dt_out = np.mean(np.diff(t_out_mat), axis=1)
    td_int = np.max(t_int_mat, axis=1) - np.min(t_int_mat, axis=1)
    td_out = np.max(t_out_mat, axis=1) - np.min(t_out_mat, axis=1)
    var = {"dt_int": dt_int, "td_int": td_int, "dt_out": dt_out, "td_out": td_out}
    dset = dataframe_index.set_df_dict(dset, "t_var", var)

    # compute the error metrics
    err_field = dset["H_err_var"]["rms"] / dset["H_ref_var"]["rms"]
    err_power = dset["P_err_var"]["avg"] / dset["P_ref_var"]["avg"]
    var = {"power": err_power, "field": err_field}
    dset = dataframe_index.set_df_dict(dset, "err_var", var)

    return dset
