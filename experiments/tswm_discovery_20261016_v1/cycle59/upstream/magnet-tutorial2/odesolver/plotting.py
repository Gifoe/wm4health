"""
Various functions for analyzing a dataset:
    - Plot and display a single signal.
    - Plot and display a complete dataset.
    - Display the model parameters.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import matplotlib.pyplot as plt


def _get_err_metrics(err_vec):
    """
    Compute the error metrics for an error vector.
    """

    var = {
        "max": np.max(np.abs(err_vec)),
        "mean": np.mean(np.abs(err_vec)),
        "rms": np.sqrt(np.mean(err_vec**2)),
        "prc_50": np.percentile(np.abs(err_vec), 50),
        "prc_95": np.percentile(np.abs(err_vec), 95),
    }

    return var


def _disp_err_metrics(name, test, train):
    """
    Display the error metrics (test and train sets).
    """

    print(f"{name}")
    print(f"    rms = {100 * test['rms']:.2f} / {100 * train['rms']:.2f} %%")
    print(f"    mean = {100 * test['mean']:.2f} / {100 * train['mean']:.2f} %%")
    print(f"    max = {100 * test['max']:.2f} / {100 * train['max']:.2f} %%")
    print(f"    prc_50 = {100 * test['prc_50']:.2f} / {100 * train['prc_50']:.2f} %%")
    print(f"    prc_95 = {100 * test['prc_95']:.2f} / {100 * train['prc_95']:.2f} %%")


def _disp_var_metrics(name, var, scl, unit):
    """
    Display base metrics (RMS, average, and peak-peak) for a signal.
    """

    print(f"{name}")
    print(f"    rms = {scl * var['rms'].item():+.2f} {unit}")
    print(f"    pkpk = {scl * var['pkpk'].item():+.2f} {unit}")
    print(f"    avg = {scl * var['avg'].item():+.2f} {unit}")


def get_plot_single(name, dser):
    """
    Plot and display a single signal:
        - Plot the waveforms.
        - Plot the hysteresis.
        - Display the metrics.
    """

    # extract the vectors
    t_out_mat = dser["t_out_mat"].to_numpy()
    dBdt_ref_mat = dser["dBdt_ref_mat"].to_numpy()
    B_ref_mat = dser["B_ref_mat"].to_numpy()
    H_ref_mat = dser["H_ref_mat"].to_numpy()
    H_cmp_mat = dser["H_cmp_mat"].to_numpy()
    H_err_mat = dser["H_err_mat"].to_numpy()

    # display the metrics
    print(f"================================================= {name}")
    print(f"sample indices")
    print(f"    is_test = {dser['stat_var']['is_test'].item()}")
    print(f"    is_train = {dser['stat_var']['is_train'].item()}")
    print(f"sample status")
    print(f"    idx_global = {dser['idx_var']['idx_global'].item()}")
    print(f"    idx_local = {dser['idx_var']['idx_local'].item()}")
    _disp_var_metrics("dBdt_ref metrics", dser["dBdt_ref_var"], 1e-3, "mT/us")
    _disp_var_metrics("B_ref metrics", dser["B_ref_var"], 1e3, "mT")
    _disp_var_metrics("H_ref metrics", dser["H_ref_var"], 1e0, "A/m")
    _disp_var_metrics("H_cmp metrics", dser["H_cmp_var"], 1e0, "A/m")
    _disp_var_metrics("H_err metrics", dser["H_err_var"], 1e0, "A/m")
    _disp_var_metrics("P_ref metrics", dser["P_ref_var"], 1e-3, "mW/cm3")
    _disp_var_metrics("P_cmp metrics", dser["P_cmp_var"], 1e-3, "mW/cm3")
    _disp_var_metrics("P_err metrics", dser["P_err_var"], 1e-3, "mW/cm3")
    print(f"timing metrics")
    print(f"    dt_int = {1e9 * dser['t_var']['dt_int'].item():+.2f} ns")
    print(f"    td_int = {1e6 * dser['t_var']['td_int'].item():+.2f} us")
    print(f"    dt_out = {1e9 * dser['t_var']['dt_out'].item():+.2f} ns")
    print(f"    td_out = {1e6 * dser['t_var']['td_out'].item():+.2f} us")
    print(f"error metrics")
    print(f"    power = {100 * dser['err_var']['power'].item():+.2f} %%")
    print(f"    field = {100 * dser['err_var']['field'].item():+.2f} %%")
    print(f"================================================= {name}")

    # create the plots
    (fig_w, ax_w) = plt.subplots(3, 1, num=name + " / waveform")
    (fig_h, ax_h) = plt.subplots(1, 1, num=name + " / hysteresis")

    # plot the flux density derivative
    ax_w[0].plot(1e6 * t_out_mat, 1e-3 * dBdt_ref_mat, "b", label="reference")
    ax_w[0].set_xlabel("t (us)")
    ax_w[0].set_ylabel("dB/dt (mT/us)")
    ax_w[0].set_title("Flux Density Derivative")
    ax_w[0].legend()
    ax_w[0].grid()

    # plot the magnetic flux density
    ax_w[1].plot(1e6 * t_out_mat, 1e3 * B_ref_mat, "b", label="reference")
    ax_w[1].set_xlabel("t (us)")
    ax_w[1].set_ylabel("B (mT)")
    ax_w[1].set_title("Magnetic Flux Density")
    ax_w[1].legend()
    ax_w[1].grid()

    # plot the magnetic field
    ax_w[2].plot(1e6 * t_out_mat, 1e0 * H_ref_mat, "b", label="reference")
    ax_w[2].plot(1e6 * t_out_mat, 1e0 * H_cmp_mat, "r", label="model")
    ax_w[2].plot(1e6 * t_out_mat, 1e0 * H_err_mat, "y", label="error")
    ax_w[2].set_xlabel("t (us)")
    ax_w[2].set_ylabel("H (A/m)")
    ax_w[2].set_title("Magnetic Field")
    ax_w[2].legend()
    ax_w[2].grid()

    # plot the hysteresis
    ax_h.plot(1e0 * H_ref_mat, 1e3 * B_ref_mat, "b", label="reference")
    ax_h.plot(1e0 * H_cmp_mat, 1e3 * B_ref_mat, "r", label="model")
    ax_h.set_xlabel("H (A/m)")
    ax_h.set_ylabel("B (mT)")
    ax_h.set_title("Hysteresis")
    ax_h.legend()
    ax_h.grid()

    # fix the plot layout
    fig_w.tight_layout()
    fig_h.tight_layout()


def get_plot_all(name, dset):
    """
    Plot and display a complete dataset:
        - Plot the error map.
        - Display the metrics.
    """

    # extract the vectors
    is_test = dset["stat_var"]["is_test"].to_numpy()
    is_train = dset["stat_var"]["is_train"].to_numpy()
    err_power = dset["err_var"]["power"].to_numpy()
    err_field = dset["err_var"]["field"].to_numpy()

    # compute the error metrics across the complete dataset
    test_power = _get_err_metrics(err_power[is_test])
    train_power = _get_err_metrics(err_power[is_train])
    test_field = _get_err_metrics(err_field[is_test])
    train_field = _get_err_metrics(err_field[is_train])

    # display the metrics
    print(f"================================================= {name}")
    print(f"data size")
    print(f"    test = {np.count_nonzero(is_test)}")
    print(f"    train = {np.count_nonzero(is_train)}")
    _disp_err_metrics("power error", test_power, train_power)
    _disp_err_metrics("field error", test_field, train_field)
    print(f"================================================= {name}")

    # scatter plot with the error map
    (fig, ax) = plt.subplots(1, 1, num=name + " / error")
    ax.plot(1e2 * err_power[is_test], 1e2 * err_field[is_test], "ob", mfc="b", label="test")
    ax.plot(1e2 * err_power[is_train], 1e2 * err_field[is_train], "or", mfc="r", label="train")
    ax.set_xlabel("Error / Power (%)")
    ax.set_ylabel("Error / Field (%)")
    ax.set_title("Error Map")
    ax.legend()
    ax.grid()
    fig.tight_layout()


def get_disp_param(name, param):
    """
    Display the model parameters.
    """

    print(f"================================================= {name}")
    for key, val in param.items():
        if np.isscalar(val):
            print(f"{key} = {val:.3e}")
        else:
            val = " , ".join([f"{tmp:.3e}" for tmp in val])
            print(f"{key} = [{val}]")
    print(f"================================================= {name}")
