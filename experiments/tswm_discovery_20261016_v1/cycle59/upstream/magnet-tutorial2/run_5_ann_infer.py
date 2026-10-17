"""
Load a trained neural network-based model and a dataset.
Evaluate the model with respect to the dataset.
Show and save the results.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import matplotlib.pyplot as plt
from odesolver import inference
from odesolver import serialize
from odesolver import plotting
import model_ann


if __name__ == "__main__":
    # options for the ODE solver
    ode = {
        "dt_step": 50e-9,                           # timestep for the ODE solver
        "dt_add": 5e-6,                             # buffer time to add for the ODE solver
        "max_steps": 32384,                         # maximum number of integration steps
        "solver": "Dopri5",                         # name of the ODE solver (standard solver)
        "adjoint": "RecursiveCheckpointAdjoint",    # adjoint solver (backward diff)
    }

    # options for sampling the signals from the dataset
    sig = {
        "r_int": 1.0,     # resampling factor for the interpolation of the excitation
        "r_out": 0.6,     # resampling factor for saving the results
        "n_wait": 1.0,    # number of periods before starting to save the results
        "n_out": 1.0,     # number of periods to be saved in the results
    }

    # load the dataset
    raw = serialize.load_dataframe("data/raw_magnet.pkl")

    # load the trained model
    (model, param, const) = model_ann.get_ann_dual()
    param = serialize.load_equinox("data/param_ann.eqx", param)

    # inference of the model with the provided dataset
    dset = inference.get_infer("infer", ode, sig, model, const, raw, param)

    # plot the error metrics for the complete dataset
    plotting.get_plot_all("all", dset)

    # plot single waveforms
    plotting.get_plot_single("single / a", dset.iloc[+0])
    plotting.get_plot_single("single / b", dset.iloc[-1])

    # save the dataset with the inference data
    serialize.write_dataframe("data/dset_ann.pkl", dset)

    # show the plots
    plt.show()
