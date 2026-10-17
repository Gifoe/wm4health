"""
Load an untrained neural network-based model and a dataset.
Train the model with respect to the dataset.
Save the trained model parameters.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import optax
import jax.random as jr
from odesolver import serialize
from odesolver import training
import model_ann


def _get_optimizer(fo, lr):
    """
    Get the options for the training optimizer.
    """

    # Optax optimizer algorithm
    optax_obj = optax.adabelief(learning_rate=lr)

    # options for the optimizer (Optax optimizer)
    optimizer = {
        "batch_size": 16,              # batch size for the dataloader
        "max_steps": 128,              # maximum number of training iterations
        "loader_key": jr.key(1234),    # random key used for the dataloader
        "optax_obj": optax_obj,        # Optax optimizer algorithm
        "conv_data": {                 # options for detecting convergence
            "conv_steps": 16,          # number of iterations used to check for convergence
            "conv_rate": 0.01,         # convergence rate to detect a stalled training
            "loss_target": 0.02,       # target loss value for stopping the training
        }
    }

    # options controlling the training process
    opt = {
        "fact_power": 0.0+fo,         # weight for the error on the power losses
        "fact_field": 1.0-fo,         # weight for the error on the magnetic field
        "frac_valid": 0.33,           # fraction of the data to be used as validation
        "split_key": jr.key(1234),    # random key used for splitting the validation set
        "optimizer": optimizer,       # options for the optimizer (Optax optimizer)
    }

    return opt


def _get_integrator(n_wait, n_out):
    """
    Get the ODE solver and sampling options.
    """

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
        "r_int": 1.0,        # resampling factor for the interpolation of the excitation
        "r_out": 0.3,        # resampling factor for saving the results
        "n_wait": n_wait,    # number of periods before starting to save the results
        "n_out": n_out,      # number of periods to be saved in the results
    }

    return ode, sig


if __name__ == "__main__":
    # load the dataset
    raw = serialize.load_dataframe("data/raw_magnet.pkl")

    # load the untrained model
    (model, param, const) = model_ann.get_ann_dual()

    # train the model for a given number of periods
    def get_opt_step(name, param_tmp, n_wait, n_out, fo, lr):
        # get the training options
        (ode, sig) = _get_integrator(n_wait, n_out)
        opt = _get_optimizer(fo, lr)

        # train the model
        (param_tmp, sol_tmp) = training.get_train_ann(
            name, ode, sig, opt, model, const, raw, param_tmp,
        )

        return param_tmp, sol_tmp

    # train the model in several steps
    #   - start with a very short prediction horizon
    #   - start with using the magnetic field as a metrics
    #   - slowly increase the prediction horizon towards steady-state
    #   - slowly introduce the power losses as a metrics
    (param, _) = get_opt_step("step_1", param, 0.0, 0.1, 0.0, 25e-3)
    (param, _) = get_opt_step("step_2", param, 0.0, 0.2, 0.1, 15e-3)
    (param, _) = get_opt_step("step_3", param, 0.0, 0.5, 0.2, 10e-3)
    (param, _) = get_opt_step("step_4", param, 0.0, 1.0, 0.3, 8e-3)
    (param, _) = get_opt_step("step_5", param, 1.0, 1.0, 0.5, 3e-3)

    # save the trained model parameters
    serialize.write_equinox("data/param_ann.eqx", param)
