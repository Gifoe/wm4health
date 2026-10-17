"""
Load an untrained equation-based model and a dataset.
Train the model with respect to the dataset.
Save the trained model parameters.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import optax
import optimistix
import jax.random as jr
import numpy.random as nr
from odesolver import serialize
from odesolver import training
import model_eqn


def _get_optimizer(method):
    """
    Get the options for the training optimizer.
    """

    # options for the optimizer (various optimizer)
    if method == "lat":
        # latin hypercube sampling (SciPy, gradient free)
        optimizer = {
            "max_val": 10.0,                # clamp loss values greater than this value
            "sampler": "hypercube",         # sampling method ("hypercube" or "uniform")
            "size": 25,                     # number of samples to be tested
            "rng": nr.default_rng(1234),    # seed for the random generator
        }

        # backward differentiation can be used
        fwd = False
    elif method == "evo":
        # Scipy optimizer options
        alg_options = {
            "maxiter": 10,
            "popsize": 15,
            "tol": 1e-4,
            "atol": 1e-4,
            "rng": nr.default_rng(1234),
        }

        # differential evolution (SciPy, gradient free)
        optimizer = {
            "max_val": 10.0,               # clamp loss values greater than this value
            "alg_options": alg_options,    # options for differential evolution
            "conv_data": {                 # options for detecting convergence
                "conv_steps": 8,           # number of iterations used to check for convergence
                "conv_rate": 0.01,         # convergence rate to detect a stalled training
                "loss_target": 0.02,       # target loss value for stopping the training
            },
        }

        # backward differentiation can be used
        fwd = False
    elif method == "opt":
        # Optax optimizer algorithm
        optax_obj = optax.adabelief(learning_rate=0.25)

        # first order gradient descent (Optax, gradient based)
        optimizer = {
            "batch_size": 16,              # batch size for the dataloader
            "max_steps": 64,               # maximum number of training iterations
            "loader_key": jr.key(1234),    # random key used for the dataloader
            "optax_obj": optax_obj,        # Optax optimizer algorithm
            "conv_data": {                 # options for detecting convergence
                "conv_steps": 8,           # number of iterations used to check for convergence
                "conv_rate": 0.01,         # convergence rate to detect a stalled training
                "loss_target": 0.02,       # target loss value for stopping the training
            },
        }

        # backward differentiation can be used
        fwd = False
    elif method == "min":
        # Optimistix minimization solver
        optx_obj = optimistix.BFGS(rtol=1e-4, atol=1e-4)

        # minimization solver (Optimistix, gradient based)
        optimizer = {
            "max_steps": 25,         # maximum number of training iterations
            "optx_obj": optx_obj,    # Optimistix optimizer algorithm
        }

        # backward differentiation can be used
        fwd = False
    elif method == "lsq":
        # Optimistix square solver
        optx_obj = optimistix.LevenbergMarquardt(rtol=1e-4, atol=1e-4)

        # least square solver (Optimistix, gradient based)
        optimizer = {
            "max_steps": 25,         # maximum number of training iterations
            "optx_obj": optx_obj,    # Optimistix optimizer algorithm
        }

        # forward-mode differentiation must be used
        fwd = True
    else:
        raise ValueError("invalid method")

    # options controlling the training process
    opt = {
        "fact_power": 0.5,            # weight for the error on the power losses
        "fact_field": 0.5,            # weight for the error on the magnetic field
        "frac_valid": 0.33,           # fraction of the data to be used as validation
        "split_key": jr.key(1234),    # random key used for splitting the validation set
        "method": method,             # name of the optimizer (various methods)
        "optimizer": optimizer,       # options for the optimizer (various methods)
    }

    return opt, fwd


def _get_integrator(fwd):
    """
    Get the ODE solver and sampling options.
    """

    # get the ODE solver and adjoint (foward or backward diff)
    if fwd:
        solver = "Dopri5"
        adjoint = "ForwardMode"
    else:
        solver = "Dopri5"
        adjoint = "RecursiveCheckpointAdjoint"

    # options for the ODE solver
    ode = {
        "dt_step": 50e-9,      # timestep for the ODE solver
        "dt_add": 5e-6,        # buffer time to add for the ODE solver
        "max_steps": 32384,    # maximum number of integration steps
        "solver": solver,      # name of the ODE solver (standard solver)
        "adjoint": adjoint,    # adjoint solver (foward or backward diff)
    }

    # options for sampling the signals from the dataset
    sig = {
        "r_int": 1.0,     # resampling factor for the interpolation of the excitation
        "r_out": 0.3,     # resampling factor for saving the results
        "n_wait": 1.0,    # number of periods before starting to save the results
        "n_out": 1.0,     # number of periods to be saved in the results
    }

    return ode, sig


if __name__ == "__main__":
    # load the dataset
    raw = serialize.load_dataframe("data/raw_magnet.pkl")

    # load the untrained model
    (model, param, const, bnd) = model_eqn.get_eqn_nonlinear()

    # train the model with a given optimizer
    def get_opt_step(name, param_tmp, method):
        # get the training options
        (opt, fwd) = _get_optimizer(method)
        (ode, sig) = _get_integrator(fwd)

        # train the model
        (param_tmp, sol_tmp) = training.get_train_eqn(
            name, ode, sig, bnd, opt, model, const, raw, param_tmp,
        )

        return param_tmp, sol_tmp

    # find an initial value with latin hypercube sampling
    (param, _) = get_opt_step("step_init", param, "lat")

    # find the optimal value with gradient descent
    (param, _) = get_opt_step("step_finish", param, "opt")

    # save the trained model parameters
    serialize.write_equinox("data/param_eqn.eqx", param)
