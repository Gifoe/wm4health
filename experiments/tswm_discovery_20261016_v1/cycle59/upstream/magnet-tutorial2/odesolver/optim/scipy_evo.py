"""
Optimizer based on the SciPy Differential Evolution solver:
    - This solver is not gradient based (robust but slow).
    - This solver is strictly enforcing the bounds.
    - This solver is using a scalar objective function.
    - This solver is using a validation set.
    - This solver is not using batches.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as sopt
from odesolver.utils import conv_tracker


def get_optim(lb, ub, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn, idx_val):
    """
    Optimizer based on the SciPy Differential Evolution solver.
    """

    # extract the data
    alg_options = optimizer["alg_options"]
    conv_data = optimizer["conv_data"]
    max_val = optimizer["max_val"]

    # cast to NumPy types
    lb = np.array(lb, dtype=np.float32)
    ub = np.array(ub, dtype=np.float32)

    # wrap the bounds into the required format
    var_bnd = list(zip(lb, ub, strict=True))

    # get the convergence tracker
    conv_obj = conv_tracker.ConvTracker(conv_data)

    # get the optimization function (scalar)
    def get_scalar(value, idx_all):
        # get the parameter scaling
        param = fct_decode(value)
        penalty = fct_penalty(value)

        # get the loss value (training set)
        residuum = fct_opt(param, idx_all)
        loss = jnp.sqrt(jnp.mean(penalty * residuum))

        # fix invalid value
        loss = jnp.nan_to_num(loss, nan=max_val, posinf=max_val, neginf=max_val)
        loss = jnp.minimum(loss, max_val)

        return loss

    # get the optimization function (vector)
    def get_solver(value_vec):
        # cast to JAX types
        value_vec = jnp.array(value_vec.transpose(), dtype=jnp.float32)

        # evaluate with JAX
        get_eval = lambda value: get_scalar(value, idx_trn)
        loss_vec = jax.vmap(get_eval)(value_vec)

        # cast to numpy
        loss_vec = np.array(loss_vec, dtype=np.float32)

        return loss_vec

    # get the iteration callback
    def get_callback(value, conv=0.0):
        # cast to JAX types
        value = jnp.array(value, dtype=jnp.float32)

        # get the parameter scaling
        penalty = fct_penalty(value)

        # get the loss values
        loss_trn = get_scalar(value, idx_trn)
        loss_val = get_scalar(value, idx_val)

        # detect the convergence
        stop = conv_obj.set_step(value, penalty, loss_trn, loss_val)
        if stop:
            raise sopt.NoConvergence("convergence achieved")

    # solve the problem
    try:
        sopt.differential_evolution(
            get_solver,
            var_bnd,
            polish=False,
            vectorized=True,
            updating="deferred",
            callback=get_callback,
            **alg_options,
        )
    except sopt.NoConvergence:
        pass

    # get the best trial
    value = conv_obj.get_best()
    sol = conv_obj.get_sol()

    return value, sol
