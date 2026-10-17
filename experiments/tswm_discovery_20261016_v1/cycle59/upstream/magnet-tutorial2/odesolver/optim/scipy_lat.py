"""
Optimizer based on the SciPy Latin Hypercube sampler:
    - This solver is meant to generate initial values.
    - This solver is strictly enforcing the bounds.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax
import jax.numpy as jnp
import scipy.stats.qmc as qmc


def get_optim(lb, ub, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn):
    """
    Optimizer based on the SciPy Latin Hypercube sampler.
    """

    # extract the data
    max_val = optimizer["max_val"]
    sampler = optimizer["sampler"]
    size = optimizer["size"]
    rng = optimizer["rng"]

    # initial
    x0 = (lb + ub) / 2.0

    # get the sampling
    if sampler == "uniform":
        value_vec = rng.random((size, len(x0)))
    elif sampler == "hypercube":
        sampler = qmc.LatinHypercube(d=len(x0), rng=rng)
        value_vec = sampler.random(n=size)
    else:
        raise ValueError("invalid sampler")

    # cast to JAX types
    value_vec = jnp.array(value_vec, dtype=jnp.float32)

    # scale the value in the bounds
    value_vec = value_vec * (ub - lb) + lb

    # get the optimization function
    def get_solver(value):
        # get the parameter scaling
        param = fct_decode(value)
        penalty = fct_penalty(value)

        # get the loss value
        residuum = fct_opt(param, idx_trn)
        loss = jnp.sqrt(jnp.mean(penalty * residuum))

        # fix invalid value
        loss = jnp.nan_to_num(loss, nan=max_val, posinf=max_val, neginf=max_val)
        loss = jnp.minimum(loss, max_val)

        # log the iteration
        jax.debug.print(
            "step / penalty = {:.4f} / loss = {:.4f}",
            penalty, loss,
        )

        return loss

    # solve the different parameter values
    loss_vec = jax.vmap(get_solver)(value_vec)

    # find the optimal value
    idx = jnp.argmin(loss_vec)
    value = value_vec[idx]

    # assign the details
    sol = {"value_vec": value_vec, "loss_vec": loss_vec}

    return value, sol
