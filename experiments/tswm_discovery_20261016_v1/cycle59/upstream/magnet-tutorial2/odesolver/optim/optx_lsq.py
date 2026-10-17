"""
Optimizer based on the Optimistix least-square solver:
    - This solver is gradient based (forward-differentiation).
    - This solver is enforcing the bounds with a penalty.
    - This solver is using a vector objective function.
    - This solver is not using a validation set.
    - This solver is not using batches.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax
import jax.numpy as jnp
import optimistix as optx


def get_optim(value, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn):
    """
    Optimizer based on the Optimistix least-square solver.
    """

    # extract the data
    max_steps = optimizer["max_steps"]
    optx_obj = optimizer["optx_obj"]

    # get the optimization function
    def get_solver(value, _):
        # get the parameter scaling
        param = fct_decode(value)
        penalty = fct_penalty(value)

        # get the loss value
        residuum = fct_opt(param, idx_trn)
        loss = penalty * residuum

        # log the iteration
        jax.debug.print(
            "step / penalty = {:.4f} / loss = {:.4f}",
            penalty, jnp.sqrt(jnp.mean(loss)),
        )

        return loss

    # solve the problem
    sol = optx.least_squares(
        get_solver,
        optx_obj,
        value,
        max_steps=max_steps,
        throw=False,
    )
    value = sol.value

    return value, sol
