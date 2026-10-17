"""
Inference of a model with respect to a dataset:
    - Parse the dataset.
    - Evaluate the model.
    - Set the solution.
    - Set the metrics.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import time
import jax
import jax.numpy as jnp
from odesolver.compute import odesolve
from odesolver.compute import dataset


def get_infer(name, ode, sig, model, const, raw, param):
    """
    Inference of a model with respect to a dataset.
    """

    # init the timing
    timestamp = time.time()
    jax.debug.print("================================================= {:s} / {:.2f}", name, time.time() - timestamp)

    # load the dataset
    jax.debug.print("parse the dataset")
    dset = dataset.parse_dataset(raw, sig)
    t_out_mat = jnp.array(dset["t_out_mat"], dtype=jnp.float32)
    t_int_mat = jnp.array(dset["t_int_mat"], dtype=jnp.float32)
    dBdt_int_mat = jnp.array(dset["dBdt_int_mat"], dtype=jnp.float32)

    jax.debug.print("solve the equations")
    H_cmp_mat = odesolve.get_waveform(model, ode, t_int_mat, t_out_mat, dBdt_int_mat, const, param)

    jax.debug.print("set the solution")
    dset_sol = dataset.set_solution(dset, H_cmp_mat)

    jax.debug.print("set the metrics")
    dset_sol = dataset.set_metrics(dset_sol)

    # end the timing
    jax.debug.print("================================================= {:s} / {:.2f}", name, time.time() - timestamp)

    return dset_sol
