"""
Train a model with respect to a dataset:
    - Train an equation-based model.
    - Train a neural network-based model.

The dataset is split into a training and validation sets.
The convergence of the optimizer is monitored.

For the equation based models, the following features are available:
    - Various solvers (Optax, Optimistix, SciPy)
    - Scaling of the variables.
    - Penalty for bound violations.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import time
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.tree as jtr
import jax.flatten_util as jfl
from odesolver.compute import odesolve
from odesolver.compute import dataset
from odesolver.utils import softjax_fct
from odesolver.optim import scipy_evo
from odesolver.optim import scipy_lat
from odesolver.optim import optax_grad
from odesolver.optim import optx_min
from odesolver.optim import optx_lsq


def _get_train_indices(key, dset, frac_valid):
    """
    Split the dataset into a training and validation sets.
    """

    # extract the dataset labels
    is_test = jnp.array(dset["stat_var"]["is_test"], dtype=jnp.bool)
    is_train = jnp.array(dset["stat_var"]["is_train"], dtype=jnp.bool)
    idx_fitting = jnp.flatnonzero(is_train)

    # all the samples should be assigned
    assert jnp.count_nonzero(is_test) > 0, "invalid empty test dataset"
    assert jnp.count_nonzero(is_train) > 0, "invalid empty train dataset"
    assert jnp.all(is_train ^ is_test), "invalid train/valid dataset"

    # permute and split the data
    idx_fitting = jr.permutation(key, idx_fitting)
    n_valid = int(round(len(idx_fitting) * frac_valid))
    idx_val = idx_fitting[:n_valid]
    idx_trn = idx_fitting[n_valid:]

    # check that all the sets are not empty
    assert len(idx_trn) > 0, "invalid empty test dataset"
    assert len(idx_val) > 0, "invalid empty train dataset"

    return idx_trn, idx_val


def _get_disp_param(param):
    """
    Display the model parameters.
    """

    for key, val in param.items():
        if jnp.isscalar(val):
            print(f"{key} = {val:.3e}")
        else:
            val = " , ".join([f"{tmp:.3e}" for tmp in val])
            jax.debug.print("{:s} = [{:s}]", key, val)


def _get_scale_param(param, bnd):
    """
    Functions for scaling the variables and bounds for the equation-based models.
        - Extracting a param vector into a param dict (fct_decode).
        - Computing a penalty for bound violations (fct_penalty).
        - Computing the initial value and bounds.
    """

    # extract the data
    param_min = bnd["param_min"]
    param_max = bnd["param_max"]
    scaling = bnd["scaling"]

    # get the scaling and offset for the scaling values
    var_scale = scaling["var_scale"]
    var_shift = scaling["var_shift"]

    # get the scaling and offset for the bound violations
    bnd_scale = scaling["bnd_scale"]
    bnd_shift = scaling["bnd_shift"]

    # softness factor for the bound penalty
    bnd_soft = scaling["bnd_soft"]

    # get the tree definition
    param = dict(sorted(param.items()))
    param_min = dict(sorted(param_min.items()))
    param_max = dict(sorted(param_max.items()))
    assert jtr.structure(param) == jtr.structure(param_min), "invalid structure for the parameters"
    assert jtr.structure(param) == jtr.structure(param_max), "invalid structure for the parameters"

    # flatten the tree to arrays
    (param, fct_unravel) = jfl.ravel_pytree(param)
    (param_min, _) = jfl.ravel_pytree(param_min)
    (param_max, _) = jfl.ravel_pytree(param_max)

    # compute the linear transform from the original value to normalized intervals
    slope = (param_max - param_min) / 2.0
    offset = (param_max + param_min) / 2.0

    # compute the bounds for the solver (scaled values)
    lb = -jnp.ones(len(param_min)) * var_scale + var_shift
    ub = +jnp.ones(len(param_max)) * var_scale + var_shift

    # scale the initial values to normalized intervals
    value = (param - offset) / slope

    # scale the normalized intervals to the solver ranges
    value = value * var_scale + var_shift

    # function for computing a penalty for bound violations
    def fct_penalty(value_tmp):
        # linear penalty function for the bound violations
        p_ub = softjax_fct.maximum(0.0, value_tmp - ub, bnd_soft)
        p_lb = softjax_fct.minimum(0.0, value_tmp - lb, bnd_soft)

        # add an offset and a scaling factor for the bound violations
        penalty_tmp = bnd_shift + bnd_scale * jnp.sum(p_ub - p_lb)

        return penalty_tmp

    # function for extracting a param vector into a param dict
    def fct_decode(value_tmp):
        # unscale the solver ranges to the normalized intervals
        value_tmp = (value_tmp - var_shift) / var_scale

        # unscale the normalized intervals to the original values
        param_tmp = value_tmp * slope + offset

        # cast to the original format
        param_tmp = fct_unravel(param_tmp)

        return param_tmp

    return fct_decode, fct_penalty, value, lb, ub


def _get_optim_objective(model, ode, dset, const, fact_power, fact_field):
    """
    Get the function returning the residuum across the dataset for given parameters.
    """

    # load the dataset
    t_out_mat = jnp.array(dset["t_out_mat"], dtype=jnp.float32)
    t_int_mat = jnp.array(dset["t_int_mat"], dtype=jnp.float32)
    dBdt_int_mat = jnp.array(dset["dBdt_int_mat"], dtype=jnp.float32)
    dBdt_ref_mat = jnp.array(dset["dBdt_ref_mat"], dtype=jnp.float32)
    H_ref_mat = jnp.array(dset["H_ref_mat"], dtype=jnp.float32)

    # compute the residuum for given parameters and dataset indices
    def fct_opt(param, idx):
        # get the power and field relative errors
        (err_power_vec, err_field_vec) = odesolve.get_error(
            model, ode,
            t_int_mat[idx], t_out_mat[idx],
            dBdt_int_mat[idx], dBdt_ref_mat[idx], H_ref_mat[idx],
            const, param,
        )

        # assemble the errors into a single residuum
        residuum = fact_power * err_power_vec + fact_field * err_field_vec

        return residuum

    return fct_opt


def get_train_eqn(name, ode, sig, bnd, opt, model, const, raw, param):
    """
    Train an equation-based model with respect to a dataset.
    """

    # extract data
    fact_power = opt["fact_power"]
    fact_field = opt["fact_field"]
    frac_valid = opt["frac_valid"]
    split_key = opt["split_key"]
    optimizer = opt["optimizer"]
    method = opt["method"]

    # init the timing
    timestamp = time.time()
    jax.debug.print("================================================= {:s} / {:.2f}", name, time.time() - timestamp)

    # show the parameters
    jax.debug.print("========= initial parameters")
    _get_disp_param(param)
    jax.debug.print("========= initial parameters")

    # parse the dataset
    dset = dataset.parse_dataset(raw, sig)

    # get the training and validation samples
    (idx_trn, idx_val) = _get_train_indices(split_key, dset, frac_valid)
    idx_trn_val = jnp.concatenate((idx_trn, idx_val))

    # get the scaling functions
    (fct_decode, fct_penalty, value, lb, ub) = _get_scale_param(param, bnd)

    # get the optimization objective function
    fct_opt = _get_optim_objective(model, ode, dset, const, fact_power, fact_field)

    # run the optimizer
    jax.debug.print("========= optimizer starting")
    if method == "evo":
        (value, sol) = scipy_evo.get_optim(lb, ub, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn, idx_val)
    elif method == "opt":
        (value, sol) = optax_grad.get_optim(value, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn, idx_val)
    elif method == "lat":
        (value, sol) = scipy_lat.get_optim(lb, ub, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn_val)
    elif method == "min":
        (value, sol) = optx_min.get_optim(value, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn_val)
    elif method == "lsq":
        (value, sol) = optx_lsq.get_optim(value, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn_val)
    else:
        raise ValueError("invalid method")
    jax.debug.print("========= optimizer finished")

    # decode the final values into original ranges
    keys = param.keys()
    param = fct_decode(value)
    param = {tmp: param[tmp] for tmp in keys}

    # show the parameters
    jax.debug.print("========= final parameters")
    _get_disp_param(param)
    jax.debug.print("========= final parameters")

    # end the timing
    jax.debug.print("================================================= {:s} / {:.2f}", name, time.time() - timestamp)

    return param, sol


def get_train_ann(name, ode, sig, opt, model, const, raw, param):
    """
    Train a neural network-based model with respect to a dataset.
    """

    # extract data
    fact_power = opt["fact_power"]
    fact_field = opt["fact_field"]
    frac_valid = opt["frac_valid"]
    split_key = opt["split_key"]
    optimizer = opt["optimizer"]

    # init the timing
    timestamp = time.time()
    jax.debug.print("================================================= {:s} / {:.2f}", name, time.time() - timestamp)

    # parse the dataset
    dset = dataset.parse_dataset(raw, sig)

    # get the training and validation samples
    (idx_trn, idx_val) = _get_train_indices(split_key, dset, frac_valid)

    # get the optimization objective function
    fct_opt = _get_optim_objective(model, ode, dset, const, fact_power, fact_field)

    # penalty is not present for neural networks
    penalty_tmp = 1.0

    # get the encoded values (do nothing for neural parameters)
    def fct_decode(value_tmp):
        return value_tmp

    # get the penalty term (do nothing for neural parameters)
    def fct_penalty(_):
        return penalty_tmp

    # run the optimizer
    (param, sol) = optax_grad.get_optim(param, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn, idx_val)

    # end the timing
    jax.debug.print("================================================= {:s} / {:.2f}", name, time.time() - timestamp)

    return param, sol
