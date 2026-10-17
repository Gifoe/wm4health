"""
Optimizer based on the Optax solver:
    - This solver is gradient based (backward-differentiation).
    - This solver is enforcing the bounds with a penalty.
    - This solver is using a scalar objective function.
    - This solver is using a validation set.
    - This solver is using batches.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from odesolver.utils import conv_tracker


def _get_train_step(get_solver, optax_obj, state, value, idx):
    """
    Perform a training step (gradient and update).
    """

    (loss, grads) = eqx.filter_value_and_grad(get_solver)(value, idx)
    (updates, state) = optax_obj.update(grads, state)
    value = eqx.apply_updates(value, updates)

    return state, value, loss


def _get_batch_train(get_solver, optax_obj, batch_size, key, idx_all, state, value):
    """
    Run a dataset with batching (training)
    """

    # create the data loader
    loader_trn = _get_batch_loader(batch_size, key, idx_all)

    # iteration over the batches
    count = 0
    loss = 0.0
    for idx in loader_trn:
        (state, value, loss_tmp) = _get_train_step(get_solver, optax_obj, state, value, idx)
        count += len(idx)
        loss += loss_tmp

    # compute the losses
    loss = jnp.sqrt(loss / count)

    return loss, state, value


def _get_batch_eval(get_solver, batch_size, key, idx_all, value):
    """
    Run a dataset with batching (evaluation)
    """

    # create the data loader
    loader_trn = _get_batch_loader(batch_size, key, idx_all)

    # iteration over the batches
    count = 0
    loss = 0.0
    for idx in loader_trn:
        loss_tmp = get_solver(value, idx)
        count += len(idx)
        loss += loss_tmp

    # compute the losses
    loss = jnp.sqrt(loss / count)

    return loss


def _get_batch_loader(batch_size, key, idx_all):
    """
    Create a batched loader for the data.
    Random permutation of the dataset order.
    All the batches have the same size.
    Return the indices of the batches.
    """

    # get the data and batch size
    data_size = len(idx_all)
    batch_size = min(data_size, batch_size)

    # get the indices in a random order
    idx_all = jr.permutation(key, idx_all)

    # slice into batches
    idx_pos = 0
    while (idx_pos + batch_size) <= data_size:
        idx_batch = idx_all[idx_pos : idx_pos + batch_size]
        idx_pos += batch_size
        yield idx_batch


def get_optim(value, optimizer, fct_opt, fct_decode, fct_penalty, idx_trn, idx_val):
    """
    Optimizer based on the Optax solver
    """

    # extract data
    batch_size = optimizer["batch_size"]
    max_steps = optimizer["max_steps"]
    conv_data = optimizer["conv_data"]
    optax_obj = optimizer["optax_obj"]
    loader_key = optimizer["loader_key"]

    # split the dataloader random key for the different iterations
    (init_key, loader_key) = jr.split(loader_key, num=2)
    loader_key = jr.split(loader_key, num=max_steps)

    # get the optimization function
    def get_solver(value, idx):
        # get the parameter scaling
        param = fct_decode(value)
        penalty = fct_penalty(value)

        # get the loss value
        residuum = fct_opt(param, idx)
        loss = jnp.sum(penalty * residuum)

        return loss

    # get the convergence tracker
    conv_obj = conv_tracker.ConvTracker(conv_data)

    # get the initial state
    loss_trn = _get_batch_eval(get_solver, batch_size, init_key, idx_trn, value)
    loss_val = _get_batch_eval(get_solver, batch_size, init_key, idx_val, value)
    penalty = fct_penalty(value)

    # set the initial state
    stop = conv_obj.set_step(value, penalty, loss_trn, loss_val)
    if stop:
        value = conv_obj.get_best()
        sol = conv_obj.get_sol()
        return value, sol

    # initialize the optimizer
    state = optax_obj.init(eqx.filter(value, eqx.is_inexact_array))

    # run the training process
    for key in loader_key:
        # run the training set
        (loss_trn, state, value) = _get_batch_train(get_solver, optax_obj, batch_size, key, idx_trn, state, value)

        # run the validation set
        loss_val = _get_batch_eval(get_solver, batch_size, init_key, idx_val, value)

        # update convergence
        penalty = fct_penalty(value)
        stop = conv_obj.set_step(value, penalty, loss_trn, loss_val)

        # break the iteration
        if stop:
            value = conv_obj.get_best()
            sol = conv_obj.get_sol()
            return value, sol

    # get the best trial
    value = conv_obj.get_best()
    sol = conv_obj.get_sol()

    return value, sol
