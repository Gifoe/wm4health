"""
Module defining the neural network-based model parameters.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax.nn as nn
import jax.random as jr
import equinox as eqx
from odemodel import ann_dual
from odemodel import ann_single


def get_ann_single():
    """
    Define a neural network-based model.
    """

    # create the object defining the model
    model = ann_single.Model()

    # split the key for generating the model initial values
    model_key = jr.key(1234)
    (key_1, key_2) = jr.split(model_key, num=2)

    # number of states for the ODE
    var_size = 3

    # initial value for the states
    field_init = 0

    # define the neural networks (input is augmented for the excitation)
    param = eqx.nn.MLP(
        in_size=var_size + 1,
        out_size=var_size + 0,
        depth=2,
        width_size=12,
        activation=nn.tanh,
        key=key_2,
    )

    # constant values for the model (non-optimized)
    const = {
        "scl_H": 1e2,                # scaling factor for the field
        "scl_dBdt": 10e3,            # scaling factor for the flux derivative
        "scl_dHdt": 10e6,            # scaling factor for the field derivative
        "var_size": var_size,        # number of states for the ODE
        "field_init": field_init,    # initial value for the states
    }

    return model, param, const


def get_ann_dual():
    """
    Define a neural network-based model.
    """

    # create the object defining the model
    model = ann_dual.Model()

    # split the key for generating the model initial values
    model_key = jr.key(1234)
    (key_1, key_2) = jr.split(model_key, num=2)

    # number of states for the ODE
    var_size = 3

    # initial value for the states
    field_init = 0

    # define the neural networks
    #   - mlp_src: neural network applied to the excitation
    #   - mlp_state: neural network applied to the states
    param = {
        "mlp_src": eqx.nn.MLP(
            in_size=var_size,
            out_size=var_size,
            depth=2,
            width_size=12,
            activation=nn.tanh,
            key=key_1,
        ),
        "mlp_state": eqx.nn.MLP(
            in_size=var_size,
            out_size=var_size,
            depth=2,
            width_size=12,
            activation=nn.tanh,
            key=key_2,
        ),
    }

    # constant values for the model (non-optimized)
    const = {
        "scl_H": 1e2,                # scaling factor for the field
        "scl_dBdt": 10e3,            # scaling factor for the flux derivative
        "scl_dHdt": 10e6,            # scaling factor for the field derivative
        "abs_soft": 0.01,            # softness factor for the absolute value
        "var_size": var_size,        # number of states for the ODE
        "field_init": field_init,    # initial value for the states
    }

    return model, param, const
