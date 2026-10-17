"""
Module defining the equation-based model parameters.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax.numpy as jnp
from odemodel import eqn_nonlinear
from odemodel import eqn_linear


def get_eqn_linear():
    """
    Define an equation-based linear model.
    """

    # create the object defining the model
    model = eqn_linear.Model()

    # number of states for the ODE
    var_size = 1

    # initial value for the states
    field_init = 0

    # initial values for the parameters
    #   - r_lin: value of the resistance
    #   - mu_lin: value of permeability
    #   - nan: values are created from bounds
    param = {
        "r_lin": jnp.nan * jnp.ones(var_size),
        "mu_lin": jnp.nan * jnp.ones(var_size),
    }

    # scaling factor for the variables
    #   - the variables are scaled into: [var_scale+var_shift , var_scale+var_shift]
    #   - the bounds violation penalty is scaled: bnd_shift + bnd_scale * sum(penalty_vec)
    scaling = {
        "var_scale": 10.0,    # scaling factor for the variables
        "var_shift": 0.0,     # offset value for the variables
        "bnd_scale": 0.5,     # scaling factor for the bound violations
        "bnd_shift": 1.0,     # offset value for the bound violations
        "bnd_soft": 0.1,      # softness factor for the bound violations
    }

    # lower bounds and upper bounds for the different parameters
    bnd = {
        "param_min": {
            "r_lin": 25.0 * jnp.ones(var_size),
            "mu_lin": 2000.0 * jnp.ones(var_size),
        },
        "param_max": {
            "r_lin": 100.0 * jnp.ones(var_size),
            "mu_lin": 5000.0 * jnp.ones(var_size),
        },
        "scaling": scaling,
    }

    # constant values for the model (non-optimized)
    const = {
        "var_size": var_size,        # number of states for the ODE
        "field_init": field_init,    # initial value for the states
    }

    return model, param, const, bnd


def get_eqn_nonlinear():
    """
    Define an equation-based nonlinear model.
    """

    # create the object defining the model
    model = eqn_nonlinear.Model()

    # number of states for the ODE
    var_size = 2

    # initial value for the states
    field_init = 0

    # initial values for the parameters
    #   - r_mul: value of the linear resistance
    #   - a_sat: steepness of the saturation behavior
    #   - mu_sat: initial permeability of the inductance
    #   - p_sat: exponent of the saturation behavior
    #   - nan: values are created from bounds
    param = {
        "r_mul": jnp.nan * jnp.ones(var_size),
        "r_scl": jnp.nan * jnp.ones(var_size),
        "a_sat": jnp.nan * jnp.ones(var_size),
        "mu_sat": jnp.nan * jnp.ones(var_size),
        "p_sat": jnp.nan * jnp.ones(var_size),
    }

    # scaling factor for the variables
    #   - the variables are scaled into: [var_scale+var_shift , var_scale+var_shift]
    #   - the bounds violation penalty is scaled: bnd_shift + bnd_scale * sum(penalty_vec)
    scaling = {
        "var_scale": 10.0,    # scaling factor for the variables
        "var_shift": 0.0,     # offset value for the variables
        "bnd_scale": 0.5,     # scaling factor for the bound violations
        "bnd_shift": 1.0,     # offset value for the bound violations
        "bnd_soft": 0.1,      # softness factor for the bound violations
    }

    # lower bounds and upper bounds for the different parameters
    bnd = {
        "param_min": {
            "r_mul": 1.0 * jnp.ones(var_size),
            "r_scl": 5.0 * jnp.ones(var_size),
            "a_sat": 10.0 * jnp.ones(var_size),
            "mu_sat": 2000.0 * jnp.ones(var_size),
            "p_sat": 1.0 * jnp.ones(var_size),
        },
        "param_max": {
            "r_mul": 350.0 * jnp.ones(var_size),
            "r_scl": 75.0 * jnp.ones(var_size),
            "a_sat": 150.0 * jnp.ones(var_size),
            "mu_sat": 5000.0 * jnp.ones(var_size),
            "p_sat": 5.0 * jnp.ones(var_size),
        },
        "scaling": scaling,
    }

    # constant values for the model (non-optimized)
    const = {
        "mu_min": 100.0,           # permeability lower bounds
        "mu_max": 10000.0,         # permeability upper bounds
        "mu_soft": 100.0,          # permeability clip softness
        "r_min": 1.0,              # resistance lower bounds
        "r_max": 1000.0,           # resistance upper bounds
        "r_soft": 1.0,             # resistance clip softness
        "abs_soft": 0.1,           # softness factor for the absolute value
        "abs_offset": 0.1,         # softness factor for the absolute value
        "var_size": var_size,      # number of states for the ODE
        "field_init": field_init,  # initial value for the states
    }

    return model, param, const, bnd
