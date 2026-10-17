"""
Neural network based model with a single neural networks.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax.numpy as jnp
import equinox as eqx
from odesolver import model


class Model(model.Model):
    @staticmethod
    @eqx.filter_jit
    def get_ode(t, H, const, param, interp):
        """
        Function defining the ODE.
        Return the derivative of the states.
        A neural model is considered.
        """

        # extract the constants
        scl_H = const["scl_H"]
        scl_dBdt = const["scl_dBdt"]
        scl_dHdt = const["scl_dHdt"]

        # obtain the applied excitation
        dBdt = interp(t)

        # combine and scale the states
        state = jnp.append(H / scl_H, dBdt / scl_dBdt)

        # evaluate the neural network
        dHdt = scl_dHdt * param(state)

        return dHdt

    @staticmethod
    @eqx.filter_jit
    def get_init(const, param, interp):
        """
        Function returning the initial state of the system.
        All the initial states are set to a constant value.
        """

        # extract the variables
        var_size = const["var_size"]
        field_init = const["field_init"]

        # create the initial state
        H_init = field_init * jnp.ones(var_size)

        return H_init

    @staticmethod
    @eqx.filter_jit
    def get_out(t, y, const, param, interp):
        """
        Function defining the output variables that are stored in the solution.
        This function is called after every time steps.
        For this model, the function take the mean value of the states.
        """

        return jnp.mean(y, axis=0)

    @staticmethod
    @eqx.filter_jit
    def get_sol(ts, ys, const, param, interp):
        """
        Function extracting the results from the solution.
        This function is called a single time after the integration.
        For this model, the function does nothing.
        """

        return ys
