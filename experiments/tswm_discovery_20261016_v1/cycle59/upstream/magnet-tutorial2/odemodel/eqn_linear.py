"""
Equation based model consisting from the series connection of the following elements:
    - A linear resistor.
    - A linear inductor.
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
        A linear circuit is considered.
        """

        # extract the variables
        r_lin = param["r_lin"]
        mu_lin = param["mu_lin"]

        # obtain the applied excitation
        dBdt = interp(t)

        # define the permeability
        mu0 = 4 * jnp.pi * 1e-7

        # compute the dynamic term
        dHdt = (dBdt - r_lin * H) / (mu0 * mu_lin)

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
