"""
Equation based model consisting from the series connection of the following elements:
    - A nonlinear resistor.
    - A nonlinear inductor.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax.numpy as jnp
import equinox as eqx
from odesolver import model
from odesolver.utils import softjax_fct


class Model(model.Model):
    @staticmethod
    @eqx.filter_jit
    def get_ode(t, H, const, param, interp):
        """
        Function defining the ODE.
        Return the derivative of the states.
        A nonlinear circuit is considered.
        """

        # extract the variables
        r_mul = param["r_mul"]
        r_scl = param["r_scl"]
        a_sat = param["a_sat"]
        mu_sat = param["mu_sat"]
        p_sat = param["p_sat"]

        # extract the constants
        r_min = const["r_min"]
        r_max = const["r_max"]
        r_soft = const["r_soft"]
        mu_min = const["mu_min"]
        mu_max = const["mu_max"]
        mu_soft = const["mu_soft"]
        abs_soft = const["abs_soft"]
        abs_offset = const["abs_offset"]

        # obtain the applied excitation
        dBdt = interp(t)

        # define the permeability
        mu0 = 4 * jnp.pi * 1e-7

        # get the absolute value of the field (handling of singularity)
        H_abs = softjax_fct.abs(H, abs_soft) + abs_offset

        # compute the nonlinear permeability
        mu_nli = mu_sat / (1.0 + ((H_abs / a_sat) ** p_sat))

        # compute the nonlinear resistance
        r_nli = r_mul * (1.0 + H_abs / r_scl)

        # clamp the values (avoid instabilities)
        mu_nli = softjax_fct.clip(mu_nli, mu_min, mu_max, mu_soft)
        r_nli = softjax_fct.clip(r_nli, r_min, r_max, r_soft)

        # compute the nonlinear dynamic term
        dHdt = (dBdt - r_nli * H) / (mu0 * mu_nli)

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
