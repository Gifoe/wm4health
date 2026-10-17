"""
Neural network based model with two neural networks:
    - A first network described the impact of the states.
    - A second network described the impact of the excitation.
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
        A neural model is considered.
        """

        # extract the variables
        mlp_src = param["mlp_src"]
        mlp_state = param["mlp_state"]

        # extract the constants
        scl_H = const["scl_H"]
        scl_dBdt = const["scl_dBdt"]
        scl_dHdt = const["scl_dHdt"]
        abs_soft = const["abs_soft"]

        # obtain the applied excitation
        dBdt = interp(t)

        # take the absolute value is used to enforce symmetry
        H_abs = softjax_fct.abs(H / scl_H, abs_soft)

        # impact the excitation (the absolute value is used to enforce symmetry)
        v_src = mlp_src(H_abs)

        # impact the states (the absolute value is used to enforce symmetry)
        v_state = mlp_state(H_abs)

        # compute the dynamic terms
        dHdt = scl_dHdt * (v_src * dBdt / scl_dBdt + v_state * H / scl_H)

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
