"""
Class with the abstract definition of a model (equation-based or neural network-based).
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import abc


class Model(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_ode(t, y, const, param, interp):
        """
        Function defining the ODE.
        Return the derivative of the states.
        """

        pass

    @staticmethod
    @abc.abstractmethod
    def get_init(const, param, interp):
        """
        Function returning the initial state of the system.
        """

        pass

    @staticmethod
    @abc.abstractmethod
    def get_out(t, y, const, param, interp):
        """
        Function defining the output variables that are stored in the solution.
        This function is called after every time steps.
        """

        pass

    @staticmethod
    @abc.abstractmethod
    def get_sol(ts, ys, const, param, interp):
        """
        Function extracting the results from the solution.
        This function is called a single time after the integration.
        Depending on the implementation, the model may become acausal.
        """

        pass
