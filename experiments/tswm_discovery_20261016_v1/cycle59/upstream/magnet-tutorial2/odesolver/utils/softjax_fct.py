"""
Smooth approximations of non-smooth functions.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax.nn as nn
import jax.numpy as jnp


def maximum(x, y, mu):
    """
    Smooth implementation of the maximum.
    The smoothness is controlled with mu.
    """

    out = y + mu * nn.softplus((x - y) / mu)

    return out


def minimum(x, y, mu):
    """
    Smooth implementation of the minimum.
    The smoothness is controlled with mu.
    """

    out = y - mu * nn.softplus((y - x) / mu)

    return out


def clip(x, v_min, v_max, mu):
    """
    Smooth implementation of the clip.
    The smoothness is controlled with mu.
    """

    out = maximum(x, v_min, mu) + minimum(x, v_max, mu) - x

    return out


def penalty(x, v_min, v_max, mu):
    """
    Smooth bathtub penalty function.
    The smoothness is controlled with mu.
    """

    out = maximum(0.0, x - v_min, mu) - minimum(0.0, x - v_max, mu)

    return out


def abs(x, mu):
    """
    Smooth implementation of the absolute value.
    The smoothness is controlled with mu.
    """

    out = x * jnp.tanh(x / mu)

    return out
