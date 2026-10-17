"""
Various functions for serializing / deserializing data.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import pandas
import equinox


def load_dataframe(filename):
    """
    Load a Pickle file containing a Pandas DataFrame.
    """

    return pandas.read_pickle(filename)


def write_dataframe(filename, dataframe):
    """
    Save a Pandas DataFrame as a Pickle file.
    """

    dataframe.to_pickle(filename)


def load_equinox(filename, pytree):
    """
    Load a serialized JAX Pytree from a file (with a provided schema).
    """

    return equinox.tree_deserialise_leaves(filename, pytree)


def write_equinox(filename, pytree):
    """
    Serialize a JAX Pytree to a file.
    """

    equinox.tree_serialise_leaves(filename, pytree)
