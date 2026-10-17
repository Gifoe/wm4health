"""
Load and parse the CSV files containing the measurement results.
Write the resulting dataset in a JSON/gzip file.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy.random as nr
from odesolver import serialize
from odesolver import parsing


if __name__ == "__main__":
    # tag describing the original data files
    tag_input = "data/N87_5"

    # time domain sampling intervals
    dt = 62.5e-9

    # temperature to be considered
    T = 25.0

    # size of the dataset to be created (training and test sets)
    n_train = 48
    n_test = 24

    # seed for the random selection
    rng = nr.default_rng(1234)

    # load and parse the dataset
    raw = parsing.get_data(tag_input, dt, T, n_train, n_test, rng)

    # write the results
    serialize.write_dataframe("data/raw_magnet.pkl", raw)
