"""
Class for logging and evaluating the solver convergence.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import time
import jax
import jax.numpy as jnp


class ConvTracker:
    def __init__(self, conv_data):
        """
        Constructor (set parameters and initialize data).
        """

        # set the parameters
        self.conv_steps = conv_data["conv_steps"]
        self.conv_rate = conv_data["conv_rate"]
        self.loss_target = conv_data["loss_target"]

        # variables for storing the convergence
        self.step_best = 0
        self.step_count = 0
        self.value_best = None
        self.time_last = time.time()
        self.penalty_best = jnp.inf
        self.loss_val_best = jnp.inf
        self.loss_trn_best = jnp.inf
        self.loss_trn_vec = jnp.array([], dtype=jnp.float32)
        self.loss_val_vec = jnp.array([], dtype=jnp.float32)

    def get_sol(self):
        """
        Return the convergence summary.
        """

        sol = {
            "step_count": self.step_count,
            "step_best": self.step_best,
            "value_best": self.value_best,
            "penalty_best": self.penalty_best,
            "loss_val_best": self.loss_val_best,
            "loss_trn_best": self.loss_trn_best,
            "loss_trn_vec": self.loss_trn_vec,
            "loss_val_vec": self.loss_val_vec,
        }

        return sol

    def get_best(self):
        """
        Return the parameters for the best trial.
        """

        return self.value_best

    def set_step(self, value, penalty, loss_trn, loss_val):
        """
        Set the step values, log the data, and check for convergence.
        """

        # measure step time
        step_time = time.time() - self.time_last
        self.time_last = time.time()

        # add the step count
        self.step_count += 1

        # update the array used for convergence
        self.loss_trn_vec = jnp.append(self.loss_trn_vec, loss_trn)
        self.loss_val_vec = jnp.append(self.loss_val_vec, loss_val)

        # update the best validation loss
        if loss_val < self.loss_val_best:
            self.step_best = self.step_count
            self.loss_val_best = loss_val
            self.loss_trn_best = loss_trn
            self.penalty_best = penalty
            self.value_best = value

        # compute the convergence rate
        if self.step_count >= (2 * self.conv_steps):
            loss_a = jnp.min(self.loss_val_vec[: -self.conv_steps])
            loss_b = jnp.min(self.loss_val_vec[-self.conv_steps :])
            conv_val = (loss_a - loss_b) / jnp.min(self.loss_val_vec)
        else:
            conv_val = jnp.inf

        # show the current training status
        jax.debug.print("step = {:d} / {:d}", self.step_count, self.step_best)
        jax.debug.print("    penalty = {:.4f} / {:.4f}", penalty, self.penalty_best)
        jax.debug.print("    loss_trn = {:.4f} / {:.4f}", loss_trn, self.loss_trn_best)
        jax.debug.print("    loss_val = {:.4f} / {:.4f}", loss_val, self.loss_val_best)
        jax.debug.print("    step_time = {:.4f}", step_time)
        jax.debug.print("    conv_val = {:+.4f}", conv_val)

        # break training if invalid values are found
        if not jnp.isfinite(loss_trn) or not jnp.isfinite(loss_val):
            jax.debug.print("    solver convergence: non-finite values")
            return True

        # break training if the target is reached
        if self.loss_val_best < self.loss_target:
            jax.debug.print("    solver convergence: target reached")
            return True

        # break training if converged / stalled
        if conv_val < self.conv_rate:
            jax.debug.print("    solver convergence: convergence rate")
            return True

        return False
