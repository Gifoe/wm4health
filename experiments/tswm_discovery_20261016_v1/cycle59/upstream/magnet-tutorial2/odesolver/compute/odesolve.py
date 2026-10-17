"""
Module used for the integration of the ODE models:
    - Integrate the ODEs and return the magnetic field.
    - Integrate the ODEs and return error metrics.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx


@eqx.filter_jit
def _get_solution(t_int, t_out, v_int, const, param, model, ode):
    """
    Integrate an ODE model for a given excitation.
    """

    # extract the data
    max_steps = ode["max_steps"]
    dt_step = ode["dt_step"]
    dt_add = ode["dt_add"]
    solver = ode["solver"]
    adjoint = ode["adjoint"]

    # get the ode integration bounds
    t_ode_0 = 0.0
    t_ode_1 = jnp.max(t_out) + dt_add

    # get the ode interpolation bounds
    t_int_0 = jnp.min(t_int)
    t_int_1 = jnp.max(t_int)

    # create a periodic interpolant with the input signal
    interp_obj = dfx.LinearInterpolation(t_int, v_int)
    interp = lambda t: interp_obj.evaluate((t + t_int_0) % (t_int_1 - t_int_0))

    # pack the static arguments for passing them to the ODE solver
    args = (const, param, interp)

    # get the definition of the ODE (return the derivative of the states)
    ode_fct = lambda t, y, args: model.get_ode(t, y, args[0], args[1], args[2])

    # get the output function (return the variables to be stored)
    out_fct = lambda t, y, args: model.get_out(t, y, args[0], args[1], args[2])

    # get the ODE solver
    solver_fct = getattr(dfx, solver)
    solver_obj = solver_fct()

    # get the adjoint for the derivatives
    adjoint_fct = getattr(dfx, adjoint)
    adjoint_obj = adjoint_fct()

    # get the ODE objects
    ctrl_obj = dfx.ConstantStepSize()
    save_obj = dfx.SaveAt(fn=out_fct, ts=t_out)
    term_obj = dfx.ODETerm(ode_fct)

    # get the initial condition
    y_init = model.get_init(const, param, interp)

    # solve the ODE
    sol = dfx.diffeqsolve(
        term_obj,
        solver_obj,
        t0=t_ode_0,
        t1=t_ode_1,
        dt0=dt_step,
        y0=y_init,
        args=args,
        saveat=save_obj,
        adjoint=adjoint_obj,
        stepsize_controller=ctrl_obj,
        max_steps=max_steps,
    )

    # extract the results from the solution
    u_out = model.get_sol(sol.ts, sol.ys, const, param, interp)

    return u_out


@eqx.filter_jit
def _get_objective(dBdt_ref, H_ref, H_cmp):
    """
    Compute the error metrics between the model and the dataset:
        - Compute the relative error of the active power (losses).
        - Compute the relative RMS error for the magnetic field.
    """

    # compute the field error
    H_err = H_cmp - H_ref

    # compute the rms values of the fields
    H_err_rms_sq = jnp.mean(H_err**2)
    H_ref_rms_sq = jnp.mean(H_ref**2)

    # compute the active power
    P_ref = jnp.mean(dBdt_ref * H_ref)
    P_cmp = jnp.mean(dBdt_ref * H_cmp)

    # compute the squared relative error for the power
    err_power = ((P_cmp - P_ref) / P_ref) ** 2

    # compute the squared relative error for the field
    err_field = H_err_rms_sq / H_ref_rms_sq

    return err_power, err_field


@eqx.filter_jit
def get_waveform(model, ode, t_int_mat, t_out_mat, dBdt_int_mat, const, param):
    """
    Integrate the ODEs for the complete dataset and return the magnetic field.
    """

    # compute the solution for a single index
    def get_slice(t_int_tmp, t_out_tmp, dBdt_int_tmp):
        H_cmp_tmp = _get_solution(t_int_tmp, t_out_tmp, dBdt_int_tmp, const, param, model, ode)
        return H_cmp_tmp

    # map the complete solution
    H_cmp_mat = jax.vmap(get_slice)(t_int_mat, t_out_mat, dBdt_int_mat)

    return H_cmp_mat


@eqx.filter_jit
def get_error(model, ode, t_int_mat, t_out_mat, dBdt_int_mat, dBdt_ref_mat, H_ref_mat, const, param):
    """
    Integrate the ODEs for the complete dataset and return the error metrics.
    """

    # compute the solution for a single index
    def get_slice(t_int_tmp, t_out_tmp, dBdt_int_tmp, dBdt_ref_tmp, H_ref_tmp):
        H_cmp_tmp = _get_solution(t_int_tmp, t_out_tmp, dBdt_int_tmp, const, param, model, ode)
        (err_power, err_field) = _get_objective(dBdt_ref_tmp, H_ref_tmp, H_cmp_tmp)
        return err_power, err_field

    # map the complete solution
    (err_power_vec, err_field_vec) = jax.vmap(get_slice)(t_int_mat, t_out_mat, dBdt_int_mat, dBdt_ref_mat, H_ref_mat)

    return err_power_vec, err_field_vec
