# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:51:57 2020

analytics.py contains functions used for analysis of bubble-growth models.

@author: Andy
"""

import sys
sys.path.append('../libs/')

import numpy as np
import matplotlib.pyplot as plt

# imports custom libraries
import bubble
import bubbleflow
import finitediff as fd

# CONVERSIONS
s_2_us = 1E6
s_2_ms = 1E3
m_2_um = 1E6
m_2_nm = 1E9

########################### FUNCTION DEFINITIONS ##############################



def compare_dcdr(N_list, dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v, R_max,
                    polyol_data_file, eos_co2_file, dt_max_list=None):
    """
    Compares concentration gradient at interface of bubble b/w Epstein-Plesset
    model and numerical model.
    """
    # initializes list of numerically computed concentration gradients
    dcdr_num_list = []
    # initializes list of times [s]
    t_num_list = []

    # first performs Epstein-Plesset computation as benchmark
    t_eps, m, D, p, p_bub, if_tension,\
    c_s, c_bulk, R, rho_co2 = bubble.grow(dt, t_nuc, p_s, R_nuc, p_atm, L,
                                        p_in, v, polyol_data_file, eos_co2_file)
    # computes concentration gradient at bubble interface
    dcdr_eps = bubble.calc_dcdr_eps(c_bulk, c_s, R, D, np.asarray(t_eps) - t_nuc)

    # then performs numerical computation for different grid spacings
    for i, N in enumerate(N_list):
        if dt_max_list is not None:
            dt_max = dt_max_list[i]
        else:
            dt_max = None
        # performs simulation
        t_flow, c, t_num, m, D, p, \
        p_bub, if_tension, c_bub, \
        c_bulk, R, rho_co2, v, r_arr = bubbleflow.numerical_eps_pless_fix_D( \
                                        dt, t_nuc, p_s, R_nuc, L, p_in, v,
                                        R_max, N, polyol_data_file, eos_co2_file,
                                        dt_max=dt_max)

        # uses 2nd-order Taylor stencil
        dr = r_arr[1] - r_arr[0]
        dcdr_num_list += [[fd.dydx_fwd_2nd(c[i][0], c[i][1], c[i][2], dr) \
                            for i in range(len(c))]]
        # saves list of times [s]
        t_num_list += [t_num]

    return t_eps, dcdr_eps, t_num_list, dcdr_num_list


def fit_growth_to_pt(t_bubble, R_bubble, t_nuc_lo, t_nuc_hi, growth_fn, args,
                     i_t_nuc, sigma_R=0.01, ax=None, max_iter=12):
    """
    Fits the bubble growth to a given bubble radius at a given time. Plots
    the different trajectories if an axis handle is given.
    """
    # inserts place-holder (0) for nucleation time in arguments list
    args.insert(i_t_nuc, 0)
    # initializes plot to show the trajectories of different guesses
    if ax is not None:
        ax.plot(t_bubble*s_2_ms, R_bubble*m_2_um, 'g*', ms=12, label='fit pt')

    # initializes counter of number of iterations
    n_iter = 0

    # computes bubble growth trajectory with lowest nucleation time
    args[i_t_nuc] = t_nuc_lo
    t, m, D, p, p_bubble, if_tension, c_s, \
                                c_bulk, R, rho_co2 = growth_fn(*tuple(args))

        # finds index of timeline corresponding to measurement of bubble size
    i_bubble = next(i for i in range(len(t)) if t[i] >= t_bubble)
    R_bubble_pred = R[i_bubble]
    if R_bubble_pred < R_bubble:
        print('Predicted bubble radius is larger than fit for lowest nucleation time. Terminating early.')
        results = (t, m, D, p, p_bubble, if_tension, c_s, R, rho_co2)
        return t_nuc_lo, results

    # computes error in using lowest nucleation time
    err_R = np.abs(R_bubble_pred - R_bubble)/R_bubble # fractional error in bubble radius

    # bisection algorithm searches for nucleation time yielding accurate R
    while err_R > sigma_R:
        # calculates new nucleation time as middle of the two bounds (bisection algorithm)
        t_nuc = (t_nuc_lo + t_nuc_hi)/2
        # computes bubble growth trajectory with new bubble nucleation time
        args[i_t_nuc] = t_nuc
        t, m, D, p, p_bubble, if_tension, \
                        c_s, c_bulk, R, rho_co2 = growth_fn(*tuple(args))
        # finds index of timeline corresponding to measurement of bubble size
        i_bubble = next(i for i in range(len(t)) if t[i] >= t_bubble)
        R_bubble_pred = R[i_bubble]
        err_R = np.abs(R_bubble_pred - R_bubble)/R_bubble # fractional error in bubble radius
        # predicted bubble radius too large means nucleation time is too early, so we raise the lower bound
        if R_bubble_pred > R_bubble:
            t_nuc_lo = t_nuc
        # otherwise, nucleation time is too late, so we lower the upper bound
        else:
            t_nuc_hi = t_nuc

        print('t_nuc = {0:.3f} ms and error in R is {1:.4f}.'.format(t_nuc*s_2_ms, err_R))

        # plots the guessed growth trajectory
        if ax is not None:
            ax.plot(np.array(t)*s_2_ms, np.array(R)*m_2_um,
                    label=r'$t_{nuc}=$' + '{0:.3f} ms'.format(t_nuc*s_2_ms))

        n_iter += 1
        if n_iter == max_iter:
            print('Max iterations {0:d} reached but tolerance of R {1:.4f} not achieved.'.format(max_iter, sigma_R))
            print('Nucleation time is {0:.3f} ms.'.format(t_nuc*s_2_ms))
            break

    if n_iter < max_iter:
        print('Error in bubble radius is below tolerance of {0:.4f} for nucleation time t = {1:.3f} ms' \
          .format(sigma_R, t_nuc*s_2_ms))
    if ax is not None:
        # formats plot of guessed trajectories
        ax.set_yscale('log')
        ax.set_xlabel(r'$t$ [ms]', fontsize=16)
        ax.set_ylabel(r'$R(t)$ [$\mu$m]', fontsize=16)
        ax.set_title('Growth Trajectory for Different Nucleation Times', fontsize=20)

        # creates legend to the right of the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        legend_x = 1
        legend_y = 0.5
        plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))

    results = (t, m, D, p, p_bubble, if_tension, c_s, c_bulk, R, rho_co2)

    return t_nuc, results


def time_step_convergence(growth_model, dt_list, t_nuc, p_s, R_nuc, p_atm, L,
                          p_in, v, polyol_data_file, eos_co2_file, adaptive_dt,
                          implicit):
    """
    Runs bubble growth model with different time steps to look for convergence.

    Parameters
    ----------

    Returns
    -------

    """
    # initializes lists of parameters to save
    t_list = []
    m_list = []
    D_list = []
    p_list = []
    p_bubble_list = []
    if_tension_list = []
    c_s_list = []
    R_list = []
    rho_co2_list = []


    # solves bubble growth for different time steps
    for dt in dt_list:
        results = growth_model(dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v,
                               polyol_data_file, eos_co2_file,
                               adaptive_dt=adaptive_dt, implicit=implicit)
        t, m, D, p, p_bubble, if_tension, c_s, R, rho_co2 = results
        t_list += [t]
        m_list += [m]
        D_list += [D]
        p_list += [p]
        p_bubble_list += [p_bubble]
        if_tension_list += [if_tension]
        c_s_list += [c_s]
        R_list += [R]
        rho_co2_list += [rho_co2]
        print('completed iteration for dt = {0:f} us.'.format(dt*s_2_us))

    result = (t_list, m_list, D_list, p_list, p_bubble_list, if_tension_list,
            c_s_list, R_list, rho_co2_list)

    return result


def tol_R_convergence(growth_model, tol_R_list, t_nuc, p_s, R_nuc, p_atm, L,
                          p_in, v, polyol_data_file, eos_co2_file, implicit,
                          alpha=1.3, dt0=1E-6):
    """
    Runs bubble growth model with different tolerances on how much the radius
    can vary by decreasing the time step by a factor of 2. Only uses the
    "adaptive_dt" mode of the bubble growth model.

    Parameters
    ----------

    Returns
    -------

    """
    # initializes lists of parameters to save
    t_list = []
    m_list = []
    p_list = []
    p_bubble_list = []
    if_tension_list = []
    c_s_list = []
    R_list = []
    rho_co2_list = []


    # solves bubble growth for different time steps
    for tol_R in tol_R_list:
        results = growth_model(dt0, t_nuc, p_s, R_nuc, p_atm, L, p_in, v,
                               polyol_data_file, eos_co2_file, adaptive_dt=True,
                               implicit=implicit, tol_R=tol_R, alpha=alpha)
        t, m, D, p, p_bubble, if_tension, c_s, R, rho_co2 = results
        t_list += [t]
        m_list += [m]
        p_list += [p]
        p_bubble_list += [p_bubble]
        if_tension_list += [if_tension]
        c_s_list += [c_s]
        R_list += [R]
        rho_co2_list += [rho_co2]
        print('completed iteration for tol_R = {0:f}.'.format(tol_R))

    result = (t_list, m_list, D, p_list, p_bubble_list, if_tension_list,
            c_s_list, R_list, rho_co2_list)

    return result


def sweep(param_list, growth_fn_wrapper, args, param_name='', units='', conv=1):
    """
    Runs growth function for each of the given parameters and saves results
    in list of lists for easy analysis with plot.series() and plot.diff().

    Parameters
    ----------
    param_list : list
        List of values of the parameter to sweep through.
    growth_fn_wrapper : function
        Function wrapper to call growth functions with the given arguments
        (args) and value of the varied parameter.
    args : tuple
        Tuple of arguments required by the growth_fn_wrapper
    param_name : string
        Name of parameter (used for reporting completion of each parameter value)
    units : string
        Units of parameter (used for reporting completion of each parameter value)
    conv : float
        Conversion factor to get unit of parameter from SI units to the given
        unit.

    Returns
    -------
    result : tuple of lists of lists
        Tuple of each property computed by growth function. Each element of the
        tuple is a list of lists. Each list represents the values at each time
        step for a given value of the varied parameter in the same order as
        provided in param_list.
    """
        # initializes lists of parameters to save
    t_list = []
    m_list = []
    p_list = []
    p_bubble_list = []
    if_tension_list = []
    c_s_list = []
    R_list = []
    rho_co2_list = []


    # solves bubble growth for different time steps
    for param in param_list:
        results = growth_fn_wrapper(param, args)
        t, m, D, p, p_bubble, if_tension, c_s, c_bulk, R, rho_co2 = results
        t_list += [t]
        m_list += [m]
        p_list += [p]
        p_bubble_list += [p_bubble]
        if_tension_list += [if_tension]
        c_s_list += [c_s]
        R_list += [R]
        rho_co2_list += [rho_co2]
        print('completed iteration for {0:s} = {1:f} {2:s}.'.format(param_name,
              param*conv, units))

    result = (t_list, m_list, D, p_list, p_bubble_list, if_tension_list,
            c_s_list, R_list, rho_co2_list)

    return result


def d_tolman(d_tolman, args):
    """
    Wrapper for growth function with varied Tolman length.
    """
    dt0, t_nuc, p_s, R_nuc, p_atm, L, p_in, v, \
    polyol_data_file, eos_co2_file, adaptive_dt, \
    implicit, tol_R, alpha = args
    return bubble.grow(dt0, t_nuc, p_s, R_nuc, p_atm, L, p_in, v,
                       polyol_data_file, eos_co2_file, adaptive_dt=True,
                       implicit=implicit, tol_R=tol_R, alpha=alpha,
                       d_tolman=d_tolman)



def diffusivity(D, args):
    """
    Wrapper for growth function with varied Tolman length.
    """
    dt0, t_nuc, p_s, R_nuc, p_atm, L, p_in, v, \
    polyol_data_file, eos_co2_file, adaptive_dt, \
    implicit, tol_R, alpha, d_tolman = args
    return bubble.grow(dt0, t_nuc, p_s, R_nuc, p_atm, L, p_in, v,
                     polyol_data_file, eos_co2_file, adaptive_dt=adaptive_dt,
                     implicit=implicit, d_tolman=d_tolman,
                     tol_R=tol_R, alpha=alpha, D=D)
