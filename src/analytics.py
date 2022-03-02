# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:51:57 2020

analytics.py contains functions used for analysis of bubble-growth models.

@author: Andy
"""

# standard libraries
import os
import pickle as pkl
import time

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt

# imports custom libraries
import objproc as op
import bubble
import bubbleflow
# from libs library
import sys
sys.path.append('../libs/')
import finitediff as fd
import plot.bubble as pltb
from conversions import *


########################### FUNCTION DEFINITIONS ##############################


def calc_dcdr_eps_fix_D(N_list, R_max, t_nuc, eps_params, dt_max_list=None):
    """
    Calculates concentration gradient at interface of bubble b/w Epstein-Plesset
    model and numerical model.
    """
    # initializes list of numerically computed concentration gradients
    dcdr_num_list = []
    # initializes list of times [s]
    t_num_list = []

    # first performs Epstein-Plesset computation as benchmark
    t_eps, m, D, p, p_bub, if_tension,\
    c_s, c_bulk, R, rho_co2 = bubble.grow(t_nuc, *eps_params)
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
        c_bulk, R, rho_co2, v, r_arr_data = bubbleflow.num_fix_D(t_nuc, eps_params, \
                                                    R_max, N, dt_max=dt_max)

        r_arr_list, r_arr_t_list = r_arr_data
        inds_r_arr = [np.where(t >= np.asarray(r_arr_t_list))[0][0] for t in t_flow]
        # list of grid for r values at each time point
        r_arr_list = [r_arr_list[i] for i in inds_r_arr]
        # dr_list = [r_arr_list[i][1] - r_arr_list[i][0] for i in inds_r_arr]

        # uses 2nd-order Taylor stencil
        # dcdr_num = [fd.dydx_fwd_2nd(c[i][0], c[i][1], c[i][2], \
        #                                     dr_list[i]) for i in range(len(c))]
        dcdr_num = [fd.dydx_non_fwd(c[i],r_arr_list[i])[0] for i in range(len(c))]
        dcdr_num_list += [dcdr_num]
        # saves list of times [s]
        t_num_list += [t_num]

    return t_eps, dcdr_eps, t_num_list, dcdr_num_list


def calc_diff(t_ref, ref, t, val):
    """
    Calculates the difference in the concentration gradient dc/dr at the
    surface of the bubble, interpolating to have matching time points.
    """
    val_arr = np.interp(t_ref, t, val)
    ref_arr = np.asarray(ref)
    diff = np.abs(val_arr - ref_arr) / ref_arr

    return diff


def compare_dcdr(num_input_list, num_fn_list, t_ref, dcdr_ref,
                    i_t_flow=0, i_c=1, i_t_num=2, i_dr=-1):
    """
    Compares concentration gradient dc/dr at the surface of the bubble for
    different numerical outputs against the reference.
    Assumes numerical functions return the same ordering of variables as output.
    """
    # initializes list of fractional differences in dc/dr from reference
    dcdr_diff_list = []
    dr_list_list = []
    t_flow_list = []
    raw_vals_list = []

    # computes dc/dr for numerical functions
    for input, fn in zip(num_input_list, num_fn_list):
        print('Computing {0:s}'.format(str(fn)))
        start_time = time.time()
        # extracts results if input is provided as a dictionary
        if isinstance(input, dict):
            output = fn(**input)
        # extracts results if input is provided as a tuple
        else:
            output = fn(*input)

        t_flow = output[i_t_flow]
        c = output[i_c]
        t_num = output[i_t_num]
        dr_list = output[i_dr]

        # uses 2nd-order Taylor stencil to compute dc/dr at r = 0
        dcdr_num = [fd.dydx_fwd_2nd(c[i][0], c[i][1], c[i][2], dr_list[i]) for \
                            i in range(len(c))]
        # collects raw values used to compute the fractional deviation in dc/dr
        raw_vals = (t_ref, dcdr_ref, t_num, dcdr_num)
        # computes fractional difference from dc/dr with E-P model
        dcdr_diff_list += [calc_diff(*raw_vals)]

        # stores values for output
        dr_list_list += [dr_list]
        t_flow_list += [t_flow]
        raw_vals_list += [raw_vals]

        print('Computation time = {0:f} s.'.format(time.time() - start_time))

    return dcdr_diff_list, dr_list_list, t_flow_list, raw_vals_list


def compare_dcdr_eps(num_input_list, num_fn_list, t_nuc, eps_params):
    """
    Compares concentration gradient dc/dr at the surface of the bubble for
    different numerical outputs against the Epstein-Plesset solution.

    Assumes numerical functions return the same ordering of variables as output.

    eps_params = (dt, p_s, R_nuc, L,
                p_in, v, polyol_data_file, eos_co2_file)
    """
    # first performs Epstein-Plesset computation as benchmark
    t_eps, m, D, p, p_bub, if_tension,\
    c_s, c_bulk, R, rho_co2 = bubble.grow(t_nuc, *eps_params)
    # computes concentration gradient at bubble interface
    dcdr_eps = bubble.calc_dcdr_eps(c_bulk, c_s, R, D, np.asarray(t_eps) - t_nuc)

    # computes fractional differences in dc/dr b/w E-P and numerical
    dcdr_diff_list = compare_dcdr(num_input_list, num_fn_list, t_eps, dcdr_eps)

    return t_eps, dcdr_diff_list


def calc_exp_ratio(t_nuc, t_fit, R_fit, t_meas, R_meas):
    """
    Calculates ratio of exponents of power-law fit for fitted model and measured data.
    """
    assert len(t_fit) > 1 and len(R_fit) > 1, 'Requires at least two fit points. One or fewer provided.'
    # removes nucleation point at the beginning
    t_fit = t_fit[1:]
    R_fit = R_fit[1:]
    assert np.min(t_meas) > t_nuc and np.min(t_fit) > t_nuc, 'Nucleation time must be before times provided.'
    # computes ratio
    exp_fit, _ = np.polyfit( np.log(t_fit - t_nuc), np.log(R_fit), 1 )
    exp_meas, _ = np.polyfit( np.log(t_meas - t_nuc), np.log(R_meas), 1 )
    exp_ratio = exp_fit / exp_meas

    return exp_ratio


def compare_R(num_input_list, num_fn_list, t_ref, R_ref,
                    i_R=10, i_t_num=2, i_dr=-1, ret_comp_time=False):
    """
    Follow up to compare_dcdr(). Compares the predicted radius of the bubble
    under different numerical schemes to a reference (typically the Epstein-
    Plesset model).

    Parameters
    ----------
    num_input_list : list
        Entries can be tuples or dictionaries of arguments for the corresponding
        function in num_fn_list or dictionar
    """
    # initializes list of fractional differences in dc/dr from reference
    R_diff_list = []
    t_num_list = []
    dr_list_list = []
    raw_vals_list = []
    comp_time_list = []

    # computes dc/dr for numerical functions
    for input, fn in zip(num_input_list, num_fn_list):
        print('Computing {0:s}'.format(str(fn)))
        start_time = time.time()
        # extracts results if input is provided as a dictionary
        if isinstance(input, dict):
            output = fn(**input)
        # extracts results if input is provided as a tuple
        else:
            output = fn(*input)
        # extracts relevant variables from the output
        R = output[i_R]
        t_num = output[i_t_num]
        dr_list = output[i_dr]

        # uses first computation as reference if no reference provided
        if t_ref is None and R_ref is None:
            t_ref = t_num
            R_ref = R
            continue

        raw_vals = (t_ref, R_ref, t_num, R)
        R_diff_list += [calc_diff(*raw_vals)]

        # stores values for output
        dr_list_list += [dr_list]
        t_num_list += [t_num]
        raw_vals_list += [raw_vals]

        comp_time = time.time() - start_time
        comp_time_list += [comp_time]
        print('Computation time = {0:f} s.\n'.format(comp_time))

    ret_vals = [R_diff_list, dr_list_list, raw_vals_list]
    if ret_comp_time:
        ret_vals += [comp_time_list]

    return ret_vals


def calc_rms_excess(t_nuc, t_fit, R_fit, t_bub, R_bub):
    """
    Calculates excess RMS error beyond that of power-law fit from
    `calc_rms_power_law`.
    """
    # calculates RMS error of data itself from power-law fit
    rms_data = calc_rms_power_law(t_nuc, t_bub, R_bub)
    # computes rms of fit
    rms_err = calc_rms_err(t_bub, R_bub, t_fit, R_fit)
    # subtracts to compute excess RMS error
    rms_excess = rms_err - rms_data

    return rms_excess


def calc_rms_power_law(t_nuc, t_meas, R_meas):
    """
    Calculates the root-mean-square error of a power-law fit to the data.
    """
    assert np.min(t_meas) > t_nuc, 'Nucleation time must be less than times of measurement.'
    # calculates RMS of data
    a, b = np.polyfit( np.log(t_meas - t_nuc), np.log(R_meas), 1 )
    R_power_law = np.exp(b) * (t_meas - t_nuc)**a
    rms_data = calc_rms_err(t_meas, R_power_law, t_meas, R_meas)

    return rms_data


def calc_rms_err(t_meas, R_meas, t_pred, R_pred):
    """
    Computes mean-squared *fractional* error of predicted bubble growth
    from measured growth (radius [m]).
    """
    # estimates corresponding fitted values
    R_interp = np.interp(t_meas, t_pred, R_pred)
    # fractional error
    err_frac = (R_interp - R_meas) / R_meas
    # takes root mean square
    err = np.sqrt(np.sum(err_frac**2)) / len(err_frac)

    return err


def calc_rms_fit(bub_data):
    """
    Calculates the RMS error of a given fit.
    """
    return calc_rms_err(bub_data['t_bub'],
                        bub_data['R_bub'],
                        bub_data['t_fit'],
                        bub_data['R_fit'])


def calc_sgn_mse(t_meas, R_meas, t_pred, R_pred):
    """
    Computes sum of signed squared error and takes mean.
    """
    # estimates corresponding fitted values
    R_interp = np.interp(t_meas, t_pred, R_pred)
    # fractional error
    err_frac = (R_interp - R_meas) / R_meas
    # signed summation
    sgn_sum = np.sum(np.sign(err_frac) * err_frac**2)
    # computes mean signed squared error
    err = np.sign(sgn_sum) * np.sqrt(np.abs(sgn_sum)) / len(err_frac)

    return err


def calc_abs_sgn_mse(t_meas, R_meas, t_pred, R_pred):
    """
    Computes absolute value of signed MSE from `calc_sgn_mse`.
    """
    return np.abs(calc_sgn_mse(t_meas, R_meas, t_pred, R_pred))


def fit_growth_to_pts(t_meas, R_meas, t_nuc_lo, t_nuc_hi, growth_fn, args,
                     i_t_nuc, err_fn=calc_abs_sgn_mse, err_tol=0.003, ax=None,
                     max_iter=15, i_t=0, i_R=-2, dict_args={}, x_lim=None,
                     y_lim=None, t_fs=18, ax_fs=16, tk_fs=14):
    """
    Finds a suitable model of bubble growth that fits measured bubble radius at
    many time points.
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
    output = growth_fn(*tuple(args), **dict_args)
    t = output[i_t]
    R = output[i_R]

    # ends fitting if maximum predictions all below minimum measured values
    R_pred = np.interp(t_meas, t, R)
    if np.sum(np.sign(R_pred - R_meas)) / len(R_pred) == -1:
        print('Predicted bubble radii are all smaller than measured values' + \
              ' for lowest nucleation time. Terminating early.')
        return t_nuc_lo, output

    # computes error
    err = err_fn(t_meas, R_meas, t, R)

    # bisection algorithm searches for nucleation time yielding accurate R
    while err > err_tol:
        # calculates new nucleation time as middle of the two bounds (bisection algorithm)
        t_nuc = (t_nuc_lo + t_nuc_hi)/2
        # computes bubble growth trajectory with new bubble nucleation time
        args[i_t_nuc] = t_nuc
        output = growth_fn(*tuple(args), **dict_args)
        # extracts time and radius of bubble growth trajectory from output
        t = output[i_t] # [s]
        R = output[i_R] # [m]

        # computes error
        err = err_fn(t_meas, R_meas, t, R)

        # determines whether to increase or decrease next nucleation time guess
        sgn_mse = calc_sgn_mse(t_meas, R_meas, t, R)
        # predicted bubble radius too large means nucleation time is too early,
        # so we raise the lower bound
        if sgn_mse > 0:
            t_nuc_lo = t_nuc
        # otherwise, nucleation time is too late, so we lower the upper bound
        else:
            t_nuc_hi = t_nuc

        print('t_nuc = {0:.3f} ms. Error is {1:.4f} and tol is {2:.4f}.' \
                .format(t_nuc*s_2_ms, err, err_tol))

        # plots the guessed growth trajectory
        if ax is not None:
            ax.plot(np.array(t)*s_2_ms, np.array(R)*m_2_um,
                    label=r'$t_{nuc}=$' + '{0:.3f} ms'.format(t_nuc*s_2_ms))

        n_iter += 1
        if n_iter == max_iter:
            print('Max iterations {0:d} reached. Error above tolerance {1:.4f}'\
                    .format(max_iter, err_tol))
            print('Nucleation time is {0:.3f} ms.'.format(t_nuc*s_2_ms))
            break

    if n_iter < max_iter:
        print('Error {0:.4f} is below tolerance of {1:.4f}'.format(err, err_tol) + \
                ' for nucleation time t = {0:.3f} ms'.format(t_nuc*s_2_ms))
    if ax is not None:
        # formats plot of guessed trajectories
        ax.set_yscale('log')
        ax.set_xlabel(r'$t$ [ms]', fontsize=ax_fs)
        ax.set_ylabel(r'$R(t)$ [$\mu$m]', fontsize=ax_fs)
        ax.tick_params(axis='both', labelsize=tk_fs)
        if title:
            ax.set_title(title, fontsize=t_fs)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        # creates legend to the right of the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        legend_x = 1
        legend_y = 0.5
        plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))

    return t_nuc, output


def fit_D_t_nuc_load(load_path, save_path, save_freq=-1,
                        n_fit=-1, show_plots=False, metatag='_meta.pkl'):
    """Wrapper for `fit_D_t_nuc` that loads data with parameters."""
    # loads data
    with open(load_path, 'rb') as f:
        data = pkl.load(f)
    # loads metadata
    with open(os.path.splitext(load_path)[0] + metatag, 'rb') as f:
        metadata = pkl.load(f)

    # collects parameters for analysis
    addl_params = {'save_path' : save_path,
                    'save_freq' : save_freq,
                    'n_fit' : n_fit,
                    'show_plots' : show_plots,
                    'data' : data}
    params = dict(metadata, **addl_params)

    return fit_D_t_nuc(**params)


def arrange_metadata(data_filename, data_dir_list, polyol_data_file,
                    eos_co2_file, frac_lo, frac_hi, D_lo, D_hi, growth_fn, dt,
                    R_nuc, fit_fn_params, exp_ratio_tol, fit_fn, L_frac,
                    min_data_pts, max_iter, i_t_nuc, i_t, i_R):
    """Arranges metadata for `fit_D_t_nuc`."""
    metadata = {'data_filename' : data_filename,
                'data_dir_list' : data_dir_list,
                'polyol_data_file' : polyol_data_file,
                'eos_co2_file' : eos_co2_file,
                'frac_lo' : frac_lo,
                'frac_hi' : frac_hi,
                'D_lo' : D_lo,
                'D_hi' : D_hi,
                'growth_fn' : growth_fn,
                'dt' : dt,
                'R_nuc' : R_nuc,
                'fit_fn_params' : fit_fn_params,
                'exp_ratio_tol' : exp_ratio_tol,
                'fit_fn' : fit_fn,
                'L_frac' : L_frac,
                'min_data_pts' : min_data_pts,
                'max_iter' : max_iter,
                'i_t_nuc' : i_t_nuc,
                'i_t' : i_t,
                'i_R' : i_R,
                }

    return metadata


def fit_D_t_nuc(data_filename, data_dir_list, polyol_data_file,
                eos_co2_file, frac_lo, frac_hi,
                D_lo, D_hi, growth_fn, dt, R_nuc, fit_fn_params,
                exp_ratio_tol, fit_fn=fit_growth_to_pts, L_frac=1,
                n_fit=-1, min_data_pts=4, max_iter=15,
                i_t_nuc=0, i_t=0, i_R=-2,
                x_lim=None, y_lim=None, show_plots=True,
                save_freq=-1, save_path=None, data={}, metatag='_meta.pkl'):
    """
    Fits effective diffusivity D and nucleation time t_nuc.

    TODO: select t_nuc_lo and t_nuc_hi with absolute deviation
    from t_center instead of fractional since deviation should
    not depend on time traveling through observation capillary.

    TODO break down 'model_output' into labeled data in dictionary
    """
    # arranges metadata
    metadata = arrange_metadata(data_filename, data_dir_list, polyol_data_file,
                    eos_co2_file, frac_lo, frac_hi, D_lo, D_hi, growth_fn, dt,
                    R_nuc, fit_fn_params, exp_ratio_tol, fit_fn, L_frac,
                    min_data_pts, max_iter, i_t_nuc, i_t, i_R)
    # saves metadata separately
    if save_path:
        with open(os.path.splitext(save_path)[0] + metatag, 'wb') as f:
            pkl.dump(metadata, f)

    # starts counting objects whose growth is modeled
    ct = 0
    # loads data from each file
    for data_dir in data_dir_list:
        # loads data
        with open(os.path.join(data_dir, data_filename), 'rb') as f:
            raw_data = pkl.load(f)

        # gets conditions of experiment
        p_in, p_sat, p_est, d, L, \
        v_max, t_center, polyol, num = op.get_conditions(raw_data['metadata'])

        # loads previous data at this distance if available
        if num in data.keys():
            vid_data = data[num]
        else:
            # creates dictionary for bubbles in current measurement video
            vid_data = {'data' : {},
                        'metadata' :
                            {'p_in' : p_in, 'p_sat' : p_sat, 'p_est' : p_est,
                            'd' : d, 'L' : L, 'v_max' : v_max,
                            't_center' : t_center, 'polyol' : polyol}
                        }

        # gets sizes of each bubble
        for ID, obj in raw_data['objects'].items():
            # skips objects that are not definitely real objects (bubbles)
            # or that have already been analyzed
            if not op.is_true_obj(obj) or (ID in vid_data['data'].keys()):
                continue

            # time of observations of bubble since entering observation capillary [s]
            t_bub = op.calc_t(obj, d, v_max)
            # bubble radius [m]
            R_bub = np.asarray(obj['props_proc']['radius [um]']) * um_2_m
            # just in case, stores width (vertical extent) and length
            # (horizontal) extent of bubbles [m]
            W_bub = np.asarray(op.calc_W(obj)) * um_2_m
            L_bub = np.asarray(op.calc_L(obj)) * um_2_m
            # gets indices of frames for bubble's fully visible early growth
            is_valid_arr = op.get_valid_idx(obj, L_frac=L_frac)
            # skips bubbles for which not enough frames of early growth were observed
            if len(t_bub[is_valid_arr]) < min_data_pts:
                continue

            print('\nAnalyzing bubble {0:d} at {1:.3f} m.\n'.format(ID, d))

            # extracts only valid measurements
            t_bub = t_bub[is_valid_arr]
            R_bub = R_bub[is_valid_arr]
            W_bub = W_bub[is_valid_arr]
            L_bub = L_bub[is_valid_arr]

            # estimates bounds on nucleation time [s]
            t_nuc_lo = frac_lo*t_center
            t_nuc_hi = frac_hi*t_center

            # sets moveable limits on effective diffusivity for binary search
            D_lo_tmp = D_lo
            D_hi_tmp = D_hi
            for _ in range(max_iter):

                # makes guess for effective diffusivity constant
                D = (D_lo_tmp + D_hi_tmp) / 2
                # packages it for solver
                dict_args = {'D' : D}
                # collects inputs -- must recollect after an.fit_growth_to_pt b/c it inserts t_nuc
                eps_params = list((dt, p_sat, R_nuc, L, p_in, v_max,
                                    polyol_data_file, eos_co2_file))

                # fits nucleation time to data [s]
                t_nuc, output = fit_fn(t_bub, R_bub, t_nuc_lo, t_nuc_hi,
                                            growth_fn, eps_params, i_t_nuc,
                                            **fit_fn_params, max_iter=max_iter,
                                            dict_args=dict_args)

                # extracts model values for time and radius
                t_fit = output[i_t]
                R_fit = output[i_R]

                # compares slopes of fit and data
                exp_ratio = calc_exp_ratio(t_nuc, t_fit, R_fit, t_bub, R_bub)

                if np.abs(exp_ratio - 1) < exp_ratio_tol:
                    print('For D = {0:g}, exponent ratio '.format(D) + \
                        '{0:.3f} deviates from 1 by less than tolerance {1:.3f}.' \
                          .format(exp_ratio, exp_ratio_tol))
                    break

                print('For D = {0:g}, exponent ratio = {1:.3f}'.format(D, exp_ratio))
                D_lo_tmp, D_hi_tmp = update_bounds_D(exp_ratio, D, D_lo_tmp,
                                                        D_hi_tmp)
                # expands bounds on D if
                if (D_hi_tmp >= D_hi) and \
                        ((D_hi_tmp - D)/D_hi_tmp < exp_ratio_tol) and \
                        (D < D_hi_tmp):
                    print('Doubling upper bound on D.')
                    D_hi_tmp *= 2
                elif (D_lo_tmp <= D_lo) and \
                        ((D - D_lo_tmp)/D_lo_tmp < exp_ratio_tol) and \
                        (D > D_lo_tmp):
                    print('Halving lower bound on D.')
                    D_lo_tmp /= 2

            # plots result
            if show_plots:
                R_i = raw_data['metadata']['object_kwargs']['R_i'] # inner stream radius [m]
                ax = pltb.fit(t_nuc, output, t_bub, R_bub, R_i)
                if x_lim:
                    ax.set_xlim(x_lim)
                if y_lim:
                    ax.set_ylim(y_lim)

            # stores results [SI units]
            vid_data['data'][ID] = {'t_nuc' : t_nuc,
                            'd_nuc' : v_max * t_nuc,
                            'D' : D,
                            't_bub' : t_bub,
                            'R_bub' : R_bub,
                            'W_bub' : W_bub,
                            'L_bub' : L_bub,
                            't_fit': t_fit,
                            'R_fit' : R_fit,
                            'model_output' : output
                            }

            # ends loop when desired number of trajectories has been fit
            ct += 1
            print('\nAnalyzed {0:d} bubbles.\n'.format(ct))

            # saves data periodically
            if ( (ct % save_freq) == 1 ) and save_path:
                print('Saving after {0:d} bubbles analyzed.\n'.format(ct))
                data[num] = vid_data
                with open(save_path, 'wb') as f:
                    pkl.dump(data, f)

            if ct == n_fit:
                break

        # stores video data under distance along capillary [m]
        data[num] = vid_data
        print('\nAnalyzed videos taken at distance {0:.3f} m.'.format(d))
        print('There are {0:d} videos to analyze.\n'.format(len(data_dir_list)))

        # must break out of two for loops when complete
        if ct == n_fit:
            break

    return data


def fit_growth_to_pt(t_bubble, R_bubble, t_nuc_lo, t_nuc_hi, growth_fn, args,
                     i_t_nuc, sigma_R=0.01, ax=None, max_iter=12, i_t=0,
                     i_R=-2, dict_args={}, x_lim=None, y_lim=None):
    """
    Fits the bubble growth to a given bubble radius at a given time. Plots
    the different trajectories if an axis handle is given.
    """
    # makes sure only one point provided
    if not isinstance(t_bubble, float):
        t_bubble = t_bubble[0]
        R_bubble = R_bubble[0]

    # inserts place-holder (0) for nucleation time in arguments list
    args.insert(i_t_nuc, 0)
    # initializes plot to show the trajectories of different guesses
    if ax is not None:
        ax.plot(t_bubble*s_2_ms, R_bubble*m_2_um, 'g*', ms=12, label='fit pt')

    # initializes counter of number of iterations
    n_iter = 0

    # computes bubble growth trajectory with lowest nucleation time
    args[i_t_nuc] = t_nuc_lo
    output = growth_fn(*tuple(args), **dict_args)
    t = output[i_t]
    R = output[i_R]

    # finds index of timeline corresponding to measurement of bubble size
    i_bubble = next(i for i in range(len(t)) if t[i] >= t_bubble)
    R_bubble_pred = R[i_bubble]
    if R_bubble_pred < R_bubble:
        print('Predicted bubble radius {0:.1f} um'.format(R_bubble_pred*m_2_um)\
            + ' is smaller than fit value {0:.1f} um'.format(R_bubble*m_2_um) \
            + ' for lowest nucleation time. Terminating early.')
        return t_nuc_lo, output

    # computes error in using lowest nucleation time
    err_R = np.abs(R_bubble_pred - R_bubble)/R_bubble # frac err in bubble rad

    # bisection algorithm searches for nucleation time yielding accurate R
    while err_R > sigma_R:
        # calculates new nucleation time as middle of the two bounds (bisection algorithm)
        t_nuc = (t_nuc_lo + t_nuc_hi)/2
        # computes bubble growth trajectory with new bubble nucleation time
        args[i_t_nuc] = t_nuc
        output = growth_fn(*tuple(args), **dict_args)
        # extracts time and radius of bubble growth trajectory from output
        t = output[i_t] # [s]
        R = output[i_R] # [m]

        # finds index of timeline corresponding to measurement of bubble size
        i_bubble = next(i for i in range(len(t)) if t[i] >= t_bubble)
        R_bubble_pred = R[i_bubble]
        err_R = np.abs(R_bubble_pred - R_bubble)/R_bubble # frac err bubble rad.
        # predicted bubble radius too large means nucleation time is too early,
        # so we raise the lower bound
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
        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)

        # creates legend to the right of the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        legend_x = 1
        legend_y = 0.5
        plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))

    return t_nuc, output


def time_step_convergence(growth_model, dt_list, t_nuc, p_s, R_nuc, L,
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
        results = growth_model(dt, t_nuc, p_s, R_nuc, L, p_in, v,
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


def tol_R_convergence(growth_model, tol_R_list, t_nuc, p_s, R_nuc, L,
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
        results = growth_model(dt0, t_nuc, p_s, R_nuc, L, p_in, v,
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
    dt0, t_nuc, p_s, R_nuc, L, p_in, v, \
    polyol_data_file, eos_co2_file, adaptive_dt, \
    implicit, tol_R, alpha = args
    return bubble.grow(dt0, t_nuc, p_s, R_nuc, L, p_in, v,
                       polyol_data_file, eos_co2_file, adaptive_dt=True,
                       implicit=implicit, tol_R=tol_R, alpha=alpha,
                       d_tolman=d_tolman)



def diffusivity(D, args):
    """
    Wrapper for growth function with varied diffusivity.
    """
    dt0, t_nuc, p_s, R_nuc, L, p_in, v, \
    polyol_data_file, eos_co2_file, adaptive_dt, \
    implicit, tol_R, alpha, d_tolman = args
    return bubble.grow(dt0, t_nuc, p_s, R_nuc, L, p_in, v,
                     polyol_data_file, eos_co2_file, adaptive_dt=adaptive_dt,
                     implicit=implicit, d_tolman=d_tolman,
                     tol_R=tol_R, alpha=alpha, D=D)


def update_bounds_D(exp_ratio, D, D_lo, D_hi):
    """
    Updates bounds on guess for effective diffusivity based on ratio of
    exponents of power-law fits.
    """
    # decreases D if fitted slope is too high
    if exp_ratio > 1:
        D_hi = D
    # increases D if fitted slope is too low
    else:
        D_lo = D

    return D_lo, D_hi
