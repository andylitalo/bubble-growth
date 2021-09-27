"""Contains functions for modeling bubble growth during sheath flow.

It combines functions from bubble.py for bubble growth and diffn.py for
diffusion in the bulk fluid to provide a more complete model of each phenomenon
and their interaction.

Date created : November 24, 2020
Author : Andy Ylitalo
"""
# adds path to general libraries
import sys
sys.path.append('../libs/')

# imports standard libraries
import numpy as np
import time

# TODO remove***
import matplotlib.pyplot as plt

# imports custom libraries
import bubble
import diffn
import polyco2

# CONSTANTS
from constants import *
# CONVERSIONS
from conversions import *


############################ FUNCTION DEFINITIONS ##############################


def get_dr(r_arr):
    """Calculates smallest grid spacing."""
    return np.min(np.diff(r_arr))


def grow(dt_sheath, dt, dcdt_fn, R_o, N, eta_i, eta_o, d, L, Q_i, Q_o, p_s,
         dc_c_s_frac, t_nuc, R_nuc, polyol_data_file, eos_co2_file,
         bc_specs_list, if_tension_model='lin', d_tolman=0, adaptive_dt=True,
         implicit=False, tol_R=0.001, alpha=0.3, drop_t_term=False, R_min=0,
         D=-1, i_c_bulk=7):
    """
    Grows bubble based on combined diffusion model of bulk and bubble interface.

    Assumes:
    -Bubble "consumes" polyol as it grows, so its growth does not directly
    affect the CO2 concentration profile

    dt_sheath : float
        Time step for modeling diffusion in sheath flow without a bubble [s].
        Typically larger than dt.
    dt : float
        Initial time step for modeling bubble growth [s]. Typically smaller than
        dt_sheath.
    dcdt_fn : function
        Function for computing the time derivative of the concentration at each
        point on the grid, dc/dt(r) [kg/m^3 / s]
    R_o : float
        Outer stream radius [m]
    N : int
        number of mesh elements in grid (so number of mesh points is N+1)
    eta_i, eta_o : float
        viscosity of (inner)/(outer) stream [Pa.s]
    d : float
        distance downstream from entrance to observation capillary [m]
    L : float
        length of observation capillary [m]
    Q_i, Q_o : float
        flow rate of (inner)/(outer) stream [uL/min]
    p_s : float
        saturation pressure of CO2 in polyol [Pa]
    dc_c_s_frac : float

    """
    # initializes parameters for CO2 diffusion in sheath flow
    # uses t_f = d/v the final time is the distance to which to perform the
    # calculation divided by the velocity
    t, c, r_arr, dp, R_i, v, \
    c_0, c_s, t_f, fixed_params_flow = diffn.init(R_min, R_o, N, eta_i, eta_o,
                                            d, L, Q_i, Q_o, p_s, dc_c_s_frac,
                                            polyol_data_file)

    # initializes parameters for bubble growth
    # c_s is concentration in saturated inner stream = c_bulk from bubble.grow
    # v is computed with diffn.init above
    p_in = P_ATM - dp
    t_bub, m, D, p, p_bub, if_tension, c_bub, \
    c_s, R, rho_co2, _, fixed_params_bub = bubble.init(p_in, p_s, t_nuc,
                                            R_nuc, v, L, D, polyol_data_file,
                                            eos_co2_file, if_tension_model,
                                            d_tolman, implicit)

    # TIME STEPPING -- PRE-BUBBLE
    # applies Euler's method to estimate bubble growth over time
    # the second condition provides cutoff for shrinking the bubble
    while t[-1] <= t_nuc:
        # calculates properties after one time step
        props_flow = diffn.time_step(dt_sheath, t[-1], r_arr, c[-1], dcdt_fn,
                            bc_specs_list, R[-1], fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t, c)


    # TIME STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t[-1] <= t_f:
        # BUBBLE GROWTH
        if adaptive_dt:
            dt, props_bub = bubble.adaptive_time_step(dt, t_bub[-1], m[-1],
                                                p[-1], if_tension[-1], R[-1],
                                                rho_co2[-1], fixed_params_bub,
                                                tol_R, alpha, drop_t_term)
        else:
            # calculates properties after one time step
            props_bub = bubble.time_step(dt, t_bub[-1], m[-1], p[-1], if_tension[-1],
                                    R[-1], rho_co2[-1], fixed_params_bub,
                                    drop_t_term=drop_t_term)
        # updates properties of bubble at new time step
        bubble.update_props(props_bub, t_bub, m, p, p_bub, if_tension, c_bub, R,
                            rho_co2)

        # SHEATH FLOW
        # updates array of radii

        # calculates properties after one time step with updated
        # boundary conditions
        props_flow = diffn.time_step(dt, t[-1], r_arr, c[-1], dcdt_fn,
                            bc_bub_cap(c_bub[-1]), R[-1], fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t, c)

    return t, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_s, R, rho_co2, R_i, v


def bc_bub_cap(c_bub, c_max=0):
    """
    Fixes concentration of CO2 at inner wall of capillary to c_max (default 0).
    Fixed concentration at bubble surface (saturation concentration).

    Parameters
    ----------
    c_bub : float
        Concentration of CO2 at interface of bubble [kg/m^3].
    c_max : float, optional
        Concentration at inner wall of capillary, default 0 [kg/m^3]

    Returns
    -------
    bc : list of two 3-tuples
        List of boundary conditions. First is at surface of bubble, second at
        inner wall of capillary. Each tuple contains the function for applying
        the boundary condition, the index of the grid to apply it at, and the
        value to set.
    """
    return [(diffn.dirichlet, 0, c_bub), (diffn.dirichlet, -1, c_max)]


def bc_cap(c_max=0):
    """
    Fixes concentration of CO2 at inner wall of capillary to c_max (default 0).
    Applies symmetry condition at center of capillary by setting dc/dr = 0.

    Parameters
    ----------
    c_max : float, optional
        Concentration at inner wall of capillary, default 0 [kg/m^3]

    Returns
    -------
    bc : list of two 3-tuples
        List of boundary conditions. First is at center of capillary, second at
        inner wall of capillary. Each tuple contains the function for applying
        the boundary condition, the index of the grid to apply it at, and the
        value to set.
    """
    # artificial array for computing derivative at indices 0 and 1.
    r_arr = np.array([0, 1])
    return [(diffn.neumann, 0, 1, 0, r_arr), (diffn.dirichlet, -1, c_max)]


def num_fix_D(t_nuc, eps_params, R_max, N, adaptive_dt=True,
                    if_tension_model='lin', implicit=False,
                    d_tolman=5E-9, tol_R=0.001, alpha=0.3, D=-1, dt_max=None,
                    R_min=0, dcdt_fn=diffn.calc_dcdt_sph_fix_D,
                    time_step_fn=bubble.time_step_dcdr, legacy_mode=False,
                    grid_fn=diffn.make_r_arr_lin, grid_params={}, adapt_freq=1,
                    remesh_fn=None, remesh_params={}, remesh_freq=1000):
    """
    Performs numerical computation of Epstein-Plesset model for comparison.
    Once confirmed to provide accurate estimation of Epstein-Plesset result,
    this will be modified to include the effects of a concentration-dependent
    diffusivity in num_vary_D().

    Parameters
    ----------
    eps_params : 8-tuple
        Positional parameters for bubble.grow() Epstein-Plesset model
    R_max : float
        Maximum value of radius in grid [m]
    N : int
        mesh size (so # of grid points = N+1)
    grid_fn : function (Optional)
        Function for producing mesh grid of the radial dimension (lin, log, etc.)

    Return
    ------
    t_flow : list of floats
        times at which concentration in flow was evaluated [s]
    c : list of lists of floats
        list of concentrations at each grid point for each time point in t_flow
        [kg/m^3]
    t_bub : list of floats
        times at which the bubble growth was evaluated [s]
    m : list of floats
        mass of CO2 in bubble at each time in t_bub [kg]
    D : float
        diffusivity [m^2/s]
    p : list of floats
        pressure in channel at each time point in t_bub [Pa]
    p_bub : list of floats
        pressure inside bubble at each time point in t_bub (includes Laplace
        pressure) [Pa]
    if_tension : list of floats
        interfacial tension at interface of bubble at each time point in t_bub
        [N/m]
    c_bub : list of floats
        concentration of CO2 at surface of bubble at each time point in t_bub
        [kg/m^3]
    c_bulk : float
        concentration of CO2 in the bulk liquid [kg/m^3]
    R : list of floats
        radius of bubble at each time point in t_bub [m]
    rho_co2 : list of floats
        density of CO2 inside the bubble at each time point in t_bub [kg/m^3]
    v : float
        maximum velocity of inner stream assuming Poiseuille flow [m/s]
    dr_list : list of floats
        grid spacing at each time in t_flow [m]

    See Also
    --------
    num_vary_D : same but allows D to vary with concentration D(c)
    """
    # extracts parameters used in Epstein-Plesset model
    dt, p_s, R_nuc, L, p_in, v, polyol_data_file, eos_co2_file = eps_params
    # INITIALIZES BUBBLE PARAMETERS
    t_bub, m, D, p, p_bub, if_tension, c_bub, \
    c_bulk, R, rho_co2, _, fixed_params_tmp = bubble.init(p_in, p_s, t_nuc,
                                            R_nuc, v, L, D, polyol_data_file,
                                            eos_co2_file, if_tension_model,
                                            d_tolman, implicit)
    # extracts relevant parameters from bubble initiation
    _, D, p_in, p_s, v, L, _, c_s_interp_arrs, \
    if_interp_arrs, f_rho_co2, _, _ = fixed_params_tmp
    # collects parameters relevant for bubble growth
    fixed_params_bub = (D, p_in, p_s, v, L, c_s_interp_arrs, if_interp_arrs,
                        f_rho_co2, d_tolman)
    # fixed parameters for flow
    fixed_params_flow = (D,)
    # INITIALIZES PARAMETERS FOR DIFFUSION IN BULK
    # starts at nucleation time since we do not consider diffusion before bubble
    t_flow = [t_nuc]
    # creates mesh grid
    r_arr = grid_fn(N, R_max, **grid_params)
    # initializes list of the grids and times at which grids change
    r_arr_list = [r_arr]
    r_arr_t_list = [t_flow[-1]]
    # initializes concentration profile as uniformly bulk concentration [kg/m^3]
    c = [c_bulk*np.ones(len(r_arr))]

    # final time of model [s]
    t_f = L/v

    # TIME STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t_bub[-1] <= t_f:
        # collects parameters for bubble growth
        args_bub = (r_arr, c[-1], *fixed_params_bub)
        inputs_bub = (t_bub[-1], m[-1], if_tension[-1], R[-1],
                                rho_co2[-1])
        # BUBBLE GROWTH
        if adaptive_dt and (len(t_bub)%adapt_freq == 0):
            dt, updated_inputs_bub, outputs_bub = bubble.adaptive_time_step(dt,
                                                    inputs_bub, args_bub,
                                                    time_step_fn, tol_R, alpha,
                                                    dt_max=dt_max,
                                                    legacy_mode=legacy_mode)
        else:
            # calculates properties after one time step
            updated_inputs_bub, outputs_bub = time_step_fn(dt, inputs_bub, args_bub)

        # updates properties of bubble at new time step
        props_bub = (*updated_inputs_bub, *outputs_bub)
        props_lists_bub = [t_bub, m, if_tension, R, rho_co2, p, p_bub, c_bub]
        bubble.update_props(props_bub, props_lists_bub)

        ######### SHEATH FLOW #############
        # first considers remeshing to adapt to changing gradient
        start = time.time()
        if remesh_fn is not None and (len(t_bub)%remesh_freq == 5):
            remeshed, r_arr, c[-1] = remesh_fn(r_arr, c[-1], **remesh_params)

            # only saves new grid if it remeshed and is not the first data point
            # (first data point is already saved before this loop)
            if remeshed and len(t_flow) > 1:
                print('remeshed')
                print('t', t_bub[-1] - t_nuc)
                dt_max = update_dt_max(get_dr(r_arr_list[-1]), get_dr(r_arr), dt_max)
                # ensures new time step is shorter than maximum allowed, o/w
                # the solution becomes unstable
                dt = min(dt, dt_max)
                r_arr_list += [r_arr]
                r_arr_t_list += [t_flow[-1]]

        # calculates properties after one time step with updated
        # boundary conditions
        # adds bubble radius R to grid of radii since r_arr starts at bubble
        # interface
        # for now, uses same grid spacing for each time step
        # computes time step
        props_flow = diffn.time_step(dt, t_flow[-1], r_arr, c[-1], dcdt_fn,
                        bc_bub_cap(c_bub[-1], c_max=c_bulk), R[-1], fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t_flow, c)

        if np.max(c[-1]) > c[-1][-1]:
            print('unstable')
            print(np.max(c[-1]))


    return t_flow, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_bulk, R, \
                rho_co2, v, (r_arr_list, r_arr_t_list)


def num_vary_D(t_nuc, eps_params, R_max, N, dc_c_s_frac=0.01,
                 dt_max=None, D_fn=polyco2.calc_D_lin,
                 adaptive_dt=True, adapt_freq=5, legacy_mode=False,
                 if_tension_model='lin', implicit=False, d_tolman=5E-9,
                 tol_R=0.001, alpha=0.3, D=-1, R_i=np.inf,
                 R_min=0, dcdt_fn=diffn.calc_dcdt_sph_vary_D,
                 time_step_fn=bubble.time_step_dcdr,
                 grid_fn=diffn.make_r_arr_lin, grid_params={},
                 remesh_fn=diffn.remesh, remesh_params={}, remesh_freq=25):
    """
    Peforms numerical computation of diffusion into bubble from bulk accounting
    for effect of concentration of CO2 on the local diffusivity D.

    Has a feature allowing the user to "halve" the grid (decimating every other
    point) to speed up computation as the diffusion boundary layer widens.
    Simply set the optional parameter "half_grid" to True.
    """
    # extracts parameters used in Epstein-Plesset model
    dt, p_s, R_nuc, L, p_in, v, polyol_data_file, eos_co2_file = eps_params
    # INITIALIZES PARAMETERS FOR DIFFUSION IN BULK
    # assumes no sheath (i.e. R_i = R_o)
    t_flow, c, r_arr, _, _, \
    t_f, fixed_params = diffn.init_sub(R_min, R_i, R_max, N, L, v, p_s,
                                        dc_c_s_frac, polyol_data_file, t_i=t_nuc)
    dc, interp_arrs = fixed_params
    # extracts "bulk" concentration from outer edge of initial profile [kg/m^3]
    c_wall = c[0][-1]

    # INITIALIZES BUBBLE PARAMETERS
    t_bub, m, D, p, p_bub, if_tension, c_bub, \
    c_bulk, R, rho_co2, _, fixed_params_tmp = bubble.init(p_in, p_s, t_nuc,
                                            R_nuc, v, L, -1, polyol_data_file,
                                            eos_co2_file, if_tension_model,
                                            d_tolman, implicit)
    # extracts relevant parameters from bubble initiation
    _, _, p_in, p_s, v, L, _, c_s_interp_arrs, \
    if_interp_arrs, f_rho_co2, d_tolman, _ = fixed_params_tmp

    # fixes parameters for flow
    # last two params are placeholders for R_i, eta_ratio used in sheath_incompressible
    fixed_params_flow = (dc, D_fn, R_max, 1)
    # initializes list of diffusivities
    D = [D]
    # creates mesh grid
    r_arr = grid_fn(N, R_max, **grid_params)
    # initializes list of the grids and times at which grids change
    r_arr_list = [r_arr]
    r_arr_t_list = [t_flow[-1]]

    n_steps = 0
    # TIME-STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t_bub[-1] <= t_f:
        # collects parameters for time-stepping method
        D += [D_fn(c_bub[-1])]
        fixed_params_bub = (D[-1], p_in, p_s, v, L, c_s_interp_arrs,
                                if_interp_arrs, f_rho_co2, d_tolman)
        ########### BUBBLE GROWTH #########
        # collects parameters for bubble growth
        args_bub = (r_arr, c[-1], *fixed_params_bub)
        inputs_bub = (t_bub[-1], m[-1], if_tension[-1], R[-1],
                                rho_co2[-1])
        # BUBBLE GROWTH
        if adaptive_dt and (len(t_bub)%adapt_freq == 0):
            dt, updated_inputs_bub, outputs_bub = bubble.adaptive_time_step(dt,
                                                    inputs_bub, args_bub,
                                                    time_step_fn, tol_R, alpha,
                                                    dt_max=dt_max,
                                                    legacy_mode=legacy_mode)
        else:
            # calculates properties after one time step
            updated_inputs_bub, outputs_bub = time_step_fn(dt, inputs_bub, args_bub)

        # updates properties of bubble at new time step
        props_bub = (*updated_inputs_bub, *outputs_bub)
        props_lists_bub = [t_bub, m, if_tension, R, rho_co2, p, p_bub, c_bub]
        bubble.update_props(props_bub, props_lists_bub)

        ######### SHEATH FLOW #############
        # first considers coarsening the grid by half if resolution of
        # first considers remeshing to adapt to changing gradient
        if remesh_fn is not None and (len(t_bub)%remesh_freq == 0):
            remeshed, r_arr, c[-1] = remesh_fn(r_arr, c[-1], **remesh_params)

            # only saves new grid if it remeshed and is not the first data point
            # (first data point is already saved before this loop)
            if remeshed and len(t_flow) > 1:
                print('remeshed')
                dt_max = update_dt_max(get_dr(r_arr_list[-1]), get_dr(r_arr), dt_max)
                # ensures new time step is shorter than maximum allowed, o/w
                # the solution becomes unstable
                dt = min(dt, dt_max)
                print('t', t_flow[-1] - t_nuc, 'dt', dt, 'dt_max', dt_max)
                r_arr_list += [r_arr]
                r_arr_t_list += [t_flow[-1]]

        # calculates properties after one time step with updated
        # boundary conditions
        # adds bubble radius R to grid of radii since r_arr starts at bubble
        # interface
        props_flow = diffn.time_step(dt, t_flow[-1], r_arr, c[-1], dcdt_fn,
                bc_bub_cap(c_bub[-1], c_max=c_wall), R[-1], fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t_flow, c)

        n_steps += 1
        if n_steps % 10000 == 0:
            print('t', t_flow[-1], 'dt', dt, 'n_steps', n_steps,
                'dr', r_arr[1] - r_arr[0], 'c_max', np.max(c[-1]), 'c_min', np.min(c[-1]))

    r_arr_data = (r_arr_list, r_arr_t_list)

    return t_flow, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_bulk, R, \
            rho_co2, v, r_arr_data


def sheath_incompressible(t_nuc, eps_params, R_max, N, R_i, dt_sheath,
                dc_c_s_frac=0.01, D_fn=polyco2.calc_D_lin, adaptive_dt=True,
                adapt_freq=1, legacy_mode=False, if_tension_model='lin',
                implicit=False, d_tolman=5E-9, tol_R=0.001, alpha=0.3, D=-1,
                eta_ratio=1, t_i=0,
                t_f=None, R_min=0, dcdt_fn=diffn.calc_dcdt_sph_vary_D,
                time_step_fn=bubble.time_step_dcdr,
                grid_fn=diffn.make_r_arr_lin, grid_params={},
                remesh_fn=None, remesh_params={}, remesh_freq=25):
    """
    Couples the diffusion in the sheath flow around the bubble with the growth
    of the bubble and assumes that the sheath is incompressible and that the
    walls of the fluid are pushed outward as the bubble grows.

    The only difference from num_vary_D() is that R_i is a required parameter
    rather than default infinity. This function just puts a new name on the
    computation.

    ***Note: increasing the adapt_freq above 1 (i.e., not adjusting the time
    step each time) can lead to failure if the model undergoes a change
    requiring high time resolution in between calls to adaptive_time_step().

    ***d_tolman MUST be greater than 0 to model nanometer-sized bubbles. 5 nm
    is suggested based on rule of thumb from Valeriy Ginzburg (a few atomic
    layers, see 20200608...txt)

    eta_ratio : float
        inner stream viscosity eta_i / outer stream viscosity eta_o
    """
    # sets maximum time step as sheath flow time step
    dt_max = dt_sheath
    # extracts parameters used in Epstein-Plesset model
    dt0, p_s, R_nuc, L, p_in, v, polyol_data_file, eos_co2_file = eps_params
    # INITIALIZES PARAMETERS FOR DIFFUSION IN BULK
    # assumes no sheath (i.e. R_i = R_o)
    t_flow, c, r_arr, _, _, \
    t_L, fixed_params = diffn.init_sub(R_min, R_i, R_max, N, L, v, p_s,
                                        dc_c_s_frac, polyol_data_file, t_i=t_i)

    if t_f is None:
        t_f = t_L
    dc, interp_arrs = fixed_params
    # extracts "bulk" concentration from outer edge of initial profile [kg/m^3]
    c_wall = c[0][-1]

    # INITIALIZES BUBBLE PARAMETERS
    t_bub, m, D, p, p_bub, if_tension, c_bub, \
    c_bulk, R, rho_co2, _, fixed_params_tmp = bubble.init(p_in, p_s, t_nuc,
                                            R_nuc, v, L, -1, polyol_data_file,
                                            eos_co2_file, if_tension_model,
                                            d_tolman, implicit)
    # extracts relevant parameters from bubble initiation
    _, _, p_in, p_s, v, L, _, c_s_interp_arrs, \
    if_interp_arrs, f_rho_co2, d_tolman, _ = fixed_params_tmp
    # declares fixed flow parameters # TODO--make compatible with (dc, D_fn) format
    # see diffn.calc_dcdt_sph_vary_D_nonuniform()
    fixed_params_flow = (dc, D_fn, R_i, eta_ratio)
    # initializes boundary conditions for sheath flow (no bubble)
    bc_specs_list = bc_cap(c_max=c_wall)
    # initializes list of diffusivities
    D = [D]
    # creates mesh grid
    r_arr = grid_fn(N, R_max, **grid_params)
    # initializes list of the grids and times at which grids change
    r_arr_list = [r_arr]
    r_arr_t_list = [t_flow[-1]]

    # initializes time step at sheath time step [s]
    dt = dt_sheath
    # counter for marking progress
    ctr = 0

    just_nucleated = False

    # TIME-STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t_flow[-1] <= t_f:
        # print('c, rho_co2, t, R, m', c[-1][0], rho_co2[-1], t_flow[-1], R[-1], m[-1])
        # plots concentration profile when mass begins to decrease
        if len(m) > 1 and m[-1] < m[-2]:
            print('mass in bubble decreased--density higher than bulk')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(r_arr*1E6, c[-1], '^')
            # ax.set_xscale('log')
            # ax.set_xlim([1E-3, 150])
            print('rho_co2', rho_co2[-1])
            break
        # prints out progress
        if t_flow[-1] >= ctr/10*t_f:
            print('{0:d}% complete, t = {1:.3f} ms.'.format(10*ctr, t_flow[-1]*s_2_ms))
            ctr += 1

        ########### BUBBLE GROWTH #########
        # only grows bubble if past nucleation time
        if t_flow[-1] >= t_nuc:
            just_nucleated = t_flow[-1] - dt < t_nuc
            if just_nucleated:
                # first time step should be the one from the Epstein-Plesset params
                dt = dt0
                # remesh to resolve bubble interface
                r_arr, c[-1] = diffn.regrid(r_arr, c[-1], grid_fn, N, R_max, grid_params,
                                            remesh_params['interp_kind'])
                # incorporates results more broadly
                dt_max = update_dt_max(get_dr(r_arr_list[-1]), get_dr(r_arr), dt_max)
                # ensures new time step is shorter than maximum allowed, o/w
                # the solution becomes unstable
                dt = min(dt, dt_max)
                r_arr_list += [r_arr]
                r_arr_t_list += [t_flow[-1]]

            if rho_co2[-1] > 500:
                print('warning: carbon dioxide in bubble is liquid-like; rho = {0:.1f} kg/m^3'.format(rho_co2[-1]))

            # collects parameters for time-stepping method
            D += [D_fn(c_bub[-1])]
            fixed_params_bub = (D[-1], p_in, p_s, v, L, c_s_interp_arrs,
                                    if_interp_arrs, f_rho_co2, d_tolman)
            ########### BUBBLE GROWTH #########
            # collects parameters for bubble growth
            args_bub = (r_arr, c[-1], *fixed_params_bub)
            inputs_bub = (t_bub[-1], m[-1], if_tension[-1], R[-1],
                                    rho_co2[-1])
            if adaptive_dt and (len(t_bub)%adapt_freq == 0):
                dt, updated_inputs_bub, \
                outputs_bub = bubble.adaptive_time_step(dt,
                                                    inputs_bub, args_bub,
                                                    time_step_fn, tol_R, alpha,
                                                    dt_max=dt_max,
                                                    legacy_mode=legacy_mode)
            else:
                # calculates properties after one time step
                updated_inputs_bub, outputs_bub = time_step_fn(dt, inputs_bub, args_bub)

            # updates properties of bubble at new time step
            props_bub = (*updated_inputs_bub, *outputs_bub)
            props_lists_bub = [t_bub, m, if_tension, R, rho_co2, p, p_bub, c_bub]
            bubble.update_props(props_bub, props_lists_bub)
            # updates boundary condition with concentration at bubble surface
            bc_specs_list = bc_bub_cap(c_bub[-1], c_max=c_wall)

        ######### SHEATH FLOW #############
        # first considers coarsening the grid by half if resolution of
        # first considers remeshing to adapt to changing gradient
        if remesh_fn is not None and (len(t_flow)%remesh_freq == 0) and \
                                                            not just_nucleated:
            r_arr_prev = r_arr.copy()
            c_prev = c[-1]
            # attempts remeshing
            remeshed, r_arr, c[-1] = remesh_fn(r_arr, c[-1], **remesh_params)

            #if it remeshed and is not the first data point
            # (first data point is already saved before this loop)
            if remeshed and len(t_flow) > 1:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(r_arr_prev*1E6, c_prev, '^')
                ax.set_xscale('log')
                ax.set_xlim([1E-3, 150])
                ax.set_title('before remeshing')
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(r_arr*1E6, c[-1], '^')
                ax.set_xscale('log')
                ax.set_xlim([1E-3, 150])
                ax.set_title('after remeshing')

                # updates maximum time step based on new grid
                dt_max = update_dt_max(get_dr(r_arr_list[-1]), get_dr(r_arr), dt_max)
                # ensures new time step is shorter than maximum allowed, o/w
                # the solution becomes unstable
                dt = min(dt, dt_max)
                r_arr_list += [r_arr]
                r_arr_t_list += [t_flow[-1]]
                print('remeshed')

        # if the next time step will surpass the nucleation time for the first
        # time, shorten it so it exactly reaches the nucleation time
        if (t_flow[-1] + dt >= t_nuc) and (t_flow[-1] < t_nuc):
            # shortens time step [s]
            dt = t_nuc - t_flow[-1]

        # calculates properties after one time step with updated
        # boundary conditions
        # adds bubble radius R to grid of radii since r_arr starts at bubble
        # interface

        props_flow = diffn.time_step(dt, t_flow[-1], r_arr, c[-1], dcdt_fn,
                        bc_specs_list, R[-1], fixed_params_flow)

        if np.any(np.isnan(c[-1])):
            print('c[0:2]', props_flow[-1][:2])
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t_flow, c)

        if np.any(np.isnan(c[-1])):
            print('nan in c[-1]')
            print('after update')
            print('c[0:2]', props_flow[-1][:2])
            break

    r_arr_data = (r_arr_list, r_arr_t_list)

    return t_flow, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_bulk, R, \
            rho_co2, v, r_arr_data


def update_dr_dt(dr, r_arr, dt_max):
    """
    Updates values of grid spacing dr and maximum time step dt_max.
    """
    dr_new = np.min(np.diff(r_arr))
    if dt_max is not None:
        print('update dt_max')
        dt_max *= (dr_new / dr)**2

    return dr_new, dt_max


def update_dt_max(dr_prev, dr, dt_max):
    """
    Updates values of grid spacing dr and maximum time step dt_max.
    """
    if dt_max is not None:
        dt_max *= (dr / dr_prev)**2
    return dt_max