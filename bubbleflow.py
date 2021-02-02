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
                            bc_specs_list, fixed_params_flow)
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
                            bc_bub_cap(c_bub[-1]), fixed_params_flow)
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
                    d_tolman=0, tol_R=0.001, alpha=0.3, D=-1, dt_max=None,
                    R_min=0, dcdt_fn=diffn.calc_dcdt_sph_fix_D,
                    time_step_fn=bubble.time_step_dcdr, legacy_mode=False,
                    grid_fn=diffn.make_r_arr_lin, grid_params={}, adapt_freq=5,
                    remesh_fn=diffn.remesh, remesh_params={}, remesh_freq=1000):
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
    if_interp_arrs, f_rho_co2, d_tolman, _ = fixed_params_tmp
    # collects parameters relevant for bubble growth
    fixed_params_bub = (D, p_in, p_s, v, L, c_s_interp_arrs, if_interp_arrs,
                        f_rho_co2, d_tolman)

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
        if remesh_fn is not None and (len(t_bub)%remesh_freq == 0):
            print('consider remeshing')
            remeshed, r_arr, c[-1] = remesh_fn(r_arr, c[-1], **remesh_params)
            # only saves new grid if it remeshed and is not the first data point
            # (first data point is already saved before this loop)
            if remeshed and len(t_flow) > 1:
                print('remeshed')
                dt_max = update_dt_max(get_dr(r_arr_list[-1]), get_dr(r_arr), dt_max)
                r_arr_list += [r_arr]
                r_arr_t_list += [t_flow[-1]]

        # calculates properties after one time step with updated
        # boundary conditions
        # adds bubble radius R to grid of radii since r_arr starts at bubble
        # interface
        # for now, uses same grid spacing for each time step
        # computes time step
        fixed_params_flow = (D, R[-1])
        props_flow = diffn.time_step(dt, t_flow[-1], r_arr, c[-1], dcdt_fn,
                        bc_bub_cap(c_bub[-1], c_max=c_bulk), fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t_flow, c)

    return t_flow, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_bulk, R, \
                rho_co2, v, (r_arr_list, r_arr_t_list)


def num_vary_D(t_nuc, eps_params, R_max, N, dc_c_s_frac,
                 dt_max=None, D_fn=polyco2.calc_D_lin,
                 adaptive_dt=True, adapt_freq=5, legacy_mode=False,
                 if_tension_model='lin', implicit=False, d_tolman=0,
                 tol_R=0.001, alpha=0.3, D=-1, R_i=np.inf,
                 R_min=0, dcdt_fn=diffn.calc_dcdt_sph_vary_D,
                 time_step_fn=bubble.time_step_dcdr,
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

    # initializes list of diffusivities
    D = [D]
    # initializes list of grid spacings [m]
    dr_list = [r_arr[1] - r_arr[0]]

    # TIME-STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t_bub[-1] <= t_f:
        # collects parameters for time-stepping method
        D += [D_fn(c_bub[-1])]
        fixed_params_bub = (D[-1], p_in, p_s, v, L, c_s_interp_arrs,
                                if_interp_arrs, f_rho_co2, d_tolman)
        time_step_params = (t_bub[-1], m[-1], if_tension[-1], R[-1],
                                rho_co2[-1], r_arr, c[-1], fixed_params_bub)
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
        # concentration gradient is sufficient
        dr = r_arr[1] - r_arr[0]
        # remeshes
        if remesh_fn is not None and (len(t_bub)%remesh_freq == 0):
            r_arr, c[-1] = remesh_fn(r_arr, c[-1], **remesh_params)
            dr, dt_max = update_dr_dt(dr, r_arr, dt_max)
            # retroactively updates dr list in case grid was halved
            dr_list[-1] = dr
        # stores grid spacing
        dr_list += [dr]
        # calculates properties after one time step with updated
        # boundary conditions
        # adds bubble radius R to grid of radii since r_arr starts at bubble
        # interface
        fixed_params_flow = (R[-1], dc, D_fn)
        props_flow = diffn.time_step(dt, t_flow[-1], r_arr, c[-1], dcdt_fn,
                        bc_bub_cap(c_bub[-1], c_max=c_wall), fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t_flow, c)

    return t_flow, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_bulk, R, \
            rho_co2, v, dr_list


def sheath_incompressible(t_nuc, eps_params, R_max, N, dc_c_s_frac, R_i, dt_sheath,
                 D_fn=polyco2.calc_D_lin,
                 adaptive_dt=True, if_tension_model='lin', implicit=False,
                 d_tolman=0, tol_R=0.001, alpha=0.3, D=-1, t_i=0,
                 R_min=0, dcdt_fn=diffn.calc_dcdt_sph_vary_D,
                 time_step_fn=bubble.time_step_dcdr,
                 remesh_fn=diffn.remesh, remesh_params={}):
    """
    Couples the diffusion in the sheath flow around the bubble with the growth
    of the bubble and assumes that the sheath is incompressible and that the
    walls of the fluid are pushed outward as the bubble grows.

    The only difference from num_vary_D() is that R_i is a required parameter
    rather than default infinity. This function just puts a new name on the
    computation.
    """
    # extracts parameters used in Epstein-Plesset model
    dt0, p_s, R_nuc, L, p_in, v, polyol_data_file, eos_co2_file = eps_params
    # INITIALIZES PARAMETERS FOR DIFFUSION IN BULK
    # assumes no sheath (i.e. R_i = R_o)
    t_flow, c, r_arr, _, _, \
    t_f, fixed_params = diffn.init_sub(R_min, R_i, R_max, N, L, v, p_s,
                                        dc_c_s_frac, polyol_data_file, t_i=t_i)
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

    # initializes boundary conditions for sheath flow (no bubble)
    bc_specs_list = bc_cap(c_max=c_wall)
    # initializes list of diffusivities
    D = [D]
    # initializes list of grid spacings [m]
    dr_list = [r_arr[1]-r_arr[0]]
    dr = dr_list[-1]
    # initializes time step at sheath time step [s]
    dt = dt_sheath
    # counter for marking progress
    ctr = 0

    # TIME-STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t_flow[-1] <= t_f:

        # prints out progress
        if t_flow[-1] >= ctr/10*t_f:
            print('{0:d}% complete, t = {1:.3f} ms.'.format(10*ctr, t_flow[-1]*s_2_ms))
            ctr += 1

        ########### BUBBLE GROWTH #########
        # only grows bubble if past nucleation time
        if t_flow[-1] >= t_nuc:
            # first time step should be the one from the Epstein-Plesset params
            if t_flow[-1] - dt < t_nuc:
                dt = dt0
            # collects parameters for time-stepping method
            D += [D_fn(c_bub[-1])]
            fixed_params_bub = (D[-1], p_in, p_s, v, L, c_s_interp_arrs,
                                    if_interp_arrs, f_rho_co2, d_tolman)
            time_step_params = (t_bub[-1], m[-1], if_tension[-1], R[-1],
                                    rho_co2[-1], r_arr, c[-1], fixed_params_bub)
            if adaptive_dt:
                dt, props_bub = bubble.adaptive_time_step(dt, time_step_params,
                                                        time_step_fn, tol_R,
                                                        alpha, dt_max=dt_sheath)
            else:
                # calculates properties after one time step
                props_bub = time_step_fn(dt, *time_step_params)

            # updates properties of bubble at new time step
            bubble.update_props(props_bub, t_bub, m, p, p_bub, if_tension,
                                c_bub, R, rho_co2)
            # updates boundary condition with concentration at bubble surface
            bc_specs_list = bc_bub_cap(c_bub[-1], c_max=c_wall)

            # once bubble has nucleated considers coarsening the grid by half
            # if resolution of concentration gradient is sufficient
            dr = r_arr[1] - r_arr[0]
            # remeshes
            if remesh_fn is not None:
                r_arr, c[-1] = remesh_fn(r_arr, c[-1], **remesh_params)
                dr, dt_sheath = update_dr_dt(dr, r_arr, dt_sheath)
                # retroactively updates dr list in case grid was halved
                dr_list[-1] = dr
            dr_list += [dr]
        ######### SHEATH FLOW #############
        # if the next time step will surpass the nucleation time for the first
        # time, shorten it so it exactly reaches the nucleation time
        if (t_flow[-1] + dt >= t_nuc) and (t_flow[-1] < t_nuc):
            # shortens time step [s]
            dt = t_nuc - t_flow[-1]

        # stores grid spacing
        dr_list += [dr]
        # calculates properties after one time step with updated
        # boundary conditions
        # adds bubble radius R to grid of radii since r_arr starts at bubble
        # interface
        fixed_params_flow = (R[-1], dc, D_fn)
        props_flow = diffn.time_step(dt, t_flow[-1], r_arr, c[-1], dcdt_fn,
                        bc_specs_list, fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t_flow, c)

    return t_flow, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_bulk, R, \
            rho_co2, v, dr_list


def update_dr_dt(dr, r_arr, dt_max):
    """
    Updates values of grid spacing dr and maximum time step dt_max.
    """
    dr_new = np.min(np.diff(r_arr))
    if dt_max is not None:
        print('update dt_max')
        print(dt_max)
        print(dr_new)
        print(dr)
        dt_max *= (dr_new / dr)**2

    return dr_new, dt_max


def update_dt_max(dr_prev, dr, dt_max):
    """
    Updates values of grid spacing dr and maximum time step dt_max.
    """
    if dt_max is not None:
        dt_max *= (dr / dr_prev)**2
    return dt_max
