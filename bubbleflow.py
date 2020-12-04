"""
@bubbleflow.py contains functions for modeling bubble growth during flow
through the sheath flow in the microfluidic channel. It combines functions from
bubble.py for bubble growth and diffn.py for diffusion in the bulk fluid to
provide a more complete model of each phenomenon and their interaction.

@date November 24, 2020
@author Andy Ylitalo
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


def grow(dt_sheath, dt, dcdt_fn, R_o, N, eta_i, eta_o, d, L, Q_i, Q_o, p_s,
         dc_c_s_frac, t_nuc, R_nuc, polyol_data_file, eos_co2_file, bc_specs_list,
         if_tension_model='lin', d_tolman=0, adaptive_dt=True, implicit=False,
         tol_R=0.001, alpha=0.3, drop_t_term=False, R_min=0, D=-1,
         i_c_bulk=7):
    """
    Grows bubble based on combined diffusion model of bulk and bubble interface.

    ASSUMPTIONS
    -Bubble "consumes" polyol as it grows, so its growth does not directly
    affect the CO2 concentration profile

    dt_sheath : float
        Time step for modeling diffusion in sheath flow without a bubble [s].
        Typically larger than dt.
    dt : float
        Time step for modeling bubble growth [s]. Typically smaller than dt_sheath.
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
    c_s, R, rho_co2, _, fixed_params_bub = bubble.init(p_in, P_ATM, p_s, t_nuc,
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

        # updates bulk concentration used in bubble model
        # i_R = np.argmin(np.abs(r_arr - R[-1]))
        # c_R = c[-1][i_R]
        # fixed_params_bub = list(fixed_params_bub)
        # fixed_params_bub[i_c_bulk] = c_R
        # print(c_R)

    return t, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_s, R, rho_co2, R_i, v


def bc_bub_cap(c_bub, c_max=0):
    """
    c_bub is concentration of CO2 at interface of bubble [kg/m^3].

    Fixes concentration at inner wall of capillary to c_max (default 0).
    Fixed concentration at bubble surface (saturation concentration).
    """
    return [(diffn.dirichlet, 0, c_bub), (diffn.dirichlet, -1, c_max)]


def halve_grid(arr):
    """
    Halves the number of points in the grid by removing every other point.
    Assumes grid has N + 1 elements, where N is divisible by 2.
    The resulting grid will have N/2 + 1 elements.
    """
    print('halving grid')
    return arr[::2]


def numerical_eps_pless_fix_D(dt, t_nuc, p_s, R_nuc, L, p_in, v, R_max, N,
                     polyol_data_file, eos_co2_file, adaptive_dt=True,
                     if_tension_model='lin', implicit=False, d_tolman=0,
                     tol_R=0.001, alpha=0.3, D=-1, dt_max=None,
                     R_min=0, dcdt_fn=diffn.calc_dcdt_sph_fix_D,
                     time_step_fn=bubble.time_step_dcdr):
    """
    Performs numerical computation of Epstein-Plesset model for comparison.
    Once confirmed to provide accurate estimation of Epstein-Plesset result,
    this will be modified to include the effects of a concentration-dependent
    diffusivity in numerical_eps_pless_vary_D().
    """
    # INITIALIZES BUBBLE PARAMETERS
    t_bub, m, D, p, p_bub, if_tension, c_bub, \
    c_bulk, R, rho_co2, _, fixed_params_tmp = bubble.init(p_in, P_ATM, p_s, t_nuc,
                                            R_nuc, v, L, D, polyol_data_file,
                                            eos_co2_file, if_tension_model,
                                            d_tolman, implicit)
    # extracts relevant parameters from bubble initiation
    _, D, p_in, p_s, p_atm, v, L, _, c_s_interp_arrs, \
    if_interp_arrs, f_rho_co2, d_tolman, _ = fixed_params_tmp
    # collects parameters relevant for bubble growth
    fixed_params_bub = (D, p_in, p_s, v, L, c_s_interp_arrs, if_interp_arrs,
                        f_rho_co2, d_tolman)

    # INITIALIZES PARAMETERS FOR DIFFUSION IN BULK
    # starts at nucleation time since we do not consider diffusion before bubble
    t_flow = [t_nuc]
    r_arr = np.linspace(R_min, R_max, N+1)
    c = [c_bulk*np.ones(N+1)]

    # final time of model [s]
    t_f = L/v

    # TIME STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t_bub[-1] <= t_f:
        # collects parameters for bubble growth
        time_step_params = (t_bub[-1], m[-1], if_tension[-1], R[-1],
                                rho_co2[-1], r_arr, c[-1], fixed_params_bub)
        # BUBBLE GROWTH
        if adaptive_dt:
            dt, props_bub = bubble.adaptive_time_step(dt, time_step_params,
                                                    time_step_fn, tol_R, alpha,
                                                    dt_max=dt_max)
        else:
            # calculates properties after one time step
            props_bub = time_step_fn(dt, *time_step_params)

        # updates properties of bubble at new time step
        bubble.update_props(props_bub, t_bub, m, p, p_bub, if_tension, c_bub, R,
                            rho_co2)

        # SHEATH FLOW
        # calculates properties after one time step with updated
        # boundary conditions
        # adds bubble radius R to grid of radii since r_arr starts at bubble
        # interface
        fixed_params_flow = (D, R[-1])
        props_flow = diffn.time_step(dt, t_flow[-1], r_arr, c[-1], dcdt_fn,
                        bc_bub_cap(c_bub[-1], c_max=c_bulk), fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t_flow, c)

    return t_flow, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_bulk, R, \
                rho_co2, v, r_arr


def numerical_eps_pless_vary_D(dt, t_nuc, p_s, R_nuc, L, p_in, v, R_max, N,
                             polyol_data_file, eos_co2_file, dc_c_s_frac,
                             adaptive_dt=True,
                             if_tension_model='lin', implicit=False, d_tolman=0,
                             tol_R=0.001, alpha=0.3, D=-1, dt_max=None,
                             R_min=0, dcdt_fn=diffn.calc_dcdt_sph_vary_D,
                             time_step_fn=bubble.time_step_dcdr,
                             D_fn=polyco2.calc_D_lin,
                             half_grid=False, pts_per_grad=20):
    """
    Peforms numerical computation of diffusion into bubble from bulk accounting
    for effect of concentration of CO2 on the local diffusivity D.
    """
    # INITIALIZES PARAMETERS FOR DIFFUSION IN BULK
    # assumes no sheath (i.e. R_i = R_o)
    t_flow, c, r_arr, _, _, \
    t_f, fixed_params = diffn.init_sub(R_min, R_max, R_max, N, L, v, p_s,
                                        dc_c_s_frac, polyol_data_file, t_i=t_nuc)
    dc, interp_arrs = fixed_params
    # INITIALIZES BUBBLE PARAMETERS
    t_bub, m, D, p, p_bub, if_tension, c_bub, \
    c_bulk, R, rho_co2, _, fixed_params_tmp = bubble.init(p_in, P_ATM, p_s, t_nuc,
                                            R_nuc, v, L, -1, polyol_data_file,
                                            eos_co2_file, if_tension_model,
                                            d_tolman, implicit)
    # extracts relevant parameters from bubble initiation
    _, _, p_in, p_s, p_atm, v, L, _, c_s_interp_arrs, \
    if_interp_arrs, f_rho_co2, d_tolman, _ = fixed_params_tmp

    # prepares arrays for interpolating D(c)
    D_params = polyco2.lin_fit_D_c(*interp_arrs)
    # initializes list of diffusivities
    D = []
    # initializes list of grid spacings [m]
    dr_list = [r_arr[1]-r_arr[0]]

    # TIME-STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t_bub[-1] <= t_f:
        # collects parameters for time-stepping method
        D += [D_fn(c_bub[-1], D_params)]
        fixed_params_bub = (D[-1], p_in, p_s, v, L, c_s_interp_arrs, if_interp_arrs,
                            f_rho_co2, d_tolman)
        time_step_params = (t_bub[-1], m[-1], if_tension[-1], R[-1],
                                rho_co2[-1], r_arr, c[-1], fixed_params_bub)
        # BUBBLE GROWTH
        if adaptive_dt:
            dt, props_bub = bubble.adaptive_time_step(dt, time_step_params,
                                                    time_step_fn, tol_R, alpha,
                                                    dt_max=dt_max)
        else:
            # calculates properties after one time step
            props_bub = time_step_fn(dt, *time_step_params)

        # updates properties of bubble at new time step
        bubble.update_props(props_bub, t_bub, m, p, p_bub, if_tension, c_bub, R,
                            rho_co2)

        # SHEATH FLOW
        # first considers coarsening the grid by half if resolution of
        # concentration gradient is sufficient
        if half_grid:
            dr = r_arr[1] - r_arr[0]
            dcdr = c[-1][1] / dr # assumes c(r=0) = 0
            if 2*dr < c_bulk / (pts_per_grad * dcdr):
                r_arr = halve_grid(r_arr)
                c[-1] = halve_grid(c[-1])
                dr = r_arr[1] - r_arr[0]
                # quadruples maximum time step since limit on time step is
                # proportional to spatial resolution squared
                if dt_max is not None:
                    dt_max *= 4

            dr_list += [dr]

        # calculates properties after one time step with updated
        # boundary conditions
        # adds bubble radius R to grid of radii since r_arr starts at bubble
        # interface
        fixed_params_flow = (R[-1], dc, D_params, D_fn)
        props_flow = diffn.time_step(dt, t_flow[-1], r_arr, c[-1], dcdt_fn,
                        bc_bub_cap(c_bub[-1], c_max=c_bulk), fixed_params_flow)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t_flow, c)

    # returns list of grid spacings if grid-halving; o/w returns full grid
    if half_grid:
        r_param = dr_list
    else:
        r_param = r_arr

    return t_flow, c, t_bub, m, D, p, p_bub, if_tension, c_bub, c_bulk, R, \
            rho_co2, v, r_param
