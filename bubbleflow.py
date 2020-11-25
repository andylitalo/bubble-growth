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

# CONSTANTS
P_ATM = 101.3E3 #[Pa]


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


def numerical_eps_pless_fix_D(dt, t_nuc, p_s, R_nuc, L, p_in, v, R_max, N,
                     polyol_data_file, eos_co2_file, adaptive_dt=True,
                     if_tension_model='lin', implicit=False, d_tolman=0,
                     tol_R=0.001, alpha=0.3, D=-1,
                     R_min=0, dcdt_fn=diffn.calc_dcdt_sph_fix_D,
                        time_step_fn=bubble.time_step_dcdr_fix_D):
    """
    Performs numerical computation of Epstein-Plesset model for comparison.
    Once confirmed to provide accurate estimation of Epstein-Plesset result,
    this will be modified to include the effects of a concentration-dependent
    diffusivity.
    """
    # INITIALIZES BUBBLE PARAMETERS
    t_bub, m, D, p, p_bub, if_tension, c_bub, \
    c_bulk, R, rho_co2, _, fixed_params_bub = bubble.init(p_in, P_ATM, p_s, t_nuc,
                                            R_nuc, v, L, D, polyol_data_file,
                                            eos_co2_file, if_tension_model,
                                            d_tolman, implicit)

    # INITIALIZES PARAMETERS FOR DIFFUSION IN BULK
    # starts at nucleation time since we do not consider diffusion before bubble
    t_flow = [t_nuc]
    r_arr = np.linspace(R_min, R_max, N+1)
    c = [c_bulk*np.ones(N+1)]
    # only fixed parameter for fixed diffusivity is the diffusivity [m^2/s]
    fixed_params_flow = (D)

    # final time of model [s]
    t_f = L/v

    # TIME STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t_bub[-1] <= t_f:
        time_step_params = (t_bub[-1], m[-1], if_tension[-1], R[-1],
                                rho_co2[-1], r_arr, c[-1], fixed_params_bub)
        # BUBBLE GROWTH
        if adaptive_dt:
            dt, props_bub = bubble.adaptive_time_step(dt, time_step_params,
                                                    time_step_fn, tol_R, alpha)
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
