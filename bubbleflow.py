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

# imports custom libraries
import bubble
import diffn




def grow(dt_sheath, dt):
    """
    Grows bubble based on combined diffusion model of bulk and bubble interface.

    dt_sheath : float
        Time step for modeling diffusion in sheath flow without a bubble [s].
        Typically larger than dt.
    dt : float
        Time step for modeling bubble growth [s]. Typically smaller than dt_sheath.
    """
    # initializes parameters for bubble growth
    t_b, m, D, p, p_bubble, if_tension, c_bubble, \
    c_bulk, R, rho_co2, t_f, fixed_params = bubble.init(p_in, p_atm, p_s, t_nuc,
                                            R_nuc, v, L, D, c_bulk,
                                            polyol_data_file, eos_co2_file,
                                            if_tension_model, d_tolman, implicit)

    # initializes parameters for CO2 diffusion in sheath flow
    # use t_f (final time) = L/v from bubble instead of d/v from diffn
    t, c, r_arr, R_i, v, \
    c_0, c_s, _, fixed_params = init(R_min, R_o, N, eta_i, eta_o, d, L, Q_i,
                                        Q_o, p_s, dc_c_s_frac, polyol_data_file)
    # TIME STEPPING -- PRE-BUBBLE
    # applies Euler's method to estimate bubble growth over time
    # the second condition provides cutoff for shrinking the bubble
    while t[-1] <= t_nuc:
        # calculates properties after one time step
        props_flow = diffn.time_step(dt_sheath, t[-1], r_arr, c[-1], dcdt_fn,
                            bc_specs_list, fixed_params)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t, c)

    # TIME STEPPING -- BUBBLE NUCLEATES AND GROWS
    while t[-1] <= t_f:
        # BUBBLE GROWTH
        if adaptive_dt:
            dt, props_bub = bubble.adaptive_time_step(dt, t_b, m, p, if_tension, R,
                                                rho_co2, fixed_params, tol_R,
                                                alpha, drop_t_term)
        else:
            # calculates properties after one time step
            props_bub = bubble.time_step(dt, t_b[-1], m[-1], p[-1], if_tension[-1],
                                    R[-1], rho_co2[-1], fixed_params,
                                    drop_t_term=drop_t_term)
        bubble.update_props(prop_bub, t_b, m, p, p_bubble, if_tension, c_s, R, rho_co2)

        # SHEATH FLOW
        # calculates properties after one time step
        props_flow = diffn.time_step(dt, t[-1], r_arr, c[-1], dcdt_fn,
                            bc_specs_list, fixed_params)
        # stores properties at new time step in lists
        diffn.update_props(props_flow, t, c)

    return t, c, t_b, m, D, p, p_bubble, if_tension, c_s, c_bulk, R, rho_co2
