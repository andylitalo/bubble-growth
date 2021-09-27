import numpy as np
import scipy.optimize
import scipy.interpolate
import pandas as pd
import matplotlib.pyplot as plt

# CONVERSIONS
s_2_ms = 1000
m_2_um = 1E6
kPa_2_Pa = 1000
gmL_2_kgm3 = 1000
cm2s_2_m2s = 1E-4


def eps_pless_p_if_1(dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v, polyol_data_file, eos_co2_file):
    """
    Solves for bubble growth based on Epstein and Plesset (1950) with
    modifications for changing pressure (p) and interfacial tension (if_tension).
    
    Parameters:
        dt : float
            time step [s]
        t_nuc : float
            time of bubble nucleation measured after entering observation capillary [s]
        p_s : float
            saturation pressure of CO2 in polyol [Pa]
        R_nuc : float
            approximate radius of initial bubble nucleus, based on Dr. Huikuan Chao's string method model [m]
        p_atm : float
            atmospheric pressure [Pa], assumed to be the pressure at the outlet
        L : float
            length of observation capillary [m]
        p_in : float
            pressure at inlet [m], calculated using flow_eqns
        v : float
            velocity of inner stream [m/s], calculated using flow_eqns
        polyol_data_file : string
            name of file containing polyol data [.csv]
        eos_co2_file : string
            File name for equation of state data table [.csv]
            
    Returns:
        t : list of N floats
            times at which numerical model was evaluated, measured relative to time of entering observation capillary [s]
        m : list of N floats
            mass of CO2 enclosed in bubble at each time step [g]
        D : list of N floats
            diffusivity at each time step estimated from experimental G-ADSA measurements (averaged exp & sqrt) [m^2/s]
        p : list of N floats
            pressure at each time step assuming linear pressure drop along channel [Pa]
        if_tension : list of N floats
            interfacial tension along bubble surface at each time step based on G-ADSA measurements [N/m]
        c_s : list of N floats
            saturation concentrations of CO2 in polyol at each time step based 
            on G-ADSA solubility and density measurements [kg CO2 / m^3 polyol-CO2]
        R : list of N floats
            radius of bubble at each time step solved self-consistently with modified Epstein-Plesset (1950) [m]
        rho_co2 : list of N floats
            density of CO2 in bubble at each time step based on pressure and CO2 equation of state [kg/m^3]
    """    
    # initializes lists of key bubble properties
    p = [calc_p(p_in, p_atm, v, t_nuc, L)]
    D = [calc_D(p[0], polyol_data_file)]
    if_tension = [calc_if_tension(p[0], polyol_data_file)]
    c_s = [calc_c_s(p[0], polyol_data_file)]
    R = [R_nuc]
    # initial bubble density [kg/m^3]
    f_rho_co2 = interp_rho_co2(eos_co2_file)
    rho_co2 = [f_rho_co2(p[0] + 2*if_tension[0]/R[0])]
    # initializes bubble mass of CO2 [kg]
    m = [rho_co2[0]*4*np.pi/3*(R[0])**3]

    # initializes timeline [s]
    t = [t_nuc]
    # defines final time [s]
    t_f = L/v
    
    # applies Euler's method to estimate bubble growth over time
    while t[-1] < t_f-dt:
        t += [t[-1]+dt] # increments time forward [s]
        m += [m[-1] + dt*calc_dmdt(D[-1], p_s, p[-1], R[-1], t[-1], 
              polyol_data_file)] # Euler's method for mass [g]
        D += [calc_D(p[-1], polyol_data_file)] # interpolates new diffusivity [m^2/s]
        p += [calc_p(p_in, p_atm, v, t[-1], L)] # computes new pressure along observation capillary [Pa]
        if_tension += [calc_if_tension(p[-1], polyol_data_file)] # interpolates interfacial tension along bubble [N/m]
        c_s += [calc_c_s(p[-1], polyol_data_file)] # interpolates saturation concentration of CO2 [kg CO2 / m^3 polyol-CO2]
        R += [calc_R(m[-1], p[-1], R[-1], if_tension[-1], eos_co2_file)] # self-consistently solves for bubble radius [m]
        rho_co2 += [f_rho_co2(p[-1] + 2*if_tension[-1]/R[-1])] # saves CO2 density in bubble [kg/m^3]

    return t, m, D, p, if_tension, c_s, R, rho_co2


def eps_pless_p_if_2(dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v, 
                     polyol_data_file, eos_co2_file):
    """
    Solves for bubble growth based on Epstein and Plesset (1950) with
    modifications for changing pressure (p) and interfacial tension (if_tension).
    ***Difference from eps_pless_p_if_1: Assumes that 
    (solubility of CO2 in the parent phase) / (density of CO2 in the bubble)
    is constant.
    
    Parameters:
        dt : float
            time step [s]
        t_nuc : float
            time of bubble nucleation measured after entering observation capillary [s]
        p_s : float
            saturation pressure of CO2 in polyol [Pa]
        R_nuc : float
            approximate radius of initial bubble nucleus, based on Dr. Huikuan Chao's string method model [m]
        p_atm : float
            atmospheric pressure [Pa], assumed to be the pressure at the outlet
        L : float
            length of observation capillary [m]
        p_in : float
            pressure at inlet [m], calculated using flow_eqns
        v : float
            velocity of inner stream [m/s], calculated using flow_eqns
        polyol_data_file : string
            name of file containing polyol data [.csv]
        eos_co2_file : string
            File name for equation of state data table [.csv]
            
    Returns:
        t : list of N floats
            times at which numerical model was evaluated, measured relative to time of entering observation capillary [s]
        m : list of N floats
            mass of CO2 enclosed in bubble at each time step [g]
        D : list of N floats
            diffusivity at each time step estimated from experimental G-ADSA measurements (averaged exp & sqrt) [m^2/s]
        p : list of N floats
            pressure at each time step assuming linear pressure drop along channel [Pa]
        if_tension : list of N floats
            interfacial tension along bubble surface at each time step based on G-ADSA measurements [N/m]
        c_s : list of N floats
            saturation concentrations of CO2 in polyol at each time step based 
            on G-ADSA solubility and density measurements [kg CO2 / m^3 polyol-CO2]
        R : list of N floats
            radius of bubble at each time step solved self-consistently with modified Epstein-Plesset (1950) [m]
        rho_co2 : list of N floats
            density of CO2 in bubble at each time step based on pressure and CO2 equation of state [kg/m^3]
    """    
    # initializes lists of key bubble properties
    p = [calc_p(p_in, p_atm, v, t_nuc, L)]
    D = [calc_D(p[0], polyol_data_file)]
    if_tension = [calc_if_tension(p[0], polyol_data_file)]
    c_s = [calc_c_s(p[0], polyol_data_file)]
    R = [R_nuc]
    # initial bubble density [kg/m^3]
    f_rho_co2 = interp_rho_co2(eos_co2_file)
    rho_co2 = [f_rho_co2(p[0] + 2*if_tension[0]/R[0])]
    # initializes bubble mass of CO2 [kg]
    m = [rho_co2[0]*4*np.pi/3*(R[0])**3]

    # initializes timeline [s]
    t = [t_nuc]
    # defines final time [s]
    t_f = L/v
    
    # applies Euler's method to estimate bubble growth over time
    while t[-1] < t_f-dt:
        t += [t[-1]+dt] # increments time forward [s]
        m += [m[-1] + dt*calc_dmdt(D[-1], p_s, p[-1], R[-1],
                                   t[-1], polyol_data_file, 
                                   add_p_young_laplace=True)] # Euler's method for mass [g], adds p_Y-L for c_s_p
        D += [calc_D(p[-1], polyol_data_file)] # interpolates diffusivity in parent phase [m^2/s]
        p += [calc_p(p_in, p_atm, v, t[-1], L)] # computes new pressure along observation capillary [Pa]
        if_tension += [calc_if_tension(p[-1], polyol_data_file)] # interpolates interfacial tension along bubble [N/m]
        c_s += [calc_c_s(p[-1] + 2*if_tension[-1]/R[-1], polyol_data_file)] # interpolates sat. conc. of CO2 [kg CO2 / m^3 polyol-CO2]
        R += [calc_R(m[-1], p[-1], R[-1], if_tension[-1], eos_co2_file)] # self-consistently solves for bubble radius [m]
        rho_co2 += [f_rho_co2(p[-1] + 2*if_tension[-1]/R[-1])] # saves CO2 density in bubble [kg/m^3]
        
    return t, m, D, p, if_tension, c_s, R, rho_co2
    
    
    
def eps_pless_p_if_3(dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v, 
                     polyol_data_file, eos_co2_file):
    """
    Solves for bubble growth based on Epstein and Plesset (1950) with
    modifications for changing pressure (p) and interfacial tension (if_tension).
    Difference from eps_pless_p_if_1: t = t - t_nuc, such that the first point 
    is defined to have t = 0.
    
    Parameters:
        dt : float
            time step [s]
        t_nuc : float
            time of bubble nucleation measured after entering observation capillary [s]
        p_s : float
            saturation pressure of CO2 in polyol [Pa]
        R_nuc : float
            approximate radius of initial bubble nucleus, based on Dr. Huikuan Chao's string method model [m]
        p_atm : float
            atmospheric pressure [Pa], assumed to be the pressure at the outlet
        L : float
            length of observation capillary [m]
        p_in : float
            pressure at inlet [m], calculated using flow_eqns
        v : float
            velocity of inner stream [m/s], calculated using flow_eqns
        polyol_data_file : string
            name of file containing polyol data [.csv]
        eos_co2_file : string
            File name for equation of state data table [.csv]
            
    Returns:
        t : list of N floats
            times at which numerical model was evaluated, measured relative to time of entering observation capillary [s]
        m : list of N floats
            mass of CO2 enclosed in bubble at each time step [g]
        D : list of N floats
            diffusivity at each time step estimated from experimental G-ADSA measurements (averaged exp & sqrt) [m^2/s]
        p : list of N floats
            pressure at each time step assuming linear pressure drop along channel [Pa]
        if_tension : list of N floats
            interfacial tension along bubble surface at each time step based on G-ADSA measurements [N/m]
        c_s : list of N floats
            saturation concentrations of CO2 in polyol at each time step based 
            on G-ADSA solubility and density measurements [kg CO2 / m^3 polyol-CO2]
        R : list of N floats
            radius of bubble at each time step solved self-consistently with modified Epstein-Plesset (1950) [m]
        rho_co2 : list of N floats
            density of CO2 in bubble at each time step based on pressure and CO2 equation of state [kg/m^3]
    """    
    # initializes lists of key bubble properties
    p = [calc_p(p_in, p_atm, v, t_nuc, L)]
    D = [calc_D(p[0], polyol_data_file)]
    if_tension = [calc_if_tension(p[0], polyol_data_file)]
    c_s = [calc_c_s(p[0], polyol_data_file)]
    R = [R_nuc]
    p_bubble = [p[0] + 2*if_tension[0]/R[0]]
    # initial bubble density [kg/m^3]
    f_rho_co2 = interp_rho_co2(eos_co2_file)
    rho_co2 = [f_rho_co2(p[0] + 2*if_tension[0]/R[0])]
    # initializes bubble mass of CO2 [kg]
    m = [rho_co2[0]*4*np.pi/3*(R[0])**3]

    # initializes timeline [s]
    t = [t_nuc]
    # defines final time [s]
    t_f = L/v
    
    # applies Euler's method to estimate bubble growth over time
    while t[-1] < t_f-dt:
        t += [t[-1]+dt] # increments time forward [s]
        m += [m[-1] + dt*calc_dmdt(D[-1], p_s, p[-1], R[-1], t[-1]-t_nuc, 
              polyol_data_file)] # Euler's method for mass [g], t_nuc -> t = 0
        D += [calc_D(p[-1], polyol_data_file)] # interpolates new diffusivity [m^2/s]
        p += [calc_p(p_in, p_atm, v, t[-1], L)] # computes new pressure along observation capillary [Pa]
        if_tension += [calc_if_tension(p[-1], polyol_data_file)] # interpolates interfacial tension along bubble [N/m]
        c_s += [calc_c_s(p[-1], polyol_data_file)] # interpolates saturation concentration of CO2 [kg CO2 / m^3 polyol-CO2]
        R += [calc_R(m[-1], p[-1], R[-1], if_tension[-1], eos_co2_file)] # self-consistently solves for bubble radius [m]
        p_bubble += [p[-1] + 2*if_tension[-1]/R[-1]]
        rho_co2 += [f_rho_co2(p[-1] + 2*if_tension[-1]/R[-1])] # saves CO2 density in bubble [kg/m^3]
        
    return t, m, D, p, p_bubble, if_tension, c_s, R, rho_co2


def eps_pless_p_if_4(dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v, 
                     polyol_data_file, eos_co2_file, adaptive_dt=True,
                     if_tension_model='lin', implicit=False):
    """
    Solves for bubble growth based on Epstein and Plesset (1950) with
    modifications for changing pressure (p) and interfacial tension (if_tension).
    
    Difference from eps_pless_p_if_3: 
        -uses pressure inside bubble to estimate interfacial tension. 
            Eventually will have DFT look-up table.???
        -Also, doesn't update diffusivity, but keeps the saturation value based 
            on the assumption that the concentration of CO2 in the bulk doesn't 
            change substantially (which seems consistent with our observation 
            that a bubble  removes only a layer of < 2 um of CO2 from its 
            surroundings).
        -The code is also faster because I do not load the data from the .csv 
            files every time I want to make an interpolation, but instead load 
            them once at the beginning and just interpolate with the resulting 
            numpy arrays.
        -Additionally, uses t_i to calculate dm/dt_i in the explicit Euler method,
            instead of t_{i+1}. This requires integrating the 1/np.sqrt(t) in the 
            first time step.
            
    NOTE: the implicit solving does not work. See grow() in bubble.py instead.
    
    Parameters:
        dt : float
            time step [s]
        t_nuc : float
            time of bubble nucleation measured after entering 
            observation capillary [s]
        p_s : float
            saturation pressure of CO2 in polyol [Pa]
        R_nuc : float
            approximate radius of initial bubble nucleus, based on 
            Dr. Huikuan Chao's string method model [m]
        p_atm : float
            atmospheric pressure [Pa], assumed to be the pressure at the outlet
        L : float
            length of observation capillary [m]
        p_in : float
            pressure at inlet [m], calculated using flow_eqns
        v : float
            velocity of inner stream [m/s], calculated using flow_eqns
        polyol_data_file : string
            name of file containing polyol data [.csv]
        eos_co2_file : string
            File name for equation of state data table [.csv]
            
    Returns:
        t : list of N floats
            times at which numerical model was evaluated, measured relative 
            to time of entering observation capillary [s]
        m : list of N floats
            mass of CO2 enclosed in bubble at each time step [g]
        D : list of N floats
            diffusivity at each time step estimated from experimental G-ADSA 
            measurements (averaged exp & sqrt) [m^2/s]
        p : list of N floats
            pressure at each time step assuming linear pressure drop along 
            channel [Pa]
        p_bubble : list of N floats
            pressure inside bubble at each time step
        if_tension : list of N floats
            interfacial tension along bubble surface at each time step based on
            G-ADSA measurements [N/m]
        c_s : list of N floats
            saturation concentrations of CO2 in polyol at each time step based 
            on G-ADSA solubility and density measurements 
            [kg CO2 / m^3 polyol-CO2]
        R : list of N floats
            radius of bubble at each time step solved self-consistently with 
            modified Epstein-Plesset (1950) [m]
        rho_co2 : list of N floats
            density of CO2 in bubble at each time step based on pressure and 
            CO2 equation of state [kg/m^3]
    """ 
    # creates interpolation fn for density of CO2 based on equation of state
    f_rho_co2 = interp_rho_co2(eos_co2_file)
    # initializes lists of key bubble properties
    p = [calc_p(p_in, p_atm, v, t_nuc, L)]
    D = [calc_D(p_s, polyol_data_file)] # assumes diffusivity of saturation pressure
    c_s = [calc_c_s(p[0], polyol_data_file)]
    R = [R_nuc]
    # solves for initial mass and pressure in bubble self-consistently
    m0 = 4*np.pi/3*(R_nuc**3)*f_rho_co2(p[0])
    p_bubble0 = p[0]
    m_init, p_bubble_init = calc_m_p_bubble(R[0], p[0], m0, p_bubble0,
                                            eos_co2_file, polyol_data_file)
    print(m_init, p_bubble_init)
    p_bubble = [p_bubble_init]
    m = [m_init]
    if_tension = [calc_if_tension(p_bubble[0], polyol_data_file)]
    # initial bubble density [kg/m^3]
    rho_co2 = [f_rho_co2(p_bubble[0])]

    # initializes timeline [s]
    t = [t_nuc]
    # defines final time [s]
    t_f = L/v
    
    # prep arrays for interpolation
    c_s_interp_arrs = calc_c_s_prep(polyol_data_file)
    if_interp_arrs = calc_if_tension_prep(polyol_data_file, 
                                          if_tension_model=if_tension_model)
    # computes bulk CO2 concentration
    c_bulk = np.interp(p_s, *c_s_interp_arrs)
        
    # TODO adaptive time-stepping
    if adaptive_dt:
        dt_min = dt
        dt_max = 0.001
        R_thresh = 1E-6
    
    # applies Euler's method to estimate bubble growth over time
    while t[-1] <= t_f-dt:
        print(m[-1], R[-1], p_bubble[-1])
        
        # TODO trial smart time-stepping--flips at threshold radius
        if adaptive_dt:
            if R[-1] < R_thresh:
                dt = dt_min
            else:
                dt = min(dt*R[-1]/R[-2], dt_max)
            
            
        t += [t[-1]+dt] # increments time forward [s]
        D += [D[-1]] # keeps same diffusivity [m^2/s]
        p += [calc_p(p_in, p_atm, v, t[-1], L)] # computes new pressure along observation capillary [Pa]
        c_s += [np.interp(p[-1], *c_s_interp_arrs)] # interpolates saturation concentration of CO2 [kg CO2 / m^3 polyol-CO2]
        # guess for self-consistently solving for radius and pressure of bubble
        R0 = (3/(4*np.pi)*m[-1]/rho_co2[-1])**(1./3)
        p_bubble0 = p_bubble[-1] #p[-1] + 2*if_tension[-1]/R0
        if implicit:
            # self-consistently solves implicit Euler equation
            m0 = m[-1]
            soln = calc_m_R_p_bubble(m0, R0, p_bubble0, c_bulk, c_s[-1], D[-1], 
                                     m[-1], p[-1], t[-1], dt, if_interp_arrs, 
                                     eos_co2_file)
            m_curr, R_curr, p_bubble_curr = soln
            m += [m_curr]
        else:
            # updates mass with explicit Euler method--inputs are i^th terms,
            # so we pass in R[-1] since R has not been updated to R_{i+1} yet
            # TODO integrate for first time step 1/np.sqrt(t) term
            m += [m[-1] + dt*calc_dmdt(D[-2], p_s, p[-2], R[-1], t[-2]-t_nuc, 
                  polyol_data_file, c_s_interp_arrs=c_s_interp_arrs, dt=dt)]
            # self-consistently solves for radius and pressure of bubble
            R_curr, p_bubble_curr = calc_R_p_bubble(m[-1], p[-1], R0, p_bubble0, 
                                                     eos_co2_file, polyol_data_file,
                                                     interp_arrs=if_interp_arrs)
        p_bubble += [p_bubble_curr]
        R += [R_curr]
        if_tension += [np.interp(p_bubble[-1], *if_interp_arrs)] # interpolates interfacial tension along bubble [N/m]
        rho_co2 += [f_rho_co2(p_bubble[-1])] # saves CO2 density in bubble [kg/m^3]
        
    return t, m, D, p, p_bubble, if_tension, c_s, R, rho_co2


def calc_c_s(p, polyol_data_file):
    """
    Estimates the saturation concentration of CO2 in a polyol solution using 
    interpolated measurements of solubility and specific volume.
    
    If p is above the experimentally measured range, returns the maximum 
    measured saturation concentration to avoid errors (this is preferable since
    we are just trying to make some rough estimates as a demonstration of this 
    method right now. More precise measurements in the future will require 
    a different approach).
    
    Parameters:
        p : float
            pressure at which to estimate the saturation concentration [Pa]
        polyol_data_file : string
            name of file containing polyol data [.csv]
            
    Returns:
        c_s : float
            concentration of CO2 in polyol-CO2 solution [kg/m^3] at the given pressure p
    """
    # loads thermophysical property data from file
    df = pd.read_csv(polyol_data_file)
    p_arr = kPa_2_Pa*df['p actual [kPa]'].to_numpy(dtype=float) # measured pressures from experiment [Pa]   
    solub_arr = df['solubility [w/w]'].to_numpy(dtype=float) # measured solubility [w/w]
    spec_vol_arr = df['specific volume (fit) [mL/g]'].to_numpy(dtype=float) # fitted specific volume [mL/g]
    density_arr = gmL_2_kgm3/spec_vol_arr # density of polyol-CO2 [kg/m^3]
    # computes saturation concentration of CO2 in polyol [kg CO2 / m^3 solution]
    c_s_arr = solub_arr*density_arr
    
    # removes data points with missing measurements
    not_nan = [i for i in range(len(c_s_arr)) if not np.isnan(c_s_arr[i])]
    p_arr = p_arr[not_nan]
    c_s_arr = c_s_arr[not_nan]
    # concatenate 0 to pressure and saturation concentration for low values of p
    p_arr = np.concatenate((np.array([0]), p_arr))
    c_s_arr = np.concatenate((np.array([0]), c_s_arr))
    
    # orders saturation concentration in order of increasing pressure
    inds = np.argsort(p_arr)
    # limits the pressure to the maximum measured value [Pa]
    p = min(p, np.max(p_arr))
    # interpolates value to match the given pressure [kg CO2 / m^3 solution]
    c_s = np.interp(p, p_arr[inds], c_s_arr[inds])
    
    return c_s


def calc_c_s_prep(polyol_data_file):
    """
    Performs calculations that only need to be done once before the 
    interpolation.
    """
    # loads thermophysical property data from file
    df = pd.read_csv(polyol_data_file)
    p_arr = kPa_2_Pa*df['p actual [kPa]'].to_numpy(dtype=float) # measured pressures from experiment [Pa]   
    solub_arr = df['solubility [w/w]'].to_numpy(dtype=float) # measured solubility [w/w]
    spec_vol_arr = df['specific volume (fit) [mL/g]'].to_numpy(dtype=float) # fitted specific volume [mL/g]
    density_arr = gmL_2_kgm3/spec_vol_arr # density of polyol-CO2 [kg/m^3]
    # computes saturation concentration of CO2 in polyol [kg CO2 / m^3 solution]
    c_s_arr = solub_arr*density_arr
    
    # removes data points with missing measurements
    not_nan = [i for i in range(len(c_s_arr)) if not np.isnan(c_s_arr[i])]
    p_arr = p_arr[not_nan]
    c_s_arr = c_s_arr[not_nan]
    # concatenate 0 to pressure and saturation concentration to cover low values of p
    p_arr = np.concatenate((np.array([0]), p_arr))
    c_s_arr = np.concatenate((np.array([0]), c_s_arr))
    # orders saturation concentration in order of increasing pressure
    inds = np.argsort(p_arr)
    
    return p_arr[inds], c_s_arr[inds]


def calc_D(p, polyol_data_file):
    """
    Estimates the diffusivity of CO2 in polyol at the given pressure
    by interpolating available measurements using G-ADSA. The two methods
    used to estimate the diffusivity, square-root fit of the initial transient
    and exponential fit of the final plateau are averaged to reduce the effects
    of noise/experimental error.
    
    If p is above the experimentally measured range, it is replaced with the 
    maximum pressure for which there is a measurement. If below, it is replaced with 
    the minimum. This is preferable since we are just trying to make some rough 
    estimates as a demonstration of this method right now. More precise measurements 
    in the future will require a more sophisticated approach).
    
    Parameters:
        p : float
            pressure at which to estimate the saturation concentration [Pa]
        polyol_data_file : string
            name of file containing polyol data [.csv]
            
    Returns:
        D : float
            diffusivity of CO2 in polyol [m^2/s] at the given pressure p
    """
    # loads thermophysical property data from file
    df = pd.read_csv(polyol_data_file)
    p_arr = kPa_2_Pa*df['p actual [kPa]'].to_numpy(dtype=float) # measured pressures from experiment [Pa]    
    D_sqrt_arr = cm2s_2_m2s*df['diffusivity (sqrt) [cm^2/s]'].to_numpy(dtype=float) # diff. measured by sqrt transient [m^2/s]
    D_exp_arr = cm2s_2_m2s*df['diffusivity (exp) [cm^2/s]'].to_numpy(dtype=float) # diff. measured by exp plateau [m^2/s]
    
    # averages sqrt and exponential estimates of diffusivity [m^2/s]
    D_arr = (D_sqrt_arr + D_exp_arr)/2
    # removes data points with missing measurements
    not_nan = [i for i in range(len(D_arr)) if not np.isnan(D_arr[i])]
    p_arr = p_arr[not_nan]
    D_arr = D_arr[not_nan]
    # orders saturation concentration in order of increasing pressure
    inds = np.argsort(p_arr)
    # limits the pressure to be within the minimum and maximum measured values [Pa]
    p = min(p, np.max(p_arr))
    p = max(p, np.min(p_arr))
    # interpolates diffusivity [m^2/s] to match the given pressure
    D = np.interp(p, p_arr[inds], D_arr[inds])
    
    return D

    
def calc_if_tension(p, polyol_data_file, R=-1, delta=5E-9):
    """
    Estimates the interfacial tension between the CO2-rich and polyol-rich 
    phases under equilibrium coexistence between CO2 and polyol at the given 
    pressure by interpolating available measurements using G-ADSA.
    
    Providing a value for the radius invokes the use of the Tolman length delta
    to correct for the effects of curvature on the interfacial tension.
    
    If p is above the experimentally measured range, it is replaced with the 
    maximum pressure for which there is a measurement. If below, it is replaced with 
    the minimum. This is preferable since we are just trying to make some rough 
    estimates as a demonstration of this method right now. More precise measurements 
    in the future will require a more sophisticated approach).
    
    Parameters:
        p : float
            pressure at which to estimate the saturation concentration [Pa]
        polyol_data_file : string
            name of file containing polyol data [.csv]
            
    Returns:
        if_tension : float
            interfacial tension between CO2-rich and polyol-rich phases [N/m] at the given pressure p
    """
    p_arr, if_tension_arr = calc_if_tension_prep(polyol_data_file)
    # interpolates interfacial tension [N/m] to match the given pressure
    if_tension = np.interp(p, p_arr, if_tension_arr)
    
    return if_tension


def calc_if_tension_prep(polyol_data_file, p_min=0, p_max=4E7, if_tension_model='lin'):
    """
    Performs calculations that only need to be done once before interpolation
    """
    # loads thermophysical property data from file
    df = pd.read_csv(polyol_data_file)
    p_arr = 1000*df['p actual [kPa]'].to_numpy(dtype=float) # measured pressures from experiment [Pa]    
    if_tension_arr = 1E-3*df['if tension [mN/m]'].to_numpy(dtype=float) # measured interfacial tension [N/m]
    
    # removes data points with missing measurements
    not_nan = [i for i in range(len(if_tension_arr)) if not np.isnan(if_tension_arr[i])]
    p_arr = p_arr[not_nan]
    if_tension_arr = if_tension_arr[not_nan]
    # orders saturation concentration in order of increasing pressure
    inds = np.argsort(p_arr)
    p_mid = p_arr[inds]
    if_tension_mid = if_tension_arr[inds]
    # extrapolates pressure beyond range of data
    a, b = np.polyfit(p_arr, if_tension_arr, 1)
    p_small = np.linspace(p_min, p_mid[0], 10)
    if_tension_small = a*p_small + b
    p_big = np.linspace(p_mid[-1], p_max, 100)
    if if_tension_model == 'lin':
        if_tension_big = a*p_big + b
        # change negative values to 0
        if_tension_big *= np.heaviside(if_tension_big, 1)
    elif if_tension_model == 'ceil':
        if_tension_big = np.min(if_tension_mid)*np.ones([len(p_big)])
    else:
        print('calc_if_tension_prep does not recognize the given if_tension_model.')
    
    return np.concatenate((p_small, p_mid, p_big)), \
            np.concatenate((if_tension_small, if_tension_mid, if_tension_big))   


def interp_rho_co2(eos_co2_file):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    webbook.nist.gov at desired temperature.
    The density is returned in term of kg/m^3 as a function of pressure in Pa.
    PARAMETERS:
        eos_co2_file : string
            File name for equation of state data table [.csv]
    RETURNS:
        rho : interpolation function
            density in kg/m^3 of co2 @ given temperature
    """
    # dataframe of appropriate equation of state (eos) data from NIST
    df_eos = pd.read_csv(eos_co2_file, header=0)
    # get list of pressures of all data points [Pa]
    p_co2 = 1000*df_eos['Pressure (kPa)'].to_numpy(dtype=float)
    # get corresponding densities of CO2 [kg/m^3]
    rho_co2 = 1000*df_eos['Density (g/ml)'].to_numpy(dtype=float)
    # remove repeated entries
    p_co2, inds_uniq = np.unique(p_co2, return_index=True)
    rho_co2 = rho_co2[inds_uniq]
    # create interpolation function [kg/m^3]
    rho_min = np.min(rho_co2)
    rho_max = np.max(rho_co2)
    f_rho = scipy.interpolate.interp1d(p_co2, rho_co2, bounds_error=False,
                                       fill_value=(rho_min, rho_max))

    return f_rho


def calc_dmdt(D, p_s, p, R, t, polyol_data_file, add_p_young_laplace=False,
              c_s_interp_arrs=None, tol=1E-10, dt=-1):
    """
    Calculates the time-derivative of the mass enclosed inside a
    CO2 bubble under the given conditions. The formula is modified
    from the widely used Epstein-Plesset formula (1950) to include
    the effect of changing pressure.
    
    Parameters:
        D : float
            diffusivity of CO2 in polyol [m^2/s]
        p_s : float
            saturation pressure of CO2 in polyol [Pa]
        p : float
            pressure at current point along the observation capillary [Pa]
        R : float
            current radius of the bubble [m]
        t : float
            current time [s]
        polyol_data_file : string
            name of file containing polyol data [.csv]
        add_p_young_laplace : bool
            *legacy* used by some codes to test effect of adding Young-Laplace
            pressure when determining the saturation concentration c_s--failed
        c_s_interp_arrs : 2-tuple of numpy arrays of floats, (p_arr, c_s_arr)
            p_arr contains pressure [Pa] and c_s_arr contains saturation
            concentrations of CO2 in the bulk phase at those pressures [kg/m^3]
        tol : float
            tolerance below which 1/np.sqrt(t) term might cause numerical 
            difficulties, so it is integrated before adding to dm/dt.
        dt : float
            time step [s]. Only used if positive.
            
    Returns:
        dmdt : float
            Time-derivative of the mass enclosed in the bubble [kg/s]
    """
    # adds Young-Laplace pressure due to interfacial tension
    if add_p_young_laplace:
        p += 2*calc_if_tension(p, polyol_data_file)/R
    # computes saturation concentrations [kg CO2 / m^3 solution]
    if c_s_interp_arrs is None:
        c_bulk = calc_c_s(p_s, polyol_data_file) # at saturation pressure of CO2--this is concentration in bulk of parent phase
        c_s = calc_c_s(p, polyol_data_file) # at current pressure
    else:
        p_arr, c_s_arr = c_s_interp_arrs
        c_bulk = np.interp(p_s, p_arr, c_s_arr)
        c_s = np.interp(p, p_arr, c_s_arr)
    
    # computes time-derivative of the mass enclosed in the bubble [kg/s] based on modified Epstein-Plesset (1950)
    dmdt1 = (4*np.pi*R**2*D) * (c_bulk - c_s) * (1/R)
    if t < tol and dt > 0:
        # integrates term and divides by dt (assumes Explicit Euler)
        dmdt2 = (4*np.pi*R**2*D) * (c_bulk - c_s) * \
                    (1 / np.sqrt(np.pi*D)) * (2/np.sqrt(dt))
    else:
        dmdt2 = (4*np.pi*R**2*D) * (c_bulk - c_s) * (1 / np.sqrt(np.pi*D*t)) 
    dmdt = dmdt1 + dmdt2
    
    return dmdt


def calc_R(m, p, R0, if_tension, eos_co2_file):
    """
    Calculates the radius of the bubble self-consistently under 
    the given conditions. The formula is modified from the widely 
    used Epstein-Plesset formula (1950) to include the effects of 
    changing pressure and interfacial tension, which are significant
    in our system due to the large pressure drops and nanoscopic 
    bubble nuclei.
    
    Parameters:
        m : float
            mass of CO2 enclosed in bubble [kg]
        p : float
            pressure at current point along observation capillary [Pa]
        R0 : float
            initial guess for the radius of the bubble (recommended to be previous radius) [m]
        if_tension : float
            interfacial tension [N/m] between CO2-rich and polyol-rich phases under equilibrium at given pressure
        eos_co2_file : string
            File name for equation of state data table [.csv]
            
    Returns:
        R : float
            radius of the bubble [m]
    """
    # creates interpolation function for the density of pure CO2 [kg/m^3] as a function of pressure [Pa]
    rho_co2 = interp_rho_co2(eos_co2_file)
    
    def f(R, args):
        """Function to self-consistently solve for R."""
        m, p, if_tension = args
        # checks for negative values and prints out warning message if so
        if p + 2*if_tension/R < 0:
            print('pressure when negative in f() of interp_rho_co2().')
            
        return R - ( 3/(4*np.pi)*(m/rho_co2(p + 2*if_tension/R)) )**(1/3.)
    
    # solves for R with nonlinear solver
    args = (m, p, if_tension) # fixed arguments for solving R
    soln = scipy.optimize.root(f, R0, args=(args,)) # solves for R
    R = soln.x[0] # extracts the radius R [m]
    
    return R


def calc_m_p_bubble(R, p, m0, p_bubble0, eos_co2_file, polyol_data_file):
    """
    Calculates the radius of the bubble and its internal pressure
    self-consistently given the mass of CO2 inside the bubble and the
    pressure in the bulk. Both are dependent on each other since a smaller
    interfacial tension decreases the pressure inside the bubble due to the
    Laplace pressure, thereby decreasing the density of CO2 (which increases
    the Laplace pressure) while also increasing the radius of the bubble
    (which decreases the Laplace pressure).
    
    The formula is modified from the widely 
    used Epstein-Plesset formula (1950) to include the effects of 
    changing pressure and interfacial tension, which are significant
    in our system due to the large pressure drops and nanoscopic 
    bubble nuclei.
    
    Parameters:
        R : float
            radius of bubble [m]
        p : float
            pressure in the bulk at current point along observation capillary [Pa]
        m0 : float
            initial guess for the mass of CO2 enclosed in the bubble
            (recommended to be previous mass) [kg]
        p0 : float
            initial guess for the pressure inside the bubble [Pa]
        eos_co2_file : string
            File name for equation of state data table [.csv]
            
    Returns:
        m : float
            mass of CO2 enclosed in the bubble [kg]
        p_bubble : float
            pressure inside the bubble [Pa]
    """
    def f(variables, args):
        """Function to self-consistently solve for R and pressure."""
        m, p_bubble = variables
        R, p, rho_co2, polyol_data_file = args
        if_tension = calc_if_tension(p_bubble, polyol_data_file)
        
        return scf_bubble_fn(if_tension, m, p, p_bubble, R, rho_co2)
    
    # creates interpolation function for the density of pure CO2 [kg/m^3] as a 
    # function of pressure [Pa]
    rho_co2 = interp_rho_co2(eos_co2_file)
    # solves for R with nonlinear solver
    args = (R, p, rho_co2, polyol_data_file) # fixed arguments for solving R
    soln = scipy.optimize.root(f, (m0, p_bubble0), args=(args,)) # solves for R
    m, p_bubble = soln.x # extracts the radius [m] and interfacial tension [N/m]
    
    return m, p_bubble
    

def calc_R_p_bubble(m, p, R0, p_bubble0, eos_co2_file, polyol_data_file,
                    interp_arrs=None):
    """
    Calculates the radius of the bubble and its internal pressure
    self-consistently given the mass of CO2 inside the bubble and the
    pressure in the bulk. Both are dependent on each other since a smaller
    interfacial tension decreases the pressure inside the bubble due to the
    Laplace pressure, thereby decreasing the density of CO2 (which increases
    the Laplace pressure) while also increasing the radius of the bubble
    (which decreases the Laplace pressure).
    
    The formula is modified from the widely 
    used Epstein-Plesset formula (1950) to include the effects of 
    changing pressure and interfacial tension, which are significant
    in our system due to the large pressure drops and nanoscopic 
    bubble nuclei.
    
    Parameters
    ----------
    m : float
        mass of CO2 enclosed in bubble [kg]
    p : float
        pressure in the bulk at current point along observation capillary [Pa]
    R0 : float
        initial guess for the radius of the bubble (recommended to be previous radius) [m]
    p0 : float
        initial guess for the pressure inside the bubble [Pa]
    eos_co2_file : string
        File name for equation of state data table [.csv]
            
    Returns
    -------
    R : float
        radius of the bubble [m]
    p_bubble : float
        pressure inside the bubble [Pa]
    """
    def f(variables, args):
        """Function to self-consistently solve for R and pressure."""
        R, p_bubble = variables
        m, p, rho_co2, polyol_data_file, interp_arrs = args
        if interp_arrs is None:
            # interpolates interfacial tension from model
            if_tension = calc_if_tension(p_bubble, polyol_data_file)
        else:
            p_arr, if_tension_arr = interp_arrs
            if_tension = np.interp(p_bubble, p_arr, if_tension_arr)
        
        return scf_bubble_fn(if_tension, m, p, p_bubble, R, rho_co2)
    
    # creates interpolation function for the density of pure CO2 [kg/m^3] as a 
    # function of pressure [Pa]
    rho_co2 = interp_rho_co2(eos_co2_file)  
    # solves for R with nonlinear solver
    args = (m, p, rho_co2, polyol_data_file, interp_arrs) # fixed arguments for solving R
    soln = scipy.optimize.root(f, (R0, p_bubble0), args=(args,)) # solves for R
    R, p_bubble = soln.x

    return R, p_bubble


def calc_m_R_p_bubble(m0, R0, p_bubble0, c_bulk, c_s, D, m_prev, p, t, dt, 
                      if_interp_arrs, eos_co2_file):
    """
    Used for implicit Euler time-stepping.
    Calculates the radius of the bubble, its internal pressure, and its mass
    self-consistently given the mass of CO2 inside the bubble at the previous
    time step and the pressure in the bulk. 
    
    Implicit Euler Method:
        m_{i+1} = m_i + dm/dt|_{t_{i+1}}*dt_i
    
    Parameters
    ----------
    m0 : float
        guess for mass of CO2 enclosed in bubble in next time step m_{i+1} [kg]
    R0 : float
        initial guess for the radius of the bubble for the next time step 
        R_{i+1} [m] (recommended to be previous radius)
    p0 : float
        initial guess for the pressure inside the bubble at the next time step
        p_bubble_{i+1} [Pa]
    c_bulk : float
        concentration of CO2 in the bulk parent phase [kg/m^3]
    c_s : float
        saturation concentration of CO2 at the pressure for the next time step
        p_{i+1} (given as p in args) [kg/m^3]
    D : float
        diffusivity of CO2 in the bulk parent phase [m^2/s]
    m_prev : float
        mass of CO2 enclosed in the bubble at the previous time step m_i [kg]
    p : float
        pressure in the bulk parent phase at the next time step p_{i+1} [Pa]
    t : float
        time at next time step t_{i+1} [s]
    dt : float
        time step [s]
    if_interp_arrs : 2-tuple of numpy arrays of floats, (p_arr, if_tension_arr)
        p_arr contains pressures [Pa] and if_tension_arr contains the
        corresponding interfacial tensions [N/m]
    eos_co2_file : string
        File name for equation of state data table [.csv]
            
    Returns
    -------
    m : float
        Mass for next time step, m_{i+1}
    R : float
        radius of the bubble for the next time step R_{i+1} [m]
    p_bubble : float
        pressure inside the bubble for the next time step p_bubble_{i+1} [Pa]
    """
    def f(variables, args):
        """Function to self-consistently solve for R and pressure."""
        m, R, p_bubble = variables
        c_bulk, c_s, D, m_prev, p, t, dt, rho_co2, if_interp_arrs = args
        p_arr, if_tension_arr = if_interp_arrs
        if_tension = np.interp(p_bubble, p_arr, if_tension_arr)
        
        return scf_bubble_impl(m, R, p_bubble, c_bulk, c_s, D, if_tension, 
                               m_prev, p, t, dt, rho_co2)
    
    # creates interpolation function for the density of pure CO2 [kg/m^3] as a 
    # function of pressure [Pa]
    rho_co2 = interp_rho_co2(eos_co2_file)  
    # solves for R with nonlinear solver
    args = (c_bulk, c_s, D, m_prev, p, t, dt, rho_co2, if_interp_arrs) # fixed arguments for solving R
    soln = scipy.optimize.root(f, (m0, R0, p_bubble0), args=(args,)) # solves for R
    m, R, p_bubble = soln.x

    return m, R, p_bubble


def calc_p(p_in, p_atm, v, t, L):
    """
    Calculates the pressure down the observation capillary
    assuming a linear pressure drop based on the estimated
    inlet (p_in) and outlet (p_atm) pressures.
    
    Parameters:
        p_in : float
            inlet pressure, estimated using the "flow_eqns" [Pa]
        p_atm : float
            outlet pressure, estimated to be atmospheric pressure [Pa]
        v : float
            velocity of inner stream, estimated using "flow_eqns" [m/s]
        t : float
            time since entering the observation capillary [s]
        L : float
            length of observation capillary [m]
            
    Returns:
        p : float
            pressure at current point along observation capillary [Pa]    
    """
    return p_in + v*t/L*(p_atm-p_in)

def flow_eqns(eta_i, eta_o, L, p_atm, p_in, Q_i, Q_o, R_i, R_o, v):
    """
    Defines equations derived from Navier-Stokes equations in 
    Stokes flow for sheath flow down a cylindrical pipe. The
    derivation is given in YlitaloA_candidacy_report.pdf.
    
    Parameters
    ----------
    p_in : float
        inlet pressure [Pa]
    v : float
        velocity of center stream [m/s]
    R_i : float
        radius of inner stream [m]
    p_atm : float
        outlet pressure, assumed to be atmospheric [Pa]
    L : float
        length of observation capillary [m]
    eta_i : float
        viscosity of inner stream of polyol-CO2 [Pa.s]
    eta_o : float
        viscosity of outer stream of pure polyol [Pa.s]            
    Q_i : float
        inner stream flow rate, supplied by ISCO 100 DM [m^3/s]
    Q_o : float
        outer stream flow rate, supplied by ISCO 260 D [m^3/s]
    R_o : float
        radius of outer stream (half of inner diameter of observation capillary) [m]
    
    Returns
    -------
    res : 3-tuple of floats
        Residuals of the three flow equations (Poiseuille flow from Stokes
        equation + 2 BCs: Q_i through the inner stream, and Q_o through the 
        outer stream BC)
    """
    # boundary condition that the outer stream has flow rate Q_o
    res1 = Q_o - np.pi*(p_in-p_atm)*(R_o**2 - R_i**2)**2/(8*eta_o*L)
    # boundary condition that the inner stream has flow rate Q_i
    res2 = (p_in - p_atm)/L - (8*eta_i*Q_i)/ \
                ( np.pi*R_i**2*(2*(R_o**2 - R_i**2)*eta_i/eta_o + R_i**2) )
    # residual from Stokes flow (v = w_i(r = 0) for w_i z-velocity from Stokes)
    res3 = v - (p_in - p_atm)/L*( (R_o**2 - R_i**2)/(4*eta_o) + \
                R_i**2/(4*eta_i) )
    res = (res1, res2, res3)
    
    return res


def flow_eqns_input(variables, args):
    """
    Formats the input to flow_eqns for use in scipy.optimize.root, which
    requires functions to have the format foo(vars, args).
    Formatting is performed by merging variables and arguments according to 
    the ordering given as the last argument so that the arguments passed to 
    flow_eqns are ordered properly (alphabetically).
    
    Parameters
    ----------
    variables : 3-tuple of floats
        Quantities to solve for with scipy.optimize.root
    args : 8-tuple of 7 floats followed by a list of 10 ints
        The first 7 floats are the remaining quantities required for flow_eqns,
        which are provided by the user and held constant. 
        The list of 10 ints at the end gives the ordering of each variable in
        alphabetical order so that when variables and args are merged, they are
        in the proper order required for flow_eqns.
        
        Example:
            variables = (p_in, v, R_i)
            args = (p_atm, L, eta_i, eta_o, Q_i, Q_o, R_cap, ordering)
            ordering = [5, 6, 4, 3, 0, 7, 8, 2, 9, 1]
            
    Returns
    -------
    res : 3-tuple of floats
        Residuals of the Stokes flow equation and boundary conditions
        calculated in flow_eqns.
    """
    # extracts the ordering of the variables/args
    ordering = np.array(args[-1])
    # merges variables and args (except for the ordering)
    unordered_args = list(variables) + list(args[:-1])
    # orders the merged variables and args in alphabetical order, which
    # requires numpy arrays
    ordered_args = list(np.array(unordered_args)[ordering])
    
    return flow_eqns(*ordered_args)
    

def fit_growth_to_pt(t_bubble, R_bubble, t_nuc_lo, t_nuc_hi, dt, p_s, R_nuc, p_atm, L, p_in, v,
                     polyol_data_file, eos_co2_file, tol_R=0.01, ax=None,
                     eps_pless_fn=eps_pless_p_if_3):
    """
    Fits the bubble growth to a given bubble radius at a given time. Plots
    the different trajectories if an axis handle is given.
    """ 
    # initializes plot to show the trajectories of different guesses
    if ax is not None:
        ax.plot(t_bubble*s_2_ms, R_bubble*m_2_um, 'g*', ms=12, label='fit pt')
    
    err_R = 1 # junk error to get the loop started
    while err_R > tol_R:
        # calculates new nucleation time as middle of the two bounds (bisection algorithm)
        t_nuc = (t_nuc_lo + t_nuc_hi)/2
        # computes bubble growth trajectory with new bubble nucleation time
        t, m, D, p, p_bubble, if_tension, c_s, R, rho_co2 = \
                eps_pless_fn(dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v,
                                 polyol_data_file, eos_co2_file)
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

        print('t_nuc = {0:.4f} s and error in R is {1:.2f}.'.format(t_nuc, err_R))

        # plots the guessed growth trajectory
        if ax is not None:
            ax.plot(np.array(t)*s_2_ms, np.array(R)*m_2_um, label=r'$t_{nuc}=$' + '{0:.4f} s'.format(t_nuc))

    if ax is not None:
        # formats plot of guessed trajectories
        ax.set_yscale('log')
        ax.set_xlabel(r'$t$ [s]', fontsize=16)
        ax.set_ylabel(r'$R(t)$ [$\mu$m]', fontsize=16)
        ax.set_title('Growth Trajectory for Different Nucleation Times', fontsize=20)
        plt.legend()
        
    return t_nuc


def scf_bubble_fn(if_tension, m, p, p_bubble, R, rho_co2):
    """
    Computes residual of the equations governing bubble size:
        1) mass = volume*density (conservation of mass + volume of a sphere)
        2) p_bubble = p + 2*if_tension/R (Young-Laplace)
    Used in calc_R_p_bubble and calc_m_p_bubble for self-consistent 
    computation using scipy.optimize.root.
    
    Parameters
    ----------
    if_tension : float
        Interfacial tension along surface of bubble [N/m]
    m : float
        mass of CO2 enclosed in bubble [kg]
    p : float
        pressure in bulk governed by microfluidic flow [Pa]
    p_bubble : float
        pressure inside bubble, p + Young-Laplace pressure [Pa]
    R : float
        radius of bubble [m]
    rho_co2 : interpolation function
        Function to interpolate density of CO2 given pressure according to 
        equation of state obtained from NIST database. See interp_rho_co2()
        
    Returns
    -------
    res : 2-tuple of floats
        residuals from the two constraints enforced (conservation of mass
        for a sphere of gas and Young-Laplace pressure)
    """
    # residual for calculation of R based on spherical geometry
    res1  = R - ( 3/(4*np.pi)*(m/rho_co2(p_bubble)) )**(1/3.)
    # residual for pressure based on Laplace pressure + bulk pressure
    res2 = p_bubble - (p + 2*if_tension/R)
    res = (res1, res2)
    
    return res


def scf_bubble_impl(m, R, p_bubble, c_bulk, c_s, D, if_tension, m_prev,
                    p, t, dt, rho_co2):
    """
    Computes residual of the equations governing bubble size:
        1) mass = volume*density (conservation of mass + volume of a sphere)
        2) p_bubble = p + 2*if_tension/R (Young-Laplace)
        3) implicit Euler using Epstein-Plesset: 
            m_{i+1} = m_i + dm/dt|_{t_{i+1}}*dt_i
    Used in calc_m_R_p_bubble for self-consistent 
    computation using scipy.optimize.root.
    
    Note that this solves for R_{i+1}, m_{i+1}, and p_bubble,i+1 because it is
    used for an implicit method.
    
    Parameters
    ----------
    m : float
        mass of CO2 enclosed in bubble [kg]
    R : float
        radius of bubble [m]
    p_bubble : float
        pressure inside bubble, p + Young-Laplace pressure [Pa]
    c_bulk : float
        concentration of CO2 in the bulk [kg/m^3]
    c_s : float
        saturation concentration of CO2 at the pressure p [kg/m^3]
    D : float
        diffusivity of CO2 in polyol, usually estimated to be diffusivity in
        bulk [m^2/s]
    if_tension : float
        Interfacial tension along surface of bubble [N/m]
    m_prev : float
        mass in the previous time step
    p : float
        pressure in bulk governed by microfluidic flow [Pa]       
    t : float
        time (t_{i+1} in the implicit Euler method) [s]
    dt : float
        time step [s]
    rho_co2 : interpolation function
        Function to interpolate density of CO2 given pressure according to 
        equation of state obtained from NIST database. See interp_rho_co2()
        
    Returns
    -------
    res : 3-tuple of floats
        residuals from the three constraints enforced (conservation of mass
        for a sphere of gas, Young-Laplace pressure, and implicit Euler
        with Epstein-Plesset)
    """
    # residual for calculation of R based on spherical geometry
    res1  = R - ( 3/(4*np.pi)*(m/rho_co2(p_bubble)) )**(1/3.)
    # residual for pressure based on Laplace pressure + bulk pressure
    res2 = p_bubble - (p + 2*if_tension/R)
    # residual for implicit Euler computation with Epstein-Plesset
    dmdt = 4*np.pi*R**2*D*(c_bulk - c_s)*(1/R + 1/np.sqrt(np.pi*D*t))
    res3 = m - (m_prev + dmdt*dt)
    res = (res1, res2, res3)
    
    return res