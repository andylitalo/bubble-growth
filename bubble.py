"""
Code for modeling bubble growth.

@author: Andy Ylitalo
"""

# adds path to general libraries
import sys
sys.path.append('../libs/')

import numpy as np
import scipy.optimize
import scipy.interpolate
import pandas as pd
import matplotlib.pyplot as plt

# custuom libraries
import geo

# CONVERSIONS
s_2_ms = 1000
m_2_um = 1E6
kPa_2_Pa = 1000
gmL_2_kgm3 = 1000
cm2s_2_m2s = 1E-4




def grow(dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v,
                     polyol_data_file, eos_co2_file, adaptive_dt=True,
                     if_tension_model='lin', implicit=False, d_tolman=0,
                     tol_R=0.001, alpha=0.3, D=-1, drop_t_term=False):
    """
    Solves for bubble growth based on Epstein and Plesset (1950) with
    modifications for changing pressure (p) and interfacial tension (if_tension).

    Difference from eps_pless_p_if_4:
        -Uses Tolman length correction of interfacial tension

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
        d_tolman : float
            Tolman length for correction of interfacial tension due to
            curvature [m].

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
    # prep arrays for interpolation
    c_s_interp_arrs = calc_c_s_prep(polyol_data_file)
    if_interp_arrs = calc_if_tension_prep(polyol_data_file,
                                          if_tension_model=if_tension_model)
    c_bulk = np.interp(p_s, *c_s_interp_arrs) # computes bulk CO2 concentration
    if D == -1:
        D = calc_D(p_s, polyol_data_file) # assumes diffusivity of saturation pressure
    # creates interpolation fn for density of CO2 based on equation of state
    f_rho_co2 = interp_rho_co2(eos_co2_file)

    # initializes lists of key bubble properties
    p = [calc_p(p_in, p_atm, v, t_nuc, L)]
    c_s = [np.interp(p[0], *c_s_interp_arrs)]
    R = [R_nuc]
    # solves for initial mass and pressure in bubble self-consistently
    m0 = geo.v_sphere(R_nuc)*f_rho_co2(p[0])
    p_bubble0 = p[0]
    m_init, p_bubble_init = calc_m_p_bubble(R[0], p[0], m0, p_bubble0,
                                            if_interp_arrs, f_rho_co2, d_tolman)
    p_bubble = [p_bubble_init]
    m = [m_init]
    if_tension = [calc_if_tension(p_bubble[0], if_interp_arrs, R[0],
                                  d_tolman=d_tolman)]
    # initial bubble density [kg/m^3]
    rho_co2 = [f_rho_co2(p_bubble[0])]

    # calcultes the time at which pressure reaches saturation pressure
    t_s = calc_t_s(p_in, p_atm, p_s, v, L)
    # ensures that nucleation time occurs after supersaturation achieved
    t_nuc = max(t_s, t_nuc)
    # initializes timeline [s]
    t = [t_nuc]
    # defines final time [s]
    t_f = L/v

    # collects fixed parameters
    fixed_params = (t_nuc, D, p_in, p_s, p_atm, v, L, c_bulk, c_s_interp_arrs,
                    if_interp_arrs, f_rho_co2, d_tolman, implicit)
    # applies Euler's method to estimate bubble growth over time
    # the second condition provides cutoff for shrinking the bubble
    while t[-1] <= t_f:
        if adaptive_dt:
            # calculates properties for two time steps
            props_a = time_step(dt, t[-1], m[-1], p[-1], if_tension[-1],
                              R[-1], rho_co2[-1], fixed_params,
                              drop_t_term=drop_t_term)
            props_b = time_step(2*dt, t[-1], m[-1], p[-1], if_tension[-1],
                              R[-1], rho_co2[-1], fixed_params,
                              drop_t_term=drop_t_term)
            # extracts estimated value of the radius
            R_a = props_a[-2]
            R_b = props_b[-2]
            # checks if the discrepancy in the radius is below tolerance
            if np.abs( (R_a - R_b) / R_a) <= tol_R:
                # increases time step for next calculation
                dt *= (1 + alpha)
                # takes properties calculated with smaller time step
                props = props_a
                # saves new properties and advances forward
                update_props(props, t, m, p, p_bubble, if_tension, c_s, R, rho_co2)
            else:
                # does not advance forward, reduces time step
                dt /= 2
        else:
            # calculates properties after one time step
            props = time_step(dt, t[-1], m[-1], p[-1], if_tension[-1], R[-1],
                            rho_co2[-1], fixed_params, drop_t_term=drop_t_term)
            # saves properties
            update_props(props, t, m, p, p_bubble, if_tension, c_s, R, rho_co2)

    return t, m, D, p, p_bubble, if_tension, c_s, R, rho_co2


def calc_c_s(p, polyol_data_file):
    """
    Estimates the saturation concentration of CO2 in a polyol solution using
    interpolated measurements of solubility.

    Parameters:
        p : float
            pressure at which to estimate the saturation concentration [Pa]
        polyol_data_file : string
            name of file containing polyol data [.csv]

    Returns:
        c_s : float
            concentration of CO2 in polyol-CO2 solution [kg/m^3] at the given pressure p
    """
    p_arr, c_s_arr = calc_c_s_prep(polyol_data_file)
    # interpolates value to match the given pressure [kg CO2 / m^3 solution]
    c_s = np.interp(p, p_arr, c_s_arr)

    return c_s


def calc_c_s_prep(polyol_data_file):
    """
    Estimates arrays of values of solubility for different pressures using
    measurements of solubility and specific volume from G-ADSA.

    If p is above the experimentally measured range, returns the maximum
    measured saturation concentration to avoid errors (this is preferable since
    we are just trying to make some rough estimates as a demonstration of this
    method right now. More precise measurements in the future will require
    a different approach).
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


def calc_if_tension(p, if_interp_arrs, R, d_tolman=0):
    """
    Estimates the interfacial tension given arrays of values.

    Providing a value for the radius invokes the use of the Tolman length delta
    to correct for the effects of curvature on the interfacial tension.

    Parameters:
        p : float
            pressure at which to estimate the saturation concentration [Pa]
        p_arr : (N) numpy array of floats
            pressures [Pa]
        if_arr : (N) numpy array of floats
            interfacial tension at the pressures in p_arr [N/m].
        R : float
            radius of curvature (assumed to be same in both directions) [m].
            If R <= 0, ignored.
        d_tolman : float
            Tolman length [m]. If R <= 0, ignored.

    Returns:
        if_tension : float
            interfacial tension between CO2-rich and polyol-rich phases [N/m] at the given pressure p
    """
    # interpolates interfacial tension [N/m] to match the given pressure
    if_tension = np.interp(p, *if_interp_arrs)/(1 + 2*d_tolman/R)
    return if_tension


def calc_if_tension_prep(polyol_data_file, p_min=0, p_max=4E7,
                         if_tension_model='lin'):
    """
    Estimates the interfacial tension between the CO2-rich and polyol-rich
    phases under equilibrium coexistence between CO2 and polyol at the given
    pressure by interpolating available measurements using G-ADSA. Provides
    arrays for interpolation using calc_if_tension().

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
    f_rho_co2 = scipy.interpolate.interp1d(p_co2, rho_co2, bounds_error=False,
                                       fill_value=(rho_min, rho_max))

    return f_rho_co2


def calc_dmdt(D, p_s, p, R, t, t_nuc, dt, c_s_interp_arrs, tol=1E-9,
                drop_t_term=False):
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
        current time since entering channel [s]
    t_nuc : float
        Nucleation time (measured since entering channel) [s]
    dt : float
        time step [s]. Only used if positive.
    c_s_interp_arrs : 2-tuple of numpy arrays of floats, (p_arr, c_s_arr)
        p_arr contains pressure [Pa] and c_s_arr contains saturation
        concentrations of CO2 in the bulk phase at those pressures [kg/m^3]
    tol : float
        tolerance of t [s] below which 1/np.sqrt(t) term might cause numerical
        difficulties, so it is integrated before adding to dm/dt.

    Returns:
    dmdt : float
        Time-derivative of the mass enclosed in the bubble [kg/s]
    """
    # shifts time to start from nucleation time
    t -= t_nuc
    # computes saturation concentrations [kg CO2 / m^3 solution]
    c_bulk = np.interp(p_s, *c_s_interp_arrs)
    c_s = np.interp(p, *c_s_interp_arrs)

    # computes time-derivative of the mass enclosed in the bubble [kg/s] based
    # on modified Epstein-Plesset (1950)
    dmdt1 = (4*np.pi*R**2*D) * (c_bulk - c_s) * (1/R)
    if drop_t_term:
        dmdt2 = 0
    elif t < tol and dt > 0:
        # integrates term from t->t+dt and divides by dt (assumes expl Euler)
        dmdt2 = (4*np.pi*R**2*D) * (c_bulk - c_s) * \
                    (1 / np.sqrt(np.pi*D)) * (2*(np.sqrt(t+dt)-np.sqrt(t))/dt)
    else:
        dmdt2 = (4*np.pi*R**2*D) * (c_bulk - c_s) * (1 / np.sqrt(np.pi*D*t))

    dmdt = dmdt1 + dmdt2

    return dmdt


def calc_m_p_bubble(R, p, m0, p_bubble0, if_interp_arrs, f_rho_co2, d_tolman):
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
    if_interp_arrs : 2-tuple of (N) numpy arrays of floats
        p_arr pressures [Pa] and if_tension_arr interfacial tensions at those
        pressures [N/m].
    eos_co2_file : string
        File name for equation of state data table [.csv]

    Returns:
    m : float
        mass of CO2 enclosed in the bubble [kg]
    p_bubble : float
        pressure inside the bubble [Pa]
    """
    def f(variables, args):
        """reorganizes variables and arguments for scipy.optimize.root"""
        m, p_bubble = variables
        R, p, f_rho_co2, if_interp_arrs, d_tolman = args

        return scf_bubble_fn(R, p_bubble, m, p, f_rho_co2, if_interp_arrs,
                             d_tolman)

    # solves for R with nonlinear solver
    args = (R, p, f_rho_co2, if_interp_arrs, d_tolman)
    soln = scipy.optimize.root(f, (m0, p_bubble0), args=(args,))
    m, p_bubble = soln.x

    return m, p_bubble


def calc_p_laplace(p, if_interp_arrs, R, d_tolman):
    """
    Computes the Laplace pressure caused by interfacial tension. Includes
    first-order curvature correction from Tolman JCP 1949.
    """
    return 2*calc_if_tension(p, if_interp_arrs, R, d_tolman=d_tolman)/R


def calc_R_p_bubble(m, p, R0, p_bubble0, if_interp_arrs, f_rho_co2,
                    d_tolman):
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
    # solves for R with nonlinear solver
    args = (m, p, f_rho_co2, if_interp_arrs, d_tolman)
    def f(variables, args):

        return scf_bubble_fn(*variables, *args)
    soln = scipy.optimize.root(f, (R0, p_bubble0), args=(args,)) # solves for R
    R, p_bubble = soln.x

    return R, p_bubble


def calc_m_R_p_bubble(m0, R0, p_bubble0, c_bulk, c_s, D, m_prev, p, t, dt,
                      if_interp_arrs, f_rho_co2, d_tolman):
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
    # solves for R with nonlinear solver
    args = (c_bulk, c_s, D, m_prev, p, t, dt, f_rho_co2, if_interp_arrs, d_tolman)
    def f(variables, args):
        return scf_bubble_impl(*variables, *args)
    soln = scipy.optimize.root(f, (m0, R0, p_bubble0), args=(args,))
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


def calc_t_s(p_in, p_atm, p_s, v, L):
    """
    Calculates time at which pressure reaches saturation pressure.
    """
    return (p_in - p_s)/(p_in - p_atm)*L/v


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


def scf_bubble_fn(R, p_bubble, m, p, f_rho_co2, if_interp_arrs, d_tolman):
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
    f_rho_co2 : interpolation function
        Function to interpolate density of CO2 given pressure according to
        equation of state obtained from NIST database. See interp_rho_co2()

    Returns
    -------
    res : 2-tuple of floats
        residuals from the two constraints enforced (conservation of mass
        for a sphere of gas and Young-Laplace pressure)
    """
    # residual for calculation of R based on spherical geometry
    res1  = R - ( 3/(4*np.pi)*(m/f_rho_co2(p_bubble)) )**(1/3.)
    # residual for pressure based on Laplace pressure + bulk pressure
#    if_tension = np.interp(p_bubble, *if_interp_arrs)
#    res2 = p_bubble - (p + 2*if_tension/R)
    res2 = p_bubble - (p + calc_p_laplace(p, if_interp_arrs, R, d_tolman))
    res = (res1, res2)

    return res


def scf_bubble_impl(m, R, p_bubble, c_bulk, c_s, D, m_prev,
                    p, t, dt, f_rho_co2, if_interp_arrs, d_tolman):
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
    m_prev : float
        mass in the previous time step
    p : float
        pressure in bulk governed by microfluidic flow [Pa]
    t : float
        time (t_{i+1} in the implicit Euler method) [s]
    dt : float
        time step [s]
    f_rho_co2 : interpolation function
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
    res1  = R - ( 3/(4*np.pi)*(m/f_rho_co2(p_bubble)) )**(1/3.)
    # residual for pressure based on Laplace pressure + bulk pressure
    res2 = p_bubble - (p + calc_p_laplace(p_bubble, if_interp_arrs, R,
                                          d_tolman))
    # residual for implicit Euler computation with Epstein-Plesset
    dmdt = 4*np.pi*R**2*D*(c_bulk - c_s)*(1/R + 1/np.sqrt(np.pi*D*t))
    res3 = m - (m_prev + dmdt*dt)
    res = (res1, res2, res3)

    return res


def time_step(dt, t_prev, m_prev, p_prev, if_tension_prev, R_prev, rho_co2_prev,
           fixed_params, drop_t_term=False):
    """
    """
    t_nuc, D, p_in, p_s, p_atm, v, L, c_bulk, c_s_interp_arrs, \
            if_interp_arrs, f_rho_co2, d_tolman, implicit = fixed_params
    t = t_prev + dt # increments time forward [s]
    p = calc_p(p_in, p_atm, v, t, L) # computes new pressure along observation capillary [Pa]
    c_s = np.interp(p, *c_s_interp_arrs) # interpolates saturation concentration of CO2 [kg CO2 / m^3 polyol-CO2]
    # guess for self-consistently solving for radius and pressure of bubble
    R0 = (3/(4*np.pi)*m_prev/rho_co2_prev)**(1./3) #p[-1] + 2*if_tension[-1]/R0
    p_bubble0 = p + 2*if_tension_prev/R0 #p_bubble[-1]

    # updates mass with explicit Euler method--inputs are i^th terms,
    # so we pass in R[-1] since R has not been updated to R_{i+1} yet
    m = m_prev + dt*calc_dmdt(D, p_s, p_prev, R_prev, t_prev, t_nuc, dt,
                              c_s_interp_arrs, drop_t_term=drop_t_term)
    # self-consistently solves for radius and pressure of bubble
    R, p_bubble = calc_R_p_bubble(m, p, R0, p_bubble0, if_interp_arrs,
                                  f_rho_co2, d_tolman)

    if implicit:
        # uses explicit time-stepping result for initial guess of implicit
        m0 = m
        R0 = R
        p_bubble0 = p_bubble
        # self-consistently solves implicit Euler equation
        soln = calc_m_R_p_bubble(m0, R0, p_bubble0, c_bulk, c_s, D, m_prev, p,
                                 t-t_nuc, dt, if_interp_arrs, f_rho_co2, d_tolman)
        m, R, p_bubble = soln

    if_tension = calc_if_tension(p_bubble, if_interp_arrs, R, d_tolman=d_tolman) # [N/m]]
    rho_co2 = f_rho_co2(p_bubble) # [kg/m^3]

    return dt, t, m, p, p_bubble, if_tension, c_s, R, rho_co2


def update_props(props, t, m, p, p_bubble, if_tension, c_s, R, rho_co2):
    """
    """
    t += [props[1]]
    m += [props[2]]
    p += [props[3]]
    p_bubble += [props[4]]
    if_tension += [props[5]]
    c_s += [props[6]]
    R += [props[7]]
    rho_co2 += [props[8]]


############################### LEGACY CODE ###################################


def fit_growth_to_pt(t_bubble, R_bubble, t_nuc_lo, t_nuc_hi, dt, p_s, R_nuc,
                     p_atm, L, p_in, v, polyol_data_file, eos_co2_file,
                     tol_R=0.01, ax=None, growth_fn=grow):
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
        t, m, D, p, p_bubble, if_tension, c_s, R, f_rho_co2 = \
                growth_fn(dt, t_nuc, p_s, R_nuc, p_atm, L, p_in, v,
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
