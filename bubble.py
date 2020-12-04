"""
Code for modeling bubble growth.

@author: Andy Ylitalo
"""

# adds path to general libraries
import sys
sys.path.append('../libs/')

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# custuom libraries
import geo
import polyco2
import flow
import finitediff as fd
from constants import * # includes constants

# CONVERSIONS
s_2_ms = 1000
m_2_um = 1E6
kPa_2_Pa = 1000
gmL_2_kgm3 = 1000
cm2s_2_m2s = 1E-4



def time_step(dt, t_prev, m_prev, p_prev, if_tension_prev, R_prev, rho_co2_prev,
           fixed_params, drop_t_term):
    """
    Advances system forward by one time step.

    Must come before grow() so it can be the default time_step_fn parameter.
    """
    t_nuc, D, p_in, p_s, v, L, c_bulk, c_s_interp_arrs, \
            if_interp_arrs, f_rho_co2, d_tolman, implicit = fixed_params
    t = t_prev + dt # increments time forward [s]
    p = flow.calc_p(p_in, P_ATM, v, t, L) # computes new pressure along observation capillary [Pa]
    c_s = np.interp(p, *c_s_interp_arrs) # interpolates saturation concentration of CO2 [kg CO2 / m^3 polyol-CO2]
    # guess for self-consistently solving for radius and pressure of bubble
    R0 = (3/(4*np.pi)*m_prev/rho_co2_prev)**(1./3) #p[-1] + 2*if_tension[-1]/R0
    p_bub0 = p + 2*if_tension_prev/R0 #p_bub[-1]

    # updates mass with explicit Euler method--inputs are i^th terms,
    # so we pass in R[-1] since R has not been updated to R_{i+1} yet
    m = m_prev + dt*calc_dmdt(D, p_s, p_prev, R_prev, t_prev, t_nuc, dt,
                              c_s_interp_arrs, drop_t_term=drop_t_term)
    # self-consistently solves for radius and pressure of bubble
    R, p_bub = calc_R_p_bub(m, p, R0, p_bub0, if_interp_arrs,
                                  f_rho_co2, d_tolman)

    if implicit:
        # uses explicit time-stepping result for initial guess of implicit
        m0 = m
        R0 = R
        p_bub0 = p_bub
        # self-consistently solves implicit Euler equation
        soln = calc_m_R_p_bub(m0, R0, p_bub0, c_bulk, c_s, D, m_prev, p,
                                 t-t_nuc, dt, if_interp_arrs, f_rho_co2, d_tolman)
        m, R, p_bub = soln

    if_tension = polyco2.calc_if_tension(p_bub, if_interp_arrs, R, d_tolman=d_tolman) # [N/m]]
    rho_co2 = f_rho_co2(p_bub) # [kg/m^3]

    return dt, t, m, p, p_bub, if_tension, c_s, R, rho_co2


def grow(dt, t_nuc, p_s, R_nuc, L, p_in, v, polyol_data_file, eos_co2_file,
         adaptive_dt=True, if_tension_model='lin', implicit=False, d_tolman=0,
         tol_R=0.001, alpha=0.3, D=-1, drop_t_term=False, time_step_fn=time_step):
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
        L : float
            length of observation capillary [m]
        p_in : float
            pressure at inlet [m], calculated using flow.sheath_eqns
        v : float
            velocity of inner stream [m/s], calculated using flow.sheath_eqns
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
        p_bub : list of N floats
            pressure inside bubble at each time step
        if_tension : list of N floats
            interfacial tension along bubble surface at each time step based on
            G-ADSA measurements [N/m]
        c_s : list of N floats
            saturation concentrations of CO2 in polyol at each time step based
            on G-ADSA solubility and density measurements
            [kg CO2 / m^3 polyol-CO2]
        c_bulk : float
            bulk concentration of CO2 in polyol at given saturation pressure
            [kg CO2 / m^3 polyol-CO2]
        R : list of N floats
            radius of bubble at each time step solved self-consistently with
            modified Epstein-Plesset (1950) [m]
        rho_co2 : list of N floats
            density of CO2 in bubble at each time step based on pressure and
            CO2 equation of state [kg/m^3]
    """
    t, m, D, p, p_bub, if_tension, c_s, \
    c_bulk, R, rho_co2, t_f, fixed_params = init(p_in, p_s, t_nuc, R_nuc,
                                            v, L, D, polyol_data_file,
                                            eos_co2_file, if_tension_model,
                                            d_tolman, implicit)

    # applies Euler's method to estimate bubble growth over time
    # the second condition provides cutoff for shrinking the bubble
    while t[-1] <= t_f:
        time_step_params = (t[-1], m[-1], p[-1],
                                    if_tension[-1], R[-1], rho_co2[-1],
                                    fixed_params, drop_t_term)
        if adaptive_dt:
            dt, props = adaptive_time_step(dt, time_step_params, time_step_fn, tol_R, alpha)
        else:
            # calculates properties after one time step
            props = time_step_fn(dt, *time_step_params)

        # saves properties
        update_props(props, t, m, p, p_bub, if_tension, c_s, R, rho_co2)

    return t, m, D, p, p_bub, if_tension, c_s, c_bulk, R, rho_co2


def adaptive_time_step(dt, time_step_params, time_step_fn, tol_R, alpha, dt_max=None):
    """
    """
    while True:
        # calculates properties for two time steps
        props_a = time_step_fn(dt, *time_step_params)
        props_b = time_step_fn(2*dt, *time_step_params)
        # extracts estimated value of the radius (index -2 in properties list)
        R_a = props_a[-2]
        R_b = props_b[-2]
        # checks if the discrepancy in the radius is below tolerance
        if np.abs( (R_a - R_b) / R_a) <= tol_R:
            # only increases time step if it will remain below maximum
            if (dt_max is None) or (dt*(1+alpha) < dt_max):
                # increases time step for next calculation
                dt *= (1 + alpha)
            # takes properties calculated with smaller time step
            props = props_a
            # breaks loop now that time step met tolerance
            break
        else:
            # does not advance forward, reduces time step
            dt /= 2

    return dt, props


def calc_dcdr_eps(c_bulk, c_s, R, D, t):
    """
    Calculates the concentration gradient at the surface of the bubble for the
    Epstein-Plesset model.
    """
    return  (c_bulk - np.asarray(c_s))*(1/np.asarray(R) + \
                1/np.sqrt(np.pi*D*(np.asarray(t))))

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


def calc_dmdt_dcdr_fix_D(r_arr, c, R, D):
    """
    Calculates the time-derivative of the mass enclosed inside a
    CO2 bubble under the given conditions. The formula is modified
    from the widely used Epstein-Plesset formula (1950) to include
    the effect of changing pressure.

    Parameters
    ----------
    r_arr : (N x 1) numpy array of floats
        grid of radii where r = 0 corresponds to surface of bubble [m]
    c : (N x 1) numpy array of floats
        concentration of CO2 at each point in r_arr [kg/m^3]
    D : float
        diffusivity of CO2 in polyol [m^2/s]
    R : float
        current radius of the bubble [m]

    Returns:
    dmdt : float
        Time-derivative of the mass enclosed in the bubble [kg/s]
    """
    # concentration gradient at interface of bubble [kg/m^3 / m]
    # 2nd-order Taylor scheme
    dcdr = fd.dydx_fwd_2nd(c[0], c[1], c[2], r_arr[1]-r_arr[0])
    dmdt = (4*np.pi*R**2*D)*dcdr

    return dmdt


def calc_m_p_bub(R, p, m0, p_bub0, if_interp_arrs, f_rho_co2, d_tolman):
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
    p_bub : float
        pressure inside the bubble [Pa]
    """
    def f(variables, args):
        """reorganizes variables and arguments for scipy.optimize.root"""
        m, p_bub = variables
        R, p, f_rho_co2, if_interp_arrs, d_tolman = args

        return scf_bubble_fn(R, p_bub, m, p, f_rho_co2, if_interp_arrs,
                             d_tolman)

    # solves for R with nonlinear solver
    args = (R, p, f_rho_co2, if_interp_arrs, d_tolman)
    soln = scipy.optimize.root(f, (m0, p_bub0), args=(args,))
    m, p_bub = soln.x

    return m, p_bub


def calc_p_laplace(p, if_interp_arrs, R, d_tolman):
    """
    Computes the Laplace pressure caused by interfacial tension. Includes
    first-order curvature correction from Tolman JCP 1949.
    """
    return 2*polyco2.calc_if_tension(p, if_interp_arrs, R, d_tolman=d_tolman)/R


def calc_R_p_bub(m, p, R0, p_bub0, if_interp_arrs, f_rho_co2,
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
    p_bub : float
        pressure inside the bubble [Pa]
    """
    # solves for R with nonlinear solver
    args = (m, p, f_rho_co2, if_interp_arrs, d_tolman)
    def f(variables, args):

        return scf_bubble_fn(*variables, *args)
    soln = scipy.optimize.root(f, (R0, p_bub0), args=(args,)) # solves for R
    R, p_bub = soln.x

    return R, p_bub


def calc_m_R_p_bub(m0, R0, p_bub0, c_bulk, c_s, D, m_prev, p, t, dt,
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
        p_bub_{i+1} [Pa]
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
    p_bub : float
        pressure inside the bubble for the next time step p_bub_{i+1} [Pa]
    """
    # solves for R with nonlinear solver
    args = (c_bulk, c_s, D, m_prev, p, t, dt, f_rho_co2, if_interp_arrs, d_tolman)
    def f(variables, args):
        return scf_bubble_impl(*variables, *args)
    soln = scipy.optimize.root(f, (m0, R0, p_bub0), args=(args,))
    m, R, p_bub = soln.x

    return m, R, p_bub


def init(p_in, p_s, t_nuc, R_nuc, v, L, D, polyol_data_file,
            eos_co2_file, if_tension_model, d_tolman, implicit):
    """
    Initializes parameters used in grow() for bubble growth.

    """
    # prep arrays for interpolation
    c_s_interp_arrs = polyco2.load_c_s_arr(polyol_data_file)
    if_interp_arrs = polyco2.load_if_tension_arr(polyol_data_file,
                                          if_tension_model=if_tension_model)
    c_bulk = np.interp(p_s, *c_s_interp_arrs) # computes bulk CO2 concentration
    if D == -1:
        D = polyco2.calc_D(p_s, polyol_data_file) # assumes diffusivity of saturation pressure
    # creates interpolation fn for density of CO2 based on equation of state
    f_rho_co2 = polyco2.interp_rho_co2(eos_co2_file)

    # initializes lists of key bubble properties
    p = [flow.calc_p(p_in, P_ATM, v, t_nuc, L)]
    c_s = [np.interp(p[0], *c_s_interp_arrs)]
    R = [R_nuc]
    # solves for initial mass and pressure in bubble self-consistently
    m0 = geo.v_sphere(R_nuc)*f_rho_co2(p[0])
    p_bub0 = p[0]
    m_init, p_bub_init = calc_m_p_bub(R[0], p[0], m0, p_bub0,
                                            if_interp_arrs, f_rho_co2, d_tolman)
    p_bub = [p_bub_init]
    m = [m_init]
    if_tension = [polyco2.calc_if_tension(p_bub[0], if_interp_arrs, R[0],
                                  d_tolman=d_tolman)]
    # initial bubble density [kg/m^3]
    rho_co2 = [f_rho_co2(p_bub[0])]

    # calcultes the time at which pressure reaches saturation pressure
    t_s = flow.calc_t_s(p_in, P_ATM, p_s, v, L)
    # ensures that nucleation time occurs after supersaturation achieved
    t_nuc = max(t_s, t_nuc)
    # initializes timeline [s]
    t = [t_nuc]
    # defines final time [s]
    t_f = L/v

    # collects fixed parameters
    fixed_params = (t_nuc, D, p_in, p_s, v, L, c_bulk, c_s_interp_arrs,
                    if_interp_arrs, f_rho_co2, d_tolman, implicit)

    return t, m, D, p, p_bub, if_tension, c_s, c_bulk, R, rho_co2, t_f, \
            fixed_params


def scf_bubble_fn(R, p_bub, m, p, f_rho_co2, if_interp_arrs, d_tolman):
    """
    Computes residual of the equations governing bubble size:
        1) mass = volume*density (conservation of mass + volume of a sphere)
        2) p_bub = p + 2*if_tension/R (Young-Laplace)
    Used in calc_R_p_bub and calc_m_p_bub for self-consistent
    computation using scipy.optimize.root.

    Parameters
    ----------
    if_tension : float
        Interfacial tension along surface of bubble [N/m]
    m : float
        mass of CO2 enclosed in bubble [kg]
    p : float
        pressure in bulk governed by microfluidic flow [Pa]
    p_bub : float
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
    res1  = R - ( 3/(4*np.pi)*(m/f_rho_co2(p_bub)) )**(1/3.)
    # residual for pressure based on Laplace pressure + bulk pressure
#    if_tension = np.interp(p_bub, *if_interp_arrs)
#    res2 = p_bub - (p + 2*if_tension/R)
    res2 = p_bub - (p + calc_p_laplace(p, if_interp_arrs, R, d_tolman))
    res = (res1, res2)

    return res


def scf_bubble_impl(m, R, p_bub, c_bulk, c_s, D, m_prev,
                    p, t, dt, f_rho_co2, if_interp_arrs, d_tolman):
    """
    Computes residual of the equations governing bubble size:
        1) mass = volume*density (conservation of mass + volume of a sphere)
        2) p_bub = p + 2*if_tension/R (Young-Laplace)
        3) implicit Euler using Epstein-Plesset:
            m_{i+1} = m_i + dm/dt|_{t_{i+1}}*dt_i
    Used in calc_m_R_p_bub for self-consistent
    computation using scipy.optimize.root.

    Note that this solves for R_{i+1}, m_{i+1}, and p_bub,i+1 because it is
    used for an implicit method.

    Parameters
    ----------
    m : float
        mass of CO2 enclosed in bubble [kg]
    R : float
        radius of bubble [m]
    p_bub : float
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
    res1  = R - ( 3/(4*np.pi)*(m/f_rho_co2(p_bub)) )**(1/3.)
    # residual for pressure based on Laplace pressure + bulk pressure
    res2 = p_bub - (p + calc_p_laplace(p_bub, if_interp_arrs, R,
                                          d_tolman))
    # residual for implicit Euler computation with Epstein-Plesset
    dmdt = 4*np.pi*R**2*D*(c_bulk - c_s)*(1/R + 1/np.sqrt(np.pi*D*t))
    res3 = m - (m_prev + dmdt*dt)
    res = (res1, res2, res3)

    return res


def time_step_dcdr(dt, t_prev, m_prev, if_tension_prev, R_prev,
                        rho_co2_prev, r_arr, c_arr, fixed_params):
    """
    Advances system forward by one time step using dc/dr direct calculation.
    This only considers the concentration at the surface of the bubble, so it
    assumes constant diffusivity whether it varies with concentration or not.
    """
    # some of the fixed parameters needed for time_step() are not required here
    D, p_in, p_s, v, L, c_s_interp_arrs, \
    if_interp_arrs, f_rho_co2, d_tolman = fixed_params
    t = t_prev + dt # increments time forward [s]
    p = flow.calc_p(p_in, P_ATM, v, t, L) # computes new pressure along observation capillary [Pa]
    c_s = np.interp(p, *c_s_interp_arrs) # interpolates saturation concentration of CO2 [kg CO2 / m^3 polyol-CO2]
    # guess for self-consistently solving for radius and pressure of bubble
    R0 = (3/(4*np.pi)*m_prev/rho_co2_prev)**(1./3) #p[-1] + 2*if_tension[-1]/R0
    p_bub0 = p + 2*if_tension_prev/R0 #p_bub[-1]



    # updates mass with explicit Euler method--inputs are i^th terms,
    # so we pass in R[-1] since R has not been updated to R_{i+1} yet
    m = m_prev + dt*calc_dmdt_dcdr_fix_D(r_arr, c_arr, R_prev, D)
    # self-consistently solves for radius and pressure of bubble
    R, p_bub = calc_R_p_bub(m, p, R0, p_bub0, if_interp_arrs,
                                  f_rho_co2, d_tolman)

    if_tension = polyco2.calc_if_tension(p_bub, if_interp_arrs, R, d_tolman=d_tolman) # [N/m]]
    rho_co2 = f_rho_co2(p_bub) # [kg/m^3]

    return dt, t, m, p, p_bub, if_tension, c_s, R, rho_co2


def update_props(props, t, m, p, p_bub, if_tension, c_s, R, rho_co2):
    """
    """
    t += [props[1]]
    m += [props[2]]
    p += [props[3]]
    p_bub += [props[4]]
    if_tension += [props[5]]
    c_s += [props[6]]
    R += [props[7]]
    rho_co2 += [props[8]]


############################### LEGACY CODE ###################################


def fit_growth_to_pt(t_bubble, R_bubble, t_nuc_lo, t_nuc_hi, dt, p_s, R_nuc,
                     L, p_in, v, polyol_data_file, eos_co2_file,
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
        t, m, D, p, p_bub, if_tension, c_s, R, f_rho_co2 = \
                growth_fn(dt, t_nuc, p_s, R_nuc, L, p_in, v,
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
