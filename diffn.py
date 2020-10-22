"""
@brief diffusion.py contains methods used to model diffusion of CO2 in polyol.

These methods can be used to help model bubble growth and to model the diffusion
of CO2 from the inner stream to the outer stream.

@author Andy Ylitalo
@date October 21, 2020
"""
# adds path to import other libraries
import sys
sys.path.append('../libs/')


# imports standard libraries
import numpy as np


# imports custom libraries
import polyco2




############################# FUNCTION DEFINITIONS #############################


def apply_bc(c, bc_specs):
    """
    Applies boundary condition to the concentration list.

    Parameters
    ----------
    c : list of N+1 floats
        concentration at each grid point [kg CO2 / m^3 polyol-CO2]
    bc_specs : list
        contains specifications for desired boundary condition
    """
    bc_fn = bc_specs[0]
    bc_fn(c, *bc_specs[1:])


def dirichlet(c, i, val):
    """
    Applies Dirichlet (constant value) boundary condition at desired boundary.

    Parameters
    ----------
    c : list of N+1 floats
        concentration at each grid point [kg CO2 / m^3 polyol-CO2]
    i : int
        index at which to apply boundary condition
    val : float
        value to enforce at boundary
    """
    c[i] = val


def calc_dcdt_cyl(r_arr, c_arr, fixed_params):
    """
    Computes the time derivative of concentration dc/dt in cylindrical
    coordinates assuming concentration dependent diffusivity constant.

    In cylindrical coordinates Fick's law of dc/dt = div(D(c)grad(c)) becomes:

            dc/dt = 1/r*D(c)*dc/dr + dD/dc*(dc/dr)^2 + D(c)*d2c/dr2

    assuming cylindrical symmetry.

    Parameters
    ----------
    r_arr : numpy array of N+1 floats
        radial coordinates of points in mesh [m]
    c_arr : numpy array of N+1 floats
        concentrations of CO2 at each of the points in r_arr
        [kg CO2 / m^3 polyol-co2]

    fixed_params: list
        dc : float
            step size of concentration used to estimate dD/dc by forward
            difference formula [kg CO2 / m^3 polyol-CO2]
        polyol_data_file : string
            name of file containing polyol data [.csv]

    Returns
    -------
    dcdt : numpy array of N-1 floats
        time derivatives of concentration at each of the interior mesh points
        (the concentrations at the end points at i=0 and i=N+1 are computed by
        the boundary conditions)
    """
    # extracts fixed parameters
    dc, interp_arrs = fixed_params
    # extracts just the concentrations that will change (i = 1...N-1)
    c_arr_c = c_arr[1:-1]
    # and their corresponding radii
    r_arr_c = r_arr[1:-1]
    # and their corresponding grid spacings
    dr_arr = r_arr[2:] - r_arr[1:-1]

    # FIRST TERM: 1/r * D(c) * dc/dr
    # computes diffusivity at each point
    D_arr = np.asarray([polyco2.calc_D_of_c_raw(c, *interp_arrs) \
                                                    for c in c_arr_c])
    # computes spatial derivative of concentration dc/dr with leapfrog method
    dcdr_arr = (c_arr[2:] - c_arr[:-2]) / (2*dr_arr)
    term1 = 1/r_arr_c * D_arr * dcdr_arr

    # SECOND TERM: dD/dc * (dc/dr)^2
    # computes dD/dc [m^2/s / kg/m^3]
    dDdc_arr = np.asarray([polyco2.calc_dDdc_raw(c, dc, *interp_arrs) \
                                                    for c in c_arr_c])
    term2 = dDdc_arr * (dcdr_arr)**2

    # THIRD TERM: D(c) * d2c/dr2
    # computes second spatial derivative of concentration with central
    # difference formula
    d2cdr2_arr = (c_arr[2:] - 2*c_arr[1:-1] + c_arr[:-2]) / (dr_arr**2)
    term3 = D_arr * d2cdr2_arr

    dcdt = term1 + term2 + term3

    return dcdt


def go(dt, t_f, r_arr, c_0, dcdt_fn, bc_specs_list, dc, polyol_data_file):
    """
    Diffuses CO2 throughout system.


    Parameters
    ----------
    dt : float
        time step (might be adjusted if adaptive time-stepping) [s]
    t_f : float
        final time [s]
    r_arr : numpy array of N+1 floats
        mesh points [m]
    c_0 : list of N+1 floats
        initial concentration profile of CO2 at grid points
    dcdt_fn : function
        function to compute the time derivative of concentration dcdt
    bc_specs_list : list of tuples
        List of tuples of specs for boundary conditions (first element of each
        tuple is the function for applying the boundary condition)
    dc : float
        Step size in concentration for estimating dD/dc with forward difference
        formula [kg CO2 / m^3 polyol-CO2]
    polyol_data_file : string
        name of file containing polyol-CO2 data [.csv]

    Returns
    -------
    t : list of M floats
        times at which each of the M concentration profiles are evaluated
    c : list of M lists of N+1 floats
        concentration profiles over time [kg CO2 / m^3 polyol-CO2], M time
        steps and N points in space
    """
    # initializes system
    t = [0]
    c = [c_0]

    # loads arrays for interpolation
    p_arr, D_sqrt_arr, D_exp_arr = polyco2.load_D_arr(polyol_data_file)
    c_s_arr, p_s_arr = polyco2.load_c_s_arr(polyol_data_file)
    interp_arrs = (c_s_arr, p_s_arr, p_arr, D_sqrt_arr, D_exp_arr)
    # stores fixed parameters
    fixed_params = (dc, interp_arrs)
    # applies Euler's method to estimate bubble growth over time
    # the second condition provides cutoff for shrinking the bubble
    while t[-1] <= t_f:
        # calculates properties after one time step
        props = time_step(dt, t[-1], r_arr, c[-1], dcdt_fn,
                            bc_specs_list, fixed_params)
        # stores properties at new time step in lists
        update_props(props, t, c)
        print('t = {0:.2g} s of {1:.2g} s.'.format(t[-1], t_f))

    return t, c


def neumann(c, i, j, dcdr, r_arr):
    """
    Applies Neumann boundary condition at desired index.

    The condition is

                    dc/dr_i = (c_i - c_j) / (r_i - r_j)

    so we can solve for the concentration at c[i] (the boundary) as

                    c[i] = dc/dr_i * (r_i - r_j) + c_j

    Parameters
    ----------
    c : list of N+1 floats
        concentration of CO2 at each grid point [kg CO2 / m^3 polyol-CO2]
    i : int
        index at which to *set* the boundary condition
    j : int
        index used to compute the derivative to compute the value at i
    dcdr : float
        value of dc/dr spatial derivative of the concentration at the boundary
        [kg CO2 / m^3 polyol-CO2]
    r_arr : numpy array of N+1 floats
        mesh points [m]
    """
    # solves boundary condition for the value at the boundary
    c[i] = dcdr * (r_arr[i] - r_arr[j]) + c[j]


def time_step(dt, t_prev, r_arr, c_prev, dcdt_fn, bc_specs_list, fixed_params):
    """
    Advances system forward by one time step.

    Uses explicit Euler method and requested function for computing time
    derivative.

    Parameters
    ----------
    dt : float
        time step [s]
    t_prev : float
        previous time [s]
    r_arr : numpy array of N+1 floats
        grid points [m]
    c_prev : list of N+1 floats
        previous concentration profile at grid points [kg CO2 / m^3 polyol-CO2]
    dcdt_fn : function
        function to compute dc/dt [kg/m^3 / s]
    bc_specs_list : list of tuples
        specifications for the boundary conditions to be applied (first item in
        each tuple of specs is the boundary condition function, like neumann or
        dirichlet)
    fixed_params : list
        In this case, only contains
        polyol_data_file : string
            name of file containing polyol data [.csv]

    Returns
    -------
    dt_curr : float
        time step (might be different than input if adaptive time-stepping used)
        [s]
    t_curr : float
        current time [s] (t_prev + dt)
    c_curr : list of N+1 floats
        current concentration profile at grid points [kg CO2 / m^3 polyol-CO2]
    """
    # increments time forward [s]
    t_curr = t_prev + dt
    # no adaptive time-stepping
    dt_curr = dt
    # extracts array of relevant concentrations
    c_prev_arr = np.array(c_prev[1:-1])

    # updates concentration with explicit Euler method
    c_curr_arr = c_prev_arr + dt*dcdt_fn(r_arr, np.array(c_prev), fixed_params)
    # adds end points for the application of the boundary conditions
    c_curr = [0] + list(c_curr_arr) + [0]

    # applies boundary conditions
    for bc_specs in bc_specs_list:
        apply_bc(c_curr, bc_specs)

    return dt_curr, t_curr, c_curr


def update_props(props, t, c):
    """
    """
    t += [props[1]]
    c += [props[2]]
