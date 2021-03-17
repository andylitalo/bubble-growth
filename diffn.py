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
import pandas as pd
import scipy.optimize
import scipy.interpolate

# imports custom libraries
import polyco2
import flow
import finitediff as fd


# conversions
from conversions import *

# global constants for diffusion model
filepath_D_c='../g-adsa_results/D_c_power_law.csv'
df_D = pd.read_csv(filepath_D_c)
D0, A_p, k_p = df_D['p']
D0, A_dp, k_dp = df_D['dp']

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
        interp_arrs : tuple of numpy arrays
            arrays for interpolations in polyco2.calc_D_of_c_raw():
            c_s_arr, p_s_arr, p_arr, D_exp_arr, D_sqrt_arr

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


def calc_dcdt_sph_fix_D(xi_arr, c_arr, R, D):
    """
    Computes the time derivative of concentration dc/dt in spherical
    coordinates assuming concentration dependent diffusivity constant.

    In spherical coordinates Fick's law of dc/dt = div(D(c)grad(c)) becomes:

            dc/dt = 2/r*D(c)*dc/dr + dD/dc*(dc/dr)^2 + D(c)*d2c/dr2

    assuming spherical symmetry.

    Parameters
    ----------
    xi_arr : numpy array of N+1 floats
        radial coordinates of points in mesh measured from radius of bubble [m]
    c_arr : numpy array of N+1 floats
        concentrations of CO2 at each of the points in r_arr
        [kg CO2 / m^3 polyol-co2]
    fixed_params: list
        D : float
            diffusivity of CO2 in polyol (fixed) [m^2/s]
        R : float
            radius of bubble [m]

    Returns
    -------
    dcdt : numpy array of N-1 floats
        time derivatives of concentration at each of the interior mesh points
        (the concentrations at the end points at i=0 and i=N+1 are computed by
        the boundary conditions)
    """
    # extracts fixed parameters
    r_arr = xi_arr + R
    # and their corresponding radii
    r_arr_c = r_arr[1:-1]
    # and their corresponding grid spacings
    dr = r_arr[1] - r_arr[0]

    # FIRST TERM: 2/r * D(c) * dc/dr
    # computes spatial derivative of concentration dc/dr with central difference
    dcdr_arr = fd.dydx_cd_2nd(c_arr, dr)
    term1 = 2/r_arr_c * D * dcdr_arr

    # SECOND TERM: D(c) * d2c/dr2
    # computes second spatial derivative of concentration with central
    # difference formula
    d2cdr2_arr = fd.d2ydx2_cd_2nd(c_arr, dr)
    term2 = D * d2cdr2_arr

    dcdt = term1 + term2

    return dcdt

def calc_dcdt_sph_fix_D_nonuniform(xi_arr, c_arr, R, D):
    """
    Computes the time derivative of concentration dc/dt in spherical
    coordinates assuming concentration dependent diffusivity constant.
    ***Allows for a grid with nonuniform spacing.***

    In spherical coordinates Fick's law of dc/dt = div(D(c)grad(c)) becomes:

            dc/dt = 2/r*D(c)*dc/dr + dD/dc*(dc/dr)^2 + D(c)*d2c/dr2

    assuming spherical symmetry.

    Parameters
    ----------
    xi_arr : numpy array of N+1 floats
        radial coordinates of points in mesh measured from radius of bubble [m]
        *May be a nonuniform grid
    c_arr : numpy array of N+1 floats
        concentrations of CO2 at each of the points in r_arr
        [kg CO2 / m^3 polyol-co2]
    fixed_params: list
        D : float
            diffusivity of CO2 in polyol (fixed) [m^2/s]
        R : float
            radius of bubble [m]

    Returns
    -------
    dcdt : numpy array of N-1 floats
        time derivatives of concentration at each of the interior mesh points
        (the concentrations at the end points at i=0 and i=N+1 are computed by
        the boundary conditions)
    """
    r_arr = xi_arr + R
    # and their corresponding radii
    r_arr_c = r_arr[1:-1]

    # FIRST TERM: 2/r * D(c) * dc/dr
    dcdr_arr = fd.dydx_non_1st(c_arr, r_arr)
    term1 = 2/r_arr_c * D * dcdr_arr

    # SECOND TERM: D(c) * d2c/dr2
    d2cdr2_arr = fd.d2ydx2_non_1st(c_arr, r_arr)
    term2 = D * d2cdr2_arr

    dcdt = term1 + term2

    return dcdt


def calc_dcdt_sph_vary_D(xi_arr, c_arr, R, fixed_params):
    """
    Computes the time derivative of concentration dc/dt in spherical
    coordinates assuming concentration-dependent diffusivity.

    In spherical coordinates Fick's law of dc/dt = div(D(c)grad(c)) becomes:

            dc/dt = 2/r*D(c)*dc/dr + dD/dc*(dc/dr)^2 + D(c)*d2c/dr2

    assuming spherical symmetry.

    Parameters
    ----------
    xi_arr : numpy array of N+1 floats
        radial coordinates of points in mesh measured from radius of bubble [m]
    c_arr : numpy array of N+1 floats
        concentrations of CO2 at each of the points in r_arr
        [kg CO2 / m^3 polyol-co2]
    R : float
        radius of bubble [m]
    dc : float
        step size in concentration dimension [kg/m^3]
    interp_arrs : tuple of numpy arrays
        numpy arrays for interpolation

    Returns
    -------
    dcdt : numpy array of N-1 floats
        time derivatives of concentration at each of the interior mesh points
        (the concentrations at the end points at i=0 and i=N+1 are computed by
        the boundary conditions)
    """
    dc, D_fn, R_i, eta_ratio = fixed_params
    r_arr = xi_arr + R
    # and their corresponding radii
    r_arr_c = r_arr[1:-1]
    # and their corresponding grid spacings
    dr = r_arr[1] - r_arr[0]
    # and the corresponding concentrations
    c_arr_c = c_arr[1:-1]
    # computes diffusivity constant at each point on grid [m^2/s]
    D_arr = np.asarray([D_fn(c) for c in c_arr_c])
    # adjusts diffusivity of outer stream based on viscosity ratio
    D_arr[xi_arr[1:-1] >= R_i] *= eta_ratio

    # FIRST TERM: 2/r * D(c) * dc/dr
    # computes spatial derivative of concentration dc/dr with central difference
    dcdr_arr = fd.dydx_cd_2nd(c_arr, dr)
    term1 = 2/r_arr_c * D_arr * dcdr_arr

    # SECOND TERM: dD/dc * (dc/dr)^2
    # computes dD/dc [m^2/s / kg/m^3]
    dDdc_arr = np.asarray([polyco2.calc_dDdc_fn(c, dc, D_fn) \
                                                    for c in c_arr_c])
    term2 = dDdc_arr * (dcdr_arr)**2

    # THIRD TERM: D(c) * d2c/dr2
    # computes second spatial derivative of concentration with central
    # difference formula
    d2cdr2_arr = fd.d2ydx2_cd_2nd(c_arr, dr)
    term3 = D_arr * d2cdr2_arr

    dcdt = term1 + term2 + term3

    return dcdt


def calc_dcdt_sph_vary_D_nonuniform(xi_arr, c_arr, R, fixed_params):
    """
    Computes the time derivative of concentration dc/dt in spherical
    coordinates assuming concentration-dependent diffusivity.

    In spherical coordinates Fick's law of dc/dt = div(D(c)grad(c)) becomes:

            dc/dt = 2/r*D(c)*dc/dr + dD/dc*(dc/dr)^2 + D(c)*d2c/dr2

    assuming spherical symmetry. Allows for a non-uniform grid but, as a result,
    uses first-order instead of second-order finite difference schemes.

    Parameters
    ----------
    xi_arr : numpy array of N+1 floats
        radial coordinates of points in mesh measured from radius of bubble [m]
    c_arr : numpy array of N+1 floats
        concentrations of CO2 at each of the points in r_arr
        [kg CO2 / m^3 polyol-co2]
    R : float
        radius of bubble [m]
    dc : float
        step size in concentration dimension [kg/m^3]

    Returns
    -------
    dcdt : numpy array of N-1 floats
        time derivatives of concentration at each of the interior mesh points
        (the concentrations at the end points at i=0 and i=N+1 are computed by
        the boundary conditions)
    """
    dc, D_fn, R_i, eta_ratio = fixed_params
    r_arr = xi_arr + R
    # and their corresponding radii
    r_arr_c = r_arr[1:-1]
    # and their corresponding grid spacings
    dr = r_arr[1] - r_arr[0]
    # and the corresponding concentrations
    c_arr_c = c_arr[1:-1]
    # computes diffusivity constant at each point on grid [m^2/s]
    D_arr = np.asarray([D_fn(c) for c in c_arr_c])
    # scales diffusivity of outer stream by ratio of diffusivities, eta_i/eta_o
    D_arr[xi_arr[1:-1] >= R_i] *= eta_ratio
    # FIRST TERM: 2/r * D(c) * dc/dr
    # computes spatial derivative of concentration dc/dr with central difference
    dcdr_arr = fd.dydx_non_1st(c_arr, r_arr)
    term1 = 2/r_arr_c * D_arr * dcdr_arr

    # SECOND TERM: dD/dc * (dc/dr)^2
    # computes dD/dc [m^2/s / kg/m^3]
    dDdc_arr = np.asarray([polyco2.calc_dDdc_fn(c, dc, D_fn) \
                                                    for c in c_arr_c])
    term2 = dDdc_arr * (dcdr_arr)**2

    # THIRD TERM: D(c) * d2c/dr2
    # computes second spatial derivative of concentration with central
    # difference formula
    d2cdr2_arr = fd.d2ydx2_non_1st(c_arr, r_arr)
    term3 = D_arr * d2cdr2_arr

    dcdt = term1 + term2 + term3

    return dcdt


def calc_dcdt_sph_fix_D_transf(xi_arr, c_arr, R, D):
    """
    Computes the time derivative of concentration dc/dt in spherical
    coordinates assuming concentration dependent diffusivity constant.

    In spherical coordinates Fick's law of dc/dt = div(D(c)grad(c)) becomes:

            du/dt = D*d2u/dxi2

    Where u = c/r and xi = r-R

    assuming spherical symmetry.

    This transformation makes the calculation simpler and *hopefully* less prone
    to numerical errors.

    Parameters
    ----------
    xi_arr : numpy array of N+1 floats
        radial coordinates of points in mesh, measured from surface of bubble [m]
    c_arr : numpy array of N+1 floats
        concentrations of CO2 at each of the points in r_arr
        [kg CO2 / m^3 polyol-co2]
    fixed_params: list
        D : float
            diffusivity of CO2 in polyol (fixed) [m^2/s]
        R : float
            Radius of bubble [m]

    Returns
    -------
    dcdt : numpy array of N-1 floats
        time derivatives of concentration at each of the interior mesh points
        (the concentrations at the end points at i=0 and i=N+1 are computed by
        the boundary conditions)
    """
    # and their corresponding grid spacings
    dxi_arr = (xi_arr[2:] - xi_arr[0:-2])/2
    # calculates u
    r_arr = xi_arr + R
    r_arr_c = r_arr[1:-1]
    u_arr = c_arr / r_arr

    # D(c) * d2u/dxi2
    # computes second spatial derivative of concentration with central
    # difference formula
    d2udxi2_arr = (u_arr[2:] - 2*u_arr[1:-1] + u_arr[:-2]) / (dxi_arr**2)

    # computes time derivative
    dudt = D * d2udxi2_arr
    dcdt = dudt * r_arr_c # c = u*r

    return dcdt


def fixed_D(a):
    """Fixes diffusivity for comparison to num_fix_D()"""
    return 2.3497250000000002e-08


def D_p(c):
    """
    Power-law fit for D(c) fitted to *pressurization* data of 1k3f @ 30c
    (see 20201124_1k3f_D_vs_rho_co2.ipynb).
    """
    return D0 + A_p * c**k_p

def D_dp(c):
    """
    Power-law fit for D(c) fitted to *depressurization* data of 1k3f @ 30c (see
    20201124_1k3f_D_vs_rho_co2.ipynb).
    """
    return D0 + A_dp * c**k_dp

def go(dt, t_f, R_min, R_o, N, c_0, dcdt_fn, bc_specs_list,
        eta_i, eta_o, d, L, Q_i, Q_o, p_s, dc_c_s_frac, polyol_data_file):
    """
    Diffuses CO2 throughout system with concentration-dependent diffusivity.

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
    t, c, r_arr, \
    c_0, c_s, t_f, fixed_params = init(R_min, R_o, N, eta_i, eta_o, d, L, Q_i,
                                        Q_o, p_s, dc_c_s_frac, polyol_data_file)
    # applies Euler's method to estimate bubble growth over time
    # the second condition provides cutoff for shrinking the bubble
    while t[-1] <= t_f:
        # calculates properties after one time step
        props = time_step(dt, t[-1], r_arr, c[-1], dcdt_fn,
                            bc_specs_list, fixed_params)
        # stores properties at new time step in lists
        update_props(props, t, c)
        # print('t = {0:.2g} s of {1:.2g} s.'.format(t[-1], t_f))

    return t, c


def halve_grid(arr):
    """
    Halves the number of points in the grid by removing every other point.
    Assumes grid has N + 1 elements, where N is divisible by 2.
    The resulting grid will have N/2 + 1 elements.
    """
    print('halving grid')
    return arr[::2]


def manage_grid_halving(r_arr, c_arr, pts_per_grad, interp_kind=None):
    """
    Manages halving of grid by checking if the gradient is resolved by a
    sufficient number of grid points and, if not, halving the grid (decimating
    every other point) and adjusting the maximum time step accordingly.

    Note: the last array in the list of concentrations "c" is decimated in
    place.
    """
    halved = False
    # computes grid spacing at interface
    dr = r_arr[1] - r_arr[0]
    # only considers halving if array will be long enough after
    if 2*pts_per_grad < len(r_arr):
        dcdr = (c_arr[1]-c_arr[0]) / dr
        # width of region with desired number of points for gradient after halving
        delta_r = r_arr[2*pts_per_grad] - r_arr[0]
        # difference between minimum and maximum concentrations
        delta_c = np.max(c_arr) - c_arr[0]
        # halves grid if gradient has enough mesh points
        if dcdr*delta_r < delta_c:
            r_arr = halve_grid(r_arr)
            c_arr = halve_grid(c_arr)
            halved = True

    return halved, r_arr, c_arr


def init(R_min, R_o, N, eta_i, eta_o, d, L, Q_i, Q_o, p_s, dc_c_s_frac,
                        polyol_data_file, t_i=0):
    """
    Full initialization function for go().
    """
    # computes inner stream radius [m] and velocity [m/s]
    _, R_i, v = flow.get_dp_R_i_v_max(eta_i, eta_o, L, Q_i*uLmin_2_m3s,
                                        Q_o*uLmin_2_m3s, R_o, SI=True)
    t, c, r_arr, c_0, c_s, t_f, fixed_params = init_sub(R_min, R_i, R_o, N, d,
                                                        v, p_s, dc_c_s_frac,
                                                        polyol_data_file)

    return t, c, r_arr, c_0, c_s, t_f, fixed_params


def init_sub(R_min, R_i, R_o, N, d, v, p_s, dc_c_s_frac, polyol_data_file, t_i=0):
    """
    Initializes parameters for go().

    """
    # creates grid of radii from bubble radius to inner wall of capillary [m]
    r_arr = make_r_arr_lin(N, R_o, R_min=R_min)

    # creates initial concentration profile [kg CO2 / m^3 polyol-CO2]
    c_0 = np.zeros([N+1])
    c_s = polyco2.calc_c_s(p_s, polyol_data_file)
    c_0[r_arr <= R_i] = c_s
    dc = c_s*dc_c_s_frac
    t_f = d/v

    # initializes system
    t = [t_i]
    c = [c_0]

    # loads arrays for interpolation
    p_arr, D_sqrt_arr, D_exp_arr = polyco2.load_D_arr(polyol_data_file)
    c_s_arr, p_s_arr = polyco2.load_c_s_arr(polyol_data_file)
    interp_arrs = (c_s_arr, p_s_arr, p_arr, D_sqrt_arr, D_exp_arr)
    # stores fixed parameters
    fixed_params = (dc, interp_arrs)

    return t, c, r_arr, c_0, c_s, t_f, fixed_params


def make_all_r_arr(r_arr_data, t_list):
    """Makes all r arrays. ***UNTESTED***"""
    r_arr_list, r_arr_t_list = r_arr_data
    r_arr_ = [r_arr_list[np.where(np.logical_and(t >= r_arr_t_list,
                t < r_arr_t_list))[0][0]] for t in t_list]
    return r_arr_

def make_r_arr_lin(N, R_max, R_min=0):
    """Makes numpy array of radius with spacing dr from R_min to R_max."""
    r_arr = np.linspace(R_min, R_max, N+1)

    return r_arr

def make_r_arr_log(N, R_max, k=1.6, R_min=0, dr=None):
    """
    Makes numpy array of the radius with logarithmic spacing,
    dr, dr, 2dr, 4dr, 8dr, etc. from R_min to R_max.
    """
    # number of points in grid is log_k(delta_R). Use log formula for different
    # log bases
    if k == 1:
        r_arr = make_r_arr_lin(N, R_max, R_min=R_min)
    elif k < 1:
        print('k must be at least 1 in make_r_arr_log()')
        r_arr = None
    else:
        # computes d if not provided
        if dr is None:
            dr = (R_max - R_min)*(k-1) / (k**N - 1)
        # if d provided, computes k from d
        else:
            def f(k, N, R_max, R_min, dr):
                return dr - (k-1)/(k**N-1)*(R_max-R_min)

            args = (N, R_max, R_min, dr)
            k0 = ((R_max-R_min) / dr)**(1/(N-1))
            k = scipy.optimize.fsolve(f, k0, args=args)[0]
            print(k)
            # k = (1 + (R_max-R_min)/dr)**(1/N)

        # creates grid
        z = np.arange(0, N+1) # {0, 1,..., N}
        r_arr = (k**z-1)/(k-1)*dr + R_min

    return r_arr


def make_r_arr_log_res_end(N, R_max, end_pts, k=1.6, R_min=0, dr=None):
    """
    Same as make_r_arr_log() but adds more points to the last mesh element to
    help resolve the end of the domain.

    end_pts gives the number of uniformly spaced points that the last point will
    be split into.
    """
    # creates logarithmically spaced array
    r_arr_raw = make_r_arr_log(N, R_max, k=k, R_min=R_min, dr=dr)
    # adds more points to the last mesh element
    r_arr_end = np.linspace(r_arr_raw[-2], r_arr_raw[-1], end_pts+1)

    return np.concatenate((r_arr_raw[:-2], r_arr_end))


def make_r_arr_log2(N, R_max, R_min=0):
    """
    Makes numpy array of the radius with logarithmic spacing,
    dr, dr, 2dr, 4dr, 8dr, etc. from R_min to R_max.
    This one allows the user to specify the number of points. The minimum grid
    spacing is then grid length / 2^(N-1).
    """
    dr = (R_max - R_min) / 2**(N-1)
    r_list = [R_min, R_min+dr]
    for i in range(N-1):
        r_list += [r_list[-1] + (2**i)*dr]

    return np.array(r_list)


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


def regrid(grid, vals, grid_fn, N, R_max, args, interp_kind):
    """
    Interpolates valus over new grid.
    """
    print('regridding')
    # reduces the k value by fixed value
    grid_new = grid_fn(N, R_max, **args)
    grid_new[-1] = grid[-1]
    f = scipy.interpolate.interp1d(grid, vals, kind=interp_kind)
    vals_new = f(grid_new)

    return grid_new, vals_new


def remesh(grid, vals, th_lo, th_hi, unif_vals=False, second=False):
    """
    Creates a new mesh (and interpolates a new set of function values) to
    adapt to moving gradients of concentration. Pairs of points next-nearest
    neighbors with a difference in values below the low threshold will have the
    point between them removed. Pairs neighboring points with a difference in
    values above the high threshold will have a point added in the middle.

    Currently only works in 1D.

    Based on an informal discussion with Harsha Reddy (Caltech).

    Failures:
    - CubicSpline leads to instability (interpolates values that exceed bulk)
    - thresholding based on magnitude of second derivative is uncalibrated

    Parameters
    ----------
    grid : N x 1 array-like
        grid points (x values) of the mesh
    vals : N x 1 array-like
        function values at the grid points
    th_lo : float
        lower threshold for difference in values of consecutive grid points,
        below which a point is removed
    th_hi : float
        higher threshold for difference in values of consecutive grid points,
        above which points are added until the threshold is no longer exceeded
    unif_vals : bool, optional
        If True, uniformly spaces added points in value instead of in grid
        points (Default is False)

    Returns
    -------
    grid : N x 1 numpy array
        remeshed grid points
    vals : N x 1 numpy array
        function values (some interpolated with cubic spline) of remeshed grid
    """
    # initially has not remeshed
    remeshed = False
    # computes difference in consecutive values (length N-1)
    diff_arr = np.abs(np.diff(vals))

    # TODO make less heuristic
    if second:
        second_deriv = fd.d2ydx2_non_1st(np.asarray(vals), np.asarray(grid))
        diff_arr[:-1] += np.sqrt(np.abs(second_deriv)*5/1E11)
        print(diff_arr)

    # restricts remeshing to points with positive slope
    inds_nonpos = np.where(diff_arr <= 0)[0]
    if len(inds_nonpos) > 0:
        # index of last point to consider for remeshing; +1 to include last pt
        end = inds_nonpos[0]+1
        # stores original values to re-append later
        grid_end = grid[end:]
        vals_end = vals[end:]
        # cuts grid and vals arrays
        grid = grid[:end]
        vals = vals[:end]
        # cuts difference array as well to prevent removing points past end
        diff_arr = diff_arr[:end]


    # identifies where to add points (large gradient)
    inds_add = np.where(diff_arr > th_hi)[0]

    # adds points where needed
    if len(inds_add) > 0:
        # records remeshing
        remeshed = True
        # evenly spaces points in value and interpolates grid points
        if unif_vals:
            # computes a cubic spline with 0 second derivative at the ends
            # ('natural' B.C.) for interpolating grid pts from values (inverse)
            f = scipy.interpolate.CubicSpline(vals, grid, bc_type='natural')
        # evenly spaces points in grid and interpolates value
        else:
            f = scipy.interpolate.CubicSpline(grid, vals, bc_type='natural')

        # counts number of points added to array
        pts_added = 0
        # adds points element by element
        for i in inds_add:
            # shifts index each time a point is added earlier in the array
            j = i + pts_added
            # gets the difference spanned by this mesh elemnt (original indices)
            diff = diff_arr[i]
            n_pts_to_add = int(diff / th_hi)

            # evenly spaces points in value and interpolates grid points
            if unif_vals:
                # computes evenly spaced values between pair of original values
                vals_to_add = np.linspace(vals[j], vals[j+1], n_pts_to_add+2)[1:-1]
                # computes interpolated grid points from evenly spaced values
                pts_to_add = f(vals_to_add)
            # evenly spaces points in grid and interpolates value
            else:
                # computes evenly spaced grid points between pair of originals
                pts_to_add = np.linspace(grid[j], grid[j+1], n_pts_to_add+2)[1:-1]
                # interpolates values from grid points to be added
                # vals_to_add = f(pts_to_add)
                vals_to_add = np.interp(pts_to_add, grid, vals)

            # adds new grid point and value; +1 to be in middle of prev pts
            grid = np.insert(grid, j+1, pts_to_add)
            vals = np.insert(vals, j+1, vals_to_add)
            # increments total points added to grid
            pts_added += n_pts_to_add

    # computes central difference (2 points shorter than vals, grid
    diff_cd = np.abs(np.asarray(vals[2:]) - np.asarray(vals[:-2]))
    # identifies indices with small gradients to remove from grid
    # shifts by 1 since gradient has first point removed
    inds_rem = np.where(diff_cd < th_lo)[0] + 1
    # removes points
    if len(inds_rem) > 0:
        # records remeshing
        remeshed = True
        grid = np.delete(grid, inds_rem)
        vals = np.delete(vals, inds_rem)

    # restores cut parts of arrays
    if len(inds_nonpos) > 0:
        # # TODO tidy up, decompose, optimize
        # dr = grid[-1] - grid[-2]
        # n = int((grid_end[0] - grid[-1])/dr)
        # grid_mid = np.linspace(grid[-1], grid_end[0], n+2)
        # vals_mid = vals_end[0]*np.ones([len(grid_mid)])
        # grid = np.concatenate((grid[:-1], grid_mid, grid_end[1:]))
        # vals = np.concatenate((vals[:-1], vals_mid, vals_end[1:]))
        grid = np.concatenate((grid, grid_end))
        vals = np.concatenate((vals, vals_end))

    return remeshed, grid, vals


def remesh_once_manual(grid, vals, i_remesh=1, interp_kind='linear', dk=0.2,
                        k_min=1.2):
    """
    Remeshes once using manually selected time point for remeshing and grid to
    remesh to. Based on discussion with JAK on Feb. 10, 2021.
    """
    remeshed = False
    k = np.round((grid[2] - grid[1]) / (grid[1] - grid[0])*20)/20
    if k==k_min:
        return remeshed, grid, vals

    val_lo = vals[0]
    val_hi = vals[-1]
    range = val_hi - val_lo
    i_diffn = np.where(vals - val_lo >= range*(1-1/np.exp(1)))[0][0]
    if i_diffn > i_remesh:
        remeshed = True
        N = len(grid)
        R_max = np.max(grid)
        args = {'k' : k-dk}
        grid_new, vals_new = regrid(grid, vals, make_r_arr_log, N, R_max, args, interp_kind)
        if np.any(vals_new > val_hi):
            print('interpolation exceeded bulk value')
            vals_new[vals_new > val_hi] = val_hi

        grid = grid_new
        vals = vals_new

    return remeshed, grid, vals


def remesh_curv(grid, vals, interp_kind='linear', small_val=1):
    """
    Remeshes to concentrate points where the curvature is highest.

    Sources
    -------
    1. https://www.mathworks.com/matlabcentral/answers/518408-generate-a-mesh-
    with-unequal-steps-based-on-a-density-function (suggested by C. Balzer)
    """
    remeshed = True
    # even spacing
    x = np.linspace(grid[0], grid[-1], len(grid))
    dval = np.max(vals) - np.min(vals)
    dx = (x[-1] - x[0]) / len(x) # approximate scale for grid spacing
    # calculates density of mesh points based on curvature
    curv = np.abs(fd.d2ydx2_non_1st(vals, grid))/(dval/dx**2)
    grad = np.abs(fd.dydx_non_1st(np.asarray(vals), np.asarray(grid)))/(dval/dx)
    rho = np.concatenate((np.array([0]),  curv + grad**6, np.array([0])))
    # replaces zeros with small values
    rho[rho <= 1E-3] = small_val

    # computes cumulative density function
    cdf = np.cumsum(rho)
    # evenly spaced cumulative density of curvature
    eq_smpl = np.linspace(cdf[0], cdf[-1], len(cdf))

    # calculates new mesh
    f_grid = scipy.interpolate.interp1d(cdf, x, kind=interp_kind)
    grid_new = f_grid(eq_smpl)

    # interpolates corresponding values
    f_vals = scipy.interpolate.interp1d(grid, vals, kind=interp_kind)
    vals_new = f_vals(grid_new[1:-1])
    vals_new = np.concatenate((np.array([vals[0]]), vals_new, np.array([vals[-1]])))
    # checks that values fit within limits
    if np.any(vals_new > np.max(vals)):
        print('interpolation exceeded bulk value')
        vals_new[vals_new > np.max(vals)] = np.max(vals)

    return remeshed, grid_new, vals_new


def remesh_dr(grid, vals, i_remesh=1, interp_kind='linear', dr_mult=2,
                        dr_max=5E-7):
    """
    Remeshes once using manually selected time point for remeshing and grid to
    remesh to. Based on discussion with JAK on Feb. 10, 2021.
    """
    remeshed = False
    dr = grid[1] - grid[0]
    if dr > dr_max:
        return remeshed, grid, vals

    val_lo = vals[0]
    val_hi = vals[-1]
    range = val_hi - val_lo
    i_diffn = np.where(vals - val_lo >= range*(1-1/np.exp(1)))[0][0]
    if i_diffn > i_remesh:
        remeshed = True
        N = len(grid)
        R_max = np.max(grid)
        # reduces the k value by fixed value
        grid_new = make_r_arr_log(N, R_max, dr=dr_mult*dr)
        grid_new[-1] = grid[-1]
        f = scipy.interpolate.interp1d(grid, vals, kind=interp_kind)
        vals_new = f(grid_new)
        if np.any(vals_new > val_hi):
            print('interpolation exceeded bulk value')
            vals_new[vals_new > val_hi] = val_hi

        grid = grid_new
        vals = vals_new

    return remeshed, grid, vals


def time_step(dt, t_prev, r_arr, c_prev, dcdt_fn, bc_specs_list, R, fixed_params,
                apply_bc_first=True):
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

    # applies boundary conditions
    if apply_bc_first:
        for bc_specs in bc_specs_list:
            apply_bc(c_prev, bc_specs)

    # increments time forward [s]
    t_curr = t_prev + dt
    # no adaptive time-stepping
    dt_curr = dt
    # extracts array of relevant concentrations
    c_prev_arr = np.asarray(c_prev)

    # updates concentration with explicit Euler method
    c_curr_arr = c_prev_arr[1:-1] + dt*dcdt_fn(r_arr, c_prev_arr, R, fixed_params)
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
