"""objproc.py contains functions used for processing objects
generated by the object-tracking algorithm in
bubbletracking_koe/src/main.py

Author: Andy Ylitalo
Date: February 26, 2022
"""

# 3rd party libraries
import numpy as np

# custom libraries
import sys
sys.path.append('../../libs/')
import fn
from conversions import *



def calc_L(obj):
    """
    Calculates the length of the object at all observation points [um].

    Parameters
    ----------
    obj : dictionary
        TrackedObj converted to dictionary from bubbletracking_koe

    Returns
    -------
    L_list : list of floats
        Length of object in each frame with an observation [um].
    """
    # gets bounding box
    bbox_list = obj['props_raw']['bbox']
    # gets unit conversion
    pix_per_um = obj['metadata']['pix_per_um']
    # computes length
    L_list = [(c_hi - c_lo)/pix_per_um for _, c_lo, _, c_hi in bbox_list]

    return L_list


def calc_t(obj, d, v_max):
    """
    Computes time since entering observation capillary [s].

    Parameters
    ----------
    obj : dictionary
        TrackedObj converted to dictionary from bubbletracking_koe
    d : float
        Distance along observation capillary at which video was taken [m].
    v_max : float
        Estimated speed of flow at center (maximum) [m/s].

    Returns
    -------
    t : numpy array of floats
        Estimated time since flow entered observation capillary of each
        observation of the object [s].
    """
    # gets timeline of bubble (starts at zero) [s]
    frame_list = obj['props_raw']['frame']
    fps = obj['metadata']['fps']
    t_fov = (np.asarray(frame_list) - frame_list[0]) / fps
    # computes time of first observation relative to center of field of view
    num_col_frame = obj['metadata']['frame_dim'][1] # number of columns in frame
    pix_per_um = obj['metadata']['pix_per_um'] # conversion factor
    col_first_obs = obj['props_raw']['bbox'][0][1] # row_lo, *col_lo*, row_hi, col_hi
    # distance along capy of first observation [m]
    d_first_obs = d + ((col_first_obs - num_col_frame/2) / pix_per_um) * um_2_m
    t_first_obs = d_first_obs / v_max
    # adds field-of-view time to time of first observation [s]
    t = t_fov + t_first_obs

    return t


# def calc_tail(obj):
#     """Calculates the position of the tail of an object (minimum column)."""
#     # computes tail position [um]
#     bbox_list = obj['props_raw']['bbox']
#     pix_per_um = obj['metadata']['pix_per_um']
#     tail_list = [col_lo/pix_per_um for _, col_lo, _, _ in bbox_list]

#     return tail_list


def calc_W(obj):
    """
    Computes width (vertical extent) of objects since entering observation
    capillary [s].

    Parameters
    ----------
    obj : dictionary
        TrackedObj converted to dictionary from bubbletracking_koe

    Returns
    -------
    W_list : list of floats
        Width (vertical extent) of object in um.
    """
    # computes width [um]
    bbox_list = obj['props_raw']['bbox']
    pix_per_um = obj['metadata']['pix_per_um']
    W_list = [(r_hi - r_lo)/pix_per_um for r_lo, _, r_hi, _ in bbox_list]

    return W_list


def est_flow_speed(obj):
    """
    Estimates flow speed by linear extrapolation of speed from early
    growth to bubble of size 0.
    """
    L = np.asarray(calc_L(obj))
    v = np.asarray(obj['props_proc']['speed']) / obj['metadata']['pix_per_um']  * um_2_m # [m/s]
    idx = get_valid_idx(obj)
    _, flow_speed = np.polyfit(L[idx], v[idx], 1)

    return flow_speed


def get_conditions(metadata):
    """
    Gets conditions of experiment.
    ***LEGACY*** `get_vid_metadata` preferred.

    Parameters
    ----------
    metadata : dictionary
        Metadata from the data file of an experiment.

    Returns
    -------
    p_in : float
        Inlet pressure [Pa]
    p_sat : float
        Saturation pressure of polyol - CO2 mixture used in inner stream [Pa].
    p_est : float
        Estimated pressure at point of observation assuming linear pressure
        profile [Pa].
    d : float
        Distance from entrance of observation capillary to center of field of
        view [m].
    L : float
        Length of observation capillary [m].
    v_max : float
        Speed along centerline of inner stream estimated with flow eqns [m/s].
    t_center : float
        Estimated time for a fluid element along the centerline to travel from
        entrance of observation capillary to center of field of view [s].
    polyol : string
        Abbreviated name of polyol used in experiment.
    num : int
        Number of video (1-indexed in order that they were taken).
    """
    conds = {}
    # saturation pressure and units
    _, p_sat, units = fn.parse_vid_dir(metadata['vid_subdir'])
    conds['p_sat'] = p_sat
    # distance along channel [m]
    conds['d'] = metadata['object_kwargs']['d']
    # polyol
    vid_params = fn.parse_vid_path(metadata['vid_name'])
    conds['polyol'] = vid_params['prefix'].split('_')[0]
    conds['num'] = vid_params['num']
    # estimated pressure in given units
    conds['L'] = metadata['L'] # length of observation capillary [m]
    conds['p_in'] = -metadata['object_kwargs']['dp']
    conds['p_est'] = (1 - d/L)*p_in # [Pa]
    if units == 'bar':
        p_sat *= bar_2_Pa
    elif units == 'mpa':
        p_sat *= MPa_2_Pa
    elif units == 'psi':
        p_sat *= psi_2_Pa
    else:
        print('Units "{0:s}" not recognized.'.format(units))

    conds['v_max'] = metadata['object_kwargs']['v_max'] # centerline speed [m/s] est w flow eqns

    # computes time to reach center of field of view
    conds['t_center'] = d / v_max

    return conds


def get_v_init(obj):
    """
    Returns initial speed of object computed between first two observations. [m/s]
    See `est_flow_speed` for more robust estimation of flow speed.
    """
    return obj['props_proc']['speed'][0] / obj['metadata']['pix_per_um']  * um_2_m


def get_valid_idx(obj, L_frac=1):
    """
    Gets indices of frames where bubble is "valid" for fitting early growth,
    meaning the bubble is not on the border of the frame and is not too oblong.

    Parameters
    ----------
    obj : dictionary
        Bubble object converted to dictionary of image-processing data (see
        `bubbletracking_koe` github repo)
    L_frac : float, optional
        If bubble length (horizontal extent) is > this fraction of inner stream
        width, bubble is too oblong. Default is 1.

    Returns
    -------
    is_valid_arr : numpy array of bools
        True if observation of bubble at given index is valid; False if not.
    """
    # gets bubble size [um]
    L_bub = np.asarray(calc_L(obj))
    # gets valid indices (eliminates where on border or where bubble is too long
    not_on_border = np.logical_not(np.asarray(obj['props_raw']['on border']))
    R_i = obj['metadata']['R_i']*m_2_um # [um]
    not_too_long = L_bub < 2*R_i * L_frac
    is_valid_arr = np.logical_and(not_on_border, not_too_long)

    return is_valid_arr


def get_vid_metadata(metadata):
    """
    Gets the metadata from the video that is required for nucleation analysis.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    v = {}
    # saturation pressure and units
    _, p_sat, units = fn.parse_vid_dir(metadata['vid_subdir'])
    # distance along channel [m]
    v['d'] = metadata['object_kwargs']['d']
    # polyol
    vid_params = fn.parse_vid_path(metadata['vid_name'])
    v['polyol'] = vid_params['prefix'].split('_')[0]
    v['num'] = vid_params['num']
    # estimated pressure in given units
    v['L'] = metadata['L'] # length of observation capillary [m]
    v['p_in'] = -metadata['object_kwargs']['dp']
    v['p_est'] = (1 - v['d']/v['L'])*v['p_in'] # [Pa]
    if units == 'bar':
        p_sat *= bar_2_Pa
    elif units == 'mpa':
        p_sat *= MPa_2_Pa
    elif units == 'psi':
        p_sat *= psi_2_Pa
    else:
        print('Units "{0:s}" not recognized.'.format(units))
        
    v['p_sat'] = p_sat # [Pa]

    v['v_max'] = metadata['object_kwargs']['v_max'] # centerline speed [m/s] est w flow eqns
    v['n_frames'] = metadata['n_frames']
    v['eta_i'] = metadata['eta_i']
    v['eta_o'] = metadata['eta_o']
    v['R_o'] = metadata['R_o']
    v['frame_dim'] = metadata['bkgd'].shape
    v['fps'] = metadata['fps']
    v['flow_dir'] = metadata['object_kwargs']['flow_dir']
    v['pix_per_um'] = metadata['object_kwargs']['pix_per_um']
    v['R_i'] = metadata['object_kwargs']['R_i']
    v['v_interf'] = metadata['object_kwargs']['v_interf']
    
    return v    


def is_true_obj(obj, true_props=['inner stream', 'oriented', 'consecutive',
                'exited', 'centered']):
    """
    Returns True if object passes tests to be a "true" object (i.e., one worth
    analyzing and not noise) and False if not.

    Parameters
    ----------
    obj : dictionary
        TrackedObj object converted to dictionary of image-processing data (see
        `bubbletracking_koe` github repo)
    true_props : list of strings
        List of properties that must be True for object to be considered True.

    Returns
    -------
    is_true : bool
        True if object is true and False if not.
    """
    for prop in true_props:
        # if lacks one of the key props for a true object, not a true object
        try:
            props = obj['props_proc'][prop]
            if False in props:
                return False
        except:
            print('DATA ARE MISSING TRUE PROPERTIES. Please rerun analysis.')
            continue

    return True
