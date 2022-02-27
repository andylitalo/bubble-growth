"""fn.py contains useful functions.

Author: Andy Ylitalo
Date: February 24, 2022
"""

import re
import os



def parse_vid_dir(vid_dir):
    """
    Copied from the repo `andylitalo/bubbletracking_koe` in `src/genl/fn.py`.
    Parses video folder of format <yyyymmdd>_<p_sat><units>
    
    Parameters
    ----------
    vid_folder : string
        Directory in which video is saved (excludes filename)
    
    Returns
    -------
    date : string
        Date of experiment <yyyymmdd>
    p_sat : int
        Pressure at which inner stream was saturated 
    p_sat_units : string
        Units of p_sat
    """
    result = re.search('20[0-9]{6}_[0-9]{2,}[a-zA-z]{3,}', vid_dir)
    if result is None:
        return None, None, None
    else:
        vid_str = result.group()
        date = re.search('^20[0-9]{6}', vid_str).group()
        p_sat = int(re.search('_[0-9]{2,}', vid_str).group()[1:])
        units = re.search('[a-zA-z]{3,}', vid_str).group()

        return date, p_sat, units
    
    
def parse_vid_path(vid_path):
    """
    Copied from the repo `andylitalo/bubbletracking_koe` in `src/genl/fn.py`.
    Parses the video filepath to extract metadata.
        
    Parameters
    ----------
    vid_path : string
        Filepath to video of form <directory>/
        <prefix>_<fps>_<exposure time>_<inner stream flow rate [uL/min]>_
        <outer stream flow rate [uL/min]>_<distance along observation capillary [mm]>_
        <magnification of objective lens>_<number of video in series>.<ext>
    
    Returns
    -------
    params : dictionary
        Items: 'prefix' (string, usually polyol and gas), 
            'fps' (int, frames per second), 'exp_time' (float, exposure time [us]),
            'Q_i' (float, inner stream flow rate [ul/min]), 
            'Q_o' (int, outer stream flow rate [uL/min]), 
            'd' (int, distance along observation capillary [mm]),
            'mag' (int, magnification of objective lens), 'num' (int, number of video in series)
    """
    i_start = vid_path.rfind(os.path.sep)
    vid_file = vid_path[i_start+1:]
    # cuts out extension and splits by underscores
    match = re.search('_[0-9]{4,5}_[0-9]{3}(-[0-9]{1})?_[0-9]{3}_[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]+', vid_file)
    span = match.span()
    prefix = vid_file[:span[0]]
    param_str = match.group()
    tokens = param_str.split('_')

    params = {'prefix' : prefix,
              'fps' : int(tokens[1]),
              'exp_time' : read_dash_decimal(tokens[2]),
              'Q_i' : read_dash_decimal(tokens[3]),
              'Q_o' : int(tokens[4]),
              'd' : int(tokens[5]),
              'mag' : int(tokens[6]),
              'num' : int(tokens[7])}

    return params


def read_dash_decimal(num_str):
    """
    Copied from the repo `andylitalo/bubbletracking_koe` in `src/genl/fn.py`.
    Reads string as float where dash '-' is used as decimal point.
    """
    result = 0
    if '-' in num_str:
        val, dec = num_str.split('-')
        result = int(val) + int(dec)/10.0**(len(dec))
    else:
        result = int(num_str)

    return result
