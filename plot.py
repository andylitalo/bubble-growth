"""
Methods for plotting related to bubble growth.

@date: May 19, 2020
@author: Andy Ylitalo
"""


# CONVERSIONS
s_2_ms = 1000
kg_2_g = 1E3
kg_2_ng = 1E12
kgm3_2_gmL = 1E-3
m_2_mm = 1000
m_2_um = 1E6
m_2_nm = 1E9
Pa_2_bar = 1E-5
Pa_2_MPa = 1E-6
Nm_2_mNm = 1000

import numpy as np
import matplotlib.pyplot as plt


def all_props(t, t_nuc, props_list, x_log=False, y_lim=[1E-3, 1E4], 
              title='Bubble of CO2 Along Channel in V360'):
    """
    Plots all properties during bubble growth trajectory. Assumes 
    input data are in SI units and converts to more convenient units
    so all values can be plotted on the same plot.
    """
    # extract properties to plot
    R, m, p, p_bubble, rho_co2, if_tension = props_list
    # shifts and scales time [ms]
    t = (np.array(t) - t_nuc)*s_2_ms
    # creates figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plots properties
    ax.plot(t, np.array(R)*m_2_um, label=r'$R(t)$ [$\mu$m]')
    ax.plot(t, np.array(m)*kg_2_ng, label=r'$m(t)$ [ng]')
    ax.plot(t, np.array(p)*Pa_2_MPa, label=r'$p(t)$ [MPa]')
    ax.plot(t, np.array(p_bubble)*Pa_2_MPa, '--', label=r'$p_{bubble}(t)$ [MPa]')
    ax.plot(t, np.array(rho_co2)*kgm3_2_gmL, label=r'$\rho_{CO2}(t)$ [g/mL]')
    ax.plot(t, np.array(if_tension)*Nm_2_mNm, label=r'$\gamma(t)$ [mN/m]')
    # sets y-scale as logarithmic
    ax.set_yscale('log')
    # sets x-scale as logarithmic if requested
    if x_log:
        ax.set_xscale('log')
    # formats plot
    ax.set_ylim(y_lim)
    ax.set_xlabel(r'$t-t_{nuc}$ [ms]', fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', labelsize=12)

    # moves legend outside box
    # puts legend outside of plot box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.65, box.height])
    legend_x = 1
    legend_y = 0.5
    plt.legend(loc='center left', fontsize=14, bbox_to_anchor=(legend_x, legend_y))
    
    return ax


def diff(x_list, y_list, label_list, x_shift, x_conv, y_conv, x_label,
           y_label, title, x_log=False, y_log=False, frac=False):
    """
    Same as series but reformats so all the y's are (y_list[i] - y_list[0]) / 
    y_list[0].
    This means there will be one fewer series than given (because it does not
    plot the trivial case of y_list[0] - y_list[0]).
    If x_list and y_list have N entries, then label_list should have N-1 
    (excludes the "ground truth"). If the label_list has length N, then the
    ground truth will be shown with the corresponding label.
    """
    x_true = np.array(x_list[0])
    y_true = np.array(y_list[0])
    x_diff_list = []
    y_diff_list = []
        
    if len(label_list) == len(y_list):
        # adds ground truth
        x_diff_list += [x_true]
        y_diff_list += [y_true]
        
    for i in range(1, len(y_list)):
        y_interp = np.interp(x_true, x_list[i], y_list[i])
        y_diff = np.abs(y_interp - y_true)
        if frac:
            y_diff /= y_true
        x_diff_list += [x_true]
        y_diff_list += [y_diff]
        
    return series(x_diff_list, y_diff_list, label_list, x_shift, x_conv, y_conv,
                  x_label, y_label, title, x_log=x_log, y_log=y_log)
        

def series(x_list, y_list, label_list, x_shift, x_conv, y_conv, x_label,
           y_label, title, x_log=False, y_log=False):
    # creates plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, label in enumerate(label_list):
        x = x_list[i]
        y = y_list[i]
        ax.plot((np.array(x)-x_shift)*x_conv, np.array(y)*y_conv, label=label)
     
    if x_log:
        ax.set_xscale('log')
    if y_log: 
        ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_title(title, fontsize=20)
    plt.legend()
    
    return ax