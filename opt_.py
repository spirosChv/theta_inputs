#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:07:10 2022

@author: spiros
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap


def my_style():
    """
    Create custom plotting style.

    Returns
    -------
    my_style : dict
        Dictionary with matplotlib parameters.

    """
    # color pallette
    my_style = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "font.family": "Times New Roman",
        "font.weight": "bold",
        # Use 16pt font in plots, to match 16pt font in document
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 2.5,
        "lines.markersize": 10.0,
        "lines.linewidth": 2.5,
        "xtick.major.width": 2.2,
        "ytick.major.width": 2.2,
        "axes.labelweight": "bold",
        "axes.spines.right": False,
        "axes.spines.top": False
    }

    return my_style


def visualize_inputs(field, probabilities, dim):
    """
    Visualize the theoretical input priobabilities.

    Parameters
    ----------
    field : int
        The inputs corresponding to a field.
    probabilities : dict
        Dictionary with all probabilites stored.
        Each element is a 3D np.ndarray with nsyn x x_path x y_path dimensions.
    dim: tuple
        Subplot dimensions. dim[0] denotes rows, while dim[1] denotes cols.

    Returns
    -------
    None.

    """
    plt.style.use(my_style())    

    # Adopted from https://github.com/BIDS/colormap/blob/master/parula.py
    cm_data = np.loadtxt('parula.txt')
    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

    probs = probabilities[f'place_field_{field}']
    fig, axes = plt.subplots(nrows=dim[0], ncols=dim[1], figsize=(12, 8))
    for cnt, ax in enumerate(axes.flat):
        im = ax.imshow(probs[cnt, :, :].reshape(1, -1),
                       cmap=parula_map,
                       extent=[0, 100, 0, 0.2],
                       aspect=100,
                       vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_title(f'grid cell {cnt+1}')
        ax.set_xlabel('position (cm)')
    clb = fig.colorbar(im, ax=axes.ravel().tolist())
    clb.set_label('probability')
    plt.show()

    # Theoretical place-field
    plt.figure()
    probs_pc = np.sum(probs, axis=0, keepdims=True)
    plt.imshow(probs_pc.T/np.max(probs_pc),
               cmap=parula_map,
               extent=[0, 100, 0, 0.2],
               aspect=100,
               vmin=0, vmax=1)
    clb = plt.colorbar()
    clb.set_label('probability')
    plt.yticks([])
    plt.title('Theoretical place cell')
    plt.xlabel('position (cm)')
    plt.show()


def binning(a, N, method):
    """
    Binning a given vector.

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if not isinstance(N, int):
        raise ValueError('Binning size must be integer.')

    a = a.squeeze()
    L = a.shape[0]
    rem = L % N

    if rem != 0:
        raise ValueError('Not a valid binning size.')

    # find the step
    step = int(L/N)

    b = np.zeros(N)

    cnt = 0
    for i in range(0, L, step):
        if method == 'summing':
            b[cnt] = np.sum(a[i:i+step])
        elif method == 'mean':
            b[cnt] = np.mean(a[i:i+step])
        cnt += 1

    return b.reshape(-1, 1)


def spike_map(spiketimes, csum, npath_x, npath_y):
    """
    Transform spike times into a rate map.

    Parameters
    ----------
    spiketimes : TYPE
        DESCRIPTION.
    csum : TYPE
        DESCRIPTION.
    npath_x : TYPE
        DESCRIPTION.
    npath_y : TYPE
        DESCRIPTION.

    Returns
    -------
    Z : TYPE
        DESCRIPTION.

    """
    Z = np.zeros((npath_x, npath_y))

    if len(spiketimes) != 0:
        for spike in spiketimes:
            # Find the last positive index --> this is the mapping from
            # time to space

            if spike > csum[-1]:
                continue
            else:
                idxs = np.argwhere((spike - csum) > 0)
                if idxs.shape[0] == 0:
                    idx = 0
                else:
                    idx = idxs.shape[0]

                Z[idx, :] += 1

    if Z.shape[0] != npath_x or Z.shape[1] != npath_y:
        print('Error in Z dimensions')

    return Z


def make_maps(path, spiketimes, xlim=200, ylim=1, nBins=100):
    """
    Make spatial maps from spiketimes.

    Parameters
    ----------
    path : np.ndarray
        Two dimensional array of the coordinates.
    spiketimes : np.ndarray
        Spike-times in ms.
    xlim : int, optional
        The x limit of the track (length). The default is 200.
    ylim : int, optional
        The y limit of the track (width). The default is 1.
    nBins : int, optional
        The number of bins. The default is 100.

    Returns
    -------
    Zmean : np.ndarray
        Heatmap with the firing rate as function of the position.

    """

    time_array = np.bincount(path[:, 0])
    csum = np.cumsum(time_array)

    Z = spike_map(spiketimes, csum, npath_x=xlim, npath_y=ylim)
    Zbinned = binning(Z, N=nBins, method='summing')

    # Gaussian filter parameters
    sigma_c = 5.0/(xlim/nBins)
    truncate_c = 30.0/(xlim/nBins)

    time_array_sec = (time_array/1000.0).reshape(Z.shape)
    time_binned = binning(time_array_sec, nBins, 'summing')

    # Gaussian smoothing
    time_array_fil = gaussian_filter1d(time_binned, axis=0,
                                       sigma=sigma_c,
                                       mode='nearest',
                                       truncate=truncate_c)

    Zsmoothed = gaussian_filter1d(Zbinned, axis=0, sigma=sigma_c,
                                  mode='nearest', truncate=truncate_c)

    Zmean = np.divide(Zsmoothed, time_array_fil)

    return Zmean