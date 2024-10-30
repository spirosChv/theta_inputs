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


def gridfield(theta, lambda_var, xo, yo, x, y):
    """
    Generate grid-like fields in a specific location.
    Parameters
    ----------
    theta : float
        DESCRIPTION.
    lambda_var : float
        DESCRIPTION.
    xo : int
        x-coordinate of the place field.
    yo : int
        y-coordinate of the place field.
    x : int
        x point in space.
    y : int
        y point in space.
    Returns
    -------
    g : float
        firing probability at point x, y based on place field (xo, yo).
    """
    th1 = np.array(
        [
        np.cos(theta),
        np.sin(theta)
        ]
    ).reshape(-1, 1)
    th2 = np.array(
        [
        np.cos(theta + np.pi/3),
        np.sin(theta + np.pi/3)
        ]
    ).reshape(-1, 1)
    th3 = np.array(
        [
        np.cos(theta + 2*np.pi/3),
        np.sin(theta + 2*np.pi/3)
        ]
    ).reshape(-1, 1)

    x -= xo
    y -= yo

    y /= float(lambda_var)
    x /= float(lambda_var)

    p = np.array([x, y]).reshape(-1, 1)

    g = (1/4.5) * (np.cos(np.dot(p.T, th1)) +
                   np.cos(np.dot(p.T, th2)) + np.cos(np.dot(p.T, th3)) +
                   1.5).item()

    return g


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
    #cm_data = np.loadtxt('parula.txt')
    #parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

    probs = probabilities[f'place_field_{field}']
    fig, axes = plt.subplots(nrows=dim[0], ncols=dim[1], figsize=(12, 8))
    for cnt, ax in enumerate(axes.flat):
        im = ax.imshow(probs[cnt, :, :].reshape(1, -1),
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
    a : numpy.ndarray
        The input array.
    N : int
        Number of bins.
    method : str
        The method of binning to be applied.

    Raises
    ------
    ValueError
        If N is not int.

    Returns
    -------
    numpy.ndarray
        The binned array.

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
    spiketimes : numpy.ndarray
        Array with floats represents the spike times.
    csum : TYPE
        DESCRIPTION.
    npath_x : int
        The x-axis lim.
    npath_y : int
        The y-axis lim (for 2D spaces).

    Returns
    -------
    Z : numpy.ndarray
        The array of spikes in space.

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


def make_maps(path, spiketimes, xlim=200, ylim=1, num_bins=100):
    """
    Make spatial maps from spiketimes.

    Parameters
    ----------
    path : numpy.ndarray
        Two dimensional array of the coordinates.
    spiketimes : numpy.ndarray
        Spike-times in ms.
    xlim : int, optional
        The x limit of the track (length). The default is 200.
    ylim : int, optional
        The y limit of the track (width). The default is 1.
    num_bins : int, optional
        The number of bins. The default is 100.

    Returns
    -------
    Zmean : numpy.ndarray
        Heatmap with the firing rate as a function of the position.

    """

    time_array = np.bincount(path[:, 0])
    csum = np.cumsum(time_array)

    Z = spike_map(spiketimes, csum, npath_x=xlim, npath_y=ylim)
    Zbinned = binning(Z, N=num_bins, method='summing')

    # Gaussian filter parameters
    sigma_c = 5.0/(xlim/num_bins)
    truncate_c = 30.0/(xlim/num_bins)

    time_array_sec = (time_array/1000.0).reshape(Z.shape)
    time_binned = binning(time_array_sec, num_bins, method='summing')

    # Gaussian smoothing
    time_array_filtered = gaussian_filter1d(
        time_binned, axis=0,
        sigma=sigma_c,
        mode='nearest',
        truncate=truncate_c
    )

    Z_filtered = gaussian_filter1d(
        Zbinned,
        axis=0,
        sigma=sigma_c,
        mode='nearest',
        truncate=truncate_c
    )

    Z_averaged = np.divide(Z_filtered, time_array_filtered)

    return Z_averaged


def visualize_spiketimes(spikelist, type='CA3 inputs'):
    """
    Raster plot with the spike times.

    Parameters
    ----------
    spikelist : list
        Presynaptic cells' spiketimes.
    type : str
        Define the type of inputs. The default is `CA3 inputs`.

    Returns
    -------
    None.

    """
    plt.style.use(my_style())
    plt.figure()
    for i, spk in enumerate(spikelist):
        plt.scatter(spk, np.ones((len(spk)))*i, color='k')
    plt.ylabel('cell ID')
    plt.xlabel('time (ms)')
    plt.title(type)
    plt.show()
