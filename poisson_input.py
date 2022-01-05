#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 09:37:20 2021.

@author: spiros
"""

import os
import brian2
import numpy as np
import matplotlib.pyplot as plt


def theta_prob(spike, f_theta, phi_theta):
    """
    Theta-like probability.

    Parameters
    ----------
    spike : float
        The spike time in miliseconds (ms).
    f_theta : float, optional
        Theta-cycle frequency in Hz. The default is 8.
    phi_theta : float, optional
        Theta-cycle phase in radians. The default is 0.

    Returns
    -------
    p : float
        The theta-like probability. Values in [0, 1].
    """
    p = (np.sin(2.0*np.pi*f_theta*spike/1000. + phi_theta) + 1.0)/2.0

    return p


def theta_filtered_spikes(rate, nrun, N, time, delay,
                          noise=0.0, f_theta=8, phi_theta=0,
                          save=False, visualize=False,
                          fname='cell_'):
    """
    Theta-modulated Poisson spike train.

    Parameters
    ----------
    rate : float
        Poisson-distribution mean firing rate (lambda) in Hz.
    nrun : int
        Simulated run. Determines the seed.
    N : int
        Number of pre-synaptic artificial cells.
    time : float
        Simulation time in miliseconds (ms).
    delay : float
        Delay in miliseconds (ms). This shifts forward in time all spike times.
    noise : float, optional
        Range 0 to 1. Fractional randomness. 0: deterministic, 1: intervals
        have poisson distribution. The default is 0.0.
    f_theta : float, optional
        Theta-cycle frequency in Hz. The default is 8.
    phi_theta : float, optional
        Theta-cycle phase in radians. The default is 0.
    save : bool, optional
        Either to save or not the spike times in txt files.
        The default is False.
    visualize : bool, optional
        Either to visualize the spike train or not. The default is False.
    fname : str, optional
        Filename keyword of each artificial cell. The default is 'cell_'.

    Returns
    -------
    spikes_train : dict
        Dictionary with the pre-synaptic spike trains.
        keys: 'neuron_i', i-th presynaptic artificial cell.

    """
    # Random seed for reproducibility.
    brian2.seed(nrun)
    np.random.seed(nrun)

    print(f'RUN: {nrun}')

    # Create the folder to store the outputs.
    if save:
        foldername = f'rate{rate}/run_{nrun}'
        os.makedirs(foldername, exist_ok=True)

    time_input = time * brian2.ms  # simulation time
    P = brian2.PoissonGroup(N, rates=rate * brian2.Hz)
    S = brian2.SpikeMonitor(P)

    # Run the simulation
    brian2.run(time_input, report='text', report_period=10 * brian2.second)

    spike_train = {}
    for s in range(len(S.spike_trains().keys())):
        # Spike times in ms
        spiketimes = np.round(1000*S.spike_trains()[s]/brian2.second, 1)
        # Make the spike train theta modulated
        spikes = []
        for spike in spiketimes:
            p = theta_prob(spike, f_theta, phi_theta)
            r_ = np.random.rand()
            if ((p > 0.7) and (r_ < p / 2.0)) or np.random.rand() < noise:
                spikes.append(spike)

        # Add a `delay` in ms in the spiketimes.
        spikes = delay + np.array(spikes)
        # Remove all occurences of elements with value greater than the time.
        spikes = spikes[spikes <= time]
        if save:
            # Write the spike in a text document.
            np.savetxt(f"{foldername}/{fname}{s}.txt",
                       spikes,
                       fmt='%10.1f',
                       newline='\n')

        spike_train[f"neuron_{s}"] = spikes

    if visualize:
        plot_spikes(spike_train, N, time, delay, f_theta, phi_theta,
                    ids=N, fname='cell_')

    return spike_train


def plot_spikes(spike_train, neurons, time, delay, f_theta, phi_theta,
                ids, fname='cell_'):
    """
    Plot spike trains.

    Parameters
    ----------
    spikes_train : dict
        Dictionary with the pre-synaptic spike trains.
        keys: 'neuron_i', i-th presynaptic artificial cell.
    neurons : int
        Number of pre-synaptic artificial cells.
    delay : float
        Delay in miliseconds (ms). This shifts forward in time all spike times.
    f_theta : float, optional
        Theta-cycle frequency in Hz. The default is 8.
    phi_theta : float, optional
        Theta-cycle phase in radians. The default is 0.
    ids : int
        Visualize cells in the raster plot. For all, `ids=neurons`.
    fname : str, optional
        Filename keyword of each artificial cell. The default is 'cell_'.

    Returns
    -------
    None.

    """
    # color pallette.
    plt.style.use("seaborn-colorblind")
    # load the custom style.
    plt.style.use('style_.txt')

    plt.figure(figsize=(12, 10))
    random_idx = np.random.choice(range(neurons), size=ids, replace=False)
    for i, idx in enumerate(random_idx):
        plt.scatter(spike_train[f'neuron_{idx}'],
                    i*np.ones_like(spike_train[f'neuron_{idx}']),
                    color='k')
    plt.axvline(x=delay, color='r', linestyle='dashed', label='delay')
    plt.xlabel('time [ms]')
    plt.ylabel('pre-synaptic neuron id')
    plt.xlim([0, time])
    plt.legend()
    plt.show()

    # Add all spike times in one list.
    mlist = []
    for j in spike_train.keys():
        mlist += (spike_train[j] - delay).tolist()

    # Calculate the theta probability and normalize so that
    # it has area under the curve equal to one.
    spikes = np.arange(0, time, 0.05)
    prob = theta_prob(spikes, f_theta, phi_theta)
    prob /= np.trapz(prob, dx=0.05)

    plt.figure(figsize=(12, 10))
    plt.hist(np.sort(mlist), density=True, bins=100,
             label='spike times')
    plt.plot(spikes, prob, linewidth=2.5, linestyle='dashed',
             label='theta filter')
    plt.xlabel('time [ms]')
    plt.ylabel('probability density')
    plt.xlim([0, time-delay])
    plt.xticks(ticks=range(0, time-delay, int(time/10)),
               labels=[str(ii+delay) for ii in range(0, time-delay,
                                                     int(time/10))])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    spikes = theta_filtered_spikes(rate=12,
                                   nrun=1,
                                   N=1000,
                                   time=2000,
                                   delay=400,
                                   noise=0.0,
                                   f_theta=8,
                                   phi_theta=0,
                                   save=True,
                                   visualize=True)