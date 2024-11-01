#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 09:37:20 2021.

@author: spiros
"""

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def poisson_spikes(t1, t2, N, rate=10, dt=0.1, seed=None):
    """
    Poisson spike generator.

    Parameters
    ----------
    t1 : float
        Simulation time start in miliseconds (ms).
    t2 : float
        Simulation time end in miliseconds (ms).
    N : int, optional
        Number of presynaptic spikes.
    rate : float, optional
        Mean firing rate in Hz. The default is 10.
    dt : float, optional
        Time step in ms. The default is 0.1.
    seed : int, optional
        The random seed value to initialize the number generation,
        The default is None.

    Returns
    -------
    spks : TYPE
        DESCRIPTION.

    """
    rng = np.random.default_rng(seed=seed)

    spks = []
    tarray = np.arange(t1, t2+dt, dt)
    for n in range(N):
        spkt = tarray[rng.random(len(tarray)) < rate*dt/1000.]  # Determine list of times of spikes
        idx = [n]*len(spkt)  # Create vector for neuron ID number the same length as time
        spkn = np.concatenate([[idx], [spkt]], axis=0).T  # Combine the lists
        if len(spkn) > 0:
            spks.append(spkn)
    spks = np.concatenate(spks, axis=0)
    return spks


def bursting_spikes(
    rate, nrun, N, delay, dt=0.1,
    noise=0.0, burst_num=2, burst_len=50, interburst=150,
    mode='random', save=False, visualize=False, fname='cell_'
    ):
    """
    Theta-modulated Poisson spike train.

    Parameters
    ----------
    rate : float
        Poisson-distribution mean firing rate (lambda) in Hz.
    nrun : int
        Simulated run.
    N : int
        Number of pre-synaptic artificial cells.
    delay : float
        Delay in miliseconds (ms). This shifts forward in time all spike times.
    dt : float
        Time step (ms).
    burst_num : int, optional
        The number of bursts. The default is 2.
    noise : float, optional
        Range 0 to 1. Fractional randomness. 0: deterministic, 1: intervals
        have poisson distribution. The default is 0.0.
    burst_len : float, optional
        Length of the burst in ms. The default is 50.
    interburst : float, optional
        Inter-burst interval in ms. The default is 150.
    mode : str, optional
        Either `random` for random burstlen in [50,100] or `constant`.
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
    rng = np.random.default_rng(seed=nrun)

    print(f'RUN: {nrun}')

    # Create the folder to store the outputs.
    if save:
        foldername = pathlib.Path(f'rate{rate}/run_{nrun}')
        os.makedirs(foldername, exist_ok=True)

    t3 = 0
    for i in range(burst_num):
        if mode == 'random':
            burst_len = rng.integers(low=50, high=100+1)
        t1 = t3
        t2 = t1 + burst_len
        t3 = t2 + interburst
        spks = poisson_spikes(t1, t2, N, rate, dt, seed=nrun)
        spks_noise = poisson_spikes(t2, t3, N, rate, dt, seed=nrun)
        spks_added = spks_noise[rng.random(spks_noise.shape[0]) < noise, :]
        spks = np.concatenate([spks, spks_added], axis=0)
        if i == 0:
            spiketimes = spks.copy()
        else:
            spiketimes = np.concatenate([spiketimes, spks], axis=0)

    spike_train = {}
    for s in range(N):
        # Spike times in ms
        spikes = np.round(spiketimes[spiketimes[:, 0] == s][:, 1], 1)
        # Add a `delay` in ms in the spiketimes.
        spikes = delay + spikes
        # Remove all occurences of elements with value greater than the time.
        spikes = spikes[spikes <= t2 + interburst]
        if save:
            # Write the spike in a text document.
            np.savetxt(
                f"{foldername}/{fname}{s}.txt",
                spikes,
                fmt='%10.1f',
                newline='\n'
            )

        spike_train[f"neuron_{s}"] = spikes

    if visualize:
        plot_bursts(
            spike_train,
            N,
            t2+interburst,
            delay,
            ids=N,
            seed=nrun,
            fname='cell_'
        )

    return spike_train


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


def theta_filtered_spikes(
        rate, nrun, N, time, delay,
        noise=0.0, f_theta=8, phi_theta=0,
        dt=0.1, save=False, visualize=False,
        fname='cell_'
    ):
    """
    Theta-modulated Poisson spike train.

    Parameters
    ----------
    rate : float
        Poisson-distribution mean firing rate (lambda) in Hz.
    nrun : int
        Simulated run.
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
    dt : float, optional
        Time integration step in miliseconds. The default in 0.1.
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
    rng = np.random.default_rng(seed=nrun)
    print(f'RUN: {nrun}')

    # Create the folder to store the outputs.
    if save:
        foldername = pathlib.Path(f'rate{rate}/run_{nrun}')
        os.makedirs(foldername, exist_ok=True)

    # Produce the poisson distributed spikes
    spiketimes = poisson_spikes(
        t1=0,
        t2=time,
        N=N,
        rate=rate,
        dt=dt,
        seed=nrun
    )

    spike_train = {}
    for s in range(N):
        # Spike times in ms
        spikes_neuron = np.round(spiketimes[spiketimes[:, 0] == s][:, 1], 1)
        # Make the spike train theta modulated
        spikes = []
        for spike in spikes_neuron:
            p = theta_prob(spike, f_theta, phi_theta)
            r_ = rng.random()
            if ((p > 0.7) and (r_ < p / 2.0)) or rng.random() < noise:
                spikes.append(spike)

        # Add a `delay` in ms in the spiketimes.
        spikes = delay + np.array(spikes)
        # Remove all occurences of elements with value greater than the time.
        spikes = spikes[spikes <= time]
        if save:
            # Write the spike in a text document.
            np.savetxt(
                f"{foldername}/{fname}{s}.txt",
                spikes,
                fmt='%10.1f',
                newline='\n'
            )

        spike_train[f"neuron_{s}"] = spikes

    if visualize:
        plot_spikes(
            spike_train,
            N,
            time,
            delay,
            f_theta,
            phi_theta,
            ids=N,
            seed=nrun,
            fname='cell_'
        )

    return spike_train


def plot_spikes(
        spike_train, neurons, time, delay,
        f_theta, phi_theta,
        ids, seed, fname='cell_'
    ):
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
    rng = np.random.default_rng(seed=seed)

    plt.figure(figsize=(12, 10))
    random_idx = rng.choice(range(neurons), size=ids, replace=False)
    for i, idx in enumerate(random_idx):
        plt.scatter(
            spike_train[f'neuron_{idx}'],
            i*np.ones_like(spike_train[f'neuron_{idx}']),
            color='k'
        )
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
    plt.hist(
        np.sort(mlist),
        density=True,
        bins=100,
        label='spike times'
    )
    plt.plot(
        spikes,
        prob,
        linewidth=2.5,
        linestyle='dashed',
        label='theta filter'
    )
    plt.xlabel('time [ms]')
    plt.ylabel('probability density')
    plt.xlim([0, time-delay])
    plt.xticks(
        ticks=range(0, time-delay, int(time/10)),
        labels=[str(ii+delay) for ii in range(0, time-delay, int(time/10))]
    )
    plt.legend()
    plt.show()


def plot_bursts(
        spike_train, neurons, time, delay,
        ids, seed, fname='cell_'
    ):
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
    ids : int
        Visualize cells in the raster plot. For all, `ids=neurons`.
    fname : str, optional
        Filename keyword of each artificial cell. The default is 'cell_'.

    Returns
    -------
    None.

    """
    # color pallette.
    rng = np.random.default_rng(seed=seed)

    plt.figure(figsize=(12, 10))
    random_idx = rng.choice(range(neurons), size=ids, replace=False)
    for i, idx in enumerate(random_idx):
        plt.scatter(
            spike_train[f'neuron_{idx}'],
            i*np.ones_like(spike_train[f'neuron_{idx}']),
            color='k'
        )
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

    plt.figure(figsize=(12, 10))
    plt.hist(
        np.sort(mlist),
        density=True,
        bins=100,
        label='spike times'
    )
    plt.xlabel('time [ms]')
    plt.ylabel('probability density')
    plt.xlim([0, time-delay])
    plt.xticks(
        ticks=range(0, time-delay, int(time/10)),
        labels=[str(ii+delay) for ii in range(0, time-delay, int(time/10))]
    )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    spikes = bursting_spikes(
        rate=5,
        nrun=2,
        N=50,
        delay=400,
        dt=0.1,
        noise=0.0,
        burst_num=100,
        burst_len=50,
        interburst=150,
        mode='random',
        save=True,
        visualize=True
    )


    spikes = theta_filtered_spikes(
        rate=20,
        nrun=1,
        N=50,
        time=2000,
        delay=400,
        noise=0.1,
        f_theta=8,
        phi_theta=0,
        dt=0.1,
        save=False,
        visualize=True,
        fname='cell_'
    )
