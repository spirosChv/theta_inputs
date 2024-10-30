#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:08:09 2018.

Modified on Thu Apr 14 11:21:21 2022 to account for place-like generation.

@author: spiros
"""

import os
import numpy as np
from pathlib import Path
from opt_ import visualize_inputs, visualize_spiketimes, gridfield
from poisson_input import poisson_spikes

# Choose coordinates +1 for the value
# e.g., if you want 100 --> myx = 101
# Apart from the case that one dimension is one

# Set the ranodm seed for each run
my_run = 1
rng = np.random.default_rng(seed=my_run)

visualize = True
###############################################################################
################# P A T H   C O N S T R U C T I O N  ##########################
###############################################################################

# Choose grid dimensions
xlim = 200  # Choose +1 the points you want (apart from 1)
ylim = 1

# Place field coordinations; all to all combinations
x_array = [100] # range(0, xlim, 5)  # x-coordinates of place field
y_array = [1]  # y-coordinates of place field

print(f'Simulating RUN... {my_run}\n')

maindir_grid = 'grid_like_inputs'
maindir_place = 'place_like_inputs'

dirname_grid = Path(f'{maindir_grid}/run_{my_run}')
if not os.path.exists(dirname_grid):
    os.makedirs(dirname_grid)

dirname_place = Path(f'{maindir_place}/run_{my_run}')
if not os.path.exists(dirname_place):
    os.makedirs(dirname_place)

nx, ny = 0, 0  # initial values
k = 1  # step-size (1 cm)

# initial location at 0, 0
p0 = [nx, ny]

# calculate the total spatial locations
npoints = xlim*ylim

# How many random stops
nstops = rng.integers(low=0, high=10)
print(f'Number of Random Stops: {nstops}')

# locations of random stops
rand_stops = np.sort(
    rng.choice(
        range(1, xlim-2),
        size=nstops,
        replace=False
    )
)

# random stops are +/- 2 cm
rand_stops_minus = [i-1 for i in rand_stops]
rand_stops_plus = [i+1 for i in rand_stops]
rand_stops = np.concatenate([rand_stops, rand_stops_minus + rand_stops_plus])

path = []  # initialization
for i in range(1, npoints + 1):
    if (ny > ylim):
        break

    # in random stop location stay longer
    if i in rand_stops:
        sigma = 25
        mu = 200
        print('Random Stop')
    else:
        sigma = 15
        mu = 50

    # random time at each point, clipped at 10 (lower bound)
    tim_at_point = np.clip(
        int(rng.normal(loc=mu, scale=sigma)),
        a_min=10,
        a_max=None,
    )

    # store the number of timesteps stayed in the same location
    path += [p0]*tim_at_point

    # The maze is a linear track, thus increase only x-axis (nx) location
    nx += 1
    p0 = [nx, ny]

# save the path
path = np.array(path)
np.savetxt(
    Path(
        f'{dirname_grid}/path.txt'),
        path,
        fmt='%.0f',
        delimiter=' '
    )
print('Path walk has been finished.')

###############################################################################
#  G R I D  A N D  P L A C E  L I K E    I N P U T S  #########################
###############################################################################
tdelay = 300  # delay in ms
tref = 3  # refractoriness of presynaptic cells in ms
tca3_delay = 17  # delay of ca3 signal in ms
r_min_ca3, r_max_ca3 = 25, 50  # firing rate limits from CA3
sigma_ca3 = 2  # sigma of randomness in ca3 delay in ms for different ca3 cells
sigma_cell = 0.33  # extra randomness for the same presynaptic cell normal with sigma 0.33, [-0.99, 0.99]
nsynEC = 20  # Number of presynaptic cells per place field
nsynCA3 = 20  # Number of presynaptic cells per place field
noiseEC = 0.01  # noise levels
noiseCA3 = 0.01  # noise levels
jitterEC = 0.2  # jitter in EC spiketimes
jitterCA3 = 0.2  # jitter in CA3 spiketimes

theta_freq = 8  # in Hz
theta_phase = 0  # in degrees

my_field = 0
d_all = {}
spikesEC, spikesCA3 = [], []
for xxx in x_array:
    for yyy in y_array:

        my_field += 1
        folder_grid = Path(f'{dirname_grid}/place_field_{my_field}')
        if not os.path.exists(folder_grid):
            os.makedirs(folder_grid)

        folder_place = Path(f'{dirname_place}/place_field_{my_field}')
        if not os.path.exists(folder_place):
            os.makedirs(folder_place)

        # GRID-LIKE PROBABILITIES AND SPIKES
        # d is the x,y point of the grid field of dend ni
        d = np.zeros((nsynEC, xlim, ylim))
        dd = np.zeros((xlim, ylim))

        angle = 0.0
        lambda_var = 3.0
        for ni in range(nsynEC):
            lambda_var += 0.5 * (8/nsynEC)
            angle += 0.4 * (8/nsynEC)
            for x in range(xlim):
                # d is the point x,y of grid field of dend ni
                for y in range(ylim):
                    d[ni, x, y] = gridfield(angle, lambda_var, xxx, yyy, x, y)

        for ni in range(nsynEC):
            rate = rng.choice(np.arange(80, 120))
            all_spikes = poisson_spikes(0, len(path), N=1, rate=rate, dt=1)[:, 1]
            dd += d[ni, :, :]
            spikes = []
            for i in range(len(path)):  # i is time
                current_loc = path[i, :]

                probability = d[ni, current_loc[0]-1, current_loc[1]-1]
                probability *= (np.sin(2.0*np.pi*theta_freq *
                                       i/1000.0 + theta_phase)+1.0)/2.0

                r_ = rng.random()
                if (i in all_spikes) and (((probability > 0.70) and (r_ < probability / 2.0)) or (r_ < noiseEC)):

                    # spikes is a vector with the locations/timespots
                    # where there is a spike
                    spikes.append(i + tdelay)

            # Remove close in time spikes based on tref
            spikes_modified = [-(tref+1)]
            for sp in spikes:
                if sp - spikes_modified[-1] > tref:
                    spikes_modified.append(sp)

            spikes_modified = np.array(
                spikes_modified[1:]
            ).reshape(-1, 1).astype('float')
            spikes_modified += rng.standard_normal(
                (
                    spikes_modified.shape[0],
                    spikes_modified.shape[1]
                )
            )*jitterEC
            spikesEC.append(spikes_modified)
            print(f'Grid-like spikes of {ni}: {len(spikes_modified)}, rate: {rate}')
            fname_grid = Path(f'{folder_grid}/cell_{ni}.txt')
            if len(spikes_modified) == 0:
                spikes_modified = [1.]
            np.savetxt(
                fname_grid,
                spikes_modified,
                fmt='%.2d',
                delimiter=' '
            )

        # Store all probabilities on a vector for visualization purposes
        d_all[f'place_field_{my_field}'] = d

        # PLACE-LIKE PROBABILITIES AND SPIKES

        # probabilites of ca3 spikes based on grid cells
        d_ca3 = np.sum(d, axis=0)
        d_normed = d_ca3/np.max(d_ca3)

        # d is the x,y point of the grid field of dend ni
        d = np.zeros((nsynCA3, xlim, ylim))
        dd = np.zeros((xlim, ylim))
        r_delay = rng.normal(loc=0.0, scale=sigma_ca3)
        for ni in range(nsynCA3):
            rate = rng.choice(np.arange(r_min_ca3, r_max_ca3))
            all_spikes = poisson_spikes(
                t1=0,
                t2=len(path),
                N=1,
                rate=rate,
                dt=1,
                seed=my_run
            )
            all_spikes = all_spikes[:, 1]
            dd += d[ni, :, :]
            spikes = []
            for i in range(len(path)):  # i represents the time
                current_loc = path[i, :]

                probability = d_normed[current_loc[0]-1, current_loc[1]-1]
                theta_prob = (np.sin(2.0*np.pi*theta_freq * i/1000.0 + theta_phase)+1.0)/2.0

                probability *= theta_prob
                r_ = rng.random()
                if (i in all_spikes) and (((probability > 0.70) and (r_ < probability / 2.0)) or (r_ < noiseCA3)):

                    # spikes is a vector with the locations/timespots
                    # where there is a spike
                    # if your range is [a, b]
                    spikes.append(i + tdelay + int(tca3_delay + r_delay + sigma_cell*rng.standard_normal()))

            # Remove close in time spikes based on tref
            spikes_modified = [-(tref+1)]
            for sp in spikes:
                if sp - spikes_modified[-1] > tref:
                    spikes_modified.append(sp)

            spikes_modified = np.array(spikes_modified[1:]).reshape(-1, 1).astype('float')
            spikes_modified += rng.standard_normal(
                (
                    spikes_modified.shape[0],
                    spikes_modified.shape[1]
                )
            )*jitterCA3
            spikesCA3.append(spikes_modified)
            fname_place = Path(f'{folder_place}/cell_{ni}.txt')
            print(f'Place-like spikes of {ni}: {len(spikes_modified)}, rate: {rate}')
            if len(spikes_modified) == 0:
                spikes_modified = [1.]
            np.savetxt(
                fname_place,
                spikes_modified,
                fmt='%.2d',
                delimiter=' '
            )

        print(f'Done with Grid- and Place-like cells in field {my_field}')


if visualize:
    visualize_inputs(field=1, probabilities=d_all, dim=(2, 5))
    visualize_spiketimes(spikesCA3, type='CA3 inputs')
    visualize_spiketimes(spikesEC, type='EC inputs')
