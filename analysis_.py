#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:04:39 2022

@author: spiros
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from opt_ import make_maps, my_style

plt.style.use(my_style())

nrun = 1  # trial number

# Load the path file
dirname_ec = f'grid_like_inputs/run_{nrun}'
dirname_ca3 = f'place_like_inputs/run_{nrun}'

fname = Path(f'{dirname_ec}/path.txt')
path = np.loadtxt(fname).astype('int64')

# Load the grid-like inputs
nplace_field = 17  # number of place fields
step = 5  # step among place fields
nEC = 20  # total EC inputs per place field
nCA3 = 20  # total CA3 inputs per place field
npath_x = 200
npath_y = 1
fbin = 1  # binarize per 2cm, if one is the bin per 1 cm --> original values.

nfield = 1  # field to analyze and plot

Z_grid = np.zeros((int(npath_x/fbin), nEC))
for n_ec in range(nEC):
    fname2 = Path(f'{dirname_ec}/place_field_{nfield}/cell_{n_ec}.txt')
    spiketimes = np.loadtxt(fname2).astype('float')
    Z_grid[:, n_ec] = make_maps(
        path,
        spiketimes,
        xlim=npath_x,
        ylim=npath_y,
        num_bins=int(npath_x/fbin)
    ).squeeze()

# Grid-like cells
dimEC = (4, 5)
fig, axes = plt.subplots(nrows=dimEC[0], ncols=dimEC[1], figsize=(12, 8))
for cnt, ax in enumerate(axes.flat):
    im = ax.imshow(
        Z_grid[:, cnt].reshape(1, -1),
        cmap='viridis',
        extent=[0, npath_x, 0, 0.2],
        aspect=100,
        vmin=0
    )
    ax.set_yticks([])
    ax.set_title(f'grid cell {cnt+1}')
    ax.set_xlabel('position (cm)')
clb = fig.colorbar(im, ax=axes.ravel().tolist())
clb.set_label('firing rate (Hz)')
plt.show()

# Place-like cells
dimCA3 = (4, 5)
Z_place = np.zeros((int(npath_x/fbin), nCA3))
for n_ca3 in range(nCA3):
    fname2 = Path(f'{dirname_ca3}/place_field_{nfield}/cell_{n_ca3}.txt')
    spiketimes = np.loadtxt(fname2).astype('float')
    Z_place[:, n_ca3] = make_maps(
        path,
        spiketimes,
        xlim=npath_x,
        ylim=npath_y,
        num_bins=int(npath_x/fbin)
    ).squeeze()

fig, axes = plt.subplots(
    nrows=dimCA3[0],
    ncols=dimCA3[1],
    figsize=(12, 8)
)

for cnt, ax in enumerate(axes.flat):
    im = ax.imshow(
        Z_place[:, cnt].reshape(1, -1),
        cmap='viridis',
        extent=[0, npath_x, 0, 0.2],
        aspect=100,
        vmin=0
    )
    ax.set_yticks([])
    ax.set_title(f'place cell {cnt+1}')
    ax.set_xlabel('position (cm)')
clb = fig.colorbar(im, ax=axes.ravel().tolist())
clb.set_label('firing rate (Hz)')
plt.show()

print(f"field location at {np.argmax(np.sum(Z_place, axis=1))} cm")
