#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys


# Data formats
# data-format.csv: specifies precision and data dims
# <field>-data.out: a binary file storing the data within <field>


if len(sys.argv) <= 2:
    print('Usage: ./analysis.py <data files>')
    exit(-1)


headerpath = 'data-format.csv'
header = np.genfromtxt(headerpath, dtype=int, delimiter=',', skip_header=1)
use_double, mx, my, mz = header[0], header[1], header[2], header[3]

slices = []
for file in sys.argv[1:]:
    data = np.fromfile(file, dtype = np.double if use_double else np.float)
    data = np.reshape(data, (mx, my, mz), order = 'F')

    slices.append(data[:, :, int(mz/2)])

min = np.min(slices[0])
max = np.max(slices[0])

fig, ax = plt.subplots()
ims = []
for slice in slices:
    im = ax.imshow(slice, animated=True, cmap='plasma', vmin = min, vmax = max)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
plt.show()


'''
if len(sys.argv) <= 1:
    print('Usage: ./analysis.py <data files>')
    exit(-1)

headerpath = 'data-format.csv'
header = np.genfromtxt(headerpath, dtype=int, delimiter=',', skip_header=1)
use_double, mx, my, mz = header[0], header[1], header[2], header[3]

slices = []
for file in sys.argv[1:]:
    data = np.fromfile(file, dtype = np.double if use_double else np.float)
    data = np.reshape(data, (mx, my, mz))

    slices.append(data[int(mz/2), :, :])

fig, ax = plt.subplots()

min = np.min(slices[0])
max = np.max(slices[0])
im = ax.imshow(slices[0], vmin = min, vmax = max)
cb = fig.colorbar(im)
def animate(i):
    return ax.imshow(slices[i % len(slices)], vmin = min, vmax = max)

anim = animation.FuncAnimation(fig, animate, interval = 30)
plt.show()
'''


'''
if len(sys.argv) <= 2:
    print('Usage: ./analysis.py <header file> <data files>')
    exit(-1)


headerpath = sys.argv[1]
header = np.genfromtxt(headerpath, dtype=int, delimiter=',', skip_header=1)
use_double, mx, my, mz = header[0], header[1], header[2], header[3]

fig, ax = plt.subplots()
ims = []

filepaths = sys.argv[2:]
for file in filepaths:
    data = np.fromfile(file, dtype = np.double if use_double else np.float)
    data = np.reshape(data, (mx, my, mz))

    slice = data[int(mz/2), :, :]
    im = ax.imshow(slice, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
plt.show()
'''
