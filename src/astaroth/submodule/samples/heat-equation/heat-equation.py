#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn

import time

# Torch setup
device = 'cpu'
dtype = torch.float
if torch.cuda.is_available():
    import torch.cuda as cuda
    device = 'cuda'
#elif torch.backends.mps.is_available(): # TODO re-enable
#    assert(torch.backends.mps.is_built())
#    import torch.mps as mps
#    device = 'mps'
print(f'Using {device} device')

# Grid setup
nx = 256
ny = nx
nz = nx
dx = 2 * np.pi / nx
dy = 2 * np.pi / ny
dz = 2 * np.pi / nz
rr = 3

def synchronize_device():
    if device == 'cuda':
        cuda.synchronize()
    elif device == 'mps':
        mps.synchronize()

def heat_equation(field, kernel):
    f = nn.functional.pad(field, (rr,rr,rr,rr,rr,rr), mode='circular').to(dtype=dtype, device=device) # Periodic padding
    #bc = nn.ConstantPad3d(rr, 0) # Constant padding

    synchronize_device()
    start = time.time()
    res = nn.functional.conv3d(f, kernel)
    #res = nn.functional.conv3d(bc(field), kernel)
    synchronize_device()
    end = time.time()
    print(f'Convolution time elapsed: {1e3*(end - start)} ms')
    return res

def euler_step(field, kernel, dt):
    return field + heat_equation(field, kernel) * dt

# Field
temperature = 2 * np.random.random((nz, ny, nx)) - 1

# Move fields to device
temperature = torch.from_numpy(temperature).reshape(1, 1, nz, ny, nx).to(dtype=dtype, device=device)
#kernel = torch.tensor([[[[[0, (1/dy**2) * 1, 0],[(1/dx**2) * 1, (1/dx**2) * -2 + (1/dy**2) * -2, (1/dx**2) * 1],[0, (1/dy**2) * 1, 0]]]]], dtype=dtype).to(device)
#kernel = torch.from_numpy(np.random.random((3, 3, 3)).reshape(1,1,3,3,3)).to(dtype=dtype, device=device)
#ddx2 = (1/dx**2) * np.array([[[1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]]])
#ddy2 = (1/dy**2) * np.array([[[1/90], [-3/20], [3/2], [-49/18], [3/2], [-3/20], [1/90]]])
#ddz2 = (1/dz**2) * np.array([[[1/90]]], [[[-3/20]]], [[[3/2]]], [[[-49/18]]], [[[3/2]]], [[[-3/20]]], [[[1/90]]])

coeffs = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
kernel = torch.zeros((1, 1, 2*rr + 1, 2*rr + 1, 2*rr + 1)).to(dtype=dtype, device=device) 
for i in range(len(coeffs)):
    kernel[0, 0, rr, rr, i] += (1/dx**2) * coeffs[i]
    kernel[0, 0, rr, i, rr] += (1/dy**2) * coeffs[i]
    kernel[0, 0, i, rr, rr] += (1/dz**2) * coeffs[i]
print(kernel)

dt = 1e-3 * min(dx, dy, dz)

for _ in range(10):
    start = time.time()
    temperature = euler_step(temperature, kernel, dt)
    diff = time.time() - start
    print(f'Euler step time elapsed: {1e3*(diff)} ms')
    elems_per_second = ((nx * ny * nz) / diff) / 1e6;
    print(f'M elements per second: {elems_per_second}')
