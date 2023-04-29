#!/usr/bin/env python3
'''
    Copyright (C) 2014-2022, Johannes Pekkila.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
'''
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='A tool for generating images from binary slices')
parser.add_argument('--input', type=str, nargs='+', required=True, help='A list of data files')
parser.add_argument('--dims', type=int, nargs=2, required=True, help='The dimensions of a single data slice')
parser.add_argument('--dtype', type=str, default='double', help='The datatype of a single data element (default: double). Accepted values: numpy dtypes.')
parser.add_argument('--plot', action='store_true', help='Plot the results instead of writing to disk.')
parser.add_argument('--dpi', type=int, default=150, help='Set DPI of the output images')
args = parser.parse_args()


output='output'
os.system(f'mkdir -p {output}')
for file in args.input:
    data = np.fromfile(file, args.dtype)
    data = data.reshape((-1, args.dims[1], args.dims[0]))
    for field in range(0, data.shape[0]):
        slice = data[field,:,:]
        plt.imshow(slice, cmap='plasma', interpolation='nearest') #, vmin='-2.0e-2', vmax='2.0e-2'
        plt.colorbar()
        if args.plot:
            plt.show()
        else:
            plt.savefig(f'{output}/{Path(file).stem}-field-{field}.png', dpi=args.dpi)
        plt.clf()

print('do `convert output/input_files.png output/output.gif` to create an animation"')