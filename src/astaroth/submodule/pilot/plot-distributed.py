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
parser.add_argument('--subdims', type=int, nargs=2, required=True, help='The dimensions of a subslice')
parser.add_argument('--dims', type=int, nargs=2, required=True, help='The dimensions of a full data slice')
parser.add_argument('--dtype', type=str, default='double', help='The datatype of a single data element (default: double). Accepted values: numpy dtypes.')
parser.add_argument('--plot', action='store_true', help='Plot the results instead of writing to disk.')
parser.add_argument('--dpi', type=int, default=150, help='Set DPI of the output images')
args = parser.parse_args()


output='output'
os.system(f'mkdir -p {output}')

def get_file(field, position, label):
    for file in args.input:
        if field in file:
            if label in file:
                if f'at_{position[0]}_{position[1]}_' in file:
                    print(f'found {file}')
                    return file
    return None


for file in args.input:
    if not 'at_0_0_' in file:
        continue

    basename = Path(file).stem.split('-')
    field = basename[0]
    label = basename[-1]

    data = None
    for j in range(0, args.dims[1], args.subdims[1]):
        hdata = None
        for i in range(0, args.dims[0], args.subdims[0]):
            slice = get_file(field, [i, j], label)
            frame = np.fromfile(slice).reshape((args.subdims[1], args.subdims[0]))
            if hdata is None:
                hdata = frame
            else:
                hdata = np.hstack((hdata, frame))
        if data is None:
            data = hdata
        else:
            data = np.vstack((data, hdata))

    plt.imshow(data, cmap='plasma', interpolation='nearest')
    plt.colorbar()
    if args.plot:
        plt.show()
    else:
        plt.savefig(f'{output}/{Path(file).stem}.png', dpi=args.dpi)
    plt.clf()

print('do `convert output/input_files.png output/output.gif` to create an animation"')