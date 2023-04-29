#!/usr/bin/env python3
import argparse
import math
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='A tool for rendering slices from binary files')
parser.add_argument('--input', type=pathlib.Path, nargs='+', required=True, help='List of slice binaries')
parser.add_argument('--output', type=pathlib.Path, default='output-slices-rendered', help='Output dir to write rendered slices to')
parser.add_argument('--render-vectors', type=str, default='on', choices=['on', 'off'], help='Deduce which fields are vectors from field names and combine vector components')
parser.add_argument('--conserve-memory', type=str, default='off', choices=['on', 'off'], help='Conserve memory by reading files twice. Once two find vmin, vmax, and again when rendering.')
parser.add_argument('--termcolor', type=str, default='on', choices=['on', 'off'], help='Use ANSI color codes for clearer terminal output')
parser.add_argument('--dtype', type=str, default='double', help='The datatype of a single data element (default: double). Accepted values: numpy dtypes')
parser.add_argument('--dpi', type=int, default=150, help='Set DPI of the output images')
parser.add_argument('--vrange', type=float, nargs=2, required=False, help='Manually set the value range of the plots as --vrange {min} {max}')
args = parser.parse_args()

#Term colors
RD=""
GR=""
YE=""
MA=""
CY=""
CLR=""

if args.termcolor == "on":
    RD="\033[91m"
    GR="\033[92m"
    YE="\033[93m"
    MA="\033[95m"
    CY="\033[96m"
    CLR="\033[00m"

output_dir = args.output

segmented_filename_regex = re.compile(r'([^-]*)-segment-at_(\d+)_(\d+)_(\d+)-dims_(\d+)_(\d+)-step_(\d+).slice')

#Parse filenames, build up a tree structure
steps = {}
field_headers = {}
for file_path in args.input:
    filename = pathlib.PurePath(file_path).name
    m = segmented_filename_regex.match(filename)
    if m:
        field = m.group(1)
        pos = [int(m.group(2)), int(m.group(3)), int(m.group(4))]
        dims = [int(m.group(5)), int(m.group(6))]
        #step = int(m.group(7))
        step = m.group(7)

        field_headers.setdefault(field, {"vmin":math.inf, "vmax":-math.inf})

        steps.setdefault(step, {})

        z = pos[2]
        #This is the slice data, this is what will be rendered
        steps[step].setdefault(field, {"z":z, "columns":{}, "field_name":field, "step": step})

        #Z should be the same across slice segments
        #TODO: make z another hierarchy to allow rendering of multiple slice locations
        old_z = steps[step][field]["z"]
        if old_z != z:
            print("f{RD}Error:{CLR} Slice segments for field {field} step {step} have multiple dissimilar z-values: {old_z} and {z}")
            exit(1)

        steps[step][field]["columns"].setdefault(pos[0], {})
        steps[step][field]["columns"][pos[0]].setdefault(pos[1], {})
        steps[step][field]["columns"][pos[0]][pos[1]] = {"dims":dims, "file_path":file_path}
    else:
        print(f"File {file}, did not match filename regex, skipping it")

vector_fields = {}
if args.render_vectors == "on":
    print("Rendering vectors")

    for field in field_headers:
        if field[-1] in "xX":
            stem = field[:-1]
            
            if field[-1] == "x":
                yfield = stem + "y"
                zfield = stem + "z"
            elif field[-1] == "X":
                yfield = stem + "Y"
                zfield = stem + "Z"

            if yfield in field_headers and zfield in field_headers:
                vector_fields[stem] = [field, yfield, zfield]

conserve_memory = True

if args.vrange:
    vmin = args.vrange[0]
    vmax = args.vrange[1]
    print(f"Setting value range of plots to [{vmin},{vmax}]") 
    for field in field_headers:
        field_headers[field]["vmin"] = vmin
        field_headers[field]["vmax"] = vmax
else:
    print("Determining the value range of each field by taking the minimum and maximum across all slices per field.")
    #Find the minimum and maximum values fields, to define a constant range for the slices
    for step, fields in steps.items():
        for field, field_data in fields.items():
            for column in field_data["columns"].values():
                for segment in column.values():
                    file_path    = segment["file_path"]
                    segment_dims = segment["dims"]
                    segment["frame"] = np.fromfile(file_path, args.dtype).reshape((segment_dims[1], segment_dims[0]))
                    field_headers[field]["vmin"] = min(np.min(segment["frame"]), field_headers[field]["vmin"])
                    field_headers[field]["vmax"] = max(np.max(segment["frame"]), field_headers[field]["vmax"])
                    if conserve_memory:
                        del segment["frame"]


#Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

"""
combine_slice

Build a full slice from segments
"""
def combine_slice(slice_data):
    full_slice = None
    #TODO: make this fault tolerant, in case a segment is missing
    for x, column in slice_data["columns"].items():
        slice_col = None
        for y, segment in column.items():
            file_path = segment["file_path"]
            if conserve_memory:
                segment["frame"] = np.fromfile(file_path, args.dtype).reshape((segment["dims"][1], segment["dims"][0]))
            slice_col = np.vstack((slice_col, segment["frame"])) if slice_col is not None else segment["frame"]
            del segment["frame"]
        full_slice = np.hstack((full_slice, slice_col)) if full_slice is not None else slice_col
    slice_data["full_slice"] = full_slice
    del slice_data["columns"]

"""
render_slice

Render a slice (a full one)
"""
def render_slice(full_slice, field_name, step, z):
    print(f"Rendering {MA}{field_name:>20}{CLR} slice at step {CY}{int(step):<8}{CLR}...", end="")
    vmin = field_headers[field_name]["vmin"]
    vmax = field_headers[field_name]["vmax"]
    plt.imshow(full_slice, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.ylabel('y')
    plt.xlabel('x')
    title_field_name = field_name.replace("_"," ").replace("VTXBUF","")
    plt.title(f'{title_field_name}, step = {int(step)}')
    output_file = output_dir/f'{field_name}.{full_slice.shape[0]}x{full_slice.shape[1]}.z_{z}.step_{step}.png'
    print(f"writing to {GR}{output_file}{CLR}")
    plt.savefig(output_file, dpi=args.dpi)
    plt.clf()

#Render the vector fields
for vector_field, components in vector_fields.items():
    #render each slice, but keep the frames
    for step in steps:
        vector_slice = None
        z = steps[step][components[0]]["z"]
        vmax_l2_norm = 0
        for comp_field in components:
            slice_data = steps[step][comp_field]
            combine_slice(slice_data)
            render_slice(**slice_data)
            comp_squared = np.square(slice_data["full_slice"])
            #Delete scalar component, so that it doesn't get re-rendered
            del steps[step][comp_field]["full_slice"]
            del steps[step][comp_field]
            vector_slice = vector_slice + comp_squared if vector_slice is not None else comp_squared
            vmax_l2_norm += max(field_headers[comp_field]["vmax"], -field_headers[comp_field]["vmin"]) ** 2

        #All three components have now deposited their slice data
        vector_slice = np.sqrt(vector_slice)

        l2_norm_header={}
        if args.vrange:
            #If the user supplied a vrange, use that
            l2_norm_header["vmin"] = args.vrange[0]
            l2_norm_header["vmax"] = args.vrange[1]
        else:
            #Use an upper and lower bound for the L2 norm vmin and vmax
            l2_norm_header["vmin"] = 0
            l2_norm_header["vmax"] = math.sqrt(vmax_l2_norm)
            #TODO: Calculate exact vmax

        field_name = vector_field + "_L2-norm"
        field_headers[field_name] = l2_norm_header

        render_slice(vector_slice, field_name, step, z)
        del vector_slice
           

#Render remaining scalar fields
for step, fields in steps.items():
    for field, slice_data in fields.items():
        combine_slice(slice_data)
        render_slice(**slice_data)
        del slice_data["full_slice"]
