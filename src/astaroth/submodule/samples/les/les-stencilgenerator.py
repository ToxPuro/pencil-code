#!/bin/python3
# Usage: ./les-stencilgenerator.py > stencil.h

halo = 2

print(f'hostdefine STENCIL_ORDER ({2 * halo})')

print('Stencil sum {')

stencil = ""
for k in range(-halo, halo+1):
    for j in range(-halo, halo+1):
        for i in range(-halo, halo+1):
            stencil += str(f'[{i}][{j}][{k}] = 1,\n')
print(stencil[:-2])
print('}')