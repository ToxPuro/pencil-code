#!/bin/python3


print('Stencil filter {')



order = 6
for k in range(order + 1):
    for j in range(order + 1):
        for i in range(order + 1):
            print(f'[{int(k - order/2)}][{int(j- order/2)}][{int(i- order/2)}] = {1.0 / ((order+1)**3)},')


print('}')