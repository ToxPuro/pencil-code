import pylab as plt
import numpy as np 


def do_bound(coeff):
    vertex_buffer = np.zeros(7, dtype=np.float32)
    xx = np.arange(vertex_buffer.size)
    
    edge_idx = 3
    
    for dst_idx in range(3):
        i_diff = abs(edge_idx - dst_idx)
        vertex_buffer[dst_idx] = coeff*np.exp(vertex_buffer[edge_idx])
     
        print("initial",vertex_buffer)
    
        for i in range(i_diff): 
            vertex_buffer[dst_idx] = coeff*vertex_buffer[dst_idx]
            print("looped", vertex_buffer[dst_idx])
        
        vertex_buffer[dst_idx] = np.log(vertex_buffer[dst_idx]);
        print("final",vertex_buffer)

    return xx, vertex_buffer


AC_dsx = 0.04908738521
coeff1 = 1.0 - AC_dsx/(25.0*AC_dsx)
coeff2 = 1.0 - AC_dsx/(100.0*AC_dsx)


plt.figure()
xx, yy = do_bound(coeff1)
plt.plot(xx, yy)

plt.figure()
xx, yy = do_bound(coeff2)
plt.plot(xx, yy)

plt.show()

