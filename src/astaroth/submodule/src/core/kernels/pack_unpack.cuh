//  packs/unpacks buffer for inner/outer yz halos from/into global memory
#include "math_utils.h"
/*
#define DEVICE_VTXBUF_IDX(i, j, k) ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mxy))

static __device__ __forceinline__ int 
IDX(const int3 idx)        
{
    return DEVICE_VTXBUF_IDX(idx.x, idx.y, idx.z); 
}             
*/
template <int direction>  static __global__ void packUnpackPlate(AcReal* __restrict__ buffer, VertexBufferArray vba, int3 start, int3 end)
{
    const int y_block_size = end.x-start.x,               // end is exclusive!
              z_block_size = (end.y-start.y)*y_block_size,
            var_block_size = z_block_size*(end.z-start.z);

    const int vertexIdx  = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertexIdx >= var_block_size*NUM_VTXBUF_HANDLES)
        return;

    const int vba_handle  = vertexIdx / var_block_size,
              var_blockIdx= vertexIdx % var_block_size;

    const int zIdx = var_blockIdx / z_block_size,
              xyIdx= var_blockIdx % z_block_size,
              yIdx = xyIdx / y_block_size,
              xIdx = xyIdx % y_block_size;

    int vba_idx = IDX(start+(int3){xIdx,yIdx,zIdx});

    if (direction==AC_D2H){
      //vba.out[vba_handle][vba_idx] = (AcReal)(-vba_idx-1);
      buffer[vertexIdx] = vba.in[vba_handle][vba_idx];     // requires that the loaded plate is already updated in vba.out
    //   if(vba_handle == 0){
    //     printf("GPU %d,%d,%d,%d: %f\n",vba_handle,start.x+xIdx,start.y + yIdx, start.z+zIdx,vba.in[vba_handle][vba_idx]);
    //   }
      //buffer[vertexIdx] = vba.in[vba_handle][vba_idx];
    }else if (direction==AC_H2D)
      vba.in[vba_handle][vba_idx] = buffer[vertexIdx];
}
__global__ void setPlate(VertexBufferArray vba, int3 start, int3 end)
{
// sets vba[*][linear_index] = -(linear_index+1) in global memory //in inner xy halos

    const int y_block_size = end.x-start.x,               // end is exclusive!
              z_block_size = (end.y-start.y)*y_block_size,
            var_block_size = z_block_size*(end.z-start.z);
    
    const int vertexIdx  = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertexIdx >= var_block_size*NUM_VTXBUF_HANDLES)
        return;
    
    const int vba_handle  = vertexIdx / var_block_size,
              var_blockIdx= vertexIdx % var_block_size;
    
    const int zIdx = var_blockIdx / z_block_size,
              xyIdx= var_blockIdx % z_block_size,
              yIdx = xyIdx / y_block_size,
              xIdx = xyIdx % y_block_size;
    
    const int vba_idx = IDX(start+(int3){xIdx,yIdx,zIdx});
    vba.out[vba_handle][vba_idx] = (AcReal)(-vba_idx-1);
//printf("vba= %f \n",vba.out[vba_handle][vba_idx]);
}

