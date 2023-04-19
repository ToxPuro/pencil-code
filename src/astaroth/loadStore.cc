//                             loadStore.cc
//                             --------------------

/* Date:   6-Jun-2017
   Author: M. Rheinhardt
   Description: Copier functions for the different "plates" of the halo and the full inner data cube with host-device concurrency.
                Load balance yet to be established.
*/

//C libraries
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>

#define real AcReal

#include "submodule/acc-runtime/api/math_utils.h"
#include "astaroth.h"
#include "../cparam_c.h"
#include "../cdata_c.h"

const int mxy=mx*my, nxy=nx*ny;
extern int halo_xy_size, halo_xz_size, halo_yz_size;
static AcReal *halo_xy_buffer, *halo_xz_buffer, *halo_yz_buffer;
static AcReal *test_buffer, *old_mesh_buffer;
extern Node node;

const int BOT=0, TOP=1, TOT=2;
int halo_widths_x[3]={nghost,nghost,2*nghost};    // bottom and top halo width and sum of them
int halo_widths_y[3]={nghost,nghost,2*nghost};
int halo_widths_z[3]={nghost,nghost,2*nghost};

void initLoadStore()
{
//printf("lperi= %d %d %d \n", lperi[0],lperi[1],lperi[2]);
        // halo widths for undivided data cube
        if (!lperi[0]){
            if (lfirst_proc_x) halo_widths_x[BOT]=nghost+1;
            if (llast_proc_x) halo_widths_x[TOP]=nghost+1;
        }
        if (!lyinyang) {
            if (!lperi[1]){
                if (lfirst_proc_y) halo_widths_y[BOT]=nghost+1;
                if (llast_proc_y) halo_widths_y[TOP]=nghost+1;
            }
            if (!lperi[2]){
                if (lfirst_proc_z) halo_widths_z[BOT]=nghost+1;
                if (llast_proc_z) halo_widths_z[TOP]=nghost+1;
            }
        }
        halo_widths_x[TOT]=halo_widths_x[BOT]+halo_widths_x[TOP];
        halo_widths_y[TOT]=halo_widths_y[BOT]+halo_widths_y[TOP];
        halo_widths_z[TOT]=halo_widths_z[BOT]+halo_widths_z[TOP];

//printf("halo_widths_x= %d %d %d\n",halo_widths_x[BOT],halo_widths_x[TOP],halo_widths_x[TOT]);
//printf("halo_widths_y= %d %d %d\n",halo_widths_y[BOT],halo_widths_y[TOP],halo_widths_x[TOT]);
//printf("halo_widths_z= %d %d %d\n",halo_widths_z[BOT],halo_widths_z[TOP],halo_widths_x[TOT]);

        // buffer for xz and yz halos in host
        halo_xz_size = mx*nz*max(halo_widths_y[BOT],halo_widths_y[TOP])*NUM_VTXBUF_HANDLES;
        if (halo_xz_buffer==NULL) halo_xz_buffer=(AcReal*) malloc(halo_xz_size*sizeof(AcReal));

        halo_yz_size = ny*nz*max(halo_widths_x[BOT],halo_widths_x[TOP])*NUM_VTXBUF_HANDLES;
        if (halo_yz_buffer==NULL) halo_yz_buffer=(AcReal*) malloc(halo_yz_size*sizeof(AcReal));

        halo_xy_size = nx*ny*max(halo_widths_z[BOT],halo_widths_z[TOP])*NUM_VTXBUF_HANDLES;
        if (halo_xy_buffer==NULL) halo_xy_buffer=(AcReal*) malloc(halo_xy_size*sizeof(AcReal));

        int test_buffer_size = mx*my*mz*NUM_VTXBUF_HANDLES;
        if (test_buffer==NULL) test_buffer=(AcReal*) calloc(test_buffer_size,sizeof(AcReal));

        if (old_mesh_buffer==NULL) old_mesh_buffer=(AcReal*) calloc(test_buffer_size,sizeof(AcReal));
}
/****************************************************************************************************************/
void finalLoadStore()
{
        free(halo_yz_buffer);
}
/****************************************************************************************************************/
void loadOuterFront(AcMesh& mesh, Stream stream)
{
        int3 src={0,0,0};
        int num_vertices=mxy*halo_widths_z[BOT];
        acNodeLoadMeshWithOffset(node, stream, mesh, src, src, num_vertices);

//printf("front:num_vertices= %d \n",num_vertices);
        //!!!cudaHostRegister(mesh, size, cudaHostRegisterDefault);    // time-critical!
}

void loadOuterBack(AcMesh& mesh, Stream stream)
{
        int3 src={0,0,mz-halo_widths_z[TOP]};      // index from m2-halo_widths_z[TOP] to m2-1
        int num_vertices=mxy*halo_widths_z[TOP];
        acNodeLoadMeshWithOffset(node, stream, mesh, src, src, num_vertices);

//printf("back:num_vertices= %d \n",num_vertices);
        //!!!cudaHostRegister(mesh, size, cudaHostRegisterDefault);    // time-critical!
}
 
void loadOuterBot(AcMesh& mesh, Stream stream)
{
        int3 start=(int3){0, 0,                    halo_widths_z[BOT]};
        int3 end  =(int3){mx,halo_widths_y[BOT],mz-halo_widths_z[TOP]};  //end is exclusive

        acNodeLoadPlateXcomp(node, stream, start, end, &mesh, halo_xz_buffer, AC_XZ);
}

void loadOuterTop(AcMesh& mesh, Stream stream)
{
        int3 start=(int3){0, my-halo_widths_y[TOP],   halo_widths_z[BOT]};
        int3 end  =(int3){mx,my,                   mz-halo_widths_z[TOP]};  //end is exclusive
//printf("loadOuterTop: %d %d %d %d %d %d \n", start.x,end.x,start.y,end.y,start.z,end.z);
        acNodeLoadPlateXcomp(node, stream, start, end, &mesh, halo_xz_buffer, AC_XZ);
}

void loadOuterLeft(AcMesh& mesh, Stream stream)
{
    int3 start=(int3){0,                      halo_widths_y[BOT],     halo_widths_z[BOT]};
    int3 end  =(int3){halo_widths_x[BOT]-1,my-halo_widths_y[TOP]-1,mz-halo_widths_z[TOP]-1}+1;  //end is exclusive

    acNodeLoadPlate(node, stream, start, end, &mesh, halo_yz_buffer, AC_YZ);
}

void loadOuterRight(AcMesh& mesh, Stream stream)
{
    int3 start=(int3){mx-halo_widths_x[TOP],   halo_widths_y[BOT],     halo_widths_z[BOT]};
    int3 end  =(int3){mx-1,                 my-halo_widths_y[TOP]-1,mz-halo_widths_z[TOP]-1}+1; //end is exclusive

    acNodeLoadPlate(node, stream, start, end, &mesh, halo_yz_buffer, AC_YZ);
}



void loadOuterHalos(AcMesh& mesh)
{
    loadOuterFront(mesh,STREAM_DEFAULT);
    loadOuterBack(mesh,STREAM_DEFAULT);
    loadOuterTop(mesh,STREAM_DEFAULT);
    loadOuterBot(mesh,STREAM_DEFAULT);
    loadOuterLeft(mesh,STREAM_DEFAULT);
    loadOuterRight(mesh,STREAM_DEFAULT);
}

void storeInnerFront(AcMesh& mesh, Stream stream)
{
        int3 start=(int3){l1,m1,n1}-1;
        int3 end  =(int3){l2,m2,n1+halo_widths_z[BOT]-1}-1+1;   //end is exclusive

printf("storeInnerFront: start,end= %d %d %d %d %d %d \n",start.x, end.x,start.y, end.y,start.z, end.z);
        //acNodeStoreIXYPlate(node, stream, start, end, &mesh, AC_FRONT);
        acNodeStorePlate(node, stream, start, end, &mesh, halo_xy_buffer, AC_XZ);
}

void storeInnerBack(AcMesh& mesh, Stream stream)
{
        int3 start=(int3){l1,m1,n2-halo_widths_z[TOP]+1}-1;
        int3 end  =(int3){l2,m2,n2                     }-1+1;    //end is exclusive
printf("storeInnerBack: start,end= %d %d %d %d %d %d \n",start.x, end.x,start.y, end.y,start.z, end.z);
        //acNodeStoreIXYPlate(node, stream, start, end, &mesh, AC_BACK);
        acNodeStorePlate(node, stream, start, end, &mesh, halo_xy_buffer, AC_XZ);
}





void storeInnerBot(AcMesh& mesh, Stream stream)
{
        int3 start=(int3){l1,m1,n1}-1;
        int3 end  =(int3){l2,m1+halo_widths_y[BOT]-1,n2}-1+1;    //end is exclusive
printf("storeInnerBot: start,end= %d %d %d %d %d %d \n",start.x, end.x,start.y, end.y,start.z, end.z);
        acNodeStorePlate(node, stream, start, end, &mesh, halo_xz_buffer, AC_XZ);
}

// void storeInnerBot(AcMesh& mesh, Stream stream)
// {
//         int3 start=(int3){l1,m1,n1+halo_widths_z[BOT]}-1;
//         int3 end=(int3){l2,m1+halo_widths_y[BOT]-1,n2-halo_widths_z[TOP]}-1+1;   //end is exclusive

//         acNodeStorePlate(node, stream, start, end, &mesh, halo_xz_buffer, AC_XZ);
// }

void storeInnerTop(AcMesh& mesh, Stream stream)
{
        int3 start=(int3){l1,m2-halo_widths_y[TOP]+1,n1}-1;
        int3 end  =(int3){l2,m2,n2}-1+1;    //end is exclusive
        printf("storeInnerTop: start,end= %d %d %d %d %d %d \n",start.x, end.x,start.y, end.y,start.z, end.z);
        acNodeStorePlate(node, stream, start, end, &mesh, halo_xz_buffer, AC_XZ);
}

// void storeInnerTop(AcMesh& mesh, Stream stream)
// {
//         int3 start=(int3){l1,m2-halo_widths_y[TOP]+1,n1+halo_widths_z[BOT]}-1;
//         int3 end=(int3){l2,m2,n2-halo_widths_z[TOP]}-1+1;    //end is exclusive

//         acNodeStorePlate(node, stream, start, end, &mesh, halo_xz_buffer, AC_XZ);
// }

// void storeInnerLeft(AcMesh& mesh, Stream stream)
// {
//         int3 start=(int3){l1,m1,n1}-1;
//         int3 end  =(int3){l1+halo_widths_x[BOT]-1,m2,n2}-1+1;    //end is exclusive
//         printf("storeInnerLeft: start,end= %d %d %d %d %d %d \n",start.x, end.x,start.y, end.y,start.z, end.z);
//         acNodeStorePlate(node, stream, start, end, &mesh, halo_xy_buffer, AC_FRONT);
// }

void storeInnerLeft(AcMesh& mesh, Stream stream)
{
    int3 start=(int3){l1,                     m1,n1}-1;
    int3 end  =(int3){l1+halo_widths_x[BOT]-1,m2,n2}-1+1;  //end is exclusive

    acNodeStorePlate(node, stream, start, end, &mesh, halo_yz_buffer, AC_YZ);
}

void storeInnerRight(AcMesh& mesh, Stream stream)
{
        int3 start=(int3){l2-halo_widths_x[TOP]+1,m1,n1}-1;
        int3 end  =(int3){l2,m2,n2}-1+1;    //end is exclusive
printf("storeInneRight: start,end= %d %d %d %d %d %d \n",start.x, end.x,start.y, end.y,start.z, end.z);
        acNodeStorePlate(node, stream, start, end, &mesh, halo_yz_buffer, AC_YZ);
}

// void storeInnerRight(AcMesh& mesh, Stream stream)
// {
//     int3 start=(int3){l2-halo_widths_x[TOP]+1,m1+halo_widths_y[BOT],n1+halo_widths_z[BOT]}-1;
//     int3 end  =(int3){l2,                     m2-halo_widths_y[TOP],n2-halo_widths_z[TOP]}-1+1; //end is exclusive

//     acNodeStorePlate(node, stream, start, end, &mesh, halo_yz_buffer, AC_YZ);
// }

void storeAll(AcMesh& mesh, Stream stream){
    int3 start = (int3){3,3,3};
    int3 end = (int3){mx-3,my-3,mz-3};
    printf("storeAll: start,end= %d %d %d %d %d %d \n",start.x, end.x,start.y, end.y,start.z, end.z);
   // acNodeStorePlate(node, stream, start, end, &mesh, halo_xyz_buffer, AC_YZ);
}

void copy_mesh_to_buffer(AcMesh& mesh, AcReal* buffer){
    printf("Copying to mesh\n");
    for(int vtxbuf=0;vtxbuf<NUM_VTXBUF_HANDLES;vtxbuf++){
        for(int i=0;i<mx;i++){
            for(int j=0;j<my;j++){
                for(int k=0;k<mz;k++){
                    int idx_mesh = i+j*mx+k*mx*my;
                    int idx = idx_mesh+vtxbuf*mx*my*mz;
                    AcReal val = mesh.vertex_buffer[VertexBufferHandle(vtxbuf)][idx_mesh];
                    buffer[idx] = val; 
                }
            }
        }
    }
}

void copy_buffer_to_mesh(AcMesh& mesh, AcReal* buffer){
    printf("Copying to mesh\n");
    for(int vtxbuf=0;vtxbuf<NUM_VTXBUF_HANDLES;vtxbuf++){
        for(int i=0;i<mx;i++){
            for(int j=0;j<my;j++){
                for(int k=0;k<mz;k++){
                    int idx_mesh = i+j*mx+k*mx*my;
                    int idx = idx_mesh+vtxbuf*mx*my*mz;
                    AcReal val = buffer[idx];
                    mesh.vertex_buffer[VertexBufferHandle(vtxbuf)][idx_mesh] = val;
                }
            }
        }
    }
}

void check_load(int3 start, int3 end, AcMesh& mesh){
    bool same=true;
    for(int vtxbuf=0;vtxbuf<NUM_VTXBUF_HANDLES;vtxbuf++){
        for(int i=start.x;i<end.x;i++){
            for(int j=start.y;j<end.y;j++){
                for(int k=start.z;k<end.z;k++){
                    int idx_mesh = i+j*mx+k*mx*my;
                    int idx = idx_mesh+vtxbuf*mx*my*mz;
                    same &= mesh.vertex_buffer[VertexBufferHandle(vtxbuf)][idx_mesh] == test_buffer[idx];
                    if(!same){
                        printf("%d,%d,%d\n copy: %f\n mesh: %f\n",i,j,k,test_buffer[idx], mesh.vertex_buffer[VertexBufferHandle(vtxbuf)][idx_mesh]);
                    }

                }
            }
        }
    }
    if(same){
        printf("Same :)\n");
    }
    else{
        printf("Not same :(\n");
    }
}

void write_mesh_to_buffer(int3 start, int3 end, AcMesh& mesh, AcReal* buffer){
    for(int vtxbuf=0;vtxbuf<NUM_VTXBUF_HANDLES;vtxbuf++){
        for(int i=start.x;i<end.x;i++){
            for(int j=start.y;j<end.y;j++){
                for(int k=start.z;k<end.z;k++){
                    int idx_mesh = i+j*mx+k*mx*my;
                    int idx = idx_mesh+vtxbuf*mx*my*mz;
                    buffer[idx] = mesh.vertex_buffer[VertexBufferHandle(vtxbuf)][idx_mesh];
                }
            }
        }
    }
}

void print_buffer_segment(int3 start, int3 end, AcReal* buffer, int vtxbuf){
        for(int i=start.x;i<end.x;i++){
            for(int j=start.y;j<end.y;j++){
                for(int k=start.z;k<end.z;k++){
                    int idx_mesh = i+j*mx+k*mx*my;
                    int idx = idx_mesh+vtxbuf*mx*my*mz;
                    printf("CPU: %d,%d,%d: %f\n", i,j,k,buffer[idx]);
                }
            }
        }
}

void print_segment_buffer(int3 start, int3 end, AcReal* plateBuffer){
    int3 dims = end-start;
    for(int i=0;i<dims.x;i++){
        for(int j=0;j<dims.y;j++){
            for(int k=0;k<dims.z;k++){
                int idx = i+j*dims.x+k*dims.x*dims.y;
                printf("Buffer: %d,%d,%d: %f\n",i+start.x,j+start.y,k+start.z,plateBuffer[idx]);
            }
        }
    }
}
void write_pack_buffer_to_mesh(int3 start, int3 end, AcReal* plateBuffer, AcMesh& mesh){
    int3 dims = end-start;
    for(int vtxbuf=0;vtxbuf<NUM_VTXBUF_HANDLES;vtxbuf++){
        for(int i=0;i<dims.x;i++){
            for(int j=0;j<dims.y;j++){
                for(int k=0;k<dims.z;k++){
                    int buffer_idx = i+j*dims.x+k*dims.x*dims.y + vtxbuf*dims.x*dims.y*dims.z;
                    int mesh_idx = i+start.x+(j+start.y)*mx+(k+start.z)*mx*my;
                    mesh.vertex_buffer[VertexBufferHandle(vtxbuf)][mesh_idx] = plateBuffer[buffer_idx];
                }
            }
        }
    }

}

void storeInnerHalos(AcMesh& mesh)
{
   copy_mesh_to_buffer(mesh,old_mesh_buffer);
    storeInnerLeft(mesh,STREAM_1);
    storeInnerBot(mesh,STREAM_2);
    //acStore(&mesh);
    // storeInnerRight(mesh,STREAM_1);
    // storeInnerTop(mesh,STREAM_2);
    // storeInnerBack(mesh,STREAM_3);
    int3 start; 
    int3 end; 
    // write_mesh_to_buffer(start,end,mesh,old_mesh_buffer);
    // print_buffer_segment(start,end,old_mesh_buffer,0);
    // print_segment_buffer(start,end,halo_yz_buffer);

   // write_pack_buffer_to_mesh(start,end,halo_xz_buffer,mesh);
    //write_mesh_to_buffer(start,end,mesh,old_mesh_buffer);
    //     start = {3,3,3};
    // end = {131,6,131};
    // write_mesh_to_buffer(start,end,mesh,old_mesh_buffer);

    //     start = {3,3,3};
    // end = {131,131,6};
    // write_mesh_to_buffer(start,end,mesh,old_mesh_buffer);

    //     start = {128,3,3};
    // end = {131,131,131};
    // write_mesh_to_buffer(start,end,mesh,old_mesh_buffer);

    //     start = {3,128,3};
    // end = {131,131,131};
    // write_mesh_to_buffer(start,end,mesh,old_mesh_buffer);

    //     start = {3,3,128};
    // end = {131,131,131};
    // write_mesh_to_buffer(start,end,mesh,old_mesh_buffer);

    // copy_buffer_to_mesh(mesh,old_mesh_buffer);
    start = {3,3,3};
    end = {6,131,131};
    write_pack_buffer_to_mesh(start,end,halo_yz_buffer,mesh);
        start = {3,3,3};
    end = {131,6,131};
    write_pack_buffer_to_mesh(start,end,halo_xz_buffer,mesh);
    storeInnerFront(mesh,STREAM_3);
        start = {3,3,3};
    end = {131,131,6};
    write_pack_buffer_to_mesh(start,end,halo_xy_buffer,mesh);

    storeInnerRight(mesh,STREAM_1);
    storeInnerTop(mesh,STREAM_2);
    start = {128,3,3};
    end = {131,131,131};
    write_pack_buffer_to_mesh(start,end,halo_yz_buffer,mesh);
        start = {3,128,3};
    end = {131,131,131};
    write_pack_buffer_to_mesh(start,end,halo_xz_buffer,mesh);
    storeInnerBack(mesh,STREAM_3);
        start = {3,3,128};
    end = {131,131,131};
    write_pack_buffer_to_mesh(start,end,halo_xy_buffer,mesh);
   //storeAll(mesh,STREAM_1);
}

