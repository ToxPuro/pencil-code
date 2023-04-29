/*                             gpu_astaroth.cc
                               --------------------

   Date:   8-Feb-2017
   Author: M. Rheinhardt & J. Pekkilae
   Description:
           ANSI C and standard library callable function wrappers for ASTAROTH-nucleus functions to be called from Fortran.
  Comments: 
*/

//General
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define CUDA_ERRCHK(X)

#include "submodule/acc-runtime/api/math_utils.h"
#include "submodule/include/astaroth.h"
#include "submodule/build/acc-runtime/api/user_defines.h"
#define real AcReal
#define EXTERN 
#define FINT int

//PC interface
#include "PC_moduleflags.h"
#include "../cparam_c.h"
#include "../cdata_c.h"
#include "../sub_c.h"                   // provides set_dt
#include "../boundcond_c.h"             // provides boundconds[xyz] etc.
#include "../mpicomm_c.h"               // provides finalize_sendrcv_bdry
//#include "diagnostics/diagnostics.h"
#if PACKED_DATA_TRANSFERS
  #include "loadStore.h"
#endif
#if LFORCING
  #include "forcing.h"
#endif

#include "PC_module_parfuncs.h"

static AcMesh mesh;
static AcTaskGraph* graph_1;
static AcTaskGraph* graph_2;
static AcTaskGraph* graph_3;
static int pid;
Node node;
DeviceConfiguration devConfig;
int halo_xy_size=0, halo_xz_size=0, halo_yz_size=0;
static int l0=1;
#if LFORCING
static ForcingParams forcing_params;
#include "../forcing_c.h"
#endif
/***********************************************************************************************/
AcReal max_advec()
{
        AcReal maxadvec_=0.;
#if LHYDRO
        AcReal umax=acReduceVec(RTYPE_MAX,VTXBUF_UUX,VTXBUF_UUY,VTXBUF_UUZ);
#endif
        return maxadvec_;
}
/***********************************************************************************************/
AcReal max_diffus()
{
        AcReal maxdiffus_=0.;
#if LVISCOSITY
        maxdiffus_=nu*dxyz_2[nghost-1];
#endif
#if LMAGNETIC
        maxdiffus_=std::max(maxdiffus_,eta*dxyz_2[nghost-1]);
#endif
        return maxdiffus_;
}
extern "C" void helloWorld(int index){
        printf("Hello World %d",index);
}
/***********************************************************************************************/
extern "C" void notifyBoundCondDone(int bc_id, int dimension){
        printf("Notifying boundcond done: %d, dimension: %d", bc_id,dimension);
}
/***********************************************************************************************/
extern "C" void registerBoundCond(int bc_id, int dimension){
        printf("Registering bouncond: %d, dimension: %d", bc_id,dimension);
}
/***********************************************************************************************/
//Do the 'isubstep'th integration step on all GPUs on the node and handle boundaries.
//
extern "C" void substepGPU(int isubstep, bool full=false, bool early_finalize=false)
{
#if LFORCING
    //Update forcing params

    if (isubstep == itorder) 
         forcing_params.Update();  // calculate on CPU and load into GPU
#endif
    if (lfirst && ldt) {
         AcReal dt1_advec  = max_advec()/cdt;
         AcReal dt1_diffus = max_diffus()/cdtv;
         AcReal dt1_=sqrt(pow(dt1_advec,2) + pow(dt1_diffus,2));
         set_dt(dt1_);
    }
    //Transfer the updated ghost zone to the device(s) in the node 

    if (full){
        printf("CPU full check before uux :%d: %f\n",pid, mesh.vertex_buffer[VTXBUF_UUX][9+mx*9+my*mx*9]);
        printf("CPU full check before uuy :%d: %f\n",pid, mesh.vertex_buffer[VTXBUF_UUY][9+mx*9+my*mx*9]);
        printf("CPU full check before uuz :%d: %f\n",pid, mesh.vertex_buffer[VTXBUF_UUZ][9+mx*9+my*mx*9]);
        // printf("CPU full check global before 9,73,9:%d: %f\n", pid, mesh.vertex_buffer[VertexBufferHandle(0)][9+mx*73+mx*my*9]);
        // acLoad(mesh);
        //What is going to be 9,9,9 in in pid 1?
        //It is going to be 0,64,0+9,9,9-3,3,3 =6,70,6
        //This is in one cpu pid 0 going to be:  6,70,6+3,3,3 = 9,73,9
        printf("Before loading\n");
        acGridLoadMesh(STREAM_DEFAULT,mesh);
        printf("After loading :)\n");
    }

    //if (ldiagnos) timeseries_diagnostics(mesh);

    //Integrate on the GPUs in this node
    //NOTE: In Astaroth, isubstep is {0,1,2}, in PC it is {1,2,3}
//printf("isubstep,full,early_finalize= %d %d %d \n",isubstep,full,early_finalize);
    if (early_finalize) {
    // MPI communication has already finished, hence the full domain can be advanced.
      if (!full) {
#if PACKED_DATA_TRANSFERS
        //  loadOuterHalos(mesh);
//        acLoad(mesh);
    // printf("CPU check: %f\n", mesh.vertex_buffer[VertexBufferHandle(0)][9+134*9+134*134*9]);
//     acGridSynchronizeStream(STREAM_ALL);
    printf("CPU not full check before:%d: %f\n", pid,mesh.vertex_buffer[VertexBufferHandle(0)][9+mx*9+mx*my*9]);
//     printf("CPU not full check global 9,73,9 before:%d: %f\n", pid, mesh.vertex_buffer[VertexBufferHandle(0)][9+mx*73+mx*my*9]);
    acGridLoadMesh(STREAM_DEFAULT,mesh);
#else
          acLoad(mesh);
#endif
float maxbx=-1e30, maxby=-1e30, maxbz=-1e30, bb; int ii;
float maxux=-1e30, maxuy=-1e30, maxuz=-1e30, maxlnrho=-1e30;
float minbx=1e30, minby=1e30, minbz=1e30;
float minux=1e30, minuy=1e30, minuz=1e30, minlnrho=1e30;
//printf("VertexBuffer aa=%d %d %d %d \n",VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ, NUM_VTXBUF_HANDLES);
//VTXBUF_UUX,VTXBUF_UUY,VTXBUF_UUZ,VTXBUF_LNRHO,VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ, NUM_VTXBUF_HANDLES);
/*for (ii=0; ii<mw; ii++) {
  bb=mesh.vertex_buffer[VertexBufferHandle(0)][ii];
  maxux=max(maxux,bb);
  minux=min(minux,bb);

  bb=mesh.vertex_buffer[VertexBufferHandle(1)][ii];
  maxuy=max(maxuy,bb);
  minuy=min(minuy,bb);

  bb=mesh.vertex_buffer[VertexBufferHandle(2)][ii];
  maxuz=max(maxuz,bb);
  minuz=min(minuz,bb);

  bb=mesh.vertex_buffer[VertexBufferHandle(3)][ii];
  maxlnrho=max(maxlnrho,bb);
  minlnrho=min(minlnrho,bb);

  bb=mesh.vertex_buffer[VertexBufferHandle(4)][ii];
  maxbx=max(maxbx,bb);
  minbx=min(minbx,bb);

  bb=mesh.vertex_buffer[VertexBufferHandle(5)][ii];
  maxby=max(maxby,bb);
  minby=min(minby,bb);

  bb=mesh.vertex_buffer[VertexBufferHandle(6)][ii];
  maxbz=max(maxbz,bb);
  minbz=min(minbz,bb);
}*/
//printf("nach Load: min= %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f \n", minux, minuy, minuz, minlnrho, minbx, minby, minbz);
//printf("nach Load: max= %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f \n", maxux, maxuy, maxuz, maxlnrho, maxbx, maxby, maxbz);
//if (isubstep>=1) printf("outer halo load \n");
      }
      acSynchronize();
printf("isubstep= %d\n", isubstep);
//       acSynchronizeMesh();
      acGridSynchronizeStream(STREAM_ALL);

    //  acGridSwapBuffers();
    // sleep(2);
     acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, dt);
     if(isubstep == 1){
        acGridExecuteTaskGraph(graph_1,1);
     }
     if(isubstep == 2){
        acGridExecuteTaskGraph(graph_2,1);
     }
     if(isubstep == 3){
        acGridExecuteTaskGraph(graph_3,1);
     }

//       acIntegrateStep(isubstep-1, dt);
      acGridSynchronizeStream(STREAM_ALL);
//      acNodeSwapBuffers(node);
//      acGridSwapBuffers();
#if PACKED_DATA_TRANSFERS
      acGridStoreMesh(STREAM_DEFAULT,&mesh);
    //   storeInnerHalos(mesh);
//     acStore(&mesh);
    printf("CPU check after:%d: %f\n", pid, mesh.vertex_buffer[VertexBufferHandle(0)][9+mx*9+mx*my*9]);
//     printf("CPU check global 9,73,9 after:%d: %f\n", pid, mesh.vertex_buffer[VertexBufferHandle(0)][9+mx*73+mx*my*9]);

#else
      acStore(&mesh);
#endif
//for (int ii=0; ii<36; ii++) printf("%10.5f \n",mesh.vertex_buffer[VertexBufferHandle(3)][ii]);

    } else {     // end early finalize

    // MPI communication has not yet finished, hence only the inner domain can be advanced.
      int3 start=(int3){l1i+2,m1i+2,n1i+2}-1, end=(int3){l2i-2,m2i-2,n2i-2}-1+1;   // -1 shift because of C indexing convention
      if (start.x<end.x && start.y<end.y && start.z<end.z) {
        if (l0) {printf("start,end= %d %d %d %d %d %d \n",start.x,start.y,start.z,end.x,end.y,end.z);}
        acNodeIntegrateSubstep(node, STREAM_1, isubstep-1,start,end,dt);                          // +1 shift because end is exclusive
      }
#if PACKED_DATA_TRANSFERS
      int iarg1=1, iarg2=NUM_VTXBUF_HANDLES; 
      finalize_isendrcv_bdry((AcReal*) mesh.vertex_buffer[0], &iarg1, &iarg2);
      boundconds_y_c((AcReal*) mesh.vertex_buffer[0], &iarg1, &iarg2);
      boundconds_z_c((AcReal*) mesh.vertex_buffer[0], &iarg1, &iarg2);
    //   loadOuterFront(mesh,STREAM_6);

    //   loadOuterLeft(mesh,STREAM_4);
    //   loadOuterRight(mesh,STREAM_5);
    //   loadOuterBot(mesh,STREAM_2);
    //   loadOuterTop(mesh,STREAM_3);
    //   loadOuterBack(mesh,STREAM_1);
      loadOuterHalos(mesh);
      acSynchronize();
    //   acIntegrateStep(isubstep-1,dt);
    // start = {3,3,3};
    // end = {131,131,64};
    // acNodeIntegrateSubstep(node, STREAM_1, isubstep-1,start,end,dt);

    // start = {3,3,64};
    // end = {131,131,131};
    // acNodeIntegrateSubstep(node, STREAM_1, isubstep-1,start,end,dt);
    //   acSynchronizeMesh();  // only for peer-to-peer

      start=(int3){l1,m1i+2,n1i+2}-1; end=(int3){l1i+1,m2i-2,n2i-2}-1+1;   // integrate inner left plate
      if (start.x<end.x && start.y<end.y && start.z<end.z) {
      if (l0) {printf("start,end inner left= %d %d %d %d %d %d \n",start.x,start.y,start.z,end.x,end.y,end.z);}
      acNodeIntegrateSubstep(node, STREAM_4, isubstep-1, start, end, dt);
      }

      start=(int3){l2i-1,m1i+2,n1i+2}-1; end=(int3){l2,m2i-2,n2i-2}-1+1;   // integrate inner right plate
      if (start.x<end.x && start.y<end.y && start.z<end.z) {
      if (l0) {printf("start,end inner right= %d %d %d %d %d %d \n",start.x,start.y,start.z,end.x,end.y,end.z);}
      acNodeIntegrateSubstep(node, STREAM_5, isubstep-1, start, end, dt);
      }
      start=(int3){l1,m1,n1i+2}-1; end=(int3){l2,m1i+1,n2i-2}-1+1;         // integrate inner bottom plate
      if (start.x<end.x && start.y<end.y && start.z<end.z) {
      if (l0) {printf("start,end inner bottom= %d %d %d %d %d %d \n",start.x,start.y,start.z,end.x,end.y,end.z);}
      acNodeIntegrateSubstep(node, STREAM_2, isubstep-1, start, end, dt);
      }
      start=(int3){l1,m2i-1,n1i+2}-1; end=(int3){l2,m2,n2i-2}-1+1;         // integrate inner top plate
      if (start.x<end.x && start.y<end.y && start.z<end.z) {
      if (l0) {printf("start,end inner top= %d %d %d %d %d %d \n",start.x,start.y,start.z,end.x,end.y,end.z);}
      acNodeIntegrateSubstep(node, STREAM_3, isubstep-1, start, end, dt);
      }
      start=(int3){l1,m1,n1}-1; end=(int3){l2,m2,n1i+1}-1+1;               // integrate inner front plate
      if (start.x<end.x && start.y<end.y && start.z<end.z) {
      if (l0) {printf("start,end inner front= %d %d %d %d %d %d \n",start.x,start.y,start.z,end.x,end.y,end.z);}
      acNodeIntegrateSubstep(node, STREAM_6, isubstep-1, start, end, dt);
      }
      start=(int3){l1,m1,n2i-1}-1; end=(int3){l2,m2,n2}-1+1;               // integrate inner back plate
      if (start.x<end.x && start.y<end.y && start.z<end.z) {
      if (l0) {printf("start,end inner back= %d %d %d %d %d %d \n",start.x,start.y,start.z,end.x,end.y,end.z); l0=0;}
      acNodeIntegrateSubstep(node, STREAM_1, isubstep-1, start, end, dt);
      }
      ERRCHK_CUDA_KERNEL_ALWAYS();
      acNodeSwapBuffers(node);
      acSynchronize();

      storeInnerHalos(mesh);
//       storeInnerLeft(mesh,STREAM_4);
//       storeInnerRight(mesh,STREAM_5);
//       storeInnerBot(mesh,STREAM_2);
//       storeInnerTop(mesh,STREAM_3);
//       storeInnerFront(mesh,STREAM_6);
//       storeInnerBack(mesh,STREAM_1);
#endif
      acSynchronize();

    }  // end not early_finalize
}
/***********************************************************************************************/
extern "C" void registerGPU(AcReal* farray)
{
    size_t offset=0;

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        //mesh.vertex_buffer[VertexBufferHandle(i)] = (AcReal*)farray+offset;
        mesh.vertex_buffer[VertexBufferHandle(i)] = &farray[offset];
printf("&farray[offset],value,i= %p %10.4f, %d\n", &farray[offset],farray[offset],VertexBufferHandle(i));
        offset+=mw;
    }
}
/***********************************************************************************************/
extern "C" void initGPU()
{
    // Initialize GPUs in the node
    AcResult res=acCheckDeviceAvailability();
}
/***********************************************************************************************/
void setupConfig(AcMeshInfo & config){

printf("nx etc. %d %d %d %.14f %.14f %.14f \n",nxgrid,nygrid,nzgrid,dx,dy,dz);
     config.int_params[AC_nx]=nxgrid;
     config.int_params[AC_ny]=nygrid;
     config.int_params[AC_nz]=nzgrid;

//           config.int_params[AC_nx]=nx;
//      config.int_params[AC_ny]=ny;
//      config.int_params[AC_nz]=nz;

     config.int_params[AC_mx] = mx;
     config.int_params[AC_my] = my;
     config.int_params[AC_mz] = mz;
//      config.int_params[AC_nx_min] = l1;
//      config.int_params[AC_nx_max] = l2;
//      config.int_params[AC_ny_min] = m1;
//      config.int_params[AC_ny_max] = m2;
//      config.int_params[AC_nz_min] = n1;
//      config.int_params[AC_nz_max] = n2;
//      config.int_params[AC_mxy]  = mx*my;
//      config.int_params[AC_nxy]  = nx*ny;
//      config.int_params[AC_nxyz] = nw;
     config.int_params[AC_xy_plate_bufsize] = halo_xy_size;
     config.int_params[AC_xz_plate_bufsize] = halo_xz_size;
     config.int_params[AC_yz_plate_bufsize] = halo_yz_size;

     config.real_params[AC_dsx]=dx;
     config.real_params[AC_dsy]=dy;
     config.real_params[AC_dsz]=dz;
printf("%d: l1i etc. %d %d %d %d %d %d \n", pid, l1i,l2i,n1i,n2i,m1i,m2i);
printf("%d: l1 etc. %d %d %d %d %d %d \n", pid, l1,l2,n1,n2,m1,m2);
     //config.real_params[AC_inv_dsx] = 1./dx;
     //config.real_params[AC_inv_dsy] = 1./dy;
     //config.real_params[AC_inv_dsz] = 1./dz;
     config.real_params[AC_dsmin]   = std::min(dx,std::min(dy,dz));
     config.real_params[AC_xlen]=lxyz[0];
     config.real_params[AC_ylen]=lxyz[1];
     config.real_params[AC_zlen]=lxyz[2];
     config.real_params[AC_xorig]=xyz0[0];
     config.real_params[AC_yorig]=xyz0[1];
     config.real_params[AC_zorig]=xyz0[2];
printf("lxyz etc. %f %f %f %f %f %f \n",lxyz[0],lxyz[1],lxyz[2],xyz0[0],xyz0[1],xyz0[2]);
     config.real_params[AC_unit_density]=unit_density;
     config.real_params[AC_unit_velocity]=unit_velocity;
     config.real_params[AC_unit_length]=unit_length;
     config.real_params[AC_mu0]=mu0;
//printf("units etc. %lf %lf %lf \n", unit_density, unit_velocity, unit_length);

#include "PC_modulepars.h"

}
/***********************************************************************************************/
void checkConfig(AcMeshInfo & config){
//printf("setupConfig:mesh.info.real_params[AC_k1_ff]= %f \n",config.real_params[AC_k1_ff]);
#if LENTROPY
     printf("lpressuregradientgas= %d %d \n", lpressuregradient_gas, config.int_params[AC_lpressuregradient_gas]);
#endif
#if LENTROPY
     printf("chi= %f %f \n", chi, config.real_params[AC_chi]);
#endif
#if LVISCOSITY
     printf("nu= %f %f \n", nu, config.real_params[AC_nu]);
     printf("zeta= %f %f \n", zeta, config.real_params[AC_zeta]);
#endif
#if LMAGNETIC
     printf("eta= %f %f \n", eta, config.real_params[AC_eta]);
#endif
#if LEOS
     printf("cs20= %f %f \n", cs20, config.real_params[AC_cs20]);
     printf("gamma= %f %f \n", gamma, config.real_params[AC_gamma]);
     printf("cv= %f %f \n", cv, config.real_params[AC_cv]);
     printf("cp= %f %f \n", cp, config.real_params[AC_cp]);
     printf("lnTT0= %f %f \n", lnTT0, config.real_params[AC_lnTT0]);
     printf("lnrho0= %f %f \n", lnrho0, config.real_params[AC_lnrho0]);
#endif
#if LFORCING
     printf("iforcing_zsym= %f %f \n", iforcing_zsym, config.int_params[AC_iforcing_zsym]);
     printf("k1_ff= %f %f \n", k1_ff, config.real_params[AC_k1_ff]);
     printf("tforce_stop= %f %f \n", tforce_stop, config.real_params[AC_tforce_stop]);
     //printf("k1_ff,profx_ampl, val= %f %d %lf %lf\n", k1_ff, profx_ampl, profx_ampl[0], profx_ampl[nx-1]);
#endif
}
/***********************************************************************************************/
#define PUT(ptr,n_x,n_y,n_z) \
  acNodeLoadScalarArray(node, STREAM_DEFAULT, AC_##ptr, ptr, (int3){nx,ny,nz});

void loadProfiles(AcMeshInfo & config){
#if LFORCING
     PUT(profx_ampl,nx,0,0)
     PUT(profy_ampl,0,my,0)
     PUT(profz_ampl,0,0,mz)
     PUT(profx_hel,nx,0,0)
     PUT(profy_hel,0,my,0)
     PUT(profz_hel,0,0,mz)
#endif
}
/***********************************************************************************************/
extern "C" void initializeGPU(AcReal **farr_GPU_in, AcReal **farr_GPU_out)
{
    //Setup configurations used for initializing and running the GPU code
#if PACKED_DATA_TRANSFERS
        initLoadStore();
#endif
        MPI_Comm_rank(MPI_COMM_WORLD, &pid);
        setupConfig(mesh.info);
        checkConfig(mesh.info);
//printf("mesh.info.real_params[AC_k1_ff]= %f \n",mesh.info.real_params[AC_k1_ff]);
        AcResult res=acInit(mesh.info,iproc);
        node=acGetNode();
        acNodeQueryDeviceConfiguration(node, &devConfig);
        loadProfiles(mesh.info);
        acHostUpdateBuiltinParams(&mesh.info);



    acGridSetDomainDecomposition({nprocx,nprocy,nprocz});
    acGridInit(mesh.info);

        AcReal *p[2];
        if (acNodeGetVBApointers(&node, p)==AC_SUCCESS) {
          *farr_GPU_in=p[0];
          *farr_GPU_out=p[1];
printf("From grid layer: vbapointer= %p %p \n", *farr_GPU_in, *farr_GPU_out);
        } else {
          *farr_GPU_in=NULL;
          *farr_GPU_out=NULL;
        }
    VertexBufferHandle all_fields[NUM_VTXBUF_HANDLES];
    for(int i=0;i<NUM_VTXBUF_HANDLES;i++){
        all_fields[i] = (VertexBufferHandle)i;
    }
    AcTaskDefinition build_graph_1[] = {
        acCompute(KERNEL_twopass_solve_intermediate, all_fields, 0, 0),
        acCompute(KERNEL_twopass_solve_final, all_fields, 0, 0),
        // acCompute(KERNEL_singlepass_solve, all_fields, 0, 0),
        // acCompute(KERNEL_check, all_fields),
    };
    graph_1 = acGridBuildTaskGraph(build_graph_1);

    AcTaskDefinition build_graph_2[] = {
        acCompute(KERNEL_twopass_solve_intermediate, all_fields, 0, 1),
        acCompute(KERNEL_twopass_solve_final, all_fields, 0, 1),
        // acCompute(KERNEL_singlepass_solve, all_fields, 0, 1),
//         acCompute(KERNEL_check, all_fields),
    };
    graph_2 = acGridBuildTaskGraph(build_graph_2);

    AcTaskDefinition build_graph_3[] = {
        acCompute(KERNEL_twopass_solve_intermediate, all_fields, 0, 2),
        acCompute(KERNEL_twopass_solve_final, all_fields, 0, 2),
        // acCompute(KERNEL_singlepass_solve, all_fields, 0, 2),
//         acCompute(KERNEL_check, all_fields),
    };
    graph_3 = acGridBuildTaskGraph(build_graph_3);

    // initialize diagnostics
       //init_diagnostics();
}
/***********************************************************************************************/
extern "C" void copyFarray() 
{
       AcResult res=acGridStoreMesh(STREAM_DEFAULT,&mesh);
printf("store all %d \n",res); fflush(stdout);
}
/***********************************************************************************************/
extern "C" void finalizeGPU()
{
#if PACKED_DATA_TRANSFERS
       finalLoadStore();
#endif
    // Deallocate everything on the GPUs and reset
       AcResult res=acQuit();
}
/***********************************************************************************************/
