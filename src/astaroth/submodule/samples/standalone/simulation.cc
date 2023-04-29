/*
    Copyright (C) 2014-2022, Johannes Pekkila, Miikka Vaisala.

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
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "run.h"

#include "config_loader.h"
#include "errchk.h"
#include "math_utils.h"
#include "model/host_forcing.h"
#include "model/host_memory.h"
#include "model/host_timestep.h"
#include "model/model_reduce.h"
#include "model/model_rk3.h"
#include "timer_hires.h"

#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// NEED TO BE DEFINED HERE. IS NOT NOTICED BY compile_acc call.
#define LFORCING (0)
#define LSHOCK (0)

#ifdef VTXBUF_ACCRETION
#define LSINK (1)
#else
#define LSINK (0)
#endif

#ifdef BFIELDX
#define LBFIELD (1)
#else
#define LBFIELD (0)
#endif

// Write all setting info into a separate ascii file. This is done to guarantee
// that we have the data specifi information in the thing, even though in
// principle these things are in the astaroth.conf.
static inline void
write_mesh_info(const AcMeshInfo* config)
{

    FILE* infotxt;

    infotxt = fopen("purge.sh", "w");
    fprintf(infotxt, "#!/bin/bash\n");
    fprintf(infotxt, "rm *.list *.mesh *.ts purge.sh\n");
    fclose(infotxt);

    infotxt = fopen("mesh_info.list", "w");

    // Determine endianness
    unsigned int EE      = 1;
    char* CC             = (char*)&EE;
    const int endianness = (int)*CC;
    // endianness = 0 -> big endian
    // endianness = 1 -> little endian

    fprintf(infotxt, "size_t %s %lu \n", "AcRealSize", sizeof(AcReal));

    fprintf(infotxt, "int %s %i \n", "endian", endianness);

    // JP: this could be done shorter and with smaller chance for errors with the following
    // (modified from acPrintMeshInfo() in astaroth.cu)
    // MV: Now adapted into working condition. E.g. removed useless / harmful formatting.

    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        fprintf(infotxt, "int %s %d\n", intparam_names[i], config->int_params[i]);

    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        fprintf(infotxt, "int3 %s  %d %d %d\n", int3param_names[i], config->int3_params[i].x,
                config->int3_params[i].y, config->int3_params[i].z);

    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        fprintf(infotxt, "real %s %g\n", realparam_names[i], double(config->real_params[i]));

    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        fprintf(infotxt, "real3 %s  %g %g %g\n", real3param_names[i],
                double(config->real3_params[i].x), double(config->real3_params[i].y),
                double(config->real3_params[i].z));

    fclose(infotxt);
}

// This funtion writes a run state into a set of C binaries.
static inline void
save_mesh(const AcMesh& save_mesh, const int step, const AcReal t_step)
{
    FILE* save_ptr;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = acVertexBufferSize(save_mesh.info);

        const char* buffername = vtxbuf_names[w];
        char cstep[11];
        char bin_filename[80] = "\0";

        // sprintf(bin_filename, "");

        sprintf(cstep, "%d", step);

        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        strcat(bin_filename, ".mesh");

        printf("Savefile %s \n", bin_filename);

        save_ptr = fopen(bin_filename, "wb");

        // Start file with time stamp
        AcReal write_long_buf = (AcReal)t_step;
        fwrite(&write_long_buf, sizeof(AcReal), 1, save_ptr);
        // Grid data
        for (size_t i = 0; i < n; ++i) {
            const AcReal point_val = save_mesh.vertex_buffer[VertexBufferHandle(w)][i];
            AcReal write_long_buf2 = (AcReal)point_val;
            fwrite(&write_long_buf2, sizeof(AcReal), 1, save_ptr);
        }
        fclose(save_ptr);
    }
}

/*
// This funtion writes a run state into a set of C binaries.
static inline void
save_slice_cut(const AcMesh& save_mesh, const int step, const AcReal t_step)
{
    FILE* save_ptr;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = acVertexBufferSize(save_mesh.info);

        const char* buffername = vtxbuf_names[w];
        char cstep[11];
        char bin_filename_xy[80] = "\0";
        char bin_filename_xz[80] = "\0";
        char bin_filename_yz[80] = "\0";

        // sprintf(bin_filename, "");

        sprintf(cstep, "%d", step);

        strcat(bin_filename_xy, buffername);
        strcat(bin_filename_xy, "_");
        strcat(bin_filename_xy, cstep);
        strcat(bin_filename_xy, ".mxy");

        strcat(bin_filename_xz, buffername);
        strcat(bin_filename_xz, "_");
        strcat(bin_filename_xz, cstep);
        strcat(bin_filename_xz, ".mxz");

        strcat(bin_filename_yz, buffername);
        strcat(bin_filename_yz, "_");
        strcat(bin_filename_yz, cstep);
        strcat(bin_filename_yz, ".myz");

        printf("Slice files %s, %s, %s, \n",
               bin_filename_xy, bin_filename_xz, bin_filename_yz);

        save_ptr = fopen(bin_filename, "wb");

        // Start file with time stamp
        AcReal write_long_buf = (AcReal)t_step;
        fwrite(&write_long_buf, sizeof(AcReal), 1, save_ptr);
        // Grid data
        for (size_t i = 0; i < n; ++i) {
            const AcReal point_val     = save_mesh.vertex_buffer[VertexBufferHandle(w)][i];
            AcReal write_long_buf2 = (AcReal)point_val;
            fwrite(&write_long_buf2, sizeof(AcReal), 1, save_ptr);
        }
        fclose(save_ptr);
    }
}
*/

// This funtion reads a run state from a set of C binaries.
static inline void
read_mesh(AcMesh& read_mesh, const int step, AcReal* t_step)
{
    FILE* read_ptr;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = acVertexBufferSize(read_mesh.info);

        const char* buffername = vtxbuf_names[w];
        char cstep[11];
        char bin_filename[80] = "\0";

        // sprintf(bin_filename, "");

        sprintf(cstep, "%d", step);

        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        strcat(bin_filename, ".mesh");

        printf("Reading savefile %s \n", bin_filename);

        read_ptr = fopen(bin_filename, "rb");

        // Start file with time stamp
        size_t result;
        result = fread(t_step, sizeof(AcReal), 1, read_ptr);
        // Read grid data
        AcReal read_buf;
        for (size_t i = 0; i < n; ++i) {
            result = fread(&read_buf, sizeof(AcReal), 1, read_ptr);
            read_mesh.vertex_buffer[VertexBufferHandle(w)][i] = read_buf;
            if (int(result) != 1) {
                fprintf(stderr, "Reading error in %s, element %i\n", vtxbuf_names[w], int(i));
                fprintf(stderr, "Result = %i,  \n", int(result));
            }
        }
        fclose(read_ptr);
    }
}

// This function prints out the diagnostic values to std.out and also saves and
// appends an ascii file to contain all the result.
static inline void
print_diagnostics(const int step, const AcReal dt, const AcReal t_step, FILE* diag_file,
                  const AcReal sink_mass, const AcReal accreted_mass, int* found_nan)
{

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    buf_max = acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_min = acReduceVec(RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_rms = acReduceVec(RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);

    // MV: The ordering in the earlier version was wrong in terms of variable
    // MV: name and its diagnostics.
    printf("Step %d, t_step %.3e, dt %e s\n", step, double(t_step), double(dt));
    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%d %e %e %e %e %e ", step, double(t_step), double(dt), double(buf_min),
            double(buf_rms), double(buf_max));

#if LBFIELD
    buf_max = acReduceVec(RTYPE_MAX, BFIELDX, BFIELDY, BFIELDZ);
    buf_min = acReduceVec(RTYPE_MIN, BFIELDX, BFIELDY, BFIELDZ);
    buf_rms = acReduceVec(RTYPE_RMS, BFIELDX, BFIELDY, BFIELDZ);

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "bb total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));

    buf_max = acReduceVecScal(RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);
    buf_min = acReduceVecScal(RTYPE_ALFVEN_MIN, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);
    buf_rms = acReduceVecScal(RTYPE_ALFVEN_RMS, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "vA total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
#endif

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        buf_max = acReduceScal(RTYPE_MAX, VertexBufferHandle(i));
        buf_min = acReduceScal(RTYPE_MIN, VertexBufferHandle(i));
        buf_rms = acReduceScal(RTYPE_RMS, VertexBufferHandle(i));

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, vtxbuf_names[i],
               double(buf_min), double(buf_rms), double(buf_max));
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));

        if (isnan(buf_max) || isnan(buf_min) || isnan(buf_rms)) {
            *found_nan = 1;
        }
    }

    if ((sink_mass >= AcReal(0.0)) || (accreted_mass >= AcReal(0.0))) {
        fprintf(diag_file, "%e %e ", double(sink_mass), double(accreted_mass));
    }

    fprintf(diag_file, "\n");
}

static inline void
print_diagnostics_device(const Device device, const int step, const AcReal dt, const AcReal t_step,
                         FILE* diag_file, const AcReal sink_mass, const AcReal accreted_mass,
                         int* found_nan, AcMeshInfo mesh_info)
{

    const int mx   = mesh_info.int_params[AC_nx];
    const int my   = mesh_info.int_params[AC_ny];
    const int mz   = mesh_info.int_params[AC_nz];
    const int mtot = mx * my * mz;

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                      &buf_max);
    acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                      &buf_min);
    acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                      &buf_rms);

    // acDeviceReduceVec calculates only the sum on var**2
    buf_rms = sqrt(buf_rms / AcReal(mtot));

    // MV: The ordering in the earlier version was wrong in terms of variable
    // MV: name and its diagnostics.
    printf("Step %d, t_step %.3e, dt %e s\n", step, double(t_step), double(dt));
    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%d %e %e %e %e %e ", step, double(t_step), double(dt), double(buf_min),
            double(buf_rms), double(buf_max));

#if LBFIELD
    acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_MAX, BFIELDX, BFIELDY, BFIELDZ, &buf_max);
    acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_MIN, BFIELDX, BFIELDY, BFIELDZ, &buf_min);
    acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_RMS, BFIELDX, BFIELDY, BFIELDZ, &buf_rms);
    buf_rms = sqrt(buf_rms / AcReal(mtot));

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "bb total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));

    acDeviceReduceVecScal(device, STREAM_DEFAULT, RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ,
                          VTXBUF_LNRHO, &buf_max)
        acDeviceReduceVecScal(device, STREAM_DEFAULT, RTYPE_ALFVEN_MIN, BFIELDX, BFIELDY, BFIELDZ,
                              VTXBUF_LNRHO, &buf_min)
            acDeviceReduceVecScal(device, STREAM_DEFAULT, RTYPE_ALFVEN_RMS, BFIELDX, BFIELDY,
                                  BFIELDZ, VTXBUF_LNRHO, &buf_rms)
                buf_rms = sqrt(buf_rms / AcReal(mtot));

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "vA total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
#endif

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MAX, VertexBufferHandle(i), &buf_max);
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, VertexBufferHandle(i), &buf_min);
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_RMS, VertexBufferHandle(i), &buf_rms);
        buf_rms = sqrt(buf_rms / AcReal(mtot));

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, vtxbuf_names[i],
               double(buf_min), double(buf_rms), double(buf_max));
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));

        if (isnan(buf_max) || isnan(buf_min) || isnan(buf_rms)) {
            *found_nan = 1;
        }
    }

    if ((sink_mass >= AcReal(0.0)) || (accreted_mass >= AcReal(0.0))) {
        fprintf(diag_file, "%e %e ", double(sink_mass), double(accreted_mass));
    }

    fprintf(diag_file, "\n");
}

/*
    MV NOTE: At the moment I have no clear idea how to calculate magnetic
    diagnostic variables from grid. Vector potential measures have a limited
    value. TODO: Smart way to get brms, bmin and bmax.
*/

int
run_simulation(const char* config_path)
{
    /* Parse configs */
    AcMeshInfo mesh_info;
    load_config(config_path, &mesh_info);

    AcMesh* mesh = acmesh_create(mesh_info);
    // TODO: This need to be possible to define in astaroth.conf
    acmesh_init_to(INIT_TYPE_GAUSSIAN_RADIAL_EXPL, mesh);
    // acmesh_init_to(INIT_TYPE_KICKBALL, mesh);
    // acmesh_init_to(INIT_TYPE_SIMPLE_CORE, mesh); //Initial condition for a collapse test

#if LSINK
    printf("WARNING! Sink particle is under development. USE AT YOUR OWN RISK!")
        vertex_buffer_set(VTXBUF_ACCRETION, 0.0, mesh);
#endif
#if LSHOCK
    vertex_buffer_set(VTXBUF_SHOCK, 0.0, mesh);
#endif

    // Read old binary if we want to continue from an existing snapshot
    // WARNING: Explicit specification of step needed!
    const int start_step = mesh_info.int_params[AC_start_step];
    AcReal t_step        = 0.0;
    if (start_step > 0) {
        read_mesh(*mesh, start_step, &t_step);
    }

#if LSHOCK
    Device device;
    acDeviceCreate(0, mesh_info, &device);
    acDevicePrintInfo(device);
    printf("Loading mesh to GPU.\n");
    acDeviceLoadMesh(device, STREAM_DEFAULT, *mesh);
#else
    acInit(mesh_info);
    acLoad(*mesh);
#endif

    printf("Mesh loaded to GPU(s).\n");

    FILE* diag_file;
    int found_nan = 0, found_stop = 0; // Nan or inf finder to give an error signal
    diag_file = fopen("timeseries.ts", "a");

    // Generate the title row.
    if (start_step == 0) {
        fprintf(diag_file, "step  t_step  dt  uu_total_min  uu_total_rms  uu_total_max  ");
#if LBFIELD
        fprintf(diag_file, "bb_total_min  bb_total_rms  bb_total_max  ");
        fprintf(diag_file, "vA_total_min  vA_total_rms  vA_total_max  ");
#endif
        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
            fprintf(diag_file, "%s_min  %s_rms  %s_max  ", vtxbuf_names[i], vtxbuf_names[i],
                    vtxbuf_names[i]);
        }
    }
#if LSINK
    fprintf(diag_file, "sink_mass  accreted_mass  ");
#endif
    fprintf(diag_file, "\n");

    write_mesh_info(&mesh_info);

    printf("Mesh info written to file.\n");

    if (start_step == 0) {
#if LSINK
        print_diagnostics(0, AcReal(.0), t_step, diag_file, mesh_info.real_params[AC_M_sink_init],
                          0.0, &found_nan);
#else
#if LSHOCK
        print_diagnostics_device(device, 0, AcReal(.0), t_step, diag_file, -1.0, -1.0, &found_nan,
                                 mesh_info);
#else
        print_diagnostics(0, AcReal(.0), t_step, diag_file, -1.0, -1.0, &found_nan);
#endif
#endif
    }

#if LSHOCK
    const int3 start  = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end    = (int3){mesh_info.int_params[AC_mx] - NGHOST,
                               mesh_info.int_params[AC_my] - NGHOST,
                               mesh_info.int_params[AC_mz] - NGHOST};
    const int3 bindex = (int3){0, 0, 0}; // DUMMY
    const int3 b1     = (int3){0, 0, 0};
    const int3 b2     = (int3){mesh_info.int_params[AC_mx], mesh_info.int_params[AC_mx],
                               mesh_info.int_params[AC_mx]};

    acDeviceGeneralBoundconds(device, STREAM_DEFAULT, b1, b2, mesh_info, bindex);
    acDeviceStoreMesh(device, STREAM_DEFAULT, mesh);
#else
    // acBoundcondStep();
    acBoundcondStepGBC(mesh_info);
    acStore(mesh);
#endif
    if (start_step == 0) {
        save_mesh(*mesh, 0, t_step);
    }

    const int max_steps      = mesh_info.int_params[AC_max_steps];
    const int save_steps     = mesh_info.int_params[AC_save_steps];
    const int bin_save_steps = mesh_info.int_params[AC_bin_steps];

    const AcReal max_time   = mesh_info.real_params[AC_max_time];
    const AcReal bin_save_t = mesh_info.real_params[AC_bin_save_t];
    AcReal bin_crit_t       = bin_save_t;

    /* initialize random seed: */
    srand(312256655);

#if LSHOCK
#endif

    /* Step the simulation */
    AcReal accreted_mass = 0.0;
    AcReal sink_mass     = 0.0;
    AcReal uu_freefall   = 0.0;
    AcReal dt_typical    = 0.0;
    int dtcounter        = 0;

    printf("Starting simulation...\n");
    for (int i = start_step + 1; i < max_steps; ++i) {
#if LSINK

        const AcReal sum_mass = acReduceScal(RTYPE_SUM, VTXBUF_ACCRETION);
        accreted_mass         = accreted_mass + sum_mass;
        sink_mass             = 0.0;
        sink_mass             = mesh_info.real_params[AC_M_sink_init] + accreted_mass;
        acLoadDeviceConstant(AC_M_sink, sink_mass);
        vertex_buffer_set(VTXBUF_ACCRETION, 0.0,
                          mesh); // TODO THIS IS A BUG! WILL ONLY SET HOST BUFFER 0!

        int on_off_switch;
        if (i < 1) {
            on_off_switch = 0; // accretion is off till certain amount of steps.
        }
        else {
            on_off_switch = 1;
        }
        acLoadDeviceConstant(AC_switch_accretion, on_off_switch);

        // Adjust courant condition for free fall velocity
        const AcReal RR    = mesh_info.real_params[AC_soft] * mesh_info.real_params[AC_soft];
        const AcReal SQ2GM = sqrt(AcReal(2.0) * mesh_info.real_params[AC_G_const] * sink_mass);
        uu_freefall        = fabs(SQ2GM / sqrt(RR));
#else
        accreted_mass          = -1.0;
        sink_mass              = -1.0;
#endif

#if LSHOCK
        AcReal umax, shock_max;
        acDeviceReduceVec(device, STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                          &umax);
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MAX, VTXBUF_SHOCK, &shock_max);
#else
        const AcReal shock_max = 0.0;
        const AcReal umax      = acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
#endif

#if LBFIELD
#if LSHOCK
        AcReal vAmax;
        acDeviceReduceVecScal(device, STREAM_DEFAULT, RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ,
                              VTXBUF_LNRHO, &vAmax)
#else
        const AcReal vAmax = acReduceVecScal(RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ,
                                             VTXBUF_LNRHO);
#endif
            const AcReal uref = max(max(umax, uu_freefall), vAmax);
        const AcReal dt       = host_timestep(uref, vAmax, shock_max, mesh_info);
#else
        const AcReal uref      = max(umax, uu_freefall);
        const AcReal dt        = host_timestep(uref, 0.0l, shock_max, mesh_info);
#endif

#if LFORCING
        const ForcingParams forcing_params = generateForcingParams(mesh_info);
        loadForcingParamsToDevice(forcing_params);
#endif

#if LSHOCK
        for (int isubstep = 0; isubstep < 3; ++isubstep) {
            // Call only singe GPU version on for testing the shock viscosity first
            acDevice_shock_1_divu(device, STREAM_DEFAULT, start, end);
            // acDeviceSynchronizeStream(device, STREAM_ALL);
            acDeviceSwapBuffer(device, VTXBUF_SHOCK);
            acDeviceGeneralBoundconds(device, STREAM_DEFAULT, b1, b2, mesh_info, bindex);
            // acDeviceSynchronizeStream(device, STREAM_ALL);

            acDevice_shock_2_max(device, STREAM_DEFAULT, start, end);
            acDeviceSwapBuffer(device, VTXBUF_SHOCK);
            // acDeviceSynchronizeStream(device, STREAM_ALL);
            acDeviceGeneralBoundconds(device, STREAM_DEFAULT, b1, b2, mesh_info, bindex);
            // acDeviceSynchronizeStream(device, STREAM_ALL);

            acDevice_shock_3_smooth(device, STREAM_DEFAULT, start, end);
            // acDeviceSynchronizeStream(device, STREAM_ALL);
            acDeviceSwapBuffer(device, VTXBUF_SHOCK);
            // acDeviceSynchronizeStream(device, STREAM_ALL);
            acDeviceGeneralBoundconds(device, STREAM_DEFAULT, b1, b2, mesh_info, bindex);

            // RUN SOLVE
            acDeviceIntegrateSubstep(device, STREAM_DEFAULT, isubstep, start, end, dt);
            // acDeviceSynchronizeStream(device, STREAM_ALL);
            acDeviceSwapBuffers(device);
            // TO compensate
            acDeviceSwapBuffer(device, VTXBUF_SHOCK);
            acDeviceSynchronizeStream(device, STREAM_ALL);
        }
#else
        /* Uses now flexible bokundary conditions */
        // acIntegrate(dt);
        acIntegrateGBC(mesh_info, dt);
#endif

        t_step += dt;

        /* Get the sense of a typical timestep */
        if (i < start_step + 100) {
            dt_typical = dt;
        }

        /* Save the simulation state and print diagnostics */
        if ((i % save_steps) == 0) {

            /*
                print_diagnostics() writes out both std.out printout from the
                results and saves the diagnostics into a table for ascii file
                timeseries.ts.
            */
#if LSHOCK
            print_diagnostics_device(device, i, dt, t_step, diag_file, sink_mass, accreted_mass,
                                     &found_nan, mesh_info);
#else
            print_diagnostics(i, dt, t_step, diag_file, sink_mass, accreted_mass, &found_nan);
#endif
#if LSINK
            printf("sink mass is: %.15e \n", double(sink_mass));
            printf("accreted mass is: %.15e \n", double(accreted_mass));
#endif
            /*
                We would also might want an XY-average calculating funtion,
                which can be very useful when observing behaviour of turbulent
                simulations. (TODO)
            */
        }

        /* Save the simulation state and print diagnostics */
        if ((i % bin_save_steps) == 0 || t_step >= bin_crit_t) {

            /*
                This loop saves the data into simple C binaries which can be
                used for analysing the data snapshots closely.

                The updated mesh will be located on the GPU. Also all calls
                to the astaroth interface (functions beginning with ac*) are
                assumed to be asynchronous, so the meshes must be also synchronized
                before transferring the data to the CPU. Like so:

                acBoundcondStep();
                acStore(mesh);
            */
            // acBoundcondStep();
#if LSHOCK
            acDeviceGeneralBoundconds(device, STREAM_DEFAULT, b1, b2, mesh_info, bindex);
            acDeviceStoreMesh(device, STREAM_DEFAULT, mesh);
#else
            acBoundcondStepGBC(mesh_info);
            acStore(mesh);
#endif
            save_mesh(*mesh, i, t_step);

            bin_crit_t += bin_save_t;
        }

        // End loop if max time reached.
        if (max_time > AcReal(0.0)) {
            if (t_step >= max_time) {
                printf("Time limit reached! at t = %e \n", double(t_step));
                break;
            }
        }

        // End loop if dt is too low
        if (dt < dt_typical / AcReal(1e5)) {
            if (dtcounter > 10) {
                printf("dt = %e TOO LOW! Ending run at t = %#e \n", double(dt), double(t_step));
                // acBoundcondStep();
                acBoundcondStepGBC(mesh_info);
                acStore(mesh);
                save_mesh(*mesh, i, t_step);
                break;
            }
            else {
                dtcounter += 1;
            }
        }
        else {
            dtcounter = 0;
        }

        // End loop if nan is found
        if (found_nan > 0) {
            printf("Found nan at t = %e \n", double(t_step));
#if LSHOCK
            acDeviceGeneralBoundconds(device, STREAM_DEFAULT, b1, b2, mesh_info, bindex);
            acDeviceStoreMesh(device, STREAM_DEFAULT, mesh);
#else
            // acBoundcondStep();
            acBoundcondStepGBC(mesh_info);
            acStore(mesh);
#endif
            save_mesh(*mesh, i, t_step);
            break;
        }

        // End loop if STOP file is found
        if (access("STOP", F_OK) != -1) {
            found_stop = 1;
        }
        else {
            found_stop = 0;
        }

        if (found_stop == 1) {
            printf("Found STOP file at t = %e \n", double(t_step));
#if LSHOCK
            acDeviceGeneralBoundconds(device, STREAM_DEFAULT, b1, b2, mesh_info, bindex);
            acDeviceStoreMesh(device, STREAM_DEFAULT, mesh);
#else
            // acBoundcondStep();
            acBoundcondStepGBC(mesh_info);
            acStore(mesh);
#endif
            save_mesh(*mesh, i, t_step);
            break;
        }
    }

    //////Save the final snapshot
    ////acSynchronize();
    ////acStore(mesh);

    ////save_mesh(*mesh, , t_step);

#if LSHOCK
    acDeviceDestroy(device);
#else
    acQuit();
#endif
    acmesh_destroy(mesh);

    fclose(diag_file);

    return 0;
}
