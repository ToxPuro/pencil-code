/*

    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

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
#define GEN_TEST_FILE (0)

/**
    Running: mpirun -np <num processes> <executable>
*/
#if AC_MPI_ENABLED
#include "astaroth.h"
#include "astaroth_utils.h"
#include "stencil_loader.h"

#include <iostream>
#include "timer_hires.h"
#include <chrono>
#include <getopt.h>
#include <mpi.h>
#include <unistd.h>

#include <cstdarg>
#include <cstring>
#include <ctime>

#include <map>
#include <set>
#include <vector>

#include "config_loader.h"
#include "errchk.h"
#include "host_forcing.h"
#include "host_memory.h"
#include "math_utils.h"

#include "simulation_control.h"
#include "simulation_rng.h"
#include "simulation_taskgraphs.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))

// IO directories
static const char* snapshot_dir = "output-snapshots";
static const char* slice_dir    = "output-slices";

#define fprintf(...)                                                                               \
    {                                                                                              \
        int tmppid;                                                                                \
        MPI_Comm_rank(acGridMPIComm(), &tmppid);                                                   \
        if (!tmppid) {                                                                             \
            fprintf(__VA_ARGS__);                                                                  \
        }                                                                                          \
    }

// TODO: currently, fprintf has been replaced by this macro
// This is a little ugly, and could be fixed with a nicer solution

// Write all setting info into a separate ascii file. This is done to guarantee
// that we have the data specifi information in the thing, even though in
// principle these things are in the astaroth.conf.
static inline void
write_info(const AcMeshInfo* config)
{

    FILE* infotxt;

    infotxt = fopen("purge.sh", "w");
    fprintf(infotxt, "#!/bin/bash\n");
    fprintf(infotxt, "rm *.list *.mesh *.csv *.field *.ts purge.sh autotune-result.out\n");
    fprintf(infotxt, "rm -rf *output-s*\n");
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

/*
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
*/

/*
// This funtion writes a run state into a set of C binaries
// WITH MPI_IO
static inline void
save_mesh_mpi_sync(const AcMeshInfo info, const int pid, const int step, const AcReal t_step)
{
    printf("Saving snapshot at step %i \n", step);

    char cstep[11];
    // char header_filename[80] = "\0";
    sprintf(cstep, "%d", step);

    // Saves a csv file which contains relevant information about the binary
    // snapshot files at the timestep.
    if (pid == 0) {
        FILE* header_file = fopen("snapshots_info.csv", "a");

        // Header only at the step zero
        if (step == 0) {
            fprintf(header_file, "use_double, mx, my, mz, step_number, t_step \n");
        }

        fprintf(header_file, "%d, %d, %d, %d, %d, %.17e \n", sizeof(AcReal) == 8,
                info.int_params[AC_mx], info.int_params[AC_my], info.int_params[AC_mz], step,
                t_step);

        // Writes the header info. Make it into an
        // appendaple csv table which will be easy to be read into a Pandas
        // dataframe.

        fclose(header_file);
    }

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const char* buffername = vtxbuf_names[w];
        char bin_filename[80]  = "\0";

        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        // strcat(bin_filename, ".field");

        // Grid data
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, snapshot_dir, bin_filename,
                                          ACCESS_WRITE);

        printf("Savefile %s \n", bin_filename);

        acGridDiskAccessSync();
    }
}
*/

/* Calls acGridDiskAccessSync before acGridWriteMeshToDiskLaunch, but need to call
acGridDiskAccessSync after the final step or before any other IO operation. */
static inline void
save_mesh_mpi_async(const AcMeshInfo info, const char* job_dir, const int pid, const int step,
                    const AcReal simulation_time)
{
    debug_log_from_root_proc_with_sim_progress(pid,
                                               "save_mesh_mpi_async: Syncing mesh disk access\n");
    acGridDiskAccessSync();                   // NOTE: important sync
    acGridPeriodicBoundconds(STREAM_DEFAULT); // Debug, may be unneeded
    acGridSynchronizeStream(STREAM_DEFAULT);  // Debug, may be unneeded
    MPI_Barrier(acGridMPIComm());             // Debug may be unneeded

    const int num_snapshots = 2;
    const int modstep       = (step / info.int_params[AC_bin_steps]) % num_snapshots;
    log_from_root_proc_with_sim_progress(pid,
                                         "save_mesh_mpi_async: Writing snapshot to %s, timestep %d "
                                         "(slot %d of %d)\n",
                                         job_dir, step, modstep, num_snapshots);

    // Saves a csv file which contains relevant information about the binary
    // snapshot files at the timestep.
    if (pid == 0) {
        FILE* header_file = fopen("snapshots_info.csv", "a");

        // Header only at the step zero
        if (step == 0) {
            fprintf(header_file,
                    "use_double, mx, my, mz, step_number, modstep, t_step, t_step (exact)\n");
        }

        fprintf(header_file, "%d, %d, %d, %d, %d, %d, %g, %la\n", sizeof(AcReal) == 8,
                info.int_params[AC_mx], info.int_params[AC_my], info.int_params[AC_mz], step,
                modstep, simulation_time, simulation_time);

        // Writes the header info. Make it into an
        // appendaple csv table which will be easy to be read into a Pandas
        // dataframe.

        fclose(header_file);
    }
    MPI_Barrier(acGridMPIComm()); // Ensure header closes before initializing the next write

    char cstep[1024];
    sprintf(cstep, "%d", modstep);
    acGridWriteMeshToDiskLaunch(job_dir, cstep);
    log_from_root_proc_with_sim_progress(pid, "save_mesh_mpi_async: Non-blocking snapshot write "
                                              "operation started, returning\n");

    // printf("Write mesh to disk launch %s, %s \n", job_dir, cstep);
    /*
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const char* buffername = vtxbuf_names[w];
        char bin_filename[80] = "\0";


        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        //strcat(bin_filename, ".field");

        // Grid data
        //acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, job_dir, bin_filename,
    ACCESS_WRITE); acGridDiskAccessSync(); acGridWriteMeshToDiskLaunch(job_dir, step);
        // %JP TODO write async

        printf("Savefile %s \n", bin_filename);
    }*/
}

/*
// This funtion reads a run state into a set of C binaries
// WITH MPI_IO
static inline void
read_mesh_mpi(const int pid, const int step, AcReal* simulation_time)
{
    int stepnumber;
    AcReal time_at_step;
    double time;

    printf("Reading snapshot at step %i \n", step);
    char cstep[11];
    sprintf(cstep, "%d", step);

    if (pid == 0) {

        AcReal element[8];

        // Saves a csv file which contains relevant information about the binary
        // snapshot files at the timestep.
        FILE* header_file = fopen("snapshots_info.csv", "r");

        // TODO: Loop through the header file to find the step number of snapshots
        // TODO: to be read. And read the relevat other info.

        // Simple cvs file reader.
        char csv_line[256];
        while (fgets(csv_line, sizeof(csv_line), header_file) != NULL) {
            int column_index = 0;
            for (char* csv_loc = strtok(csv_line, ","); csv_loc != NULL;
                 csv_loc       = strtok(NULL, ",")) {
                printf("%s, ", csv_loc);
                element[column_index++] = atof(csv_loc);
            }
            printf("\n");
            stepnumber   = int(element[4]);
            time_at_step = element[5];
            // printf("stepnumber %i at time_at_step %e \n", stepnumber, time_at_step);

            if (stepnumber == step) {
                time = double(time_at_step);
            }
        }

        fclose(header_file);
    }

    MPI_Bcast(&time, 1, MPI_DOUBLE, 0, acGridMPIComm());

    *simulation_time = time;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const char* buffername = vtxbuf_names[w];
        char bin_filename[80]  = "\0";

        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        // strcat(bin_filename, ".field");

        // Grid data
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, snapshot_dir, bin_filename,
                                          ACCESS_READ);

        printf("Read file %s \n", bin_filename);

        acGridDiskAccessSync();
    }
}

// This funtion reads a run state from a set of C binaries.
static inline void
read_mesh(AcMesh& read_mesh, const int step, AcReal* simulation_time)
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
                result = fread(simulation_time, sizeof(AcReal), 1, read_ptr);
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
*/

static inline void
print_diagnostics_header_from_root_proc(int pid, FILE* diag_file)
{
    // Generate the file header (from root)
    if (pid == 0) {
        fprintf(diag_file, "step  t_step  dt  uu_total_min  uu_total_rms  uu_total_max  ");
#if LBFIELD
        fprintf(diag_file, "bb_total_min  bb_total_rms  bb_total_max  ");
        fprintf(diag_file, "vA_total_min  vA_total_rms  vA_total_max  ");
#endif
        for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
            fprintf(diag_file, "%s_min  %s_rms  %s_max  ", vtxbuf_names[i], vtxbuf_names[i],
                    vtxbuf_names[i]);
        }
#if LSINK
        fprintf(diag_file, "sink_mass  accreted_mass  ");
#endif
        fprintf(diag_file, "\n");
    }
}

/*
// This function prints out the diagnostic values to std.out and also saves and
// appends an ascii file to contain all the result.
// JP: MUST BE CALLED FROM PROC 0. Must be rewritten for multiple processes (this implementation
// has write race condition)
static inline void
print_diagnostics_host(const AcMesh mesh, const int step, const AcReal dt, const AcReal t_step,
                       FILE* diag_file, const AcReal sink_mass, const AcReal accreted_mass)
{

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    buf_max = acHostReduceVec(mesh, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_min = acHostReduceVec(mesh, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_rms = acHostReduceVec(mesh, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);

    // MV: The ordering in the earlier version was wrong in terms of variable
    // MV: name and its diagnostics.
    printf("Step %d, t_step %.3e, dt %e s\n", step, double(t_step), double(dt));
    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%d %e %e %e %e %e ", step, double(t_step), double(dt), double(buf_min),
            double(buf_rms), double(buf_max));
#if LBFIELD
    buf_max = acHostReduceVec(mesh, RTYPE_MAX, BFIELDX, BFIELDY, BFIELDZ);
    buf_min = acHostReduceVec(mesh, RTYPE_MIN, BFIELDX, BFIELDY, BFIELDZ);
    buf_rms = acHostReduceVec(mesh, RTYPE_RMS, BFIELDX, BFIELDY, BFIELDZ);

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "bb total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));

    buf_max = acHostReduceVecScal(mesh, RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);
    buf_min = acHostReduceVecScal(mesh, RTYPE_ALFVEN_MIN, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);
    buf_rms = acHostReduceVecScal(mesh, RTYPE_ALFVEN_RMS, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "vA total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
#endif

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        buf_max = acHostReduceScal(mesh, RTYPE_MAX, VertexBufferHandle(i));
        buf_min = acHostReduceScal(mesh, RTYPE_MIN, VertexBufferHandle(i));
        buf_rms = acHostReduceScal(mesh, RTYPE_RMS, VertexBufferHandle(i));

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, vtxbuf_names[i],
               double(buf_min), double(buf_rms), double(buf_max));
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
    }

    if ((sink_mass >= AcReal(0.0)) || (accreted_mass >= AcReal(0.0))) {
        fprintf(diag_file, "%e %e ", double(sink_mass), double(accreted_mass));
    }

    fprintf(diag_file, "\n");
}
*/

// This function prints out the diagnostic values to std.out and also saves and
// appends an ascii file to contain all the result.
// JP: EXECUTES ON MULTIPLE GPUS, MUST BE CALLED FROM ALL PROCS
static inline void
print_diagnostics(const int pid, const int step, const AcReal dt, const AcReal simulation_time,
                  FILE* diag_file, const AcReal sink_mass, const AcReal accreted_mass,
                  int* found_nan)
{

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_max);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_min);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_rms);

    acLogFromRootProc(pid, "Step %d, t_step %.3e, dt %e s\n", step, double(simulation_time),
                      double(dt));
    acLogFromRootProc(pid, "  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total",
                      double(buf_min), double(buf_rms), double(buf_max));
    if (pid == 0) {
        fprintf(diag_file, "%d %e %e %e %e %e ", step, double(simulation_time), double(dt),
                double(buf_min), double(buf_rms), double(buf_max));
    }

#if LBFIELD
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, BFIELDX, BFIELDY, BFIELDZ, &buf_max);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MIN, BFIELDX, BFIELDY, BFIELDZ, &buf_min);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, BFIELDX, BFIELDY, BFIELDZ, &buf_rms);

    acLogFromRootProc(pid, "  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "bb total",
                      double(buf_min), double(buf_rms), double(buf_max));
    if (pid == 0) {
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
    }

    acGridReduceVecScal(STREAM_DEFAULT, RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO,
                        &buf_max);
    acGridReduceVecScal(STREAM_DEFAULT, RTYPE_ALFVEN_MIN, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO,
                        &buf_min);
    acGridReduceVecScal(STREAM_DEFAULT, RTYPE_ALFVEN_RMS, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO,
                        &buf_rms);

    acLogFromRootProc(pid, "  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "vA total",
                      double(buf_min), double(buf_rms), double(buf_max));
    if (pid == 0) {
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
    }
#endif

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, VertexBufferHandle(i), &buf_max);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, VertexBufferHandle(i), &buf_min);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_RMS, VertexBufferHandle(i), &buf_rms);

        acLogFromRootProc(pid, "  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width,
                          vtxbuf_names[i], double(buf_min), double(buf_rms), double(buf_max));
        if (pid == 0) {
            fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
        }

        if (isnan(buf_max) || isnan(buf_min) || isnan(buf_rms)) {
            *found_nan = 1;
        }
    }

    if ((sink_mass >= AcReal(0.0)) || (accreted_mass >= AcReal(0.0))) {
        if (pid == 0) {
            fprintf(diag_file, "%e %e ", double(sink_mass), double(accreted_mass));
        }
    }

    if (pid == 0) {
        fprintf(diag_file, "\n");
    }

#if LSINK
    acLogFromRootProc(pid, "sink mass is: %.15e \n", double(sink_mass));
    acLogFromRootProc(pid, "accreted mass is: %.15e \n", double(accreted_mass));
#endif

    fflush(diag_file);
    fflush(stdout);
}

/*
    MV NOTE: At the moment I have no clear idea how to calculate magnetic
    diagnostic variables from grid. Vector potential measures have a limited
    value. TODO: Smart way to get brms, bmin and bmax.
*/

#include "math_utils.h"
AcReal
calc_timestep(const AcMeshInfo info)
{
    AcReal uumax     = 0.0;
    AcReal vAmax     = 0.0;
    AcReal shock_max = 0.0;
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &uumax);
    // TODO ERROR: uumax is currently only seen by the rank 0 process which will
    // lead to asyncronizity in the timestep leading to deadlocks! The resulting
    // dt or uumax and vAmax in rank 0 should be broadcasted to others.
    // SOLUTION: broadcast uumax and vAmax to all ranks
    // NOTE: It would be also possible to edit contents of
    // acGridReduceVecScal(), but for the sake of coherence, with less risk of
    // interfering things elsewhere, I have deiced to try out this approach
    // first, as it is not too complicated anyway.
    //
    // %JP: uumax, vAmax, seems to be OK now with the following bcasts
    //
    // OL: it would not be too difficult to make acGridReduceVecScal broadcast the value
    // Two changes in distributedScalarReduction are needed:
    //  1. change MPI_Reduce -> MPI_Allreduce
    //  2. remove the if (rank == 0) checks for the final RMS reduction
    //
    // This should also perform better because we will save the Bcast here
    // Right now we're doing two collective operations where one would suffice

    // MPI_Bcast to share uumax with all ranks
    MPI_Bcast(&uumax, 1, AC_REAL_MPI_TYPE, 0,
              acGridMPIComm()); // JP note: should no longer be needed, distributedScalarReduction
                                // now does MPI_Allreduce which includes the broadcast

#if LBFIELD
    // NOTE: bfield is 0 during the first step
    acGridReduceVecScal(STREAM_DEFAULT, RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO,
                        &vAmax);

    // MPI_Bcast to share vAmax with all ranks
    MPI_Bcast(&vAmax, 1, AC_REAL_MPI_TYPE, 0,
              acGridMPIComm()); // JP note: should no longer be needed, distributedScalarReduction
                                // now does MPI_Allreduce which includes the broadcast
#endif

#if LSHOCK
    acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_SHOCK, &shock_max);

    // MPI_Bcast to share vAmax with all ranks
    MPI_Bcast(&shock_max, 1, AC_REAL_MPI_TYPE, 0,
              acGridMPIComm()); // JP note: should no longer be needed, distributedScalarReduction
                                // now does MPI_Allreduce which includes the broadcast
#endif

    const long double cdt  = (long double)info.real_params[AC_cdt];
    const long double cdtv = (long double)info.real_params[AC_cdtv];
    // const long double cdts     = (long double)info.real_params[AC_cdts];
    const long double cs2_sound = (long double)info.real_params[AC_cs2_sound];
    const long double nu_visc   = (long double)info.real_params[AC_nu_visc];
    const long double eta       = (long double)info.real_params[AC_eta];
    const long double chi      = 0; // (long double)info.real_params[AC_chi]; // TODO not calculated
    const long double gamma    = (long double)info.real_params[AC_gamma];
    const long double dsmin    = (long double)info.real_params[AC_dsmin];
    const long double nu_shock = (long double)info.real_params[AC_nu_shock];

    // Old ones from legacy Astaroth
    // const long double uu_dt   = cdt * (dsmin / (uumax + cs_sound));
    // const long double visc_dt = cdtv * dsmin * dsmin / nu_visc;

    // New, closer to the actual Courant timestep
    // See Pencil Code user manual p. 38 (timestep section)
    const long double uu_dt = cdt * dsmin /
                              (fabsl((long double)uumax) +
                               sqrtl(cs2_sound + (long double)vAmax * (long double)vAmax));
    const long double visc_dt = cdtv * dsmin * dsmin /
                                (max(max(nu_visc, eta), gamma * chi) +
                                 nu_shock * (long double)shock_max);

    const long double dt = min(uu_dt, visc_dt);
    ERRCHK_ALWAYS(is_valid((AcReal)dt));
    return AcReal(dt);
}

void
dryrun(void)
{
    int pid;
    MPI_Comm_rank(acGridMPIComm(), &pid);

    // Scale the fields
    AcReal max, min, sum;
    acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)0, &max);
    acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)0, &min);
    acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)0, &sum);

    MPI_Barrier(acGridMPIComm());

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_scaling_factor, (AcReal)2.0);
    AcMeshDims dims = acGetMeshDims(acGridGetLocalMeshInfo());
    acGridLaunchKernel(STREAM_DEFAULT, scale, dims.n0, dims.n1);
    acGridSwapBuffers();
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    MPI_Barrier(acGridMPIComm());

    acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)0, &max);
    acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)0, &min);
    acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)0, &sum);
    const AcReal dt = 0.0;
    acGridIntegrate(STREAM_DEFAULT, dt);
    acGridLaunchKernel(STREAM_DEFAULT, reset, dims.n0, dims.n1);
    acGridLaunchKernel(STREAM_DEFAULT, randomize, dims.n0, dims.n1);

    MPI_Barrier(acGridMPIComm());

    // Reset the fields
    acGridLaunchKernel(STREAM_DEFAULT, reset, dims.n0, dims.n1);
    acGridSwapBuffers();
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    MPI_Barrier(acGridMPIComm());

    acGridLaunchKernel(STREAM_DEFAULT, reset, dims.n0, dims.n1);
    acGridSwapBuffers();
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    MPI_Barrier(acGridMPIComm());
}

static void
read_varfile_to_mesh_and_setup(const AcMeshInfo info, const char* file_path)
{
    // Read PC varfile to Astaroth

    const int3 nn = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 rr = (int3){3, 3, 3};

    int pid;
    MPI_Comm_rank(acGridMPIComm(), &pid);
    acLogFromRootProc(pid, "Reading varfile nn = (%d, %d, %d)\n", nn.x, nn.y, nn.z);

    // IO configuration
    const Field io_fields[] =
    { VTXBUF_UUX,
      VTXBUF_UUY,
      VTXBUF_UUZ,
      VTXBUF_LNRHO,
#if LMAGNETIC
      VTXBUF_AX,
      VTXBUF_AY,
      VTXBUF_AZ,
#endif
    };
    const size_t num_io_fields = ARRAY_SIZE(io_fields);
#if !LMAGNETIC
    WARNING("LMAGNETIC was not set, magnetic field is not read read_varfile_to_mesh_and_setup");
#endif

    acGridReadVarfileToMesh(file_path, io_fields, num_io_fields, nn, rr);

    // Scale the magnetic field
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_scaling_factor, info.real_params[AC_scaling_factor]);
    AcMeshDims dims = acGetMeshDims(acGridGetLocalMeshInfo());
    acGridLaunchKernel(STREAM_DEFAULT, scale, dims.n0, dims.n1);
    acGridSwapBuffers();

    acGridSynchronizeStream(STREAM_ALL);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);
}

/* Set step = -1 to load from the latest snapshot. step = 0 to start a new run. */
static void
read_file_to_mesh_and_setup(int* step, AcReal* simulation_time, const AcMeshInfo info)
{
    if (*step > 0) {
        ERROR("step in read_file_to_mesh (config start_step) was > 0, do not know what to do with "
              "it. Cannot restart from an arbitrary snapshot. Restart from the latest valid "
              "snapshot by setting `start_step = -1`");
    }

    // Quick hack, TODO better
    system("tail -n2 snapshots_info.csv | head -n1 > latest_snapshot.info");

    // Read the previous valid step from snapshots_info.csv
    int modstep = 0;

    if (*step < 0) {
        FILE* fp = fopen("latest_snapshot.info", "r");
        if (fp) {
            fseek(fp, 0L, SEEK_END);
            const size_t bytes = ftell(fp);
            if (bytes == 0) {
                ERROR("latest_snapshot.info was empty or invalid. Must have at least one valid "
                      "snapshot available, start from step 0 to generate");
            }
            rewind(fp);

            // Note: quick hack, hardcoded + bad practice
            int use_double, mx, my, mz;
            float approx_time;
            fscanf(fp, "%d, %d, %d, %d, %d, %d, %g, %la", &use_double, &mx, &my, &mz, step,
                   &modstep, &approx_time, simulation_time);
            fclose(fp);
        }
        else {
            ERROR("Tried to load from the latest snapshot but snapshots_info.csv is malformatted "
                  "or non-existing");
        }
    }
    ERRCHK_ALWAYS(modstep >= 0);
    ERRCHK_ALWAYS(*step >= 0);
    ERRCHK_ALWAYS(is_valid(*simulation_time));

    const size_t buflen = 128;
    char modstep_str[buflen];
    sprintf(modstep_str, "%d", modstep);

    int pid;
    MPI_Comm_rank(acGridMPIComm(), &pid);
    acLogFromRootProc(pid, "Restarting from snapshot %d (step %d, tstep %g)\n", modstep, *step,
                      (double)(*simulation_time));

    const Field io_fields[] =
    { VTXBUF_UUX,
      VTXBUF_UUY,
      VTXBUF_UUZ,
      VTXBUF_LNRHO,
#if LMAGNETIC
      VTXBUF_AX,
      VTXBUF_AY,
      VTXBUF_AZ,
#endif
    };
    const size_t num_io_fields = ARRAY_SIZE(io_fields);
#if !LMAGNETIC
    WARNING("NOTE: LMAGNETIC was not set, magnetic field is not read in "
            "read_file_to_mesh_and_setup. TODO improve: read the fields stored in the snapshot "
            "from a file instead of hardcoding it like this.");
#endif

    for (size_t i = 0; i < num_io_fields; ++i)
        acGridAccessMeshOnDiskSynchronous(io_fields[i], snapshot_dir, modstep_str, ACCESS_READ);

    // for (size_t i = 0; i < NUM_FIELDS; ++i)
    //     acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, snapshot_dir, modstep_str,
    //     ACCESS_READ);

    // Not needed for synchronous reading
    // acGridDiskAccessSync();
    // acGridPeriodicBoundconds(STREAM_DEFAULT);
    // acGridSynchronizeStream(STREAM_DEFAULT);

#if LMAGNETIC
    // Scale the magnetic field
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_scaling_factor, info.real_params[AC_scaling_factor]);
    AcMeshDims dims = acGetMeshDims(acGridGetLocalMeshInfo());
    acGridLaunchKernel(STREAM_DEFAULT, scale, dims.n0, dims.n1);
    acGridSwapBuffers();

    acGridSynchronizeStream(STREAM_ALL);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);
#endif
}

/*
static void
read_distributed_to_mesh_and_setup(void)
{
    for (size_t i = 0; i < num_io_fields; ++i) {
        const Field field = io_fields[i];
        acGridAccessMeshOnDiskSynchronousDistributed(field, snapshot_dir, vtxbuf_names[field],
                                                     ACCESS_READ);
    }
    acGridPeriodicBoundconds(STREAM_DEFAULT);
}

static void
read_collective_to_mesh_and_setup(void)
{
    for (size_t i = 0; i < num_io_fields; ++i) {
        const Field field = io_fields[i];
        acGridAccessMeshOnDiskSynchronousCollective(field, snapshot_dir, vtxbuf_names[field],
                                                    ACCESS_READ);
    }
    acGridPeriodicBoundconds(STREAM_DEFAULT);
}*/

static void
create_directory(const char* dirname)
{
    constexpr size_t cmdlen = 4096;
    static char cmd[cmdlen];

    // Improvement suggestion: use create_directory() from <filesystem> (C++17) or mkdir() from
    // <sys/stat.h> (POSIX)
    snprintf(cmd, cmdlen, "mkdir -p %s", dirname);
    system(cmd);
}

static void
create_output_directories(void)
{
    create_directory(snapshot_dir);
    create_directory(slice_dir);

    // JP: Note: striping here potentially bad practice (uncomment to enable)
    // OL: Agree, there is no guarantee that the environment uses a lustre filesystem (perhaps as an
    // option though?)
    // const int stripe_count = 48;
    // Note: striping here potentially bad practice (uncomment to enable)
    // snprintf(cmd, cmdlen, "lfs setstripe -c %d %s", stripe_count, snapshot_dir);
    // system(cmd);
}

static void
write_slices(int pid, int i)
{
    debug_log_from_root_proc_with_sim_progress(pid, "write_slices: Syncing slice disk access\n");
    acGridDiskAccessSync();
    debug_log_from_root_proc_with_sim_progress(pid, "write_slices: Slice disk access synced\n");

    char slice_frame_dir[2048];
    sprintf(slice_frame_dir, "%s/step_%012d", slice_dir, i);

    log_from_root_proc_with_sim_progress(pid, "write_slices: Creating directory %s\n",
                                         slice_frame_dir);
    // The root proc creates the frame dir and then we sync
    if (pid == 0) {
        create_directory(slice_frame_dir);
    }
    MPI_Barrier(acGridMPIComm()); // Ensure directory is created for all procs

    log_from_root_proc_with_sim_progress(pid, "write_slices: Writing slices to %s, timestep = %d\n",
                                         slice_dir, i);
    /*
    Timer t;
    timer_reset(&t);
    acGridWriteSlicesToDiskCollectiveSynchronous(slice_dir, label);
    acLogFromRootProc(pid, "Collective sync slices elapsed %g ms\n",
    timer_diff_nsec(t)/1e6);
    */

    // This label is redundant now that the step number is in the dirname
    //  JP: still useful for debugging and analysis if working in a flattened dir structure
    char label[80];
    sprintf(label, "step_%012d", i);

    acGridWriteSlicesToDiskLaunch(slice_frame_dir, label);
    log_from_root_proc_with_sim_progress(pid, "write_slices: Non-blocking slice write operation "
                                              "started, returning\n");
}

void
print_usage(const char* name)
{
    printf("Usage: ./%s "
           "[--config <config_path>] "
           "[ --run-init-kernel | --from-pc-varfile | --from-distributed-snapshot | "
           "--from-monolithic-snapshot ]"
           "\n"
           "\n"
           " -c <config_path>\n"
           " --config <config_path>\n"
           "\tread config file from directory at config_path, the default config path is: %s\n"
           "\n"
           "Mutually exclusive initial mesh load procedures\n"
           "------------------------------------------------"
           "\n"
           "  the default is --run-init-kernel\n"
           "\n"
           " -k\n"
           " --run-init-kernel\n"
           "\tRun a kernel to initialize the mesh\n"
           "\tThe kernel is currently hardcoded\n"
           "\n"
           " -p\n"
           " --from-pc-varfile\n"
           "\tLoad the mesh from a pc varfile\n"
           "\tThe path to this file is currently hardcoded\n"
           "\n"
           " -d\n --from-distributed-snapshot\n"
           "\tLoad the mesh from a distributed snapshot (one file per process)\n"
           "\tThe path to the snapshot is currently hardcoded\n"
           "\n"
           " -m\n"
           " --from-monolithic-snapshot\n"
           "\tLoad the mesh from a monolithic snapshot (one single file)\n"
           "\tThe path to the snapshot is currently hardcoded\n",
           name, AC_DEFAULT_CONFIG);
}

// Enums for mesh initialization
enum class InitialMeshProcedure {
    InitKernel,
    LoadPC_Varfile,
    LoadDistributedSnapshot,
    LoadMonolithicSnapshot,
    LoadSnapshot,
};

// Enums for actions taken in the simulation loop
enum class PeriodicAction {
    PrintDiagnostics,
    WriteSnapshot,
    WriteSlices,
    EndSimulation,
    GenerateForcing,
};

// Enums for events
enum class SimulationEvent {
    NanDetected        = 0x001,
    StopSignal         = 0x002,
    TimeLimitReached   = 0x004,
    ConfigReloadSignal = 0x008,

    EndCondition = NanDetected | StopSignal | TimeLimitReached,
    ErrorState   = NanDetected
};

void
set_event(uint16_t* events, SimulationEvent mask)
{
    *events |= (uint16_t)mask;
}

bool
check_event(uint16_t events, SimulationEvent mask)
{
    return (events & (uint16_t)mask) ? true : false;
}

int
main(int argc, char** argv)
{
    // Use multi-threaded MPI
    if (ac_MPI_Init_thread(MPI_THREAD_MULTIPLE) != AC_SUCCESS) {
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    /////////////////////////////////
    // Read command line arguments //
    /////////////////////////////////

    // The input mesh files are hardcoded at the moment, but they could easily be passed by
    // parameter Just change no_argument below to required_argument or optional_argument and copy
    // the value of optarg to a filename variable in the switch
    static struct option long_options[] = {{"config", required_argument, 0, 'c'},
                                           {"run-init-kernel", no_argument, 0, 'k'},
                                           {"from-pc-varfile", required_argument, 0, 'p'},
                                           {"from-distributed-snapshot", no_argument, 0, 'd'},
                                           {"from-monolithic-snapshot", no_argument, 0, 'm'},
                                           {"from-snapshot", no_argument, 0, 's'},
                                           {"help", no_argument, 0, 'h'}};

    const char* config_path = AC_DEFAULT_CONFIG;
    // Default mesh procedure is kernel randomize
    InitialMeshProcedure initial_mesh_procedure = InitialMeshProcedure::InitKernel;
    const char* initial_mesh_procedure_param    = nullptr;

    int opt{};
    while ((opt = getopt_long(argc, argv, "c:kpdmh", long_options, nullptr)) != -1) {
        switch (opt) {
        case 'h':
            if (pid == 0) {
                print_usage("ac_run_mpi");
            }
            return EXIT_SUCCESS;
        case 'c':
            config_path = optarg;
            break;
        case 'k':
            initial_mesh_procedure = InitialMeshProcedure::InitKernel;
            break;
        case 'p':
            initial_mesh_procedure       = InitialMeshProcedure::LoadPC_Varfile;
            initial_mesh_procedure_param = optarg;
            break;
        case 'd':
            initial_mesh_procedure = InitialMeshProcedure::LoadDistributedSnapshot;
            break;
        case 'm':
            initial_mesh_procedure = InitialMeshProcedure::LoadMonolithicSnapshot;
            break;
        case 's':
            initial_mesh_procedure = InitialMeshProcedure::LoadSnapshot;
            break;
        default:
            print_usage("ac_run_mpi");
            return EXIT_FAILURE;
        }
    }

    //////////////////////
    // Load config file //
    //////////////////////

    AcMeshInfo info;
    acLogFromRootProc(pid, "Loading config file %s\n", config_path);
    acLoadConfig(config_path, &info);

    // OL: We are calling both acLoadConfig AND set_extra_config_params (defined in config_loader.c)
    // even though acLoadConfig calls acHostUpdateBuiltinParams
    // set_extra_config_params will set some extra config parameters, namely:
    //  - AC_xlen, AC_ylen, AC_zlen
    //  - AC_xorig, AC_yorig, AC_zorig
    //  ^ these could be set in acHostUpdateBuiltinParams
    //  - AC_cs2_sound
    //  - AC_cv_sound
    //  - AC_unit_mass
    //  - AC_M_sink
    //  - AC_M_sink_init
    //  - AC_G_const
    //  - AC_sq2GM_star
    //  ^ these depend on config vals that may not be present
    //  but we could check if they are defined before attempting to set the extra params
    //  perhaps set_extra_config_params could become
    //   -> acHostUpdateAstrophysicsBuiltinParams
    set_extra_config_params(&info);
    acLogFromRootProc(pid, "Done loading config file\n");
    // TODO: to reduce verbosity, only print uninitialized value warnings for rank == 0
    // we could e.g. define a function acCheckConfig and call it:
    // if (pid == 0){
    //     acCheckConfig(&info);
    // }

    //////////////////////////////
    // Output run configuration //
    //////////////////////////////

    if (pid == 0) {
        // Write purge.sh and meshinfo.list
        acLogFromRootProc(pid, "Creating purge.sh and meshinfo.list\n");
        write_info(&info);

        // Ensure output-slices and output-snapshots exist
        acLogFromRootProc(pid, "Creating directories output-slices and output-snapshots\n");
        create_output_directories();

        // Print config to stdout
        acLogFromRootProc(pid, "Printing config to stdout\n");
        acPrintMeshInfo(info);

        acLogFromRootProc(pid, "Logging build configuration\n");
        const char* is_on  = "ON";
        const char* is_off = "OFF";

        const char* forcing_flag =
#if LFORCING
            is_on;
#else
            is_off;
#endif
        const char* sink_flag =
#if LSINK
            is_on;
#else
            is_off;
#endif
        const char* shock_flag =
#if LSHOCK
            is_on;
#else
            is_off;
#endif

        acLogFromRootProc(pid, "Forcing is: %s\n", forcing_flag);
        acLogFromRootProc(pid, "Sink is: %s\n", sink_flag);
        acLogFromRootProc(pid, "Shock is: %s\n", shock_flag);
    }
    // MPI_Barrier(acGridMPIComm()); // Ensure output directories are created before continuing

    ///////////////////////////////////////////////
    // Define variables for main simulation loop //
    ///////////////////////////////////////////////

    // Run-control variables
    // --------------------
    bool log_progress = 1;

    AcReal simulation_time = 0.0;
    int start_step         = info.int_params[AC_start_step];

    // Additional physics variables
    // ----------------------------
    AcReal sink_mass     = -1.0;
    AcReal accreted_mass = -1.0;
    // TODO: hide these in some structure

#if LSINK
    sink_mass     = info.real_params[AC_M_sink_init];
    accreted_mass = 0.0;
#endif

    // Set random seed for reproducibility
    seed_rng(312256655);

    ////////////////////////////////////////
    // Initialize internal Astaroth state //
    //  allocate mesh                     //
    //  load config to GPU                //
    ////////////////////////////////////////

    acLogFromRootProc(pid, "Initializing Astaroth (acGridInit)\n");
    acGridInit(info);

    // Compute stencil coefficients from `info` and load to device
    acLogFromRootProc(pid, "Loading stencils (load_stencil_from_config)\n");
    load_stencil_from_config(info);
    acLogFromRootProc(pid, "Stencils loaded (load_stencil_from_config)\n");

    ///////////////////////////////////////////////////
    // Test kernels: scale, solve, reset, randomize. //
    // then reset                                    //
    ///////////////////////////////////////////////////

    acLogFromRootProc(pid, "Calling dryrun to test kernels on non-initialized mesh\n");
    dryrun();

    // Load input data

    /////////////////////////////////////////////
    // Mesh initialization from file or kernel //
    /////////////////////////////////////////////

    acLogFromRootProc(pid, "Initializing mesh\n");
    switch (initial_mesh_procedure) {
    case InitialMeshProcedure::InitKernel: {
        // Randomize
        acLogFromRootProc(pid, "Scrambling mesh with some (low-quality) pseudo-random data\n");
        AcMeshDims dims = acGetMeshDims(acGridGetLocalMeshInfo());
        acGridLaunchKernel(STREAM_DEFAULT, randomize, dims.n0, dims.n1);
        acGridSwapBuffers();
        acLogFromRootProc(pid, "Communicating halos\n");
        acGridPeriodicBoundconds(STREAM_DEFAULT);

        {
            // Should some labels be printed here?
            AcReal max, min, sum;
            for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
                acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)i, &max);
                acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)i, &min);
                acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)i, &sum);
                acLogFromRootProc(pid, "max %g, min %g, sum %g\n", (double)max, (double)min,
                                  (double)sum);
            }
        }
        break;
    }
    //MV TODO: Add other initialization configurations! These are suitable only
    //MV TODO: for you who work with the forcing runs. 
    case InitialMeshProcedure::LoadPC_Varfile: {
        acLogFromRootProc(pid, "Reading mesh state from Pencil Code var file %s\n",
                          initial_mesh_procedure_param);
        if (initial_mesh_procedure_param == nullptr) {
            acLogFromRootProc(pid, "Error: no file path given");
            return EXIT_FAILURE;
        }
        read_varfile_to_mesh_and_setup(info, initial_mesh_procedure_param);
        acLogFromRootProc(pid, "Done reading Pencil Code var file\n");
        break;
    }
    /*
    case InitialMeshProcedure::LoadDistributedSnapshot: {
        acLogFromRootProc(pid, "Reading mesh state from distributed snapshot\n");
        read_distributed_to_mesh_and_setup();
        acLogFromRootProc(pid, "Done reading distributed snapshot\n");
        break;
    }
    case InitialMeshProcedure::LoadMonolithicSnapshot: {
        acLogFromRootProc(pid, "Reading mesh state monolithic snapshot\n");
        read_collective_to_mesh_and_setup();
        acLogFromRootProc(pid, "Done reading monolithic snapshot\n");
        break;
    }
    */
    case InitialMeshProcedure::LoadSnapshot: {
        acLogFromRootProc(pid, "Reading mesh file\n");
        read_file_to_mesh_and_setup(&start_step, &simulation_time, info);
        acLogFromRootProc(pid, "Done reading mesh file\n");
        break;
    }
    default:
        fprintf(stderr, "Invalid initial_mesh_procedure %d passed to ac_run_mpi\n",
                (int)initial_mesh_procedure);
        ERROR("Invalid initial_mesh_procedure");
    }

    acLogFromRootProc(pid, "Mesh initialization done\n");

    ////////////////////////////////////////////////////
    // Building the task graph (or using the default) //
    ////////////////////////////////////////////////////

    acLogFromRootProc(pid, "Setting simulation program\n");
    Simulation sim = Simulation::Default;
#if LSHOCK
    sim = Simulation::Shock_Singlepass_Solve;
#endif 
    log_simulation_choice(pid, sim);
    AcTaskGraph* simulation_graph = get_simulation_graph(pid, sim);

    ////////////////////////////////////////////////////////
    // Simulation loop setup: defining events and actions //
    ////////////////////////////////////////////////////////

    // Simulation events will be collected in the bits of this variable
    // 16 bits, so there's room for sixteen different events.
    // Upgrade to a larger int if you need more events.
    uint16_t events = 0;

    //////////////////////////////
    // Define user signal files //
    //////////////////////////////

    // Astaroth will trigger an event when a file is touched (when the modification time is updated)
    // Map filenames to events here

    std::map<SimulationEvent, UserSignalFile> signal_files;

    // STOP
    signal_files[SimulationEvent::StopSignal] = UserSignalFile("STOP");

    // RELOAD config
    signal_files[SimulationEvent::ConfigReloadSignal] = UserSignalFile("RELOAD");

    /////////////////////////////
    // Define periodic actions //
    /////////////////////////////

    // Values set here define when certain timed simulation actions should happen
    // Either periodic ones (write snapshot) or single actions (end simulation)

    // Both a period in discrete time steps and a period in simulation time can be used
    // It is enough to satisfy one of these conditions.
    // A value of zero means that the trigger is inactive.

    // These run before the simulation step
    std::map<PeriodicAction, SimulationPeriod> pre_step_actions;
    // Generate forcing
#if LFORCING
    pre_step_actions[PeriodicAction::GenerateForcing] = SimulationPeriod(info,
                                                                         AC_forcing_period_steps,
                                                                         AC_forcing_period_t);
#endif

    // These run after the simulation step
    std::map<PeriodicAction, SimulationPeriod> post_step_actions;

    // Print diagnostics
    post_step_actions
        [PeriodicAction::PrintDiagnostics] = SimulationPeriod(info, AC_save_steps,
                                                              SimulationPeriod::NoTimeParam);

    // Write snapshots
    AcReal snapshot_time_offset                      = simulation_time;
    post_step_actions[PeriodicAction::WriteSnapshot] = SimulationPeriod(info, AC_bin_steps,
                                                                        AC_bin_save_t,
                                                                        snapshot_time_offset);

    // Write slices
    post_step_actions
        [PeriodicAction::WriteSlices] = SimulationPeriod(info, AC_slice_steps,
                                                         SimulationPeriod::NoTimeParam);

    // Stop simulation after max time
    post_step_actions[PeriodicAction::EndSimulation] = SimulationPeriod(info, AC_max_steps,
                                                                        AC_max_time);

    /////////////////////////////////////////////////////////////
    // Set up certain periodic actions and run them for i == 0 //
    /////////////////////////////////////////////////////////////

    FILE* diag_file = fopen("timeseries.ts", "a");
    ERRCHK_ALWAYS(diag_file);
    // TODO: should probably always check for NaN's, not just at start_step = 0
    if (start_step == 0) {
        // TODO: calculate time step before entering loop, recalculate at end
        acLogFromRootProc(pid, "Initial state: diagnostics\n");
        print_diagnostics_header_from_root_proc(pid, diag_file);
        int found_nan = 0;
        print_diagnostics(pid, start_step, 0, simulation_time, diag_file, sink_mass, accreted_mass,
                          &found_nan);

        acLogFromRootProc(pid, "Initial state: writing mesh slices\n");
        write_slices(pid, start_step);

        acLogFromRootProc(pid, "Initial state: writing full mesh snapshot\n");
        save_mesh_mpi_async(info, snapshot_dir, pid, 0, 0.0);

        if (found_nan != 0) {
            acLogFromRootProc(pid, "Found NaN in initial state -> exiting\n");
            set_event(&events, SimulationEvent::NanDetected);
        }

#if LFORCING

        log_from_root_proc_with_sim_progress(pid, "Periodic action: Generating new forcing "
                                                  "parameters\n");
        auto forcing_params = generateForcingParams(info);
        // printForcingParams(forcing_params);
        loadForcingParamsToGrid(forcing_params);
#endif
    }
    else if (pid == 0) {
        // add newline to old diag_file from previous run
        // TODO: figure out why we're doing this? do we want a clear indication in the file that a
        // new run was started?
        // MV: Yes this was a a non-intrusive way of making it possible to
        // MV: locate where new run stars. If I remember right. 
        fprintf(diag_file, "\n");
    }

    ///////////////////////////////////////////////////////////////
    //                     Main simulation loop                  //
    ///////////////////////////////////////////////////////////////

    acLogFromRootProc(pid, "Starting simulation\n");
    set_simulation_timestamp(start_step, simulation_time);

    for (int i = start_step + 1;; ++i) {

        /////////////////////////////////////////////////////////////////////
        //                                                                 //
        // 1. Update simulation parameters on host                         //
        //                                                                 //
        /////////////////////////////////////////////////////////////////////

        // Generic parameters
        debug_log_from_root_proc_with_sim_progress(pid, "Calculating time delta\n");
        const AcReal dt = calc_timestep(info);
        debug_log_from_root_proc_with_sim_progress(pid, "Done calculating time delta, dt = %e\n",
                                                   dt);

        // Case-specific parameters
#if LSINK
        AcReal sum_mass;
        acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, VTXBUF_ACCRETION, &sum_mass);
        accreted_mass = accreted_mass + sum_mass;
        sink_mass     = info.real_params[AC_M_sink_init] + accreted_mass;

        int switch_accretion = (i < 1) ? 0 : 1;
#endif

        for (auto& [action, period] : pre_step_actions) {
            if (i - 1 != 0 && period.check(i - 1, simulation_time)) {
                switch (action) {
#if LFORCING
                case PeriodicAction::GenerateForcing: {
                    log_from_root_proc_with_sim_progress(pid, "Periodic action: Generating new "
                                                              "forcing parameters\n");
                    auto forcing_params = generateForcingParams(info);
                    // printForcingParams(forcing_params);
                    loadForcingParamsToGrid(forcing_params);
                    break;
                }
#endif
                default:
                    log_from_root_proc_with_sim_progress(pid,
                                                         "Unsupported periodic action pre sim "
                                                         "step: %lu\n",
                                                         (size_t)action);
                    break;
                }
            }
        }

        /////////////////////////////////////////////////////////////////////
        //                                                                 //
        // 2. Distribute updated state to all processes and load it to GPU //
        //                                                                 //
        /////////////////////////////////////////////////////////////////////

        // TODO, PERFORMANCE: lots of collective calls here.
        // I'm sure a lot of these could be calculated locally in each proc.
        // And for the values that do need to be distributed, they could be distributed in fewer
        // calls

        // Generic parameters
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_current_time, simulation_time);
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, dt);

        // Case-specific parameters
#if LSINK
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_M_sink, sink_mass);
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_switch_accretion, switch_accretion);
#endif
        acGridSynchronizeStream(STREAM_DEFAULT);

        /////////////////////////////////////////////////////////////////////
        //                                                                 //
        // 3. Run simulation step on GPU                                   //
        //                                                                 //
        /////////////////////////////////////////////////////////////////////

        // Execute the active task graph for 3 iterations
        // in the case that simulation_graph = acGridGetDefaultTaskGraph(), then this is equivalent
        // to acGridIntegrate(STREAM_DEFAULT, dt)
        auto start = std::chrono::high_resolution_clock::now();
        acGridExecuteTaskGraph(simulation_graph, 1);
        simulation_time += dt;
        set_simulation_timestamp(i, simulation_time);

        if (log_progress) {
            log_from_root_proc_with_sim_progress(pid, "Simulation step complete\n");
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
 
        std::cout << "Time taken in acGridExecuteTaskGraph: "
            << duration.count() << " microseconds" << std::endl;
        /////////////////////////////////////////////////////////////////////
        //                                                                 //
        // 4. Run periodic actions (can generate events)                   //
        //                                                                 //
        /////////////////////////////////////////////////////////////////////

        for (auto& [action, period] : post_step_actions) {
            if (period.check(i, simulation_time)) {
#if !(AC_VERBOSE)
                // End progress logging (which step you are at) after first period.
                if (log_progress) {
                    log_progress = false;
                    log_from_root_proc_with_sim_progress(pid,
                                                         "VERBOSE is off, not logging simulation "
                                                         "step completion for "
                                                         "step > %d\n",
                                                         i);
                }
#endif

                switch (action) {
                case PeriodicAction::PrintDiagnostics: {
                    // Print diagnostics and search for nans
                    log_from_root_proc_with_sim_progress(pid, "Periodic action: diagnostics\n");
                    int found_nan = 0;
                    print_diagnostics(pid, i, dt, simulation_time, diag_file, sink_mass,
                                      accreted_mass, &found_nan);
                    if (found_nan) {
                        set_event(&events, SimulationEvent::NanDetected);
                    }
                    /*
                    MV: We would also might want an XY-average calculating funtion,
                        which can be very useful when observing behaviour of turbulent
                        simulations. (TODO)
                    */
                    break;
                }
                case PeriodicAction::WriteSnapshot: {
                    log_from_root_proc_with_sim_progress(pid, "Periodic action: writing full mesh "
                                                              "snapshot\n");
                    save_mesh_mpi_async(info, snapshot_dir, pid, i, simulation_time);
                    break;
                }
                case PeriodicAction::WriteSlices: {
                    log_from_root_proc_with_sim_progress(pid,
                                                         "Periodic action: writing mesh slices\n");
                    acGridPeriodicBoundconds(STREAM_DEFAULT);
                    write_slices(pid, i);
                    break;
                }
                case PeriodicAction::EndSimulation: {
                    set_event(&events, SimulationEvent::TimeLimitReached);
                    break;
                }
                default:
                    log_from_root_proc_with_sim_progress(pid,
                                                         "Unsupported periodic action post sim "
                                                         "step: %lu\n",
                                                         (size_t)action);
                    break;
                }
            }
        }

        /////////////////////////////////////////////////////////////////////
        //                                                                 //
        // 5. Check input events from user                                 //
        //                                                                 //
        /////////////////////////////////////////////////////////////////////

        // Check signal files
        for (auto& [signal, file] : signal_files) {
            if (pid == 0 && file.check()) {
                log_from_root_proc_with_sim_progress(pid, "Detected file modified: %s\n",
                                                     file.file_path.c_str());
                set_event(&events, signal);
            }
        }

        /////////////////////////////////////////////////////////////////////
        //                                                                 //
        // 6. Agree on events across processes                             //
        //                                                                 //
        /////////////////////////////////////////////////////////////////////

        MPI_Allreduce(MPI_IN_PLACE, &events, sizeof(uint16_t), MPI_BYTE, MPI_BOR, acGridMPIComm());

        /////////////////////////////////////////////////////////////////////
        //                                                                 //
        // 7. Run event-triggered actions                                  //
        //                                                                 //
        /////////////////////////////////////////////////////////////////////

        for (uint16_t e = 1; e != 0; e <<= 1) {
            if (check_event(events, static_cast<SimulationEvent>(e))) {
                switch (static_cast<SimulationEvent>(e)) {
                case SimulationEvent::NanDetected: {
                    log_from_root_proc_with_sim_progress(pid, "FOUND NAN -> exiting\n");
                    break;
                }
                case SimulationEvent::StopSignal: {
                    log_from_root_proc_with_sim_progress(pid, "Got STOP signal -> exiting\n");
                    break;
                }
                case SimulationEvent::TimeLimitReached: {
                    int max_step = post_step_actions[PeriodicAction::EndSimulation].step_period;
                    if (i == max_step) {
                        log_from_root_proc_with_sim_progress(pid,
                                                             "Max time steps reached (%d == %d) -> "
                                                             "exiting\n",
                                                             i, max_step);
                    }
                    AcReal max_time = post_step_actions[PeriodicAction::EndSimulation].time_period;
                    if (max_time > 0 && simulation_time >= max_time) {
                        log_from_root_proc_with_sim_progress(pid,
                                                             "Time limit reached (%e >= %e ) -> "
                                                             "exiting\n",
                                                             simulation_time, max_time);
                    }
                    break;
                }
                case SimulationEvent::ConfigReloadSignal: {
                    log_from_root_proc_with_sim_progress(pid, "Got RELOAD signal -> reloading "
                                                              "config file\n");

                    AcMeshInfo new_info;
                    log_from_root_proc_with_sim_progress(pid, "Synchronizing procs\n");
                    MPI_Barrier(acGridMPIComm());
                    log_from_root_proc_with_sim_progress(pid, "Reloading config file\n");
                    acLoadConfig(config_path, &new_info);
                    set_extra_config_params(&new_info);

                    // TODO: refactor this big mess of runtime checks into a function that returns a
                    // bool Check differences to current config
                    std::vector<const char*> nan_params;
                    bool configs_differ = false;

                    // TODO: define run constants, these cannot be changed in a reload
                    bool changed_run_constants = false;

                    std::set<size_t> int_run_constants;
                    int_run_constants.insert(AC_nx);
                    int_run_constants.insert(AC_ny);
                    int_run_constants.insert(AC_nz);

                    int_run_constants.insert(AC_dsx);
                    int_run_constants.insert(AC_dsy);
                    int_run_constants.insert(AC_dsz);

                    int_run_constants.insert(AC_start_step);
                    int_run_constants.insert(AC_init_type);

                    for (size_t int_param = 0; int_param < NUM_INT_PARAMS; int_param++) {
                        int old_value = info.int_params[int_param];
                        int new_value = new_info.int_params[int_param];
                        if (old_value != new_value) {
                            configs_differ         = true;
                            const char* param_name = intparam_names[int_param];
                            acLogFromRootProc(pid, "about to update %s from %d to %d\n", param_name,
                                              old_value, new_value);
                            if (int_run_constants.count(int_param) != 0) {
                                acLogFromRootProc(pid, "ERROR: Can't update %s during a run\n",
                                                  param_name);
                                changed_run_constants = true;
                            }
                        }
                    }
                    for (size_t int3_param = 0; int3_param < NUM_INT3_PARAMS; int3_param++) {
                        int3 old_value = info.int3_params[int3_param];
                        int3 new_value = new_info.int3_params[int3_param];
                        if (old_value != new_value) {
                            configs_differ         = true;
                            const char* param_name = int3param_names[int3_param];
                            acLogFromRootProc(pid,
                                              "about to update %s: from (%d, %d, %d) to (%d, %d, "
                                              "%d)\n",
                                              param_name, old_value.x, old_value.y, old_value.z,
                                              new_value.x, new_value.y, new_value.z);
                        }
                    }
                    for (size_t real_param = 0; real_param < NUM_REAL_PARAMS; real_param++) {
                        AcReal old_value = info.real_params[real_param];
                        AcReal new_value = new_info.real_params[real_param];
                        if (old_value != new_value && !(isnan(old_value) && isnan(new_value))) {
                            configs_differ         = true;
                            const char* param_name = realparam_names[real_param];
                            acLogFromRootProc(pid, "about to update %s from %e to %e\n", param_name,
                                              old_value, new_value);
                            if (isnan(new_value)) {
                                acLogFromRootProc(pid,
                                                  "WARNING: updated value of %s parsed as NAN. "
                                                  "Check the config file!\n",
                                                  param_name);
                                nan_params.push_back(param_name);
                            }
                        }
                    }
                    for (size_t real3_param = 0; real3_param < NUM_REAL3_PARAMS; real3_param++) {
                        AcReal3 old_value = info.real3_params[real3_param];
                        AcReal3 new_value = new_info.real3_params[real3_param];
                        if ((old_value.x != new_value.x &&
                             !(isnan(old_value.x) && isnan(new_value.x))) ||
                            (old_value.y != new_value.y &&
                             !(isnan(old_value.y) && isnan(new_value.y))) ||
                            (old_value.z != new_value.z &&
                             !(isnan(old_value.z) && isnan(new_value.z)))) {
                            configs_differ         = true;
                            const char* param_name = real3param_names[real3_param];
                            acLogFromRootProc(pid,
                                              "about to update %s: from (%d, %d, %d) to (%d, %d, "
                                              "%d)\n",
                                              param_name, old_value.x, old_value.y, old_value.z,
                                              new_value.x, new_value.y, new_value.z);
                            if (isnan(new_value.x) || isnan(new_value.y) || isnan(new_value.z)) {
                                acLogFromRootProc(pid,
                                                  "WARNING: updated value of %s parsed as NAN. "
                                                  "Check the config file!\n",
                                                  param_name);
                                nan_params.push_back(param_name);
                            }
                        }
                    }

                    if (!nan_params.empty()) {
                        log_from_root_proc_with_sim_progress(pid, "ERROR reloading config file: "
                                                                  "NaN's in updated values -> not "
                                                                  "reloading config.\n");
                        log_from_root_proc_with_sim_progress(pid, "The following parameters were "
                                                                  "parsed as NaN:\n");
                        for (auto param_name : nan_params) {
                            acLogFromRootProc(pid, "  %s\n", param_name);
                        }
                        log_from_root_proc_with_sim_progress(pid,
                                                             "Please edit the config file and try "
                                                             "again. Aborting config reload.\n");
                        break;
                    }

                    if (changed_run_constants) {
                        log_from_root_proc_with_sim_progress(pid, "Tried to change run constant -> "
                                                                  "not reloading config.\n");
                        log_from_root_proc_with_sim_progress(pid,
                                                             "Please edit the config file and try "
                                                             "again. Aborting config reload.\n");
                        break;
                    }

                    if (!configs_differ) {
                        log_from_root_proc_with_sim_progress(pid, "No changes found in new config "
                                                                  "file -> doing nothing\n");
                        break;
                    }

                    // No NaN's, we did not change run constants, and we have some changes

                    log_from_root_proc_with_sim_progress(pid, "Reloading config\n");

                    // Update periodic actions with new params
                    for (auto& [_, action] : pre_step_actions) {
                        action.update(new_info);
                    }
                    for (auto& [_, action] : post_step_actions) {
                        action.update(new_info);
                    }

                    // Decompose the config
                    AcMeshInfo submesh_info = acGridDecomposeMeshInfo(new_info);
                    acDeviceLoadMeshInfo(acGridGetDevice(), submesh_info);
                    info = new_info;
                    MPI_Barrier(acGridMPIComm());
                    log_from_root_proc_with_sim_progress(pid, "Done reloading config\n");
                    break;
                }
                default: {
                    break;
                }
                }
            }
        }

        /////////////////////////////////////////////////////////////////////
        //                                                                 //
        // 8. End simulation if an end condition has been reached          //
        //                                                                 //
        /////////////////////////////////////////////////////////////////////

        if (check_event(events, SimulationEvent::EndCondition)) {
            /*
            // JP: Commented out, will mess up slice buffering if not divisible by bin_steps
            // TODO: don't save data if save_steps has already done it
            //  Save data after the loop ends
            if (i % bin_steps != 0) {
                acGridPeriodicBoundconds(STREAM_DEFAULT);
                acGridSynchronizeStream(STREAM_DEFAULT);
                log_from_root_proc_with_sim_progress(pid, "Writing final snapshots to %s, timestep =
            %d\n", snapshot_dir, i); save_mesh_mpi_async(info, snapshot_dir, pid, i,
            simulation_time); log_from_root_proc_with_sim_progress(pid, "Done writing snapshots\n");
            }
            else {
                log_from_root_proc_with_sim_progress(pid, "Snapshots for timestep %d have already
            been written\n");
            }*/

            log_from_root_proc_with_sim_progress(pid, "Exiting simulation loop\n");
            break;
        }
        events = 0;
    }

    // Sync all pending disk accesses before exiting
    acGridDiskAccessSync();

    //////////////////////////////////
    // Simulation over, exit cleanly//
    // Deallocate resources and log //
    //////////////////////////////////
    free_simulation_graphs(pid);

    acLogFromRootProc(pid, "Calling acGridQuit\n");
    acGridQuit();
    fclose(diag_file);
    acLogFromRootProc(pid, "Calling MPI_Finalize\n");
    ac_MPI_Finalize();

    if (check_event(events, SimulationEvent::ErrorState)) {
        acLogFromRootProc(pid, "Simulation ended due to an error, goodbye :(\n");
        return EXIT_FAILURE;
    }
    acLogFromRootProc(pid, "Simulation complete, goodbye!\n");
    return EXIT_SUCCESS;
}

#else
#include <cstdio>
#include <cstdlib>

int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
