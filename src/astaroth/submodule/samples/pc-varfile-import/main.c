#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"
#include "user_defines.h"

#if !AC_MPI_ENABLED
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#else

#include <mpi.h>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

/*
cmake -DMPI_ENABLED=ON .. && make -j && $SRUNMPI4 ./pc-varfile-import\
--input=../mesh-scaler/build --volume=256,256,256
*/

// #include <math.h>

static void
merge_slices(const char* job_dir, const int label, const size_t nx, const size_t ny,
             const Field fields[], const size_t num_fields)
{
    const size_t count = nx * ny;
    AcReal* buf        = malloc(sizeof(AcReal) * count);
    ERRCHK_ALWAYS(buf);

    char outfile[4096] = {0};
    sprintf(outfile, "%s/%s.dat", job_dir, "slices");
    FILE* out = fopen(outfile, "w");
    ERRCHK_ALWAYS(out);
    for (size_t i = 0; i < num_fields; ++i) {
        const Field field = fields[i];

        char infile[4096] = {0};
        sprintf(infile, "%s/%s-%012d.slice", job_dir, vtxbuf_names[field], label);
        FILE* in = fopen(infile, "r");
        ERRCHK_ALWAYS(in);

        fread(buf, sizeof(AcReal), count, in);
        fwrite(buf, sizeof(AcReal), count, out);

        fclose(in);
    }
    fclose(out);
    free(buf);
}

int
main(void)
{
    MPI_Init(NULL, NULL);
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Modify these based on the varfile format
    // const char* file = "test.dat";
    const char* file = "/scratch/project_462000077/mkorpi/forced/mahti_4096/data/allprocs/var.dat";
    const Field fields[] =
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
#if !LMAGNETIC
    WARNING("LMAGNETIC was not set, magnetic field is not read in pc-varfile-import");
#endif
    const size_t num_fields = ARRAY_SIZE(fields);
    const int3 nn           = (int3){4096, 4096, 4096};
    // const int3 nn = (int3){2048, 2048, 8};
    const int3 rr = (int3){3, 3, 3};

    /*
    // Debug start
    const int3 mm = (int3){nn.x + 2 * rr.x, nn.y + 2 * rr.y, nn.z + 2 * rr.z};

    FILE* fp = fopen(file, "w");
    ERRCHK_ALWAYS(fp);
    const size_t count = mm.x * mm.y * mm.z;
    AcReal* buf        = malloc(sizeof(AcReal) * count);
    for (size_t k = 0; k < mm.z; ++k) {
        for (size_t j = 0; j < mm.y; ++j) {
            for (size_t i = 0; i < mm.x; ++i) {
                const size_t idx = i + j * mm.x + k * mm.x * mm.y;
                buf[idx]         = powf(i * i + j * j, 1.0 / 2.0);
            }
        }
    }
    fwrite(buf, sizeof(AcReal), count, fp);
    for (size_t k = 0; k < mm.z; ++k) {
        for (size_t j = 0; j < mm.y; ++j) {
            for (size_t i = 0; i < mm.x; ++i) {
                const size_t idx = i + j * mm.x + k * mm.x * mm.y;
                buf[idx]         = i;
            }
        }
    }
    fwrite(buf, sizeof(AcReal), count, fp);
    for (size_t k = 0; k < mm.z; ++k) {
        for (size_t j = 0; j < mm.y; ++j) {
            for (size_t i = 0; i < mm.x; ++i) {
                const size_t idx = i + j * mm.x + k * mm.x * mm.y;
                buf[idx]         = j;
            }
        }
    }
    fwrite(buf, sizeof(AcReal), count, fp);
    for (size_t k = 0; k < mm.z; ++k) {
        for (size_t j = 0; j < mm.y; ++j) {
            for (size_t i = 0; i < mm.x; ++i) {
                const size_t idx = i + j * mm.x + k * mm.x * mm.y;
                buf[idx]         = k;
            }
        }
    }
    fwrite(buf, sizeof(AcReal), count, fp);

    for (size_t i = 0; i < num_fields - 4; ++i)
        fwrite(buf, sizeof(AcReal), count, fp);

    free(buf);
    fclose(fp);
    // Debug end
    */

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info.int_params[AC_nx] = nn.x;
    info.int_params[AC_ny] = nn.y;
    info.int_params[AC_nz] = nn.z;
    acHostUpdateBuiltinParams(&info);

    // Init
    acGridInit(info);
    acGridReadVarfileToMesh(file, fields, num_fields, nn, rr);

    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        AcReal buf_max, buf_min, buf_rms;
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, i, &buf_max);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, i, &buf_min);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_RMS, i, &buf_rms);

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", 8, vtxbuf_names[i], (double)(buf_min),
               (double)(buf_rms), (double)(buf_max));
    }

    // Write snapshots
    // Create a tmpdir for output
    const int job_id = 12345;
    char job_dir[4096];
    snprintf(job_dir, 4096, "output-%d", job_id);

    char cmd[4096];
    snprintf(cmd, 4096, "mkdir -p %s", job_dir);
    system(cmd);

    // Write snapshots
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, job_dir, vtxbuf_names[i],
                                          ACCESS_WRITE);

    // Write slices
    acGridWriteSlicesToDiskLaunch(job_dir, 0);

    // Merge slices
    merge_slices(job_dir, 0, info.int_params[AC_nx], info.int_params[AC_ny], fields, num_fields);

    // Quit
    acGridQuit();
    MPI_Finalize();

    if (!pid)
        printf("OK!\n");

    return EXIT_SUCCESS;
}
#endif