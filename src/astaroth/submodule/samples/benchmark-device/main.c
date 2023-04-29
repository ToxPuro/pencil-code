#include <float.h> // FLT_EPSILON
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"

#include "timer_hires.h"

#define NSAMPLES (100)

#define ERRCHK_AC(x) ERRCHK_ALWAYS((x) == AC_SUCCESS);

static const bool verify = false;

// TODO use common sort with benchmark, benchmark-device, benchmark-node, mpi-io
void
validate(const size_t count, const double* arr)
{
    for (size_t i = 1; i < count; ++i)
        ERRCHK_ALWAYS(arr[i] >= arr[i - 1]);
}

// TODO use common sort with benchmark, benchmark-device, benchmark-node, mpi-io
void
sort(const size_t count, double* arr)
{
    for (size_t j = 0; j < count; ++j) {
        for (size_t i = j + 1; i < count; ++i) {
            if (arr[i] < arr[j]) {
                const double tmp = arr[j];
                arr[j]           = arr[i];
                arr[i]           = tmp;
            }
        }
    }
}

int
main(int argc, char** argv)
{
    cudaProfilerStop();

    printf("Num fields %lu\n", acGetNumFields());
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        size_t field;
        ERRCHK_AC(acGetFieldHandle(field_names[i], &field));
        printf("Searching field %s. Got %lu (%s)\n", field_names[i], field, field_names[field]);
    }
    size_t field;
    ERRCHK_ALWAYS(acGetFieldHandle("nonexistent", &field) == AC_FAILURE);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    // Set mesh dimensions
    if (argc != 4) {
        fprintf(stderr, "Usage: ./benchmark-device <nx> <ny> <nz>\n");
        return EXIT_FAILURE;
    }
    else {
        info.int_params[AC_nx] = atoi(argv[1]);
        info.int_params[AC_ny] = atoi(argv[2]);
        info.int_params[AC_nz] = atoi(argv[3]);
        acHostUpdateBuiltinParams(&info);
    }

    // Alloc
    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);

    // Init
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);
    acHostMeshApplyPeriodicBounds(&model);

    // Verify that the mesh was loaded and stored correctly
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acVerifyMesh("Load/Store", model, candidate);

    // Verify that boundconds work correctly
    const int3 m_min = (int3){0, 0, 0};
    const int3 m_max = (int3){
        info.int_params[AC_mx],
        info.int_params[AC_my],
        info.int_params[AC_mz],
    };
    const int3 n_min = (int3){STENCIL_ORDER / 2, STENCIL_ORDER / 2, STENCIL_ORDER / 2};
    const int3 n_max = (int3){
        n_min.x + info.int_params[AC_nx],
        n_min.y + info.int_params[AC_ny],
        n_min.z + info.int_params[AC_nz],
    };
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    acHostMeshApplyPeriodicBounds(&model);
    acVerifyMesh("Boundconds", model, candidate);

    // Verify that integration works correctly
    const AcReal dt = (AcReal)FLT_EPSILON;
    // const bool alt_integration = false; // Uncomment to test one- and two-pass integration

    // DRYRUN START
    // Optimize for the more expensive substep (second and third)
    acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 2, n_min, n_max, dt);
    for (int i = 0; i < 3; ++i) {
        acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, n_min, n_max, dt);
        acDeviceSwapBuffers(device);
        acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
    }
    // TODO START
    // create acDeviceReset or something like that
    // to flush the buffers to non-nan values (otherwise two-pass fails here)
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    acDeviceSwapBuffers(device);
    // TODO END
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
    ///////////////////////////// DRYRUN END

    const size_t nsteps = 1;
    for (size_t j = 0; j < nsteps; ++j) {
        for (int i = 0; i < 3; ++i) {
            acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, n_min, n_max, dt);
            acDeviceSwapBuffers(device);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
        }
    }

    if (verify) {
        acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);

        for (size_t j = 0; j < nsteps; ++j) {
            acHostIntegrateStep(model, dt);
            acHostMeshApplyPeriodicBounds(&model);
        }

        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
        acVerifyMesh("Integration", model, candidate);
    }

    // Warmup
    for (int j = 0; j < NSAMPLES / 10; ++j) {
        for (int step = 0; step < 3; ++step) {
            acDeviceIntegrateSubstep(device, STREAM_DEFAULT, step, n_min, n_max, dt);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
        }
    }
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    // Benchmark
    Timer t;
    // timer_reset(&t);
    double results[NSAMPLES] = {0};
#pragma unroll
    for (int j = 0; j < NSAMPLES; ++j) {
        // Substep
        acDeviceSynchronizeStream(device, STREAM_ALL);
        timer_reset(&t);
        acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 2, n_min, n_max, dt);
        acDeviceSynchronizeStream(device, STREAM_ALL);
        results[j] = timer_diff_nsec(t) / 1e6;

        // Full integration step
        // for (int i = 0; i < 3; ++i) {
        //    acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, n_min, n_max, dt);
        //    acDeviceSwapBuffers(device);
        //    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m_min, m_max);
        //}
    }
    sort(NSAMPLES, results);
    validate(NSAMPLES, results);
    const double min            = results[0];
    const double median         = NSAMPLES % 2 ? results[NSAMPLES / 2]
                                               : 0.5 * (results[NSAMPLES / 2 - 1] + results[NSAMPLES / 2]);
    const double percentile90th = results[(size_t)ceil(0.9 * NSAMPLES)];
    const double max            = results[NSAMPLES - 1];

    printf("Integration times:\n");
    printf("\tmin: %g\n", min);
    printf("\tmedian: %g\n", median);
    printf("\t90th percentile: %g\n",
           percentile90th); // Conservative, takes the first precentile >= 90%
    printf("\tmax: %g\n", max);

    // for (size_t i = 0; i < NSAMPLES; ++i)
    //     printf("%g\n", results[i]);
    // acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    // const double ms_elapsed   = timer_diff_nsec(t) / 1e6;
    // const double milliseconds = ms_elapsed / NSAMPLES;
    // printf("Average integration time: %.4g ms\n", milliseconds);

    // Write to file
    const char* benchmark_dir = "device-benchmark.csv";
    FILE* fp                  = fopen(benchmark_dir, "a");
    ERRCHK_ALWAYS(fp);
    // 'implementation,
    // maxthreadsperblock,millisecondsmin,millisecondsmedian,milliseconds90thpercentile,millisecondsmax,
    // nx, ny, nz, devices'
    const int num_devices = 1;
    fprintf(fp, "%d,%d,%g,%d,%d,%d,%d\n", IMPLEMENTATION, MAX_THREADS_PER_BLOCK, percentile90th,
            info.int_params[AC_nx], info.int_params[AC_ny], info.int_params[AC_nz], num_devices);
    fclose(fp);

    // Profile
    cudaProfilerStart();
    acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 2, n_min, n_max, dt);
    cudaProfilerStop();

    // Destroy
    acDeviceDestroy(device);
    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);

    return EXIT_SUCCESS;
}
