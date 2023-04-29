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
        fprintf(stderr, "Usage: ./benchmark-node <nx> <ny> <nz>\n");
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
    Node node;
    acNodeCreate(0, info, &node);
    acNodePrintInfo(node);
    acNodeLoadMesh(node, STREAM_DEFAULT, model);
    acNodeStoreMesh(node, STREAM_DEFAULT, &candidate);
    acVerifyMesh("Load/Store", model, candidate);

    // Verify that boundconds work correctly
    const int3 n_min = (int3){STENCIL_ORDER / 2, STENCIL_ORDER / 2, STENCIL_ORDER / 2};
    const int3 n_max = (int3){
        n_min.x + info.int_params[AC_nx],
        n_min.y + info.int_params[AC_ny],
        n_min.z + info.int_params[AC_nz],
    };
    acNodePeriodicBoundconds(node, STREAM_DEFAULT);
    acNodeStoreMesh(node, STREAM_DEFAULT, &candidate);
    acNodeSynchronizeStream(node, STREAM_DEFAULT);
    acHostMeshApplyPeriodicBounds(&model);
    acVerifyMesh("Boundconds", model, candidate);

    // Verify that integration works correctly
    const AcReal dt = (AcReal)FLT_EPSILON;
    // const bool alt_integration = false; // Uncomment to test one- and two-pass integration

    // DRYRUN START
    // Optimize for the more expensive substep (second and third)
    acNodeIntegrateSubstep(node, STREAM_DEFAULT, 2, n_min, n_max, dt);
    for (int i = 0; i < 3; ++i) {
        acNodeIntegrateSubstep(node, STREAM_DEFAULT, i, n_min, n_max, dt);
        acNodeSwapBuffers(node);
        acNodePeriodicBoundconds(node, STREAM_DEFAULT);
    }
    // TODO START
    // create acNodeReset or something like that
    // to flush the buffers to non-nan values (otherwise two-pass fails here)
    acNodeLoadMesh(node, STREAM_DEFAULT, model);
    acNodeSwapBuffers(node);
    // TODO END
    acNodeLoadMesh(node, STREAM_DEFAULT, model);
    acNodePeriodicBoundconds(node, STREAM_DEFAULT);
    ///////////////////////////// DRYRUN END

    const size_t nsteps = 1;
    for (size_t j = 0; j < nsteps; ++j) {
        for (int i = 0; i < 3; ++i) {
            acNodeIntegrateSubstep(node, STREAM_DEFAULT, i, n_min, n_max, dt);
            acNodeSwapBuffers(node);
            acNodePeriodicBoundconds(node, STREAM_DEFAULT);
        }
    }
    if (verify) {
        acNodeStoreMesh(node, STREAM_DEFAULT, &candidate);

        for (size_t j = 0; j < nsteps; ++j) {
            acHostIntegrateStep(model, dt);
            acHostMeshApplyPeriodicBounds(&model);
        }

        acNodeSynchronizeStream(node, STREAM_DEFAULT);
        acVerifyMesh("Integration", model, candidate);
    }

    // Warmup
    for (int j = 0; j < NSAMPLES / 10; ++j) {
        for (int step = 0; step < 3; ++step) {
            acNodeIntegrateSubstep(node, STREAM_DEFAULT, step, n_min, n_max, dt);
            acNodePeriodicBoundconds(node, STREAM_DEFAULT);
        }
    }
    acNodeSynchronizeStream(node, STREAM_DEFAULT);

    // Benchmark
    Timer t;
    // timer_reset(&t);
    double results[NSAMPLES] = {0};
#pragma unroll
    for (int j = 0; j < NSAMPLES; ++j) {
        // Single substep
        // acNodeIntegrateSubstep(node, STREAM_DEFAULT, 2, n_min, n_max, dt);

        // Full integration step
        acNodeSynchronizeStream(node, STREAM_ALL);
        timer_reset(&t);
        for (int i = 0; i < 3; ++i) {
            acNodeIntegrateSubstep(node, STREAM_DEFAULT, i, n_min, n_max, dt);
            acNodeSwapBuffers(node);
            acNodePeriodicBoundconds(node, STREAM_DEFAULT);
        }
        acNodeSynchronizeStream(node, STREAM_ALL);
        results[j] = timer_diff_nsec(t) / 1e6;
    }
    // acNodeSynchronizeStream(node, STREAM_DEFAULT);
    // const double ms_elapsed   = timer_diff_nsec(t) / 1e6;
    // const double milliseconds = ms_elapsed / NSAMPLES;
    // printf("Average integration time: %.4g ms\n", milliseconds);

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

    // Write to file
    const char* benchmark_dir = "node-benchmark.csv";
    FILE* fp                  = fopen(benchmark_dir, "a");
    ERRCHK_ALWAYS(fp);
    // 'implementation, maxthreadsperblock, milliseconds, nx, ny, nz, devices'
    const int num_devices = acGetNumDevicesPerNode();
    fprintf(fp, "%d,%d,%g,%d,%d,%d,%d\n", IMPLEMENTATION, MAX_THREADS_PER_BLOCK, percentile90th,
            info.int_params[AC_nx], info.int_params[AC_ny], info.int_params[AC_nz], num_devices);
    fclose(fp);

    // Profile
    cudaProfilerStart();
    acNodeIntegrateSubstep(node, STREAM_DEFAULT, 2, n_min, n_max, dt);
    cudaProfilerStop();

    // Destroy
    acNodeDestroy(node);
    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);

    return EXIT_SUCCESS;
}
