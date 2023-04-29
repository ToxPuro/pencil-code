#include <float.h> // FLT_EPSILON
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"

#include "timer_hires.h"

#if !BLUR_KERNEL_AVAILABLE
int
main(void)
{
    fprintf(stderr, "blur_kernel() was not available. Must build with `cmake "
                    "-DDSL_MODULE_DIR=<path/to/blur.ac> -DBUILD_MHD_SAMPLES=OFF ..\n");
    return EXIT_FAILURE;
}
#else
#define NSAMPLES (100)

int
main(int argc, char** argv)
{
    AcMeshInfo info;

    // Set mesh dimensions
    if (argc != 3) {
        fprintf(stderr, "Usage: ./blur <nx> <ny>\n");
        return EXIT_FAILURE;
    }
    else {
        info.int_params[AC_nx] = atoi(argv[1]);
        info.int_params[AC_ny] = atoi(argv[2]);
        info.int_params[AC_nz] = 1;
        acHostUpdateBuiltinParams(&info);
    }

    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    // acHostMeshRandomize(&mesh);
    acHostMeshSet(1, &mesh);

    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);

    // Benchmark
    Timer t;
    timer_reset(&t);

    const int3 nn_min = (int3){info.int_params[AC_nx_min], info.int_params[AC_ny_min],
                               info.int_params[AC_nz_min]};
    const int3 nn_max = (int3){info.int_params[AC_nx_max], info.int_params[AC_ny_max],
                               info.int_params[AC_nz_max]};
    const int3 start  = nn_min;
    const int3 end    = nn_max;
    acDeviceLaunchKernel(device, STREAM_DEFAULT, blur_kernel, start, end);
    acDeviceSwapBuffers(device);

    acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);
    printf("Store complete\n");

    const bool print_bounds = true;
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        for (int k = nn_min.z; k < nn_max.z; ++k) {
            printf("==== DEPTH %d ====\n", k);
            if (print_bounds) {
                for (int j = 0; j < info.int_params[AC_my]; ++j) {
                    for (int i = 0; i < info.int_params[AC_mx]; ++i) {
                        printf("%.3g ",
                               (double)mesh.vertex_buffer[w][acVertexBufferIdx(i, j, k, info)]);
                    }
                    printf("\n");
                }
            }
            else {
                for (int j = nn_min.y; j < nn_max.y; ++j) {
                    for (int i = nn_min.x; i < nn_max.x; ++i) {
                        printf("%.3g ",
                               (double)mesh.vertex_buffer[w][acVertexBufferIdx(i, j, k, info)]);
                    }
                    printf("\n");
                }
            }
        }
    }

    acDeviceDestroy(device);
    acHostMeshDestroy(&mesh);

    return EXIT_SUCCESS;
}
#endif
