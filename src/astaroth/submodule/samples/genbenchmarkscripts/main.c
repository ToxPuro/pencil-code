#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main(void)
{
    const int max_nprocs = 64;
    for (int nprocs = 1; nprocs <= max_nprocs; nprocs *= 2) {
        char filename[4096];
        sprintf(filename, "benchmark_%d.sh", nprocs);

        FILE* fp = fopen(filename, "w");
        assert(fp);

        // Boilerplate
        fprintf(fp, "#!/bin/bash\n");
        fprintf(fp, "#BATCH --job-name=astaroth\n");        // OK
        fprintf(fp, "#SBATCH --account=project_2000403\n"); // OK
        fprintf(fp, "#SBATCH --time=04:00:00\n");           // OK
        fprintf(fp, "#SBATCH --mem=0\n");                   // OK
        fprintf(fp, "#SBATCH --partition=gpu\n");           // OK
        fprintf(fp, "#SBATCH --exclusive\n");               // OK
        fprintf(fp, "#SBATCH --cpus-per-task=10\n");        // OK
        fprintf(fp, "#SBATCH --output=benchmark-%d-%%j.out\n", nprocs);
        // HACK: exclude misconfigured nodes on Puhti
        fprintf(fp, "#SBATCH -x "
                    "r04g[05-06],r02g02,r14g04,r04g07,r16g07,r18g[02-03],r15g08,r17g06,r13g04\n");
        // fprintf(fp, "#SBATCH --cpus-per-task=10\n");

        // nprocs, nodes, gpus
        const int max_gpus_per_node = 4;
        const int gpus_per_node     = nprocs < max_gpus_per_node ? nprocs : max_gpus_per_node;
        const int nodes             = (int)ceil((double)nprocs / max_gpus_per_node);
        fprintf(fp, "#SBATCH --gres=gpu:v100:%d\n", gpus_per_node); // OK
        fprintf(fp, "#SBATCH -n %d\n", nprocs);                     // OK
        fprintf(fp, "#SBATCH -N %d\n", nodes);                      // OK
        // fprintf(fp, "#SBATCH --exclusive\n");
        // if (nprocs >= 4)
        //    fprintf(fp, "#SBATCH --ntasks-per-socket=2\n");

        // Modules
        // OpenMPI
        fprintf(fp, "module load gcc/8.3.0 cuda/10.1.168 cmake openmpi/4.0.3-cuda nccl\n");
        // fprintf(fp, "export UCX_TLS=rc,sm,cuda_copy,gdr_copy,cuda_ipc\n"); //
        // https://www.open-mpi.org/fa fprintf(fp, "export PSM2_CUDA=1\nexport PSM2_GPUDIRECT=1\n");
        // if (nprocs >= 32)
        //    fprintf(fp, "export UCX_TLS=ud_x,cuda_copy,gdr_copy,cuda_ipc\n"); //
        //    https://www.open-mpi.org/fa

        // HPCX
        // fprintf(fp, "module load gcc/8.3.0 cuda/10.1.168 cmake hpcx-mpi/2.5.0-cuda nccl\n");
        // fprintf(fp, "export UCX_MEMTYPE_CACHE=n\n"); // Workaround for bug in hpcx-mpi/2.5.0

        // Profile and run
        // fprintf(fp, "mkdir -p profile_%d\n", nprocs);

        /*
        const int nx = 256; // max size 2048;
        const int ny = nx;
        const int nz = nx;

        fprintf(fp,
                //"srun nvprof --annotate-mpi openmpi -o profile_%d/%%p.nvprof ./benchmark %d %d "
                //"%d\n",
                "srun ./benchmark %d %d %d\n", nx, ny, nz);
        */
        // fprintf(fp, "srun ./benchmark %d %d %d\n", nx, ny, nz);

        const char* files[] = {
            "benchmark_decomp_1D",       "benchmark_decomp_2D",      "benchmark_decomp_3D",
            "benchmark_decomp_1D_comm",  "benchmark_decomp_2D_comm", "benchmark_decomp_3D_comm",
            "benchmark_meshsize_256",    "benchmark_meshsize_512",   "benchmark_meshsize_1024",
            "benchmark_meshsize_2048",   "benchmark_stencilord_2",   "benchmark_stencilord_4",
            "benchmark_stencilord_6",    "benchmark_stencilord_8",   "benchmark_timings_control",
            "benchmark_timings_comp",    "benchmark_timings_comm",   "benchmark_timings_default",
            "benchmark_timings_corners", "benchmark_weak_128",       "benchmark_weak_256",
            "benchmark_weak_512",
        };
        for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); ++i) {
            int nn = 256;
            if (strcmp(files[i], "benchmark_meshsize_512") == 0)
                nn = 512;
            else if (strcmp(files[i], "benchmark_meshsize_1024") == 0)
                nn = 1024;
            else if (strcmp(files[i], "benchmark_meshsize_2048") == 0)
                nn = 2048;
            else if (strcmp(files[i], "benchmark_weak_128") == 0)
                nn = 128;
            else if (strcmp(files[i], "benchmark_weak_512") == 0)
                nn = 512;

            // W/ Fredriks tunings
            // (may cause Assertion `status == UCS_OK' failed errors)
            // fprintf(fp,
            //        "$(cd %s && UCX_RNDV_THRESH=16384 UCX_RNDV_SCHEME=get_zcopy "
            //        "UCX_MAX_RNDV_RAILS=1 srun ./benchmark %d %d %d && cd ..)\n",
            //        files[i], nn, nn, nn);
            if (nodes >= 2) {
                fprintf(fp,
                        "$(cd %s && UCX_RNDV_THRESH=16384 UCX_RNDV_SCHEME=get_zcopy "
                        "UCX_MAX_RNDV_RAILS=1 srun --kill-on-bad-exit=0 ./benchmark %d %d %d && rm "
                        "-f core.* && cd ..)\n",
                        files[i], nn, nn, nn);
            }
            else {
                fprintf(fp,
                        "$(cd %s && srun --kill-on-bad-exit=0 ./benchmark %d %d %d && rm -f core.* "
                        "&& cd ..)\n",
                        files[i], nn, nn, nn);
            }
        }

        fclose(fp);
    }

    return EXIT_SUCCESS;
}
