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
#include "astaroth_utils.h"

#include "errchk.h"

static const char dataformat_path[] = "data-format.csv";

AcResult
acHostVertexBufferSet(const VertexBufferHandle handle, const AcReal value, AcMesh* mesh)
{
    const size_t n = acVertexBufferSize(mesh->info);
    for (size_t i = 0; i < n; ++i)
        mesh->vertex_buffer[handle][i] = value;

    return AC_SUCCESS;
}
AcResult
acHostMeshSet(const AcReal value, AcMesh* mesh)
{
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        acHostVertexBufferSet(w, value, mesh);

    return AC_SUCCESS;
}

AcResult
acHostMeshApplyPeriodicBounds(AcMesh* mesh)
{
    const AcMeshInfo info = mesh->info;
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const int3 start = (int3){0, 0, 0};
        const int3 end   = (int3){info.int_params[AC_mx], info.int_params[AC_my],
                                  info.int_params[AC_mz]};

        const int nx = info.int_params[AC_nx];
        const int ny = info.int_params[AC_ny];
        const int nz = info.int_params[AC_nz];

        const int nx_min = info.int_params[AC_nx_min];
        const int ny_min = info.int_params[AC_ny_min];
        const int nz_min = info.int_params[AC_nz_min];

        // The old kxt was inclusive, but our mx_max is exclusive
        const int nx_max = info.int_params[AC_nx_max];
        const int ny_max = info.int_params[AC_ny_max];
        const int nz_max = info.int_params[AC_nz_max];

        // #pragma omp parallel for
        for (int k_dst = start.z; k_dst < end.z; ++k_dst) {
            for (int j_dst = start.y; j_dst < end.y; ++j_dst) {
                for (int i_dst = start.x; i_dst < end.x; ++i_dst) {

                    // If destination index is inside the computational domain, return since
                    // the boundary conditions are only applied to the ghost zones
                    if (i_dst >= nx_min && i_dst < nx_max && j_dst >= ny_min && j_dst < ny_max &&
                        k_dst >= nz_min && k_dst < nz_max)
                        continue;

                    // Find the source index
                    // Map to nx, ny, nz coordinates
                    int i_src = i_dst - nx_min;
                    int j_src = j_dst - ny_min;
                    int k_src = k_dst - nz_min;

                    // Translate (s.t. the index is always positive)
                    i_src += nx;
                    j_src += ny;
                    k_src += nz;

                    // Wrap
                    i_src %= nx;
                    j_src %= ny;
                    k_src %= nz;

                    // Map to mx, my, mz coordinates
                    i_src += nx_min;
                    j_src += ny_min;
                    k_src += nz_min;

                    const size_t src_idx = acVertexBufferIdx(i_src, j_src, k_src, info);
                    const size_t dst_idx = acVertexBufferIdx(i_dst, j_dst, k_dst, info);
                    ERRCHK(src_idx < acVertexBufferSize(info));
                    ERRCHK(dst_idx < acVertexBufferSize(info));
                    mesh->vertex_buffer[w][dst_idx] = mesh->vertex_buffer[w][src_idx];
                }
            }
        }
    }
    return AC_SUCCESS;
}

AcResult
acHostMeshClear(AcMesh* mesh)
{
    return acHostMeshSet(0, mesh);
}

AcResult
acHostMeshWriteToFile(const AcMesh mesh, const size_t id)
{
    FILE* header = fopen(dataformat_path, "w");
    ERRCHK_ALWAYS(header);
    fprintf(header, "use_double, mx, my, mz\n");
    fprintf(header, "%d, %d, %d, %d\n", sizeof(AcReal) == 8, mesh.info.int_params[AC_mx],
            mesh.info.int_params[AC_my], mesh.info.int_params[AC_mz]);
    fclose(header);

    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        const size_t len = 4096;
        char buf[len];
        const int retval = snprintf(buf, len, "%s-%.5lu.dat", field_names[i], id);
        ERRCHK_ALWAYS(retval >= 0);
        ERRCHK_ALWAYS((size_t)retval <= len);

        FILE* fp = fopen(buf, "w");
        ERRCHK_ALWAYS(fp);

        const size_t bytes = sizeof(mesh.vertex_buffer[i][0]);
        const size_t count = acVertexBufferSize(mesh.info);
        const size_t res   = fwrite(mesh.vertex_buffer[i], bytes, count, fp);
        ERRCHK_ALWAYS(res == count);

        fclose(fp);
    }
    return AC_SUCCESS;
}

AcResult
acHostMeshReadFromFile(const size_t id, AcMesh* mesh)
{
    const size_t len = 4096;
    char buf[len];
    int use_double, mx, my, mz;

    FILE* header = fopen(dataformat_path, "r");
    ERRCHK_ALWAYS(header);
    fgets(buf, len, header);
    fscanf(header, "%d, %d, %d, %d\n", &use_double, &mx, &my, &mz);
    fclose(header);

    ERRCHK_ALWAYS(use_double == (sizeof(AcReal) == 8));
    ERRCHK_ALWAYS(mx == mesh->info.int_params[AC_mx]);
    ERRCHK_ALWAYS(my == mesh->info.int_params[AC_my]);
    ERRCHK_ALWAYS(mz == mesh->info.int_params[AC_mz]);

    for (size_t i = 0; i < NUM_FIELDS; ++i) {

        const int retval = snprintf(buf, len, "%s-%.5lu.dat", field_names[i], id);
        ERRCHK_ALWAYS(retval >= 0);
        ERRCHK_ALWAYS((size_t)retval <= len);

        FILE* fp = fopen(buf, "r");
        ERRCHK_ALWAYS(fp);

        const size_t bytes = sizeof(mesh->vertex_buffer[i][0]);
        const size_t count = acVertexBufferSize(mesh->info);
        const size_t res   = fread(mesh->vertex_buffer[i], bytes, count, fp);
        ERRCHK_ALWAYS(res == count);

        fclose(fp);
    }
    return AC_SUCCESS;
}