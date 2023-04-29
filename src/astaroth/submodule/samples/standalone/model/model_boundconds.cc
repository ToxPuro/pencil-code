/*
    Copyright (C) 2014-2022, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) amy later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT Amy WARRANTY; without even the implied warranty of
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
#include "model_boundconds.h"

#include "errchk.h"

void
boundconds(const AcMeshInfo& mesh_info, ModelMesh* mesh)
{
#pragma omp parallel for
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const int3 start = (int3){0, 0, 0};
        const int3 end   = (int3){mesh_info.int_params[AC_mx], mesh_info.int_params[AC_my],
                                  mesh_info.int_params[AC_mz]};

        const int nx = mesh_info.int_params[AC_nx];
        const int ny = mesh_info.int_params[AC_ny];
        const int nz = mesh_info.int_params[AC_nz];

        const int nx_min = mesh_info.int_params[AC_nx_min];
        const int ny_min = mesh_info.int_params[AC_ny_min];
        const int nz_min = mesh_info.int_params[AC_nz_min];

        // The old kxt was inclusive, but our mx_max is exclusive
        const int nx_max = mesh_info.int_params[AC_nx_max];
        const int ny_max = mesh_info.int_params[AC_ny_max];
        const int nz_max = mesh_info.int_params[AC_nz_max];

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

                    const size_t src_idx = acVertexBufferIdx(i_src, j_src, k_src, mesh_info);
                    const size_t dst_idx = acVertexBufferIdx(i_dst, j_dst, k_dst, mesh_info);
                    ERRCHK(src_idx < acVertexBufferSize(mesh_info));
                    ERRCHK(dst_idx < acVertexBufferSize(mesh_info));
                    mesh->vertex_buffer[w][dst_idx] = mesh->vertex_buffer[w][src_idx];
                }
            }
        }
    }
}

#if 0
void
boundconds(const AcMeshInfo& mesh_info, ModelMesh* mesh)
{
    const int mx = mesh_info.int_params[AC_mx];
    const int my = mesh_info.int_params[AC_my];
    const int mz = mesh_info.int_params[AC_mz];

    // Volatile here suppresses the warning about strict-overflow (i.e. compiler
    // wanted to optimize these loops by assuming that kxb etc never overflow)
    // However we do not need the performance improvement (~1-3%) and it's
    // not either good to
    //	a) get useless warnings originating from here
    //	b) disable the warnings completely
    volatile const int kxb = mesh_info.int_params[AC_nx_min];
    volatile const int kyb = mesh_info.int_params[AC_ny_min];
    volatile const int kzb = mesh_info.int_params[AC_nz_min];

    // The old kxt was inclusive, but our mx_max is exclusive
    volatile const int kxt = mesh_info.int_params[AC_nx_max] - 1;
    volatile const int kyt = mesh_info.int_params[AC_ny_max] - 1;
    volatile const int kzt = mesh_info.int_params[AC_nz_max] - 1;
    const int bound[3]     = {0, 0, 0};

    // Periodic boundary conditions
    if (bound[0] == 0) {
        for (int k = kzb; k <= kzt; k++) {
            for (int j = kyb; j <= kyt; j++) {
                for (int i = kxb; i <= kxb + 2; i++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (kxt + i - 2) + j * mx + k * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
                for (int i = kxt - 2; i <= kxt; i++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - kxt + 2) + j * mx + k * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
    }
    if (bound[1] == 0) {
        for (int k = kzb; k <= kzt; k++) {
            for (int i = kxb; i <= kxt; i++) {
                for (int j = kyb; j <= kyb + 2; j++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = i + (kyt + j - 2) * mx + k * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
                for (int j = kyt - 2; j <= kyt; j++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = i + (j - kyt + 2) * mx + k * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
    }

    if (bound[2] == 0) {
        for (int i = kxb; i <= kxt; i++) {
            for (int j = kyb; j <= kyt; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = i + j * mx + (kzt + k - 2) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = i + j * mx + (k - kzt + 2) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
    }

    // Copy the corners in the fully periodic case
    if (bound[0] == 0 && bound[1] == 0 && bound[2] == 0) {
        // Source corner: x=0, y=0, z=0
        for (int i = kxb; i <= kxb + 2; i++) {
            for (int j = kyb; j <= kyb + 2; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i + mx - STENCIL_ORDER) + (j + my - STENCIL_ORDER) * mx +
                                     (k + mz - STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source corner: x=1, y=0, z=0
        for (int i = kxt - 2; i <= kxt; i++) {
            for (int j = kyb; j <= kyb + 2; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - mx + STENCIL_ORDER) + (j + my - STENCIL_ORDER) * mx +
                                     (k + mz - STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source corner: x=0, y=1, z=0
        for (int i = kxb; i <= kxb + 2; i++) {
            for (int j = kyt - 2; j <= kyt; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i + mx - STENCIL_ORDER) + (j - my + STENCIL_ORDER) * mx +
                                     (k + mz - STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source corner: x=0, y=0, z=1
        for (int i = kxb; i <= kxb + 2; i++) {
            for (int j = kyb; j <= kyb + 2; j++) {
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i + mx - STENCIL_ORDER) + (j + my - STENCIL_ORDER) * mx +
                                     (k - mz + STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source corner: x=1, y=1, z=0
        for (int i = kxt - 2; i <= kxt; i++) {
            for (int j = kyt - 2; j <= kyt; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - mx + STENCIL_ORDER) + (j - my + STENCIL_ORDER) * mx +
                                     (k + mz - STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source corner: x=1, y=0, z=1
        for (int i = kxt - 2; i <= kxt; i++) {
            for (int j = kyb; j <= kyb + 2; j++) {
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - mx + STENCIL_ORDER) + (j + my - STENCIL_ORDER) * mx +
                                     (k - mz + STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source corner: x=0, y=1, z=1
        for (int i = kxb; i <= kxb + 2; i++) {
            for (int j = kyt - 2; j <= kyt; j++) {
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i + mx - STENCIL_ORDER) + (j - my + STENCIL_ORDER) * mx +
                                     (k - mz + STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source corner: x=1, y=1, z=1
        for (int i = kxt - 2; i <= kxt; i++) {
            for (int j = kyt - 2; j <= kyt; j++) {
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - mx + STENCIL_ORDER) + (j - my + STENCIL_ORDER) * mx +
                                     (k - mz + STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
    }
    else {
        ERROR("ONLY FULLY PERIODIC WORKS WITH CORNERS SO FAR! \n");
    }

    // Copy the edges in the fully periodic case
    if (bound[0] == 0 && bound[1] == 0 && bound[2] == 0) {
        // Source edge: x = 0, y = 0
        for (int i = kxb; i <= kxb + 2; i++) {
            for (int j = kyb; j <= kyb + 2; j++) {
                for (int k = kzb; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i + mx - STENCIL_ORDER) + (j + my - STENCIL_ORDER) * mx +
                                     k * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: x = 1, y = 0
        for (int i = kxt - 2; i <= kxt; i++) {
            for (int j = kyb; j <= kyb + 2; j++) {
                for (int k = kzb; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - mx + STENCIL_ORDER) + (j + my - STENCIL_ORDER) * mx +
                                     k * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: x = 0, y = 1
        for (int i = kxb; i <= kxb + 2; i++) {
            for (int j = kyt - 2; j <= kyt; j++) {
                for (int k = kzb; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i + mx - STENCIL_ORDER) + (j - my + STENCIL_ORDER) * mx +
                                     k * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: x = 1, y = 1
        for (int i = kxt - 2; i <= kxt; i++) {
            for (int j = kyt - 2; j <= kyt; j++) {
                for (int k = kzb; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - mx + STENCIL_ORDER) + (j - my + STENCIL_ORDER) * mx +
                                     k * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: x = 0, z = 0
        for (int i = kxb; i <= kxb + 2; i++) {
            for (int j = kyb; j <= kyt; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i + mx - STENCIL_ORDER) + j * mx +
                                     (k + mz - STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: x = 1, z = 0
        for (int i = kxt - 2; i <= kxt; i++) {
            for (int j = kyb; j <= kyt; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - mx + STENCIL_ORDER) + j * mx +
                                     (k + mz - STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: x = 0, z = 1
        for (int i = kxb; i <= kxb + 2; i++) {
            for (int j = kyb; j <= kyt; j++) {
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i + mx - STENCIL_ORDER) + j * mx +
                                     (k - mz + STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: x = 1, z = 1
        for (int i = kxt - 2; i <= kxt; i++) {
            for (int j = kyb; j <= kyt; j++) {
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = (i - mx + STENCIL_ORDER) + j * mx +
                                     (k - mz + STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: y = 0, z = 0
        for (int i = kxb; i <= kxt; i++) {
            for (int j = kyb; j <= kyb + 2; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = i + (j + my - STENCIL_ORDER) * mx +
                                     (k + mz - STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: y = 1, z = 0
        for (int i = kxb; i <= kxt; i++) {
            for (int j = kyt - 2; j <= kyt; j++) {
                for (int k = kzb; k <= kzb + 2; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = i + (j - my + STENCIL_ORDER) * mx +
                                     (k + mz - STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: y = 0, z = 1
        for (int i = kxb; i <= kxt; i++) {
            for (int j = kyb; j <= kyb + 2; j++) {
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = i + (j + my - STENCIL_ORDER) * mx +
                                     (k - mz + STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
        // Source edge: y = 1, z = 1
        for (int i = kxb; i <= kxt; i++) {
            for (int j = kyt - 2; j <= kyt; j++) {
                for (int k = kzt - 2; k <= kzt; k++) {
                    const int inds = i + j * mx + k * mx * my;
                    const int indr = i + (j - my + STENCIL_ORDER) * mx +
                                     (k - mz + STENCIL_ORDER) * mx * my;
                    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
                        mesh->vertex_buffer[w]
                                           [indr] = mesh->vertex_buffer[w]
                                                                       [inds];
                }
            }
        }
    }
    else {
        ERROR("ONLY FULLY PERIODIC WORKS WITH EDGES SO FAR! \n");
    }
}
#endif
