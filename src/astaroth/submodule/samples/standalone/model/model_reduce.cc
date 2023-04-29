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
#include "model_reduce.h"

#include <math.h>

#include "errchk.h"

// Function pointer definitions
typedef ModelScalar (*ReduceFunc)(const ModelScalar&, const ModelScalar&);
typedef ModelScalar (*ReduceInitialScalFunc)(const ModelScalar&);
typedef ModelScalar (*ReduceInitialVecFunc)(const ModelScalar&, const ModelScalar&,
                                            const ModelScalar&);

// clang-format off
/* Comparison funcs */
static inline ModelScalar
max(const ModelScalar& a, const ModelScalar& b) { return a > b ? a : b; }

static inline ModelScalar
min(const ModelScalar& a, const ModelScalar& b) { return a < b ? a : b; }

static inline ModelScalar
sum(const ModelScalar& a, const ModelScalar& b) { return a + b; }

/* Function used to determine the values used during reduction */
static inline ModelScalar
length(const ModelScalar& a) { return (ModelScalar)(a); }

static inline ModelScalar
length(const ModelScalar& a, const ModelScalar& b, const ModelScalar& c) { return sqrtl(a*a + b*b + c*c); }

static inline ModelScalar
squared(const ModelScalar& a) { return (ModelScalar)(a*a); }

static inline ModelScalar
squared(const ModelScalar& a, const ModelScalar& b, const ModelScalar& c) { return squared(a) + squared(b) + squared(c); }

static inline ModelScalar
exp_squared(const ModelScalar& a) { return expl(a)*expl(a); }

static inline ModelScalar
exp_squared(const ModelScalar& a, const ModelScalar& b, const ModelScalar& c) { return exp_squared(a) + exp_squared(b) + exp_squared(c); }
// clang-format on

ModelScalar
model_reduce_scal(const ModelMesh& mesh, const ReductionType& rtype, const VertexBufferHandle& a)
{
    ReduceInitialScalFunc reduce_initial;
    ReduceFunc reduce;

    bool solve_mean = false;

    switch (rtype) {
    case RTYPE_MAX:
        reduce_initial = length;
        reduce         = max;
        break;
    case RTYPE_MIN:
        reduce_initial = length;
        reduce         = min;
        break;
    case RTYPE_RMS:
        reduce_initial = squared;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_RMS_EXP:
        reduce_initial = exp_squared;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_SUM:
        reduce_initial = length;
        reduce         = sum;
        break;
    default:
        ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = acVertexBufferIdx(mesh.info.int_params[AC_nx_min],
                                              mesh.info.int_params[AC_ny_min],
                                              mesh.info.int_params[AC_nz_min], mesh.info);

    ModelScalar res;
    if (rtype == RTYPE_MAX || rtype == RTYPE_MIN)
        res = reduce_initial(mesh.vertex_buffer[a][initial_idx]);
    else
        res = 0;

    for (int k = mesh.info.int_params[AC_nz_min]; k < mesh.info.int_params[AC_nz_max]; ++k) {
        for (int j = mesh.info.int_params[AC_ny_min]; j < mesh.info.int_params[AC_ny_max]; ++j) {
            for (int i = mesh.info.int_params[AC_nx_min]; i < mesh.info.int_params[AC_nx_max];
                 ++i) {
                const int idx              = acVertexBufferIdx(i, j, k, mesh.info);
                const ModelScalar curr_val = reduce_initial(mesh.vertex_buffer[a][idx]);
                res                        = reduce(res, curr_val);
            }
        }
    }

    if (solve_mean) {
        const ModelScalar inv_n = 1.0l / mesh.info.int_params[AC_nxyz];
        return sqrtl(inv_n * res);
    }
    else {
        return res;
    }
}

ModelScalar
model_reduce_vec(const ModelMesh& mesh, const ReductionType& rtype, const VertexBufferHandle& a,
                 const VertexBufferHandle& b, const VertexBufferHandle& c)
{
    // ModelScalar (*reduce_initial)(ModelScalar, ModelScalar, ModelScalar);
    ReduceInitialVecFunc reduce_initial;
    ReduceFunc reduce;

    bool solve_mean = false;

    switch (rtype) {
    case RTYPE_MAX:
        reduce_initial = length;
        reduce         = max;
        break;
    case RTYPE_MIN:
        reduce_initial = length;
        reduce         = min;
        break;
    case RTYPE_RMS:
        reduce_initial = squared;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_RMS_EXP:
        reduce_initial = exp_squared;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_SUM:
        reduce_initial = length;
        reduce         = sum;
        break;
    default:
        ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = acVertexBufferIdx(mesh.info.int_params[AC_nx_min],
                                              mesh.info.int_params[AC_ny_min],
                                              mesh.info.int_params[AC_nz_min], mesh.info);

    ModelScalar res;
    if (rtype == RTYPE_MAX || rtype == RTYPE_MIN)
        res = reduce_initial(mesh.vertex_buffer[a][initial_idx], mesh.vertex_buffer[b][initial_idx],
                             mesh.vertex_buffer[c][initial_idx]);
    else
        res = 0;

    for (int k = mesh.info.int_params[AC_nz_min]; k < mesh.info.int_params[AC_nz_max]; k++) {
        for (int j = mesh.info.int_params[AC_ny_min]; j < mesh.info.int_params[AC_ny_max]; j++) {
            for (int i = mesh.info.int_params[AC_nx_min]; i < mesh.info.int_params[AC_nx_max];
                 i++) {
                const int idx              = acVertexBufferIdx(i, j, k, mesh.info);
                const ModelScalar curr_val = reduce_initial(mesh.vertex_buffer[a][idx],
                                                            mesh.vertex_buffer[b][idx],
                                                            mesh.vertex_buffer[c][idx]);
                res                        = reduce(res, curr_val);
            }
        }
    }

    if (solve_mean) {
        const ModelScalar inv_n = 1.0l / mesh.info.int_params[AC_nxyz];
        return sqrtl(inv_n * res);
    }
    else {
        return res;
    }
}
