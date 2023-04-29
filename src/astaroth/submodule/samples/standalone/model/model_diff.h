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
#pragma once
#include "errchk.h"

typedef long double MODEL_REAL;

typedef enum { AXIS_X, AXIS_Y, AXIS_Z, NUM_AXIS_TYPES } AxisType;

template <AxisType axis>
static inline MODEL_REAL
der_scal(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
         const MODEL_REAL* scal)
{
    MODEL_REAL f0, f1, f2, f4, f5, f6;
    MODEL_REAL ds;

    switch (axis) {
    case AXIS_X:
        f0 = scal[acVertexBufferIdx(i - 3, j, k, mesh_info)];
        f1 = scal[acVertexBufferIdx(i - 2, j, k, mesh_info)];
        f2 = scal[acVertexBufferIdx(i - 1, j, k, mesh_info)];
        f4 = scal[acVertexBufferIdx(i + 1, j, k, mesh_info)];
        f5 = scal[acVertexBufferIdx(i + 2, j, k, mesh_info)];
        f6 = scal[acVertexBufferIdx(i + 3, j, k, mesh_info)];
        ds = mesh_info.real_params[AC_dsx];
        break;
    case AXIS_Y:
        f0 = scal[acVertexBufferIdx(i, j - 3, k, mesh_info)];
        f1 = scal[acVertexBufferIdx(i, j - 2, k, mesh_info)];
        f2 = scal[acVertexBufferIdx(i, j - 1, k, mesh_info)];
        f4 = scal[acVertexBufferIdx(i, j + 1, k, mesh_info)];
        f5 = scal[acVertexBufferIdx(i, j + 2, k, mesh_info)];
        f6 = scal[acVertexBufferIdx(i, j + 3, k, mesh_info)];
        ds = mesh_info.real_params[AC_dsy];
        break;
    case AXIS_Z:
        f0 = scal[acVertexBufferIdx(i, j, k - 3, mesh_info)];
        f1 = scal[acVertexBufferIdx(i, j, k - 2, mesh_info)];
        f2 = scal[acVertexBufferIdx(i, j, k - 1, mesh_info)];
        f4 = scal[acVertexBufferIdx(i, j, k + 1, mesh_info)];
        f5 = scal[acVertexBufferIdx(i, j, k + 2, mesh_info)];
        f6 = scal[acVertexBufferIdx(i, j, k + 3, mesh_info)];
        ds = mesh_info.real_params[AC_dsz];
        break;
    default:
        ERROR("Unknown axis type");
    }
    return ((f6 - f0) + MODEL_REAL(-9.) * (f5 - f1) + MODEL_REAL(45.) * (f4 - f2)) /
           (MODEL_REAL(60.) * ds);
}

template <AxisType axis>
static inline MODEL_REAL
der2_scal(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
          const MODEL_REAL* scal)
{
    MODEL_REAL f0, f1, f2, f3, f4, f5, f6;
    MODEL_REAL ds;

    f3 = scal[acVertexBufferIdx(i, j, k, mesh_info)];

    switch (axis) {
    case AXIS_X:
        f0 = scal[acVertexBufferIdx(i - 3, j, k, mesh_info)];
        f1 = scal[acVertexBufferIdx(i - 2, j, k, mesh_info)];
        f2 = scal[acVertexBufferIdx(i - 1, j, k, mesh_info)];
        f4 = scal[acVertexBufferIdx(i + 1, j, k, mesh_info)];
        f5 = scal[acVertexBufferIdx(i + 2, j, k, mesh_info)];
        f6 = scal[acVertexBufferIdx(i + 3, j, k, mesh_info)];
        ds = mesh_info.real_params[AC_dsx];
        break;
    case AXIS_Y:
        f0 = scal[acVertexBufferIdx(i, j - 3, k, mesh_info)];
        f1 = scal[acVertexBufferIdx(i, j - 2, k, mesh_info)];
        f2 = scal[acVertexBufferIdx(i, j - 1, k, mesh_info)];
        f4 = scal[acVertexBufferIdx(i, j + 1, k, mesh_info)];
        f5 = scal[acVertexBufferIdx(i, j + 2, k, mesh_info)];
        f6 = scal[acVertexBufferIdx(i, j + 3, k, mesh_info)];
        ds = mesh_info.real_params[AC_dsy];
        break;
    case AXIS_Z:
        f0 = scal[acVertexBufferIdx(i, j, k - 3, mesh_info)];
        f1 = scal[acVertexBufferIdx(i, j, k - 2, mesh_info)];
        f2 = scal[acVertexBufferIdx(i, j, k - 1, mesh_info)];
        f4 = scal[acVertexBufferIdx(i, j, k + 1, mesh_info)];
        f5 = scal[acVertexBufferIdx(i, j, k + 2, mesh_info)];
        f6 = scal[acVertexBufferIdx(i, j, k + 3, mesh_info)];
        ds = mesh_info.real_params[AC_dsz];
        break;
    default:
        ERROR("Unknown axis type");
    }
    return (MODEL_REAL(2.) * (f0 + f6) + MODEL_REAL(-27.) * (f1 + f5) +
            MODEL_REAL(270.) * (f2 + f4) + MODEL_REAL(-490.) * f3) /
           (MODEL_REAL(180.) * ds * ds);
}

static MODEL_REAL
laplace_scal(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
             const MODEL_REAL* scal)
{
    return der2_scal<AXIS_X>(i, j, k, mesh_info, scal) +
           der2_scal<AXIS_Y>(i, j, k, mesh_info, scal) +
           der2_scal<AXIS_Z>(i, j, k, mesh_info, scal);
}

static void
laplace_vec(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
            const MODEL_REAL* vec_x, const MODEL_REAL* vec_y, const MODEL_REAL* vec_z,
            MODEL_REAL* laplace_x, MODEL_REAL* laplace_y, MODEL_REAL* laplace_z)
{
    *laplace_x = laplace_scal(i, j, k, mesh_info, vec_x);
    *laplace_y = laplace_scal(i, j, k, mesh_info, vec_y);
    *laplace_z = laplace_scal(i, j, k, mesh_info, vec_z);
}

static MODEL_REAL
div_vec(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
        const MODEL_REAL* vec_x, const MODEL_REAL* vec_y, const MODEL_REAL* vec_z)
{
    return der_scal<AXIS_X>(i, j, k, mesh_info, vec_x) +
           der_scal<AXIS_Y>(i, j, k, mesh_info, vec_y) +
           der_scal<AXIS_Z>(i, j, k, mesh_info, vec_z);
}

static void
grad(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info, const MODEL_REAL* scal,
     MODEL_REAL* res_x, MODEL_REAL* res_y, MODEL_REAL* res_z)
{
    *res_x = der_scal<AXIS_X>(i, j, k, mesh_info, scal);
    *res_y = der_scal<AXIS_Y>(i, j, k, mesh_info, scal);
    *res_z = der_scal<AXIS_Z>(i, j, k, mesh_info, scal);
}

static MODEL_REAL
vec_dot_nabla_scal(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
                   const MODEL_REAL* vec_x, const MODEL_REAL* vec_y, const MODEL_REAL* vec_z,
                   const MODEL_REAL* scal)
{
    const int idx = acVertexBufferIdx(i, j, k, mesh_info);
    MODEL_REAL ddx_scal, ddy_scal, ddz_scal;
    grad(i, j, k, mesh_info, scal, &ddx_scal, &ddy_scal, &ddz_scal);
    return vec_x[idx] * ddx_scal + vec_y[idx] * ddy_scal + vec_z[idx] * ddz_scal;
}

/*
 * =============================================================================
 * Viscosity
 * =============================================================================
 */
typedef enum { DERNM_XY, DERNM_YZ, DERNM_XZ } DernmType;

template <DernmType dernm>
static MODEL_REAL
dernm_scal(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
           const MODEL_REAL* scal)
{

    MODEL_REAL fac;

    const MODEL_REAL dsx = mesh_info.real_params[AC_dsx];
    const MODEL_REAL dsy = mesh_info.real_params[AC_dsy];
    const MODEL_REAL dsz = mesh_info.real_params[AC_dsz];

    MODEL_REAL f_p1_p1, f_m1_p1, f_m1_m1, f_p1_m1;
    MODEL_REAL f_p2_p2, f_m2_p2, f_m2_m2, f_p2_m2;
    MODEL_REAL f_p3_p3, f_m3_p3, f_m3_m3, f_p3_m3;

    switch (dernm) {
    case DERNM_XY:
        fac     = MODEL_REAL(1. / 720.) * (MODEL_REAL(1.) / dsx) * (MODEL_REAL(1.) / dsy);
        f_p1_p1 = scal[acVertexBufferIdx(i + 1, j + 1, k, mesh_info)];
        f_m1_p1 = scal[acVertexBufferIdx(i - 1, j + 1, k, mesh_info)];
        f_m1_m1 = scal[acVertexBufferIdx(i - 1, j - 1, k, mesh_info)];
        f_p1_m1 = scal[acVertexBufferIdx(i + 1, j - 1, k, mesh_info)];

        f_p2_p2 = scal[acVertexBufferIdx(i + 2, j + 2, k, mesh_info)];
        f_m2_p2 = scal[acVertexBufferIdx(i - 2, j + 2, k, mesh_info)];
        f_m2_m2 = scal[acVertexBufferIdx(i - 2, j - 2, k, mesh_info)];
        f_p2_m2 = scal[acVertexBufferIdx(i + 2, j - 2, k, mesh_info)];

        f_p3_p3 = scal[acVertexBufferIdx(i + 3, j + 3, k, mesh_info)];
        f_m3_p3 = scal[acVertexBufferIdx(i - 3, j + 3, k, mesh_info)];
        f_m3_m3 = scal[acVertexBufferIdx(i - 3, j - 3, k, mesh_info)];
        f_p3_m3 = scal[acVertexBufferIdx(i + 3, j - 3, k, mesh_info)];
        break;
    case DERNM_YZ:
        // NOTE this is a bit different from the old one, second is j+1k-1
        // instead of j-1,k+1
        fac     = MODEL_REAL(1. / 720.) * (MODEL_REAL(1.) / dsy) * (MODEL_REAL(1.) / dsz);
        f_p1_p1 = scal[acVertexBufferIdx(i, j + 1, k + 1, mesh_info)];
        f_m1_p1 = scal[acVertexBufferIdx(i, j - 1, k + 1, mesh_info)];
        f_m1_m1 = scal[acVertexBufferIdx(i, j - 1, k - 1, mesh_info)];
        f_p1_m1 = scal[acVertexBufferIdx(i, j + 1, k - 1, mesh_info)];

        f_p2_p2 = scal[acVertexBufferIdx(i, j + 2, k + 2, mesh_info)];
        f_m2_p2 = scal[acVertexBufferIdx(i, j - 2, k + 2, mesh_info)];
        f_m2_m2 = scal[acVertexBufferIdx(i, j - 2, k - 2, mesh_info)];
        f_p2_m2 = scal[acVertexBufferIdx(i, j + 2, k - 2, mesh_info)];

        f_p3_p3 = scal[acVertexBufferIdx(i, j + 3, k + 3, mesh_info)];
        f_m3_p3 = scal[acVertexBufferIdx(i, j - 3, k + 3, mesh_info)];
        f_m3_m3 = scal[acVertexBufferIdx(i, j - 3, k - 3, mesh_info)];
        f_p3_m3 = scal[acVertexBufferIdx(i, j + 3, k - 3, mesh_info)];
        break;
    case DERNM_XZ:
        fac     = MODEL_REAL(1. / 720.) * (MODEL_REAL(1.) / dsx) * (MODEL_REAL(1.) / dsz);
        f_p1_p1 = scal[acVertexBufferIdx(i + 1, j, k + 1, mesh_info)];
        f_m1_p1 = scal[acVertexBufferIdx(i - 1, j, k + 1, mesh_info)];
        f_m1_m1 = scal[acVertexBufferIdx(i - 1, j, k - 1, mesh_info)];
        f_p1_m1 = scal[acVertexBufferIdx(i + 1, j, k - 1, mesh_info)];

        f_p2_p2 = scal[acVertexBufferIdx(i + 2, j, k + 2, mesh_info)];
        f_m2_p2 = scal[acVertexBufferIdx(i - 2, j, k + 2, mesh_info)];
        f_m2_m2 = scal[acVertexBufferIdx(i - 2, j, k - 2, mesh_info)];
        f_p2_m2 = scal[acVertexBufferIdx(i + 2, j, k - 2, mesh_info)];

        f_p3_p3 = scal[acVertexBufferIdx(i + 3, j, k + 3, mesh_info)];
        f_m3_p3 = scal[acVertexBufferIdx(i - 3, j, k + 3, mesh_info)];
        f_m3_m3 = scal[acVertexBufferIdx(i - 3, j, k - 3, mesh_info)];
        f_p3_m3 = scal[acVertexBufferIdx(i + 3, j, k - 3, mesh_info)];
        break;
    default:
        ERROR("Invalid dernm type");
    }
    return fac * (MODEL_REAL(270.) * (f_p1_p1 - f_m1_p1 + f_m1_m1 - f_p1_m1) -
                  MODEL_REAL(27.) * (f_p2_p2 - f_m2_p2 + f_m2_m2 - f_p2_m2) +
                  MODEL_REAL(2.) * (f_p3_p3 - f_m3_p3 + f_m3_m3 - f_p3_m3));
}

static void
grad_div_vec(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
             const MODEL_REAL* vec_x, const MODEL_REAL* vec_y, const MODEL_REAL* vec_z,
             MODEL_REAL* gdvx, MODEL_REAL* gdvy, MODEL_REAL* gdvz)
{
    *gdvx = der2_scal<AXIS_X>(i, j, k, mesh_info, vec_x) +
            dernm_scal<DERNM_XY>(i, j, k, mesh_info, vec_y) +
            dernm_scal<DERNM_XZ>(i, j, k, mesh_info, vec_z);

    *gdvy = dernm_scal<DERNM_XY>(i, j, k, mesh_info, vec_x) +
            der2_scal<AXIS_Y>(i, j, k, mesh_info, vec_y) +
            dernm_scal<DERNM_YZ>(i, j, k, mesh_info, vec_z);

    *gdvz = dernm_scal<DERNM_XZ>(i, j, k, mesh_info, vec_x) +
            dernm_scal<DERNM_YZ>(i, j, k, mesh_info, vec_y) +
            der2_scal<AXIS_Z>(i, j, k, mesh_info, vec_z);
}

static void
S_grad_lnrho(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
             const MODEL_REAL* vec_x, const MODEL_REAL* vec_y, const MODEL_REAL* vec_z,
             const MODEL_REAL* lnrho, MODEL_REAL* sgrhox, MODEL_REAL* sgrhoy, MODEL_REAL* sgrhoz)
{
    const MODEL_REAL c23 = MODEL_REAL(2. / 3.);
    const MODEL_REAL c13 = MODEL_REAL(1. / 3.);

    const MODEL_REAL Sxx = c23 * der_scal<AXIS_X>(i, j, k, mesh_info, vec_x) -
                           c13 * (der_scal<AXIS_Y>(i, j, k, mesh_info, vec_y) +
                                  der_scal<AXIS_Z>(i, j, k, mesh_info, vec_z));
    const MODEL_REAL Sxy = MODEL_REAL(.5) * (der_scal<AXIS_Y>(i, j, k, mesh_info, vec_x) +
                                             der_scal<AXIS_X>(i, j, k, mesh_info, vec_y));
    const MODEL_REAL Sxz = MODEL_REAL(.5) * (der_scal<AXIS_Z>(i, j, k, mesh_info, vec_x) +
                                             der_scal<AXIS_X>(i, j, k, mesh_info, vec_z));

    const MODEL_REAL Syx = Sxy;
    const MODEL_REAL Syy = c23 * der_scal<AXIS_Y>(i, j, k, mesh_info, vec_y) -
                           c13 * (der_scal<AXIS_X>(i, j, k, mesh_info, vec_x) +
                                  der_scal<AXIS_Z>(i, j, k, mesh_info, vec_z));
    const MODEL_REAL Syz = MODEL_REAL(.5) * (der_scal<AXIS_Z>(i, j, k, mesh_info, vec_y) +
                                             der_scal<AXIS_Y>(i, j, k, mesh_info, vec_z));

    const MODEL_REAL Szx = Sxz;
    const MODEL_REAL Szy = Syz;
    const MODEL_REAL Szz = c23 * der_scal<AXIS_Z>(i, j, k, mesh_info,
                                                  vec_z) // replaced from "c23*der_scal<AXIS_Z>(i,
                                                         // j, k, mesh_info, vec_x)"! TODO recheck
                                                         // that ddz_uu_z is the correct one
                           - c13 * (der_scal<AXIS_X>(i, j, k, mesh_info, vec_x) +
                                    der_scal<AXIS_Y>(i, j, k, mesh_info, vec_y));

    // Grad lnrho

    MODEL_REAL glnx, glny, glnz;

    grad(i, j, k, mesh_info, lnrho, &glnx, &glny, &glnz);

    *sgrhox = Sxx * glnx + Sxy * glny + Sxz * glnz;
    *sgrhoy = Syx * glnx + Syy * glny + Syz * glnz;
    *sgrhoz = Szx * glnx + Szy * glny + Szz * glnz;
}

static void
nu_const(const int& i, const int& j, const int& k, const AcMeshInfo& mesh_info,
         const MODEL_REAL* vec_x, const MODEL_REAL* vec_y, const MODEL_REAL* vec_z,
         const MODEL_REAL* scal, MODEL_REAL* visc_x, MODEL_REAL* visc_y, MODEL_REAL* visc_z)
{
    MODEL_REAL lx, ly, lz;
    laplace_vec(i, j, k, mesh_info, vec_x, vec_y, vec_z, &lx, &ly, &lz);
    // lx = ly = lz = .0f;

    MODEL_REAL gx, gy, gz;
    grad_div_vec(i, j, k, mesh_info, vec_x, vec_y, vec_z, &gx, &gy, &gz);
    // gx = gy =gz = .0f;

    MODEL_REAL sgrhox, sgrhoy, sgrhoz;
    S_grad_lnrho(i, j, k, mesh_info, vec_x, vec_y, vec_z, scal, &sgrhox, &sgrhoy, &sgrhoz);
    // sgrhox = sgrhoy = sgrhoz = .0f;

    *visc_x = mesh_info.real_params[AC_nu_visc] *
                  (lx + MODEL_REAL(1. / 3.) * gx + MODEL_REAL(2.) * sgrhox) +
              mesh_info.real_params[AC_zeta] * gx;
    *visc_y = mesh_info.real_params[AC_nu_visc] *
                  (ly + MODEL_REAL(1. / 3.) * gy + MODEL_REAL(2.) * sgrhoy) +
              mesh_info.real_params[AC_zeta] * gy;
    *visc_z = mesh_info.real_params[AC_nu_visc] *
                  (lz + MODEL_REAL(1. / 3.) * gz + MODEL_REAL(2.) * sgrhoz) +
              mesh_info.real_params[AC_zeta] * gz;
}
