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

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "astaroth_utils.h"

#include <math.h>
#include <stdbool.h>

#include "errchk.h"
#include "memory.h" // acHostMeshCreate, acHostMeshDestroy, acHostMeshApplyPeriodicBounds

#ifdef AC_INTEGRATION_ENABLED
/*
// Standalone flags (currently defined in the DSL)
#define LDENSITY (1)
#define LHYDRO (1)
#define LMAGNETIC (1)
#define LENTROPY (1)
#define LTEMPERATURE (0)
#define LFORCING (0)
#define LUPWD (0)
#define AC_THERMAL_CONDUCTIVITY ((Scalar)(0.001)) // TODO: make an actual config parameter
#define R_PI ((Scalar)M_PI)
*/

/*
typedef AcReal Scalar;
// typedef AcReal3 Vector;
// typedef AcMatrix Matrix;

#if AC_DOUBLE_PRECISION == 1
typedef double Vector __attribute__((vector_size(4 * sizeof(double))));
#else
typedef float Vector __attribute__((vector_size(4 * sizeof(float))));

#define fabs fabsf
#define exp expf
#define sqrt sqrtf
#define cos cosf
#define sin sinf
#endif
*/
typedef long double Scalar;
typedef struct {
    Scalar x, y, z;
} Vector;

typedef struct {
    Vector row[3];
} Matrix;

#define fabs fabsl
#define exp expl
#define sqrt sqrtl
#define cos cosl
#define sin sinl

#define SCALAR_PI (M_PIl) // Long double variant

static Vector
operator-(const Vector& a)
{
    return (Vector){-a.x, -a.y, -a.z};
}

static Vector
operator+(const Vector& a, const Vector& b)
{
    return (Vector){a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vector
operator-(const Vector& a, const Vector& b)
{
    return (Vector){a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vector
operator*(const Scalar& a, const Vector& b)
{
    return (Vector){a * b.x, a * b.y, a * b.z};
}

static AcMeshInfo* mesh_info = NULL;

static inline int
getInt(const AcIntParam param)
{
    return mesh_info->int_params[param];
}

static inline Scalar
getReal(const AcRealParam param)
{
    return (Scalar)mesh_info->real_params[param];
}

static inline int
IDX(const int i, const int j, const int k)
{
    return acVertexBufferIdx(i, j, k, (*mesh_info));
}

typedef struct {
    Scalar value;
    Vector gradient;
    Matrix hessian;
#if LUPWD
    Vector upwind;
#endif
} ScalarData;

typedef struct {
    ScalarData xdata;
    ScalarData ydata;
    ScalarData zdata;
} VectorData;

static inline Scalar
first_derivative(const Scalar* pencil, const Scalar inv_ds)
{
#if STENCIL_ORDER == 2
    const Scalar coefficients[] = {0, (Scalar)(1. / 2.)};
#elif STENCIL_ORDER == 4
    const Scalar coefficients[] = {0, (Scalar)(2.0 / 3.0), (Scalar)(-1.0 / 12.0)};
#elif STENCIL_ORDER == 6
    const Scalar coefficients[] = {
        0,
        (Scalar)3.0 / (Scalar)4.0,
        (Scalar)-3.0 / (Scalar)20.0,
        (Scalar)1.0 / (Scalar)60.0,
    };
#elif STENCIL_ORDER == 8
    const Scalar coefficients[] = {
        0, (Scalar)(4.0 / 5.0), (Scalar)(-1.0 / 5.0), (Scalar)(4.0 / 105.0), (Scalar)(-1.0 / 280.0),
    };
#endif

#define MID (STENCIL_ORDER / 2)
    Scalar res = 0;

    // #pragma unroll
    for (int i = 1; i <= MID; ++i)
        // for (int i = MID; i >= 1; --i)
        res += coefficients[i] * (pencil[MID + i] - pencil[MID - i]);

    return res * inv_ds;
}

static inline Scalar
second_derivative(const Scalar* pencil, const Scalar inv_ds)
{
#if STENCIL_ORDER == 2
    const Scalar coefficients[] = {-2, 1};
#elif STENCIL_ORDER == 4
    const Scalar coefficients[] = {
        (Scalar)(-5.0 / 2.0),
        (Scalar)(4.0 / 3.0),
        (Scalar)(-1.0 / 12.0),
    };
#elif STENCIL_ORDER == 6
    const Scalar coefficients[] = {
        (Scalar)-49.0 / (Scalar)18.0,
        (Scalar)3.0 / (Scalar)2.0,
        (Scalar)-3.0 / (Scalar)20.0,
        (Scalar)1.0 / (Scalar)90.0,
    };
#elif STENCIL_ORDER == 8
    const Scalar coefficients[] = {
        (Scalar)(-205.0 / 72.0), (Scalar)(8.0 / 5.0),    (Scalar)(-1.0 / 5.0),
        (Scalar)(8.0 / 315.0),   (Scalar)(-1.0 / 560.0),
    };
#endif

#define MID (STENCIL_ORDER / 2)
    Scalar res = coefficients[0] * pencil[MID];

    // #pragma unroll
    for (int i = 1; i <= MID; ++i)
        // for (int i = MID; i >= 1; --i)
        res += coefficients[i] * (pencil[MID + i] + pencil[MID - i]);

    return res * inv_ds * inv_ds;
}

/** inv_ds: inverted mesh spacing f.ex. 1. / mesh.int_params[AC_dsx] */
static inline Scalar
cross_derivative(const Scalar* pencil_a, const Scalar* pencil_b, const Scalar inv_ds_a,
                 const Scalar inv_ds_b)
{
#if STENCIL_ORDER == 2
    const Scalar coefficients[] = {0, (Scalar)(1.0 / 4.0)};
#elif STENCIL_ORDER == 4
    const Scalar coefficients[] = {
        (Scalar)0.,
        0,
        0,
    }; // TODO correct coefficients, these are just placeholders
#elif STENCIL_ORDER == 6
    const Scalar fac            = (Scalar)1. / (Scalar)720.;
    const Scalar coefficients[] = {
        0 * fac,
        (Scalar)(270.0) * fac,
        (Scalar)(-27.0) * fac,
        (Scalar)(2.0) * fac,
    };
#elif STENCIL_ORDER == 8
    const Scalar fac            = ((Scalar)(1. / 20160.));
    const Scalar coefficients[] = {
        0 * fac,
        (Scalar)(8064.) * fac,
        (Scalar)(-1008.) * fac,
        (Scalar)(128.) * fac,
        (Scalar)(-9.) * fac,
    };
#endif

#define MID (STENCIL_ORDER / 2)
    Scalar res = (Scalar)(0.);

    // #pragma unroll
    for (int i = 1; i <= MID; ++i) {
        // for (int i = MID; i >= 1; --i) {
        res += coefficients[i] *
               (pencil_a[MID + i] + pencil_a[MID - i] - pencil_b[MID + i] - pencil_b[MID - i]);
    }
    return res * inv_ds_a * inv_ds_b;
}

static inline Scalar
derx(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i + offset - STENCIL_ORDER / 2, j, k)];

    return first_derivative(pencil, ((Scalar)1. / getReal(AC_dsx)));
}

static inline Scalar
derxx(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i + offset - STENCIL_ORDER / 2, j, k)];

    return second_derivative(pencil, ((Scalar)1. / getReal(AC_dsx)));
}

static inline Scalar
derxy(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil_a[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = (Scalar)arr[IDX(i + offset - STENCIL_ORDER / 2, //
                                           j + offset - STENCIL_ORDER / 2, k)];

    Scalar pencil_b[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = (Scalar)arr[IDX(i + offset - STENCIL_ORDER / 2, //
                                           j + STENCIL_ORDER / 2 - offset, k)];

    return cross_derivative(pencil_a, pencil_b, ((Scalar)1. / getReal(AC_dsx)),
                            ((Scalar)1. / getReal(AC_dsy)));
}

static inline Scalar
derxz(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil_a[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = (Scalar)
            arr[IDX(i + offset - STENCIL_ORDER / 2, j, k + offset - STENCIL_ORDER / 2)];

    Scalar pencil_b[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = (Scalar)
            arr[IDX(i + offset - STENCIL_ORDER / 2, j, k + STENCIL_ORDER / 2 - offset)];

    return cross_derivative(pencil_a, pencil_b, ((Scalar)1. / getReal(AC_dsx)),
                            ((Scalar)1. / getReal(AC_dsz)));
}

static inline Scalar
dery(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i, j + offset - STENCIL_ORDER / 2, k)];

    return first_derivative(pencil, ((Scalar)1. / getReal(AC_dsy)));
}

static inline Scalar
deryy(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i, j + offset - STENCIL_ORDER / 2, k)];

    return second_derivative(pencil, ((Scalar)1. / getReal(AC_dsy)));
}

static inline Scalar
deryz(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil_a[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = (Scalar)
            arr[IDX(i, j + offset - STENCIL_ORDER / 2, k + offset - STENCIL_ORDER / 2)];

    Scalar pencil_b[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = (Scalar)
            arr[IDX(i, j + offset - STENCIL_ORDER / 2, k + STENCIL_ORDER / 2 - offset)];

    return cross_derivative(pencil_a, pencil_b, ((Scalar)1. / getReal(AC_dsy)),
                            ((Scalar)1. / getReal(AC_dsz)));
}

static inline Scalar
derz(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i, j, k + offset - STENCIL_ORDER / 2)];

    return first_derivative(pencil, ((Scalar)1. / getReal(AC_dsz)));
}

static inline Scalar
derzz(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i, j, k + offset - STENCIL_ORDER / 2)];

    return second_derivative(pencil, ((Scalar)1. / getReal(AC_dsz)));
}

#if LUPWD
static inline Scalar
der6x_upwd(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar inv_ds = ((Scalar)1. / getReal(AC_dsx));

    return (Scalar)(1.0 / 60.0) * inv_ds *
           (-(Scalar)(20.0) * (Scalar)arr[IDX(i, j, k)] +
            (Scalar)(15.0) * ((Scalar)arr[IDX(i + 1, j, k)] + (Scalar)arr[IDX(i - 1, j, k)]) -
            (Scalar)(6.0) * ((Scalar)arr[IDX(i + 2, j, k)] + (Scalar)arr[IDX(i - 2, j, k)]) +
            (Scalar)arr[IDX(i + 3, j, k)] + (Scalar)arr[IDX(i - 3, j, k)]);
}

static inline Scalar
der6y_upwd(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar inv_ds = ((Scalar)1. / getReal(AC_dsy));

    return (Scalar)(1.0 / 60.0) * inv_ds *
           (-(Scalar)(20.0) * (Scalar)arr[IDX(i, j, k)] +
            (Scalar)(15.0) * ((Scalar)arr[IDX(i, j + 1, k)] + (Scalar)arr[IDX(i, j - 1, k)]) -
            (Scalar)(6.0) * ((Scalar)arr[IDX(i, j + 2, k)] + (Scalar)arr[IDX(i, j - 2, k)]) +
            (Scalar)arr[IDX(i, j + 3, k)] + (Scalar)arr[IDX(i, j - 3, k)]);
}

static inline Scalar
der6z_upwd(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar inv_ds = ((Scalar)1. / getReal(AC_dsz));

    return (Scalar)(1.0 / 60.0) * inv_ds *
           (-(Scalar)(20.0) * (Scalar)arr[IDX(i, j, k)] +
            (Scalar)(15.0) * ((Scalar)arr[IDX(i, j, k + 1)] + (Scalar)arr[IDX(i, j, k - 1)]) -
            (Scalar)(6.0) * ((Scalar)arr[IDX(i, j, k + 2)] + (Scalar)arr[IDX(i, j, k - 2)]) +
            (Scalar)arr[IDX(i, j, k + 3)] + (Scalar)arr[IDX(i, j, k - 3)]);
}
#endif

static inline Scalar
compute_value(const int i, const int j, const int k, const AcReal* arr)
{
    return (Scalar)arr[IDX(i, j, k)];
}

static inline Vector
compute_gradient(const int i, const int j, const int k, const AcReal* arr)
{
    return (Vector){derx(i, j, k, arr), dery(i, j, k, arr), derz(i, j, k, arr)};
}

#if LUPWD
static inline Vector
compute_upwind(const int i, const int j, const int k, const AcReal* arr)
{
    return (Vector){der6x_upwd(i, j, k, arr), der6y_upwd(i, j, k, arr), der6z_upwd(i, j, k, arr)};
}
#endif

static inline Matrix
compute_hessian(const int i, const int j, const int k, const AcReal* arr)
{
    Matrix hessian;

    hessian.row[0] = (Vector){derxx(i, j, k, arr), derxy(i, j, k, arr), derxz(i, j, k, arr)};
    hessian.row[1] = (Vector){hessian.row[0].y, deryy(i, j, k, arr), deryz(i, j, k, arr)};
    hessian.row[2] = (Vector){hessian.row[0].z, hessian.row[1].z, derzz(i, j, k, arr)};

    return hessian;
}

static inline ScalarData
read_scal_data(const int i, const int j, const int k, AcReal* buf[NUM_VTXBUF_HANDLES],
               const int handle)
{
    ScalarData data;

    data.value    = compute_value(i, j, k, buf[handle]);
    data.gradient = compute_gradient(i, j, k, buf[handle]);

    // No significant effect on performance even though we do not need the
    // diagonals with all arrays
    data.hessian = compute_hessian(i, j, k, buf[handle]);

#if LUPWD
    data.upwind = compute_upwind(i, j, k, buf[handle]);
#endif

    return data;
}

static inline VectorData
read_vec_data(const int i, const int j, const int k, AcReal* buf[NUM_VTXBUF_HANDLES],
              const int3 handle)
{
    VectorData data;

    data.xdata = read_scal_data(i, j, k, buf, handle.x);
    data.ydata = read_scal_data(i, j, k, buf, handle.y);
    data.zdata = read_scal_data(i, j, k, buf, handle.z);

    return data;
}

static inline Scalar
value(const ScalarData data)
{
    return data.value;
}

static inline Vector
gradient(const ScalarData data)
{
    return data.gradient;
}

static inline Matrix
hessian(const ScalarData data)
{
    return data.hessian;
}

static inline Vector
vecvalue(const VectorData data)
{
    return (Vector){value(data.xdata), value(data.ydata), value(data.zdata)};
}

static inline Vector
vecvalue_abs(const VectorData data)
{
    return (Vector){
        fabs(value(data.xdata)),
        fabs(value(data.ydata)),
        fabs(value(data.zdata)),
    };
}

static inline Matrix
gradients(const VectorData data)
{
    Matrix mat;
    mat.row[0] = gradient(data.xdata);
    mat.row[1] = gradient(data.ydata);
    mat.row[2] = gradient(data.zdata);
    return mat;
}

/*
 * =============================================================================
 * Level 0.3 (Built-in functions available during the Stencil Processing Stage)
 * =============================================================================
 */
/*
static inline Vector
operator-(const Vector a, const Vector b)
{
    return (Vector){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vector
operator+(const Vector a, const Vector b)
{
    return (Vector){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vector
operator-(const Vector a)
{
    return (Vector){-a.x, -a.y, -a.z};
}

static inline Vector operator*(const Scalar a, const Vector b)
{
    return (Vector){a * b.x, a * b.y, a * b.z};
}
*/

static inline Scalar
dot(const Vector a, const Vector b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vector
mul(const Matrix aa, const Vector x)
{
    return (Vector){dot(aa.row[0], x), dot(aa.row[1], x), dot(aa.row[2], x)};
}

static inline Vector
cross(const Vector a, const Vector b)
{
    Vector c;

    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;
}
/*
static inline bool
is_valid(const Scalar a)
{
    return !isnan(a) && !isinf(a);
}

static inline bool
is_valid(const Vector a)
{
    return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}
*/
/*
 * =============================================================================
 * Stencil Processing Stage (helper functions)
 * =============================================================================
 */
static inline Scalar
laplace(const ScalarData data)
{
    return hessian(data).row[0].x + hessian(data).row[1].y + hessian(data).row[2].z;
}

static inline Scalar
divergence(const VectorData vec)
{
    return gradient(vec.xdata).x + gradient(vec.ydata).y + gradient(vec.zdata).z;
}

static inline Vector
laplace_vec(const VectorData vec)
{
    return (Vector){laplace(vec.xdata), laplace(vec.ydata), laplace(vec.zdata)};
}

static inline Vector
curl(const VectorData vec)
{
    return (Vector){gradient(vec.zdata).y - gradient(vec.ydata).z,
                    gradient(vec.xdata).z - gradient(vec.zdata).x,
                    gradient(vec.ydata).x - gradient(vec.xdata).y};
}

static inline Vector
gradient_of_divergence(const VectorData vec)
{
    return (Vector){
        hessian(vec.xdata).row[0].x + hessian(vec.ydata).row[0].y + hessian(vec.zdata).row[0].z,
        hessian(vec.xdata).row[1].x + hessian(vec.ydata).row[1].y + hessian(vec.zdata).row[1].z,
        hessian(vec.xdata).row[2].x + hessian(vec.ydata).row[2].y + hessian(vec.zdata).row[2].z,
    };
}

// Takes uu gradients and returns S
static inline Matrix
stress_tensor(const VectorData vec)
{
    Matrix S;

    S.row[0].x = (Scalar)(2. / 3.) * gradient(vec.xdata).x -
                 (Scalar)(1. / 3.) * (gradient(vec.ydata).y + gradient(vec.zdata).z);
    S.row[0].y = (Scalar)(1. / 2.) * (gradient(vec.xdata).y + gradient(vec.ydata).x);
    S.row[0].z = (Scalar)(1. / 2.) * (gradient(vec.xdata).z + gradient(vec.zdata).x);

    S.row[1].y = (Scalar)(2. / 3.) * gradient(vec.ydata).y -
                 (Scalar)(1. / 3.) * (gradient(vec.xdata).x + gradient(vec.zdata).z);

    S.row[1].z = (Scalar)(1. / 2.) * (gradient(vec.ydata).z + gradient(vec.zdata).y);

    S.row[2].z = (Scalar)(2. / 3.) * gradient(vec.zdata).z -
                 (Scalar)(1. / 3.) * (gradient(vec.xdata).x + gradient(vec.ydata).y);

    S.row[1].x = S.row[0].y;
    S.row[2].x = S.row[0].z;
    S.row[2].y = S.row[1].z;

    return S;
}

/** Currently used only if LENTROPY (1) */
static inline __attribute__((unused)) Scalar
contract(const Matrix mat)
{
    Scalar res = 0;

    // #pragma unroll
    for (int i = 0; i < 3; ++i)
        res += dot(mat.row[i], mat.row[i]);

    return res;
}

/*
 * =============================================================================
 * Stencil Processing Stage (equations)
 * =============================================================================
 */

#if LUPWD
Vector
gradient_upwd(const ScalarData scal)
{
    return (Vector){
        scal.upwind.x,
        scal.upwind.y,
        scal.upwind.z,
    };
}

Matrix
gradients_upwd(const VectorData vec)
{
    Matrix mat;
    mat.row[0] = gradient_upwd(vec.xdata);
    mat.row[1] = gradient_upwd(vec.ydata);
    mat.row[2] = gradient_upwd(vec.zdata);
    return mat;
}
#endif

static inline Scalar
continuity(const VectorData uu, const ScalarData lnrho)
{
    return -dot(vecvalue(uu), gradient(lnrho))
#if LUPWD
           // This is a corrective hyperdiffusion term for upwinding.
           + dot(vecvalue_abs(uu), gradient_upwd(lnrho))
#endif
           - divergence(uu);
}

__attribute__((unused)) static inline Scalar
length(const Vector vec)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

static inline Scalar
reciprocal_len(const Vector vec)
{
    return (Scalar)(1.0) / sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__attribute__((unused)) static inline Vector
normalized(const Vector vec)
{
    const Scalar inv_len = reciprocal_len(vec);
    return (Vector){inv_len * vec.x, inv_len * vec.y, inv_len * vec.z};
}

#define H_CONST ((Scalar)(0.0))
#define C_CONST ((Scalar)(0.0))

static inline Vector
momentum(const VectorData uu, const ScalarData lnrho
#if LENTROPY
         ,
         const ScalarData ss
#endif
#if LMAGNETIC
         ,
         const VectorData aa
#endif
)
{
    const Matrix S         = stress_tensor(uu);
    const Scalar cs2_sound = getReal(AC_cs_sound) * getReal(AC_cs_sound);
#if LENTROPY
    const Scalar cs2 = cs2_sound *
                       exp(getReal(AC_gamma) * value(ss) / getReal(AC_cp_sound) +
                           (getReal(AC_gamma) - 1) * (value(lnrho) - getReal(AC_lnrho0)));
#else
    const Scalar cs2 = cs2_sound;
#endif

#if LENTROPY
#if LMAGNETIC
    const Vector j = ((Scalar)(1.) / getReal(AC_mu0)) *
                     (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Vector B       = curl(aa);
    const Scalar inv_rho = (Scalar)(1.) / exp(value(lnrho));
#endif
    const Vector mom = -mul(gradients(uu), vecvalue(uu)) -
                       cs2 *
                           (((Scalar)(1.) / getReal(AC_cp_sound)) * gradient(ss) + gradient(lnrho))
#if LUPWD
                       // Note: dangerous implementation, upwd calculation duplicated (w/ and w/o
                       // entropy). Need to modify both if one is modified
                       + mul(gradients_upwd(uu), vecvalue_abs(uu))
#endif
#if LMAGNETIC
                       + inv_rho * cross(j, B)
#endif
                       + getReal(AC_nu_visc) *
                             (laplace_vec(uu) + (Scalar)(1. / 3.) * gradient_of_divergence(uu) +
                              (Scalar)(2.) * mul(S, gradient(lnrho))) +
                       getReal(AC_zeta) * gradient_of_divergence(uu);
    return mom;
#else
    // !!!!!!!!!!!!!!!!%JP: NOTE TODO IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!
    // NOT CHECKED FOR CORRECTNESS: USE AT YOUR OWN RISK
#if LMAGNETIC
    const Vector j = ((Scalar)(1.) / getReal(AC_mu0)) *
                     (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Vector B       = curl(aa);
    const Scalar inv_rho = (Scalar)(1.) / exp(value(lnrho));
#endif
    const Vector mom     = -mul(gradients(uu), vecvalue(uu)) -
                       cs2 * gradient(lnrho)
#if LUPWD
                       // Note: dangerous implementation, upwd calculation duplicated (w/ and w/o
                       // entropy). Need to modify both if one is modified
                       + mul(gradients_upwd(uu), vecvalue_abs(uu))
#endif
#if LMAGNETIC
                       + inv_rho * cross(j, B)
#endif
                       + getReal(AC_nu_visc) *
                             (laplace_vec(uu) + (Scalar)(1. / 3.) * gradient_of_divergence(uu) +
                              (Scalar)(2.) * mul(S, gradient(lnrho))) +
                       getReal(AC_zeta) * gradient_of_divergence(uu);
    return mom;
#endif
}

static inline Vector
induction(const VectorData uu, const VectorData aa)
{
    // Note: We do (-nabla^2 A + nabla(nabla dot A)) instead of (nabla x (nabla
    // x A)) in order to avoid taking the first derivative twice (did the math,
    // yes this actually works. See pg.28 in arXiv:astro-ph/0109497)
    // u cross B - AC_eta * AC_mu0 * (AC_mu0^-1 * [- laplace A + grad div A ])
    const Vector B = curl(aa);
    // MV: Due to gauge freedom we can reduce the gradient of scalar (divergence) from the equation
    // const Vector grad_div = gradient_of_divergence(aa);
    const Vector lap = laplace_vec(aa);

    // Note, AC_mu0 is cancelled out
    // MV: Due to gauge freedom we can reduce the gradient of scalar (divergence) from the equation
    // const Vector ind = cross(value(uu), B) - getReal(AC_eta) * (grad_div - lap);
    const Vector ind = cross(vecvalue(uu), B) + getReal(AC_eta) * lap;

    return ind;
}

#if LENTROPY
static inline Scalar
lnT(const ScalarData ss, const ScalarData lnrho)
{
    const Scalar lnT = getReal(AC_lnT0) + getReal(AC_gamma) * value(ss) / getReal(AC_cp_sound) +
                       (getReal(AC_gamma) - (Scalar)(1.)) * (value(lnrho) - getReal(AC_lnrho0));
    return lnT;
}

// Nabla dot (K nabla T) / (rho T)
static inline Scalar
heat_conduction(const ScalarData ss, const ScalarData lnrho)
{
    const Scalar inv_cp_sound = (Scalar)(1.) / getReal(AC_cp_sound);

    const Vector grad_ln_chi = -gradient(lnrho);

    const Scalar first_term = getReal(AC_gamma) * inv_cp_sound * laplace(ss) +
                              (getReal(AC_gamma) - (Scalar)(1.)) * laplace(lnrho);
    const Vector second_term = getReal(AC_gamma) * inv_cp_sound * gradient(ss) +
                               (getReal(AC_gamma) - (Scalar)(1.)) * gradient(lnrho);
    const Vector third_term = getReal(AC_gamma) * (inv_cp_sound * gradient(ss) + gradient(lnrho)) +
                              grad_ln_chi;

    const Scalar chi = (Scalar)(AC_THERMAL_CONDUCTIVITY) /
                       (exp(value(lnrho)) * getReal(AC_cp_sound));
    return getReal(AC_cp_sound) * chi * (first_term + dot(second_term, third_term));
}

static inline Scalar
#if LMAGNETIC
entropy(const ScalarData ss, const VectorData uu, const ScalarData lnrho, const VectorData aa)
#else
entropy(const ScalarData ss, const VectorData uu, const ScalarData lnrho)
#endif
{
    const Matrix S      = stress_tensor(uu);
    const Scalar inv_pT = (Scalar)(1.) / (exp(value(lnrho)) * exp(lnT(ss, lnrho)));
#if LMAGNETIC
    const Vector j = ((Scalar)(1.) / getReal(AC_mu0)) *
                     (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
#else
    const Vector j                 = (Vector){0.0, 0.0, 0.0};
#endif
    const Scalar RHS = H_CONST - C_CONST + getReal(AC_eta) * getReal(AC_mu0) * dot(j, j) +
                       (Scalar)(2.) * exp(value(lnrho)) * getReal(AC_nu_visc) * contract(S) +
                       getReal(AC_zeta) * exp(value(lnrho)) * divergence(uu) * divergence(uu);

    return -dot(vecvalue(uu), gradient(ss)) + inv_pT * RHS + heat_conduction(ss, lnrho);
    /*
    const Matrix S = stress_tensor(uu);

    // nabla x nabla x A / mu0 = nabla(nabla dot A) - nabla^2(A)
    const Vector j = gradient_of_divergence(aa) - laplace_vec(aa);

    const Scalar inv_pT = (Scalar)(1.) / (exp(value(lnrho)) + exp(lnT(ss, lnrho)));

    return - dot(vecvalue(uu), gradient(ss))
           + inv_pT * ( H_CONST - C_CONST
                + getReal(AC_eta) * getReal(AC_mu0) * dot(j, j)
                + (Scalar)(2.) * exp(value(lnrho)) * getReal(AC_nu_visc) * contract(S)
                + getReal(AC_zeta) * exp(value(lnrho)) * divergence(uu) * divergence(uu)
            )
            + heat_conduction(ss, lnrho);
    */
}
#endif

__attribute__((unused)) static inline bool
is_valid(const Scalar a)
{
    return !isnan(a) && !isinf(a);
}

__attribute__((unused)) static inline bool
is_valid_vec(const Vector a)
{
    return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}

#if LFORCING
Vector
simple_vortex_forcing(Vector a, Vector b, Scalar magnitude)
{
    return magnitude * cross(normalized(b - a), (Vector){0, 0, 1}); // Vortex
}

Vector
simple_outward_flow_forcing(Vector a, Vector b, Scalar magnitude)
{
    return magnitude * (1 / length(b - a)) * normalized(b - a); // Outward flow
}

// The Pencil Code forcing_hel_noshear(), manual Eq. 222, inspired forcing function with adjustable
// helicity
Vector
helical_forcing(Scalar magnitude, Vector k_force, Vector xx, Vector ff_re, Vector ff_im, Scalar phi)
{
    (void)magnitude; // WARNING: unused
    xx.x = xx.x * ((Scalar)2.0 * SCALAR_PI / (getReal(AC_dsx) * getInt(AC_nx)));
    xx.y = xx.y * ((Scalar)2.0 * SCALAR_PI / (getReal(AC_dsy) * getInt(AC_ny)));
    xx.z = xx.z * ((Scalar)2.0 * SCALAR_PI / (getReal(AC_dsz) * getInt(AC_nz)));

    Scalar cos_phi     = cos(phi);
    Scalar sin_phi     = sin(phi);
    Scalar cos_k_dot_x = cos(dot(k_force, xx));
    Scalar sin_k_dot_x = sin(dot(k_force, xx));
    // Phase affect only the x-component
    // Scalar real_comp       = cos_k_dot_x;
    // Scalar imag_comp       = sin_k_dot_x;
    Scalar real_comp_phase = cos_k_dot_x * cos_phi - sin_k_dot_x * sin_phi;
    Scalar imag_comp_phase = cos_k_dot_x * sin_phi + sin_k_dot_x * cos_phi;

    Vector force = (Vector){ff_re.x * real_comp_phase - ff_im.x * imag_comp_phase,
                            ff_re.y * real_comp_phase - ff_im.y * imag_comp_phase,
                            ff_re.z * real_comp_phase - ff_im.z * imag_comp_phase};

    return force;
}

Vector
forcing(int3 globalVertexIdx, Scalar dt)
{
    Vector a = (Scalar)(.5) * (Vector){getInt(AC_nx) * getReal(AC_dsx),
                                       getInt(AC_ny) * getReal(AC_dsy),
                                       getInt(AC_nz) * getReal(AC_dsz)}; // source (origin)
    (void)a;                                                             // WARNING: not used
    Vector xx = (Vector){
        (globalVertexIdx.x - getInt(AC_nx_min)) * getReal(AC_dsx),
        (globalVertexIdx.y - getInt(AC_ny_min)) * getReal(AC_dsy),
        (globalVertexIdx.z - getInt(AC_nz_min)) * getReal(AC_dsz),
    }; // sink (current index)
    const Scalar cs2 = getReal(AC_cs2_sound);
    const Scalar cs  = sqrt(cs2);

    // Placeholders until determined properly
    Scalar magnitude = getReal(AC_forcing_magnitude);
    Scalar phase     = getReal(AC_forcing_phase);
    Vector k_force   = (Vector){getReal(AC_k_forcex), getReal(AC_k_forcey), getReal(AC_k_forcez)};
    Vector ff_re = (Vector){getReal(AC_ff_hel_rex), getReal(AC_ff_hel_rey), getReal(AC_ff_hel_rez)};
    Vector ff_im = (Vector){getReal(AC_ff_hel_imx), getReal(AC_ff_hel_imy), getReal(AC_ff_hel_imz)};

    (void)phase;   // WARNING: unused with simple forcing. Should be defined in helical_forcing
    (void)k_force; // WARNING: unused with simple forcing. Should be defined in helical_forcing
    (void)ff_re;   // WARNING: unused with simple forcing. Should be defined in helical_forcing
    (void)ff_im;   // WARNING: unused with simple forcing. Should be defined in helical_forcing

    // Determine that forcing funtion type at this point.
    // Vector force = simple_vortex_forcing(a, xx, magnitude);
    // Vector force = simple_outward_flow_forcing(a, xx, magnitude);
    Vector force = helical_forcing(magnitude, k_force, xx, ff_re, ff_im, phase);

    // Scaling N = magnitude*cs*sqrt(k*cs/dt)  * dt
    const Scalar NN = cs * sqrt(getReal(AC_kaver) * cs);
    // MV: Like in the Pencil Code. I don't understandf the logic here.
    force.x = sqrt(dt) * NN * force.x;
    force.y = sqrt(dt) * NN * force.y;
    force.z = sqrt(dt) * NN * force.z;

    if (is_valid_vec(force)) {
        return force;
    }
    else {
        return (Vector){0, 0, 0};
    }
}
#endif

static void
solve_alpha_step(AcMesh in, const int step_number, const AcReal dt, const int i, const int j,
                 const int k, AcMesh* out)
{
    const int idx = acVertexBufferIdx(i, j, k, in.info);

    const ScalarData lnrho = read_scal_data(i, j, k, in.vertex_buffer, VTXBUF_LNRHO);
    const VectorData uu    = read_vec_data(i, j, k, in.vertex_buffer,
                                           (int3){VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ});

    Scalar rate_of_change[NUM_VTXBUF_HANDLES] = {0};
    rate_of_change[VTXBUF_LNRHO]              = continuity(uu, lnrho);

#if LMAGNETIC
    const VectorData aa       = read_vec_data(i, j, k, in.vertex_buffer,
                                              (int3){VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ});
    const Vector aa_res       = induction(uu, aa);
    rate_of_change[VTXBUF_AX] = aa_res.x;
    rate_of_change[VTXBUF_AY] = aa_res.y;
    rate_of_change[VTXBUF_AZ] = aa_res.z;
#endif
#if LENTROPY
    const ScalarData ss = read_scal_data(i, j, k, in.vertex_buffer, VTXBUF_ENTROPY);
#if LMAGNETIC
    const Vector uu_res = momentum(uu, lnrho, ss, aa);
#else
    const Vector uu_res            = momentum(uu, lnrho, ss);
#endif
    rate_of_change[VTXBUF_UUX] = uu_res.x;
    rate_of_change[VTXBUF_UUY] = uu_res.y;
    rate_of_change[VTXBUF_UUZ] = uu_res.z;
#if LMAGNETIC
    rate_of_change[VTXBUF_ENTROPY] = entropy(ss, uu, lnrho, aa);
#else
    rate_of_change[VTXBUF_ENTROPY] = entropy(ss, uu, lnrho);
#endif
#else
#if LMAGNETIC
    const Vector uu_res        = momentum(uu, lnrho, aa);
#else
    const Vector uu_res = momentum(uu, lnrho);
#endif
    rate_of_change[VTXBUF_UUX] = uu_res.x;
    rate_of_change[VTXBUF_UUY] = uu_res.y;
    rate_of_change[VTXBUF_UUZ] = uu_res.z;
#endif

    // Williamson (1980) NOTE: older version of astaroth used inhomogenous
    const Scalar alpha[] = {(Scalar)(.0), (Scalar)(-5. / 9.), (Scalar)(-153. / 128.)};
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        if (step_number == 0) {
            out->vertex_buffer[w][idx] = rate_of_change[w] * (Scalar)dt;
        }
        else {
            out->vertex_buffer[w][idx] = alpha[step_number] * (Scalar)out->vertex_buffer[w][idx] +
                                         rate_of_change[w] * (Scalar)dt;
        }
    }

    if (step_number == 2) {
#if LBFIELD
        const Vector bfield              = curl(aa);
        out->vertex_buffer[BFIELDX][idx] = bfield.x;
        out->vertex_buffer[BFIELDY][idx] = bfield.y;
        out->vertex_buffer[BFIELDZ][idx] = bfield.z;
#endif
    }
}

static void
solve_beta_step(const AcMesh in, const int step_number, const AcReal dt, const int i, const int j,
                const int k, AcMesh* out)
{
    const int idx = acVertexBufferIdx(i, j, k, in.info);

    // Williamson (1980) NOTE: older version of astaroth used inhomogenous
    const Scalar beta[] = {(Scalar)(1. / 3.), (Scalar)(15. / 16.), (Scalar)(8. / 15.)};

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        out->vertex_buffer[w][idx] += beta[step_number] * (Scalar)in.vertex_buffer[w][idx];

    (void)dt; // Suppress unused variable warning if forcing not used
    if (step_number == 2) {
#if LFORCING
        Vector force = forcing((int3){i, j, k}, (Scalar)dt);
        out->vertex_buffer[VTXBUF_UUX][idx] += force.x;
        out->vertex_buffer[VTXBUF_UUY][idx] += force.y;
        out->vertex_buffer[VTXBUF_UUZ][idx] += force.z;
#endif
#if LBFIELD
        out->vertex_buffer[BFIELDX][idx] = in.vertex_buffer[BFIELDX][idx];
        out->vertex_buffer[BFIELDY][idx] = in.vertex_buffer[BFIELDY][idx];
        out->vertex_buffer[BFIELDZ][idx] = in.vertex_buffer[BFIELDZ][idx];
#endif
    }
}

// Checks whether the parameters passed in an AcMeshInfo are valid
static void
checkConfiguration(const AcMeshInfo info)
{
#if AC_VERBOSE
    for (int i = 0; i < NUM_REAL_PARAMS; ++i) {
        if (!is_valid(info.real_params[i])) {
            fprintf(stderr, "WARNING: Passed an invalid value %g to model solver (%s). Skipping.\n",
                    (double)info.real_params[i], realparam_names[i]);
        }
    }

    for (int i = 0; i < NUM_REAL3_PARAMS; ++i) {
        if (!is_valid(info.real3_params[i].x)) {
            fprintf(stderr,
                    "WARNING: Passed an invalid value %g to model solver (%s.x). Skipping.\n",
                    (double)info.real3_params[i].x, realparam_names[i]);
        }
        if (!is_valid(info.real3_params[i].y)) {
            fprintf(stderr,
                    "WARNING: Passed an invalid value %g to model solver (%s.y). Skipping.\n",
                    (double)info.real3_params[i].y, realparam_names[i]);
        }
        if (!is_valid(info.real3_params[i].z)) {
            fprintf(stderr,
                    "WARNING: Passed an invalid value %g to model solver (%s.z). Skipping.\n",
                    (double)info.real3_params[i].z, realparam_names[i]);
        }
    }
#endif

    ERRCHK_ALWAYS(is_valid((Scalar)1. / (Scalar)info.real_params[AC_dsx]));
    ERRCHK_ALWAYS(is_valid((Scalar)1. / (Scalar)info.real_params[AC_dsy]));
    ERRCHK_ALWAYS(is_valid((Scalar)1. / (Scalar)info.real_params[AC_dsz]));
    // ERRCHK_ALWAYS(is_valid(info.real_params[AC_cs2_sound]));
}

AcResult
acHostIntegrateStep(AcMesh mesh, const AcReal dt)
{
    mesh_info = &(mesh.info);

    // Setup built-in parameters
    // mesh_info->real_params[AC_inv_dsx] = (Scalar)(1.0) / mesh_info->real_params[AC_dsx];
    // mesh_info->real_params[AC_inv_dsy] = (Scalar)(1.0) / mesh_info->real_params[AC_dsy];
    // mesh_info->real_params[AC_inv_dsz] = (Scalar)(1.0) / mesh_info->real_params[AC_dsz];
    // mesh_info->real_params[AC_cs2_sound] = mesh_info->real_params[AC_cs_sound] *
    //                                       mesh_info->real_params[AC_cs_sound];
    checkConfiguration(*mesh_info);

    AcMesh intermediate_mesh;
    acHostMeshCreate(mesh.info, &intermediate_mesh);

    const int nx_min = getInt(AC_nx_min);
    const int nx_max = getInt(AC_nx_max);

    const int ny_min = getInt(AC_ny_min);
    const int ny_max = getInt(AC_ny_max);

    const int nz_min = getInt(AC_nz_min);
    const int nz_max = getInt(AC_nz_max);

    for (int step_number = 0; step_number < 3; ++step_number) {

        // Boundconds
        acHostMeshApplyPeriodicBounds(&mesh);

        // Alpha step
        // #pragma omp parallel for
        for (int k = nz_min; k < nz_max; ++k) {
            for (int j = ny_min; j < ny_max; ++j) {
                for (int i = nx_min; i < nx_max; ++i) {
                    solve_alpha_step(mesh, step_number, dt, i, j, k, &intermediate_mesh);
                }
            }
        }

        // Beta step
        // #pragma omp parallel for
        for (int k = nz_min; k < nz_max; ++k) {
            for (int j = ny_min; j < ny_max; ++j) {
                for (int i = nx_min; i < nx_max; ++i) {
                    solve_beta_step(intermediate_mesh, step_number, dt, i, j, k, &mesh);
                }
            }
        }
    }

    acHostMeshDestroy(&intermediate_mesh);
    mesh_info = NULL;
    return AC_SUCCESS;
}

#else  // AC_INTEGRATION_ENABLED == 0
AcResult
acHostIntegrateStep(AcMesh mesh, const AcReal dt)
{
    (void)mesh; // Unused
    (void)dt;   // Unused
    ERROR("Parameters required by acHostIntegrateStep not defined.");
    return AC_FAILURE;
}
#endif // AC_INTEGRATION_ENABLED
