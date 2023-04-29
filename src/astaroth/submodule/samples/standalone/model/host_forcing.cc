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
#include "host_forcing.h"

// #include "math_utils.h"
#include <cmath>
using namespace std;

// The is a wrapper for genering random numbers with a chosen system.
AcReal
get_random_number_01()
{
    // TODO: Implement better randon number generator http://www.cplusplus.com/reference/random/
    return AcReal(rand()) / AcReal(RAND_MAX);
}

static AcReal3
cross(const AcReal3& a, const AcReal3& b)
{
    AcReal3 c;

    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;
}

static AcReal
dot(const AcReal3& a, const AcReal3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static AcReal3
vec_norm(const AcReal3& a)
{
    AcReal3 c;
    AcReal norm = dot(a, a);

    c.x = a.x / sqrt(norm);
    c.y = a.y / sqrt(norm);
    c.z = a.z / sqrt(norm);

    return c;
}

static AcReal3
vec_multi_scal(const AcReal scal, const AcReal3& a)
{
    AcReal3 c;

    c.x = a.x * scal;
    c.y = a.y * scal;
    c.z = a.z * scal;

    return c;
}

// Generate forcing wave vector k_force
AcReal3
helical_forcing_k_generator(const AcReal kmax, const AcReal kmin)
{
    AcReal phi, theta, kk; // Spherical direction coordinates
    AcReal3 k_force;       // forcing wave vector

    AcReal delta_k = kmax - kmin;

    // Generate vector in spherical coordinates
    phi   = AcReal(2.0) * AcReal(M_PI) * get_random_number_01();
    theta = AcReal(M_PI) * get_random_number_01();
    kk    = delta_k * get_random_number_01() + kmin;

    // Cast into Cartesian form
    k_force = (AcReal3){AcReal(kk * sin(theta) * cos(phi)), //
                        AcReal(kk * sin(theta) * sin(phi)), //
                        AcReal(kk * cos(theta))};

    // printf("k_force.x %f, k_force.y %f, k_force.z %f \n", k_force.x, k_force.y, k_force.z);

    // Round the numbers. In that way k(x/y/z) will get complete waves.
    k_force.x = round(k_force.x);
    k_force.y = round(k_force.y);
    k_force.z = round(k_force.z);

    // printf("After rounding --> k_force.x %f, k_force.y %f, k_force.z %f \n", k_force.x,
    // k_force.y, k_force.z);

    return k_force;
}

// Generate the unit perpendicular unit vector e required for helical forcing
// Addapted from Pencil code forcing.f90 hel_vec() subroutine.
void
helical_forcing_e_generator(AcReal3* e_force, const AcReal3 k_force)
{

    AcReal3 k_cross_e         = cross(k_force, *e_force);
    k_cross_e                 = vec_norm(k_cross_e);
    AcReal3 k_cross_k_cross_e = cross(k_force, k_cross_e);
    k_cross_k_cross_e         = vec_norm(k_cross_k_cross_e);
    AcReal phi                = AcReal(2.0) * AcReal(M_PI) * get_random_number_01();
    AcReal3 ee_tmp1           = vec_multi_scal(cos(phi), k_cross_e);
    AcReal3 ee_tmp2           = vec_multi_scal(sin(phi), k_cross_k_cross_e);

    *e_force = (AcReal3){ee_tmp1.x + ee_tmp2.x, ee_tmp1.y + ee_tmp2.y, ee_tmp1.z + ee_tmp2.z};
}

// PC Manual Eq. 223
void
helical_forcing_special_vector(AcReal3* ff_hel_re, AcReal3* ff_hel_im, const AcReal3 k_force,
                               const AcReal3 e_force, const AcReal relhel)
{

    // k dot e
    AcReal3 kdote;
    kdote.x = k_force.x * e_force.x;
    kdote.y = k_force.y * e_force.y;
    kdote.z = k_force.z * e_force.z;

    // k cross e
    AcReal3 k_cross_e;
    k_cross_e.x = k_force.y * e_force.z - k_force.z * e_force.y;
    k_cross_e.y = k_force.z * e_force.x - k_force.x * e_force.z;
    k_cross_e.z = k_force.x * e_force.y - k_force.y * e_force.x;

    // k cross k cross e
    AcReal3 k_cross_k_cross_e;
    k_cross_k_cross_e.x = k_force.y * k_cross_e.z - k_force.z * k_cross_e.y;
    k_cross_k_cross_e.y = k_force.z * k_cross_e.x - k_force.x * k_cross_e.z;
    k_cross_k_cross_e.z = k_force.x * k_cross_e.y - k_force.y * k_cross_e.x;

    // abs(k)
    AcReal kabs = sqrt(k_force.x * k_force.x + k_force.y * k_force.y + k_force.z * k_force.z);

    AcReal denominator = sqrt(AcReal(1.0) + relhel * relhel) * kabs *
                         sqrt(kabs * kabs -
                              (kdote.x * kdote.x + kdote.y * kdote.y + kdote.z * kdote.z));

    // MV: I suspect there is a typo in the Pencil Code manual!
    //*ff_hel_re = (AcReal3){-relhel*kabs*k_cross_e.x/denominator,
    //                       -relhel*kabs*k_cross_e.y/denominator,
    //                       -relhel*kabs*k_cross_e.z/denominator};

    //*ff_hel_im = (AcReal3){k_cross_k_cross_e.x/denominator,
    //                       k_cross_k_cross_e.y/denominator,
    //                       k_cross_k_cross_e.z/denominator};

    // See PC forcing.f90 forcing_hel_both()
    *ff_hel_im = (AcReal3){kabs * k_cross_e.x / denominator, kabs * k_cross_e.y / denominator,
                           kabs * k_cross_e.z / denominator};

    *ff_hel_re = (AcReal3){relhel * k_cross_k_cross_e.x / denominator,
                           relhel * k_cross_k_cross_e.y / denominator,
                           relhel * k_cross_k_cross_e.z / denominator};
}

// Tool for loading forcing vector information into the device memory
// %JP: Added a generic function for loading device constants (acLoadDeviceConstant).
// This acForcingVec should go outside the core library since it references user-defined
// parameters such as AC_forcing_magnitude which may not be defined in all projects.
// host_forcing.cc is probably a good place for this.
// %JP update: moved this here out of astaroth.cu. Should be renamed such that it cannot be
// confused with actual interface functions.
// %JP update 2: deprecated acForcingVec: use loadForcingParams instead
void
DEPRECATED_acForcingVec(const AcReal forcing_magnitude, const AcReal3 k_force,
                        const AcReal3 ff_hel_re, const AcReal3 ff_hel_im,
                        const AcReal forcing_phase, const AcReal kaver)
{
    acLoadDeviceConstant(AC_forcing_magnitude, forcing_magnitude);
    acLoadDeviceConstant(AC_forcing_phase, forcing_phase);

    acLoadDeviceConstant(AC_k_forcex, k_force.x);
    acLoadDeviceConstant(AC_k_forcey, k_force.y);
    acLoadDeviceConstant(AC_k_forcez, k_force.z);

    acLoadDeviceConstant(AC_ff_hel_rex, ff_hel_re.x);
    acLoadDeviceConstant(AC_ff_hel_rey, ff_hel_re.y);
    acLoadDeviceConstant(AC_ff_hel_rez, ff_hel_re.z);

    acLoadDeviceConstant(AC_ff_hel_imx, ff_hel_im.x);
    acLoadDeviceConstant(AC_ff_hel_imy, ff_hel_im.y);
    acLoadDeviceConstant(AC_ff_hel_imz, ff_hel_im.z);

    acLoadDeviceConstant(AC_kaver, kaver);
}

void
loadForcingParamsToDevice(const ForcingParams& forcing_params)
{
    acLoadDeviceConstant(AC_forcing_magnitude, forcing_params.magnitude);
    acLoadDeviceConstant(AC_forcing_phase, forcing_params.phase);

    acLoadDeviceConstant(AC_k_forcex, forcing_params.k_force.x);
    acLoadDeviceConstant(AC_k_forcey, forcing_params.k_force.y);
    acLoadDeviceConstant(AC_k_forcez, forcing_params.k_force.z);

    acLoadDeviceConstant(AC_ff_hel_rex, forcing_params.ff_hel_re.x);
    acLoadDeviceConstant(AC_ff_hel_rey, forcing_params.ff_hel_re.y);
    acLoadDeviceConstant(AC_ff_hel_rez, forcing_params.ff_hel_re.z);

    acLoadDeviceConstant(AC_ff_hel_imx, forcing_params.ff_hel_im.x);
    acLoadDeviceConstant(AC_ff_hel_imy, forcing_params.ff_hel_im.y);
    acLoadDeviceConstant(AC_ff_hel_imz, forcing_params.ff_hel_im.z);

    acLoadDeviceConstant(AC_kaver, forcing_params.kaver);
    acSynchronizeStream(STREAM_ALL);
}

/** This function would be used in autotesting to update the forcing params of the host
    configuration. */
void
loadForcingParamsToHost(const ForcingParams& forcing_params, AcMesh* mesh)
{
    // %JP: Left some regex magic here in case we need to modify the ForcingParams struct
    // acLoadDeviceConstant\(([A-Za-z_]*), ([a-z_.]*)\);
    // mesh->info.real_params[$1] = $2;
    mesh->info.real_params[AC_forcing_magnitude] = forcing_params.magnitude;
    mesh->info.real_params[AC_forcing_phase]     = forcing_params.phase;

    mesh->info.real_params[AC_k_forcex] = forcing_params.k_force.x;
    mesh->info.real_params[AC_k_forcey] = forcing_params.k_force.y;
    mesh->info.real_params[AC_k_forcez] = forcing_params.k_force.z;

    mesh->info.real_params[AC_ff_hel_rex] = forcing_params.ff_hel_re.x;
    mesh->info.real_params[AC_ff_hel_rey] = forcing_params.ff_hel_re.y;
    mesh->info.real_params[AC_ff_hel_rez] = forcing_params.ff_hel_re.z;

    mesh->info.real_params[AC_ff_hel_imx] = forcing_params.ff_hel_im.x;
    mesh->info.real_params[AC_ff_hel_imy] = forcing_params.ff_hel_im.y;
    mesh->info.real_params[AC_ff_hel_imz] = forcing_params.ff_hel_im.z;

    mesh->info.real_params[AC_kaver] = forcing_params.kaver;
}

void
loadForcingParamsToHost(const ForcingParams& forcing_params, ModelMesh* mesh)
{
    // %JP: Left some regex magic here in case we need to modify the ForcingParams struct
    // acLoadDeviceConstant\(([A-Za-z_]*), ([a-z_.]*)\);
    // mesh->info.real_params[$1] = $2;
    mesh->info.real_params[AC_forcing_magnitude] = forcing_params.magnitude;
    mesh->info.real_params[AC_forcing_phase]     = forcing_params.phase;

    mesh->info.real_params[AC_k_forcex] = forcing_params.k_force.x;
    mesh->info.real_params[AC_k_forcey] = forcing_params.k_force.y;
    mesh->info.real_params[AC_k_forcez] = forcing_params.k_force.z;

    mesh->info.real_params[AC_ff_hel_rex] = forcing_params.ff_hel_re.x;
    mesh->info.real_params[AC_ff_hel_rey] = forcing_params.ff_hel_re.y;
    mesh->info.real_params[AC_ff_hel_rez] = forcing_params.ff_hel_re.z;

    mesh->info.real_params[AC_ff_hel_imx] = forcing_params.ff_hel_im.x;
    mesh->info.real_params[AC_ff_hel_imy] = forcing_params.ff_hel_im.y;
    mesh->info.real_params[AC_ff_hel_imz] = forcing_params.ff_hel_im.z;

    mesh->info.real_params[AC_kaver] = forcing_params.kaver;
}

ForcingParams
generateForcingParams(const AcMeshInfo& mesh_info)
{
    ForcingParams params = {};

    // Forcing properties
    AcReal relhel    = mesh_info.real_params[AC_relhel];
    params.magnitude = mesh_info.real_params[AC_forcing_magnitude];
    AcReal kmin      = mesh_info.real_params[AC_kmin];
    AcReal kmax      = mesh_info.real_params[AC_kmax];

    params.kaver = (kmax - kmin) / AcReal(2.0);

    // Generate forcing wave vector k_force
    params.k_force = helical_forcing_k_generator(kmax, kmin);

    // Randomize the phase
    params.phase = AcReal(2.0) * AcReal(M_PI) * get_random_number_01();

    // Generate e for k. Needed for the sake of isotrophy.
    AcReal3 e_force;
    if ((params.k_force.y == AcReal(0.0)) && (params.k_force.z == AcReal(0.0))) {
        e_force = (AcReal3){0.0, 1.0, 0.0};
    }
    else {
        e_force = (AcReal3){1.0, 0.0, 0.0};
    }
    helical_forcing_e_generator(&e_force, params.k_force);

    helical_forcing_special_vector(&params.ff_hel_re, &params.ff_hel_im, params.k_force, e_force,
                                   relhel);

    return params;
}
