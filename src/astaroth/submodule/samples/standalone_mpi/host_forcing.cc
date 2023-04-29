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
#include "host_forcing.h"
#include "simulation_rng.h"
#include "simulation_control.h"

#include "astaroth_utils.h"
#include "errchk.h"

// #include "math_utils.h"
#include <cmath>

#include <map>

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

    c.x = a.x / std::sqrt(norm);
    c.y = a.y / std::sqrt(norm);
    c.z = a.z / std::sqrt(norm);

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

    using k_force_params = std::pair<AcReal, AcReal>;
    static std::map<k_force_params, std::vector<AcReal3>> k_force_populations = {};

    k_force_params k_params{kmin, kmax};

    // If population has not been generated, generate it
    if (k_force_populations.count(k_params) == 0) {
        auto& pop = k_force_populations[k_params];

        AcReal min_squared = kmin * kmin;
        AcReal max_squared = kmax * kmax;

        // Take the ceil of min and floor of max to get the extreme integer values still within the
        // range
        int min_int = static_cast<int>(std::ceil(kmin));
        int max_int = static_cast<int>(std::floor(kmax));

        int min_squared_int = min_int * min_int;
        int max_squared_int = max_int * max_int;

        for (int x = -max_int; x <= max_int; x++) {
            for (int y = -max_int; y <= max_int; y++) {
                for (int z = -max_int; z <= max_int; z++) {
                    int dist_squared = x * x + y * y + z * z;
                    // Might be redundant, but for sanity's sake, check if the integer distance is
                    // equal to the square maximal integer
                    if ((min_squared <= dist_squared || min_squared_int == dist_squared) &&
                        (max_squared >= dist_squared || max_squared_int == dist_squared)) {
                        pop.push_back(AcReal3{x, y, z});
                    }
                }
            }
        }
    }

    // Select the population of k-forces based on the parameters
    const auto& pop = k_force_populations[k_params];
    std::uniform_int_distribution<uint32_t> k_distribution(0, pop.size() - 1);

    // Sample population
    size_t idx = k_distribution(get_rng());
    AcReal3 k  = pop[idx];
    //log_from_root_proc_with_sim_progress("{\"k\":[%lf,%lf,%lf]}\n", k.x, k.y, k.z);
    return k;
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
    AcReal phi                = AcReal(2.0) * AcReal(M_PI) * random_uniform_real_01();
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
    AcReal kabs = std::sqrt(k_force.x * k_force.x + k_force.y * k_force.y + k_force.z * k_force.z);

    AcReal denominator = std::sqrt(AcReal(1.0) + relhel * relhel) * kabs *
                         std::sqrt(kabs * kabs -
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
#if AC_MPI_ENABLED
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_forcing_magnitude, forcing_magnitude);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_forcing_phase, forcing_phase);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_k_forcex, k_force.x);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_k_forcey, k_force.y);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_k_forcez, k_force.z);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_rex, ff_hel_re.x);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_rey, ff_hel_re.y);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_rez, ff_hel_re.z);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_imx, ff_hel_im.x);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_imy, ff_hel_im.y);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_imz, ff_hel_im.z);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_kaver, kaver);
#else
    (void)forcing_magnitude;
    (void)k_force;
    (void)ff_hel_re;
    (void)ff_hel_im;
    (void)k_force;
    (void)forcing_phase;
    (void)kaver;
    ERROR("AC_MPI_ENABLED must be set to use DEPRECATED_acForcingVec");
#endif // AC_MPI_ENABLED
}

void
printForcingParams(const ForcingParams& forcing_params)
{
    printf(
	"Forcing parameters:\n"
	" magnitude: %lf\n"
	" phase: %lf\n"
	" k force: %lf\n"
	"          %lf\n"
	"          %lf\n"
	" ff hel real: %lf\n"
	"            : %lf\n"
	"            : %lf\n"
	" ff hel imag: %lf\n"
	"            : %lf\n"
	"            : %lf\n"
	" k aver: %lf\n"
	"\n",
	forcing_params.magnitude,
	forcing_params.phase,
	forcing_params.k_force.x,
	forcing_params.k_force.y,
	forcing_params.k_force.z,
	forcing_params.ff_hel_re.x,
	forcing_params.ff_hel_re.y,
	forcing_params.ff_hel_re.z,
	forcing_params.ff_hel_im.x,
	forcing_params.ff_hel_im.y,
	forcing_params.ff_hel_im.z,
	forcing_params.kaver
	);
}

void
loadForcingParamsToGrid(const ForcingParams& forcing_params)
{
#if AC_MPI_ENABLED
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_forcing_magnitude, forcing_params.magnitude);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_forcing_phase, forcing_params.phase);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_k_forcex, forcing_params.k_force.x);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_k_forcey, forcing_params.k_force.y);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_k_forcez, forcing_params.k_force.z);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_rex, forcing_params.ff_hel_re.x);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_rey, forcing_params.ff_hel_re.y);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_rez, forcing_params.ff_hel_re.z);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_imx, forcing_params.ff_hel_im.x);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_imy, forcing_params.ff_hel_im.y);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_ff_hel_imz, forcing_params.ff_hel_im.z);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_kaver, forcing_params.kaver);
    acGridSynchronizeStream(STREAM_ALL);
#else
    (void)forcing_params;
    ERROR("AC_MPI_ENABLED must be set to use loadForcingParamsToGrid");
#endif // AC_MPI_ENABLED
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
    params.phase = AcReal(2.0) * AcReal(M_PI) * random_uniform_real_01();

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
