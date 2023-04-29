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
#include "host_memory.h"

#include <math.h>

#include "errchk.h"

#define AC_GEN_STR(X) #X
const char* init_type_names[] = {AC_FOR_INIT_TYPES(AC_GEN_STR)};
#undef AC_GEN_STR

#define XORIG (AcReal(.5) * mesh->info.int_params[AC_nx] * mesh->info.real_params[AC_dsx])
#define YORIG (AcReal(.5) * mesh->info.int_params[AC_ny] * mesh->info.real_params[AC_dsy])
#define ZORIG (AcReal(.5) * mesh->info.int_params[AC_nz] * mesh->info.real_params[AC_dsz])

/*
#include <stdint.h>
static uint64_t ac_rand_next = 1;

static int32_t
ac_rand(void)
{
        ac_rand_next = ac_rand_next * 1103515245 + 12345;
        return (uint32_t)(ac_rand_next/65536) % 32768;
}

static void
ac_srand(const uint32_t seed)
{
        ac_rand_next = seed;
}
*/

AcMesh*
acmesh_create(const AcMeshInfo& mesh_info)
{
    AcMesh* mesh = (AcMesh*)malloc(sizeof(*mesh));
    mesh->info   = mesh_info;

    const size_t bytes = acVertexBufferSizeBytes(mesh->info);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        mesh->vertex_buffer[VertexBufferHandle(i)] = (AcReal*)malloc(bytes);
        ERRCHK(mesh->vertex_buffer[VertexBufferHandle(i)] != NULL);
    }

    return mesh;
}

void
vertex_buffer_set(const VertexBufferHandle& key, const AcReal& val, AcMesh* mesh)
{
    const int n = acVertexBufferSize(mesh->info);
    for (int i = 0; i < n; ++i)
        mesh->vertex_buffer[key][i] = val;
}

/** Inits all fields to 1. Setting the mesh to zero is problematic because some fields are supposed
    to be > 0 and the results would vary widely, which leads to loss of precision in the
    computations */
void
acmesh_clear(AcMesh* mesh)
{
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        vertex_buffer_set(VertexBufferHandle(w), 1, mesh); // Init all fields to 1 by default.
}

static AcReal
randr(void)
{
    return AcReal(rand()) / AcReal(RAND_MAX);
}

void
lnrho_step(AcMesh* mesh)
{
    const int mx = mesh->info.int_params[AC_mx];
    const int my = mesh->info.int_params[AC_my];
    const int mz = mesh->info.int_params[AC_mz];

    // const int    nx_min = mesh->info.int_params[AC_nx_min];
    // const int    nx_max = mesh->info.int_params[AC_nx_max];
    // const int    ny_min = mesh->info.int_params[AC_ny_min];
    // const int    ny_max = mesh->info.int_params[AC_ny_max];
    // const int    nz_min = mesh->info.int_params[AC_nz_min];
    // const int    nz_max = mesh->info.int_params[AC_nz_max];

    // const AcReal DX     = mesh->info.real_params[AC_dsx];
    // const AcReal DY     = mesh->info.real_params[AC_dsy];
    // const AcReal DZ     = mesh->info.real_params[AC_dsz];
    // const AcReal xmax   = DX * (nx_max - nx_min) ;
    // const AcReal zmax   = DZ * (nz_max - nz_min) ;

    // const AcReal lnrho1 = (AcReal) -1.0; // TODO mesh->info.real_params[AC_lnrho1];
    const AcReal lnrho2 = (AcReal)0.0; // TODO mesh->info.real_params[AC_lnrho2];
    // const AcReal rho1   = (AcReal) exp(lnrho1);
    // const AcReal rho2   = (AcReal) exp(lnrho2);

    // const AcReal k_pert    = (AcReal) 1.0; //mesh->info.real_params[AC_k_pert]; //Wamenumber of
    // the perturbation const AcReal k_pert    = 4.0; //mesh->info.real_params[AC_k_pert];
    // //Wamenumber of the perturbation
    // const AcReal ampl_pert = xmax/10.0; // xmax/mesh->info.real_params[AC_pert]; //Amplitude of
    // the perturbation
    // const AcReal ampl_pert = (AcReal) 0.0;//xmax/20.0; // xmax/mesh->info.real_params[AC_pert];
    // //Amplitude of the perturbation const AcReal two_pi       = (AcReal) 6.28318531;

    // const AcReal xorig  = mesh->info.real_params[AC_xorig];
    // const AcReal zorig  = mesh->info.real_params[AC_zorig];
    // const AcReal trans  = mesh->info.real_params[AC_trans];

    // AcReal       xx, zz, tanhprof, cosz_wave;

    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
            for (int i = 0; i < mx; i++) {
                int idx = i + j * mx + k * mx * my;
                // zz = DZ * AcReal(k) - zorig; // Not used
                // cosz_wave = ampl_pert*AcReal(cos(k_pert*((zz/zmax)*two_pi))); // Not used
                // xx = DX * AcReal(i) - xorig + cosz_wave; //ADD WAVE TODO // Not used
                // tanhprof = AcReal(0.5)*((rho2+rho1) + (rho2-rho1)*AcReal(tanh(xx/trans))); // Not
                // used Commented out the step function initial codition.
                // mesh->vertex_buffer[VTXBUF_LNRHO][idx] = log(tanhprof);
                mesh->vertex_buffer[VTXBUF_LNRHO][idx] = lnrho2;
            }
        }
    }
}

// This is the initial condition type for the infalling vedge in the pseudodisk
// model.
void
inflow_vedge(AcMesh* mesh)
{
    const int mx = mesh->info.int_params[AC_mx];
    const int my = mesh->info.int_params[AC_my];
    const int mz = mesh->info.int_params[AC_mz];

    // const int nx_min = mesh->info.int_params[AC_nx_min];
    // const int nx_max = mesh->info.int_params[AC_nx_max];
    // const int ny_min = mesh->info.int_params[AC_ny_min];
    // const int ny_max = mesh->info.int_params[AC_ny_max];
    // const int nz_min = mesh->info.int_params[AC_nz_min];
    // const int nz_max = mesh->info.int_params[AC_nz_max];

    // const double DX    = mesh->info.real_params[AC_dsx];
    // const double DY    = mesh->info.real_params[AC_dsy];
    const double DZ = mesh->info.real_params[AC_dsz];

    const double AMPL_UU = mesh->info.real_params[AC_ampl_uu];
    const double ANGL_UU = mesh->info.real_params[AC_angl_uu];

    const double zorig = mesh->info.real_params[AC_zorig];
    double zz;
    double trans = mesh->info.real_params[AC_trans];

    // const AcReal range = AcReal(.5);

    // const AcReal zmax  = AcReal(DZ * (nz_max - nz_min));
    // const AcReal gaussr  = zmax / AcReal(4.0);

    // for (int k = nz_min; k < nz_max; k++) {
    //    for (int j = ny_min; j < ny_max; j++) {
    //        for (int i = nx_min; i < nx_max; i++) {
    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
            for (int i = 0; i < mx; i++) {
                int idx = i + j * mx + k * mx * my;
                zz      = DZ * double(k) - zorig;
                // mesh->vertex_buffer[VTXBUF_UUX][idx] = -AMPL_UU*cos(ANGL_UU);
                mesh->vertex_buffer[VTXBUF_UUX][idx] = AcReal(-AMPL_UU * cos(ANGL_UU) *
                                                              fabs(tanh(zz / trans)));
                mesh->vertex_buffer[VTXBUF_UUY][idx] = AcReal(0.0);
                mesh->vertex_buffer[VTXBUF_UUZ][idx] = AcReal(-AMPL_UU * sin(ANGL_UU) *
                                                              tanh(zz / trans));

                // Variarion to density
                // AcReal rho = exp(mesh->vertex_buffer[VTXBUF_LNRHO][idx]);
                // NO GAUSSIAN//rho = rho*exp(-(zz/gaussr)*(zz/gaussr));
                // mesh->vertex_buffer[VTXBUF_LNRHO][idx] = log(rho + (range*rho) * (randr() -
                // AcReal(-0.5)));
            }
        }
    }
}

// This is the initial condition type for the infalling vedge in the pseudodisk
// model.
void
simple_uniform_core(AcMesh* mesh)
{
    const int mx = mesh->info.int_params[AC_mx];
    const int my = mesh->info.int_params[AC_my];
    const int mz = mesh->info.int_params[AC_mz];

    const double DX = mesh->info.real_params[AC_dsx];
    const double DY = mesh->info.real_params[AC_dsy];
    const double DZ = mesh->info.real_params[AC_dsz];

    const double ampl_lnrho = mesh->info.real_params[AC_ampl_lnrho];

    const double xorig = mesh->info.real_params[AC_xorig];
    const double yorig = mesh->info.real_params[AC_yorig];
    const double zorig = mesh->info.real_params[AC_zorig];

    const double G_const     = mesh->info.real_params[AC_G_const];
    const double M_sink_init = mesh->info.real_params[AC_M_sink_init];
    const double cs2_sound   = mesh->info.real_params[AC_cs2_sound];

    const double RR_inner_bound = mesh->info.real_params[AC_soft] / AcReal(2.0);
    const double core_coeff     = (exp(ampl_lnrho) * cs2_sound) / (double(4.0) * M_PI * G_const);

    double xx, yy, zz, RR;
    double core_profile;

    // TEMPORARY TEST INPUT PARAMETERS
    const double core_radius = DX * 32.0;
    const double trans       = DX * 12.0;
    // const double epsilon = DX*2.0;
    const double vel_scale = mesh->info.real_params[AC_ampl_uu];
    double abso_vel;

    RR = 1.0;
    printf("%e %e %e \n", RR, trans, core_radius);

    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
            for (int i = 0; i < mx; i++) {
                int idx = i + j * mx + k * mx * my;
                xx      = DX * double(i) - xorig;
                yy      = DY * double(j) - yorig;
                zz      = DZ * double(k) - zorig;

                RR = sqrt(xx * xx + yy * yy + zz * zz);

                if (RR >= RR_inner_bound) {
                    abso_vel     = vel_scale * sqrt(2.0 * G_const * M_sink_init / RR);
                    core_profile = pow(RR, -2.0); // double(1.0);
                }
                else {
                    abso_vel = vel_scale * sqrt(2.0 * AC_G_const * AC_M_sink_init / RR_inner_bound);
                    core_profile = pow(RR_inner_bound, -2.0); // double(1.0);
                }

                if (RR <= sqrt(DX * DX + DY * DY + DZ * DZ)) {
                    abso_vel = 0.0;
                    RR       = 1.0;
                }

                mesh->vertex_buffer[VTXBUF_LNRHO][idx] = AcReal(log(core_coeff * core_profile));
                mesh->vertex_buffer[VTXBUF_UUX][idx]   = AcReal(-abso_vel * (yy / RR));
                mesh->vertex_buffer[VTXBUF_UUY][idx]   = AcReal(abso_vel * (xx / RR));
                mesh->vertex_buffer[VTXBUF_UUZ][idx]   = AcReal(0.0);
            }
        }
    }
}

// This is the initial condition type for the infalling vedge in the pseudodisk
// model.
void
inflow_vedge_freefall(AcMesh* mesh)
{
    const int mx = mesh->info.int_params[AC_mx];
    const int my = mesh->info.int_params[AC_my];
    const int mz = mesh->info.int_params[AC_mz];

    // const int nx_min = mesh->info.int_params[AC_nx_min];
    // const int nx_max = mesh->info.int_params[AC_nx_max];
    // const int ny_min = mesh->info.int_params[AC_ny_min];
    // const int ny_max = mesh->info.int_params[AC_ny_max];
    // const int nz_min = mesh->info.int_params[AC_nz_min];
    // const int nz_max = mesh->info.int_params[AC_nz_max];

    const double DX = mesh->info.real_params[AC_dsx];
    // const double DY    = mesh->info.real_params[AC_dsy];
    const double DZ = mesh->info.real_params[AC_dsz];

    // const double AMPL_UU = mesh->info.real_params[AC_ampl_uu];
    const double ANGL_UU = mesh->info.real_params[AC_angl_uu];
    const double SQ2GM   = mesh->info.real_params[AC_sq2GM_star];
    // const double GM = mesh->info.real_params[AC_GM_star];
    // const double M_star  = mesh->info.real_params[AC_M_star];
    // const double G_CONST = mesh->info.real_params[AC_G_CONST];

    // const double unit_length   = mesh->info.real_params[AC_unit_length];
    // const double unit_density  = mesh->info.real_params[AC_unit_density];
    // const double unit_velocity = mesh->info.real_params[AC_unit_velocity];

    const double xorig = mesh->info.real_params[AC_xorig];
    // const double yorig = mesh->info.real_params[AC_yorig];
    const double zorig = mesh->info.real_params[AC_zorig];
    // const double trans = mesh->info.real_params[AC_trans];
    //  double xx, yy, zz, RR;
    double xx, zz, RR;
    // double delx, dely, delz;
    double delx, delz;
    // double u_x, u_y, u_z, veltot, tanhz;
    double u_x, u_z, veltot, tanhz;

    const double star_pos_x = mesh->info.real_params[AC_star_pos_x];
    const double star_pos_z = mesh->info.real_params[AC_star_pos_z];

    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
            for (int i = 0; i < mx; i++) {
                int idx = i + j * mx + k * mx * my;
                xx      = DX * double(i) - xorig;
                zz      = DZ * double(k) - zorig;

                delx = xx - star_pos_x;
                delz = zz - star_pos_z;
                // TODO: Figure out isthis needed. Now a placeholder.
                // tanhz = fabs(tanh(zz/trans));
                tanhz = 1.0;

                RR     = sqrt(delx * delx + delz * delz);
                veltot = SQ2GM / sqrt(RR); // Free fall velocity

                // Normal velocity components
                u_x = -veltot * (delx / RR);
                u_z = -veltot * (delz / RR);

                // printf("star_pos_z %e, zz %e, delz %e, RR %e\n", star_pos_z, zz, delz, RR);

                // printf("unit_length = %e, unit_density = %e, unit_velocity = %e,\n M_star = %e,
                // G_CONST = %e, GM = %e, SQ2GM = %e, \n RR = %e, u_x = %e, u_z %e\n",
                //        unit_length, unit_density,
                //        unit_velocity, M_star, G_CONST, GM, SQ2GM, RR, u_x, u_z);
                // printf("%e\n", unit_length*unit_length*unit_length);

                // Here including an angel tilt due to pseudodisk
                if (delz >= 0.0) {
                    mesh->vertex_buffer[VTXBUF_UUX][idx] = AcReal(
                        (u_x * cos(ANGL_UU) - u_z * sin(ANGL_UU)) * tanhz);
                    mesh->vertex_buffer[VTXBUF_UUY][idx] = AcReal(0.0);
                    mesh->vertex_buffer[VTXBUF_UUZ][idx] = AcReal(
                        (u_x * sin(ANGL_UU) + u_z * cos(ANGL_UU)) * tanhz);
                }
                else {
                    mesh->vertex_buffer[VTXBUF_UUX][idx] = AcReal(
                        (u_x * cos(ANGL_UU) + u_z * sin(ANGL_UU)) * tanhz);
                    mesh->vertex_buffer[VTXBUF_UUY][idx] = AcReal(0.0);
                    mesh->vertex_buffer[VTXBUF_UUZ][idx] = AcReal(
                        (-u_x * sin(ANGL_UU) + u_z * cos(ANGL_UU)) * tanhz);
                }
            }
        }
    }
}

// Only x-direction free fall
void
inflow_freefall_x(AcMesh* mesh)
{
    const int mx = mesh->info.int_params[AC_mx];
    const int my = mesh->info.int_params[AC_my];
    const int mz = mesh->info.int_params[AC_mz];

    const double DX = mesh->info.real_params[AC_dsx];

    const double SQ2GM = mesh->info.real_params[AC_sq2GM_star];
    // const double G_CONST = mesh->info.real_params[AC_G_CONST];

    const double xorig = mesh->info.real_params[AC_xorig];
    double xx, RR;
    double delx;
    double /*u_x,*/ veltot;

    const double star_pos_x = mesh->info.real_params[AC_star_pos_x];

    const double ampl_lnrho = mesh->info.real_params[AC_ampl_lnrho];

    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
            for (int i = 0; i < mx; i++) {
                int idx = i + j * mx + k * mx * my;
                xx      = DX * double(i) - xorig;

                delx = xx - star_pos_x;

                RR = fabs(delx);

                veltot = SQ2GM / sqrt(RR); // Free fall velocity

                if (isinf(veltot) == 1)
                    printf("xx %e star_pos_x %e delz %e RR %e veltot %e\n", xx, star_pos_x, delx,
                           RR, veltot);

                // Normal velocity components
                // u_x = - veltot; // Not used

                // Freefall condition
                // mesh->vertex_buffer[VTXBUF_UUX][idx] = u_x;
                // mesh->vertex_buffer[VTXBUF_UUY][idx] = 0.0;
                // mesh->vertex_buffer[VTXBUF_UUZ][idx] = 0.0;

                // Starting with steady state
                mesh->vertex_buffer[VTXBUF_UUX][idx] = 0.0;
                mesh->vertex_buffer[VTXBUF_UUY][idx] = 0.0;
                mesh->vertex_buffer[VTXBUF_UUZ][idx] = 0.0;

                mesh->vertex_buffer[VTXBUF_LNRHO][idx] = AcReal(ampl_lnrho);
            }
        }
    }
}

void
kickball(AcMesh* mesh)
{
    AcReal* uu_x = mesh->vertex_buffer[VTXBUF_UUX];
    AcReal* uu_y = mesh->vertex_buffer[VTXBUF_UUY];
    AcReal* uu_z = mesh->vertex_buffer[VTXBUF_UUZ];

    const int mx = mesh->info.int_params[AC_mx];
    const int my = mesh->info.int_params[AC_my];

    const int nx_min = mesh->info.int_params[AC_nx_min];
    const int nx_max = mesh->info.int_params[AC_nx_max];
    const int ny_min = mesh->info.int_params[AC_ny_min];
    const int ny_max = mesh->info.int_params[AC_ny_max];
    const int nz_min = mesh->info.int_params[AC_nz_min];
    const int nz_max = mesh->info.int_params[AC_nz_max];

    const double DX = mesh->info.real_params[AC_dsx];
    const double DY = mesh->info.real_params[AC_dsy];
    const double DZ = mesh->info.real_params[AC_dsz];

    const double xorig = double(XORIG) - 0.000001;
    const double yorig = double(YORIG) - 0.000001;
    const double zorig = double(ZORIG) - 0.000001;

    const double INIT_LOC_UU_X = 0.0;
    const double INIT_LOC_UU_Y = 0.0;
    const double INIT_LOC_UU_Z = 0.0;

    const double AMPL_UU    = mesh->info.real_params[AC_ampl_uu];
    const double UU_SHELL_R = 0.8;
    const double WIDTH_UU   = 0.2;

    // Outward explosion with gaussian initial velocity profile.
    int idx;
    double xx, yy, zz, rr2, rr, theta = 0.0, phi = 0.0;
    double uu_radial;

    // double theta_old = 0.0;

    for (int k = nz_min; k < nz_max; k++) {
        for (int j = ny_min; j < ny_max; j++) {
            for (int i = nx_min; i < nx_max; i++) {
                // Calculate the value of velocity in a particular radius.
                idx = i + j * mx + k * mx * my;
                // Determine the coordinates
                xx = DX * (i - nx_min) - xorig;
                xx = xx - INIT_LOC_UU_X;

                yy = DY * (j - ny_min) - yorig;
                yy = yy - INIT_LOC_UU_Y;

                zz = DZ * (k - nz_min) - zorig;
                zz = zz - INIT_LOC_UU_Z;

                rr2 = pow(xx, 2.0) + pow(yy, 2.0) + pow(zz, 2.0);
                rr  = sqrt(rr2);

                // Origin is different!
                double xx_abs, yy_abs, zz_abs;
                if (rr > 0.0) {
                    // theta range [0, PI]
                    if (zz >= 0.0) {
                        theta = acos(zz / rr);
                        if (theta > M_PI / 2.0 || theta < 0.0) {
                            printf("Explosion THETA WRONG: zz = %.3f, rr = "
                                   "%.3f, theta = %.3e/PI, M_PI = %.3e\n",
                                   zz, rr, theta / M_PI, M_PI);
                        }
                    }
                    else {
                        zz_abs = -zz; // Needs a posite value for acos
                        theta  = M_PI - acos(zz_abs / rr);
                        if (theta < M_PI / 2.0 || theta > 2 * M_PI) {
                            printf("Explosion THETA WRONG: zz = %.3f, rr = "
                                   "%.3f, theta = %.3e/PI, M_PI = %.3e\n",
                                   zz, rr, theta / M_PI, M_PI);
                        }
                    }

                    // phi range [0, 2*PI]i
                    if (xx != 0.0) {
                        if (xx < 0.0 && yy >= 0.0) {
                            //-+
                            xx_abs = -xx; // Needs a posite value for atan
                            phi    = M_PI - atan(yy / xx_abs);
                            if (phi < (M_PI / 2.0) || phi > M_PI) {
                                printf("Explosion PHI WRONG -+: xx = %.3f, yy "
                                       "= %.3f, phi = %.3e/PI, M_PI = %.3e\n",
                                       xx, yy, phi / M_PI, M_PI);
                            }
                        }
                        else if (xx > 0.0 && yy < 0.0) {
                            //+-
                            yy_abs = -yy;
                            phi    = 2.0 * M_PI - atan(yy_abs / xx);
                            if (phi < (3.0 * M_PI) / 2.0 || phi > (2.0 * M_PI + 1e-6)) {
                                printf("Explosion PHI WRONG +-: xx = %.3f, yy "
                                       "= %.3f, phi = %.3e/PI, M_PI = %.3e\n",
                                       xx, yy, phi / M_PI, M_PI);
                            }
                        }
                        else if (xx < 0.0 && yy < 0.0) {
                            //--
                            yy_abs = -yy;
                            xx_abs = -xx;
                            phi    = M_PI + atan(yy_abs / xx_abs);
                            if (phi < M_PI || phi > ((3.0 * M_PI) / 2.0 + 1e-6)) {
                                printf("Explosion PHI WRONG --: xx = %.3f, yy "
                                       "= %.3f, xx_abs = %.3f, yy_abs = %.3f, "
                                       "phi = %.3e, (3.0*M_PI)/2.0 = %.3e\n",
                                       xx, yy, xx_abs, yy_abs, phi, (3.0 * M_PI) / 2.0);
                            }
                        }
                        else {
                            //++
                            phi = atan(yy / xx);
                            if (phi < 0 || phi > M_PI / 2.0) {
                                printf("Explosion PHI WRONG --: xx = %.3f, yy = "
                                       "%.3f, phi = %.3e, (3.0*M_PI)/2.0 = %.3e\n",
                                       xx, yy, phi, (3.0 * M_PI) / 2.0);
                            }
                        }
                    }
                    else { // To avoid div by zero with atan
                        if (yy > 0.0) {
                            phi = M_PI / 2.0;
                        }
                        else if (yy < 0.0) {
                            phi = (3.0 * M_PI) / 2.0;
                        }
                        else {
                            phi = 0.0;
                        }
                    }

                    // Set zero for explicit safekeeping
                    if (xx == 0.0 && yy == 0.0) {
                        phi = 0.0;
                    }

                    uu_radial = AMPL_UU *
                                exp(-pow((rr - UU_SHELL_R), 2.0) / (2.0 * pow(WIDTH_UU, 2.0))) *
                                exp(-pow((theta - M_PI / 2.0), 2.0) /
                                    (2.0 * pow(M_PI / 4.0, 2.0))) *
                                exp(-pow((phi - M_PI / 2.0), 2.0) / (2.0 * pow(M_PI / 4.0, 2.0)));
                }
                else {
                    uu_radial = 0.0; // TODO: There will be a discontinuity in
                                     // the origin... Should the shape of the
                                     // distribution be different?
                }

                // Determine the carthesian velocity components and lnrho

                uu_x[idx] = AcReal(uu_radial * sin(theta) * cos(phi));
                uu_y[idx] = AcReal(uu_radial * sin(theta) * sin(phi));
                uu_z[idx] = AcReal(uu_radial * cos(theta));
            }
        }
    }
}

void
gaussian_radial_explosion(AcMesh* mesh)
{
    AcReal* uu_x = mesh->vertex_buffer[VTXBUF_UUX];
    AcReal* uu_y = mesh->vertex_buffer[VTXBUF_UUY];
    AcReal* uu_z = mesh->vertex_buffer[VTXBUF_UUZ];

    const int mx = mesh->info.int_params[AC_mx];
    const int my = mesh->info.int_params[AC_my];

    const int nx_min = mesh->info.int_params[AC_nx_min];
    const int nx_max = mesh->info.int_params[AC_nx_max];
    const int ny_min = mesh->info.int_params[AC_ny_min];
    const int ny_max = mesh->info.int_params[AC_ny_max];
    const int nz_min = mesh->info.int_params[AC_nz_min];
    const int nz_max = mesh->info.int_params[AC_nz_max];

    const double DX = mesh->info.real_params[AC_dsx];
    const double DY = mesh->info.real_params[AC_dsy];
    const double DZ = mesh->info.real_params[AC_dsz];

    const double xorig = double(XORIG) - 0.000001;
    const double yorig = double(YORIG) - 0.000001;
    const double zorig = double(ZORIG) - 0.000001;

    const double INIT_LOC_UU_X = 0.0;
    const double INIT_LOC_UU_Y = 0.0;
    const double INIT_LOC_UU_Z = 0.0;

    const double AMPL_UU    = mesh->info.real_params[AC_ampl_uu];
    const double UU_SHELL_R = 0.8;
    const double WIDTH_UU   = 0.2;

    // Outward explosion with gaussian initial velocity profile.
    int idx;
    double xx, yy, zz, rr2, rr, theta = 0.0, phi = 0.0;
    double uu_radial;

    // double theta_old = 0.0;

    for (int k = nz_min; k < nz_max; k++) {
        for (int j = ny_min; j < ny_max; j++) {
            for (int i = nx_min; i < nx_max; i++) {
                // Calculate the value of velocity in a particular radius.
                idx = i + j * mx + k * mx * my;
                // Determine the coordinates
                xx = DX * (i - nx_min) - xorig;
                xx = xx - INIT_LOC_UU_X;

                yy = DY * (j - ny_min) - yorig;
                yy = yy - INIT_LOC_UU_Y;

                zz = DZ * (k - nz_min) - zorig;
                zz = zz - INIT_LOC_UU_Z;

                rr2 = pow(xx, 2.0) + pow(yy, 2.0) + pow(zz, 2.0);
                rr  = sqrt(rr2);

                // Origin is different!
                double xx_abs, yy_abs, zz_abs;
                if (rr > 0.0) {
                    // theta range [0, PI]
                    if (zz >= 0.0) {
                        theta = acos(zz / rr);
                        if (theta > M_PI / 2.0 || theta < 0.0) {
                            printf("Explosion THETA WRONG: zz = %.3f, rr = "
                                   "%.3f, theta = %.3e/PI, M_PI = %.3e\n",
                                   zz, rr, theta / M_PI, M_PI);
                        }
                    }
                    else {
                        zz_abs = -zz; // Needs a posite value for acos
                        theta  = M_PI - acos(zz_abs / rr);
                        if (theta < M_PI / 2.0 || theta > 2 * M_PI) {
                            printf("Explosion THETA WRONG: zz = %.3f, rr = "
                                   "%.3f, theta = %.3e/PI, M_PI = %.3e\n",
                                   zz, rr, theta / M_PI, M_PI);
                        }
                    }

                    // phi range [0, 2*PI]i
                    if (xx != 0.0) {
                        if (xx < 0.0 && yy >= 0.0) {
                            //-+
                            xx_abs = -xx; // Needs a posite value for atan
                            phi    = M_PI - atan(yy / xx_abs);
                            if (phi < (M_PI / 2.0) || phi > M_PI) {
                                printf("Explosion PHI WRONG -+: xx = %.3f, yy "
                                       "= %.3f, phi = %.3e/PI, M_PI = %.3e\n",
                                       xx, yy, phi / M_PI, M_PI);
                            }
                        }
                        else if (xx > 0.0 && yy < 0.0) {
                            //+-
                            yy_abs = -yy;
                            phi    = 2.0 * M_PI - atan(yy_abs / xx);
                            if (phi < (3.0 * M_PI) / 2.0 || phi > (2.0 * M_PI + 1e-6)) {
                                printf("Explosion PHI WRONG +-: xx = %.3f, yy "
                                       "= %.3f, phi = %.3e/PI, M_PI = %.3e\n",
                                       xx, yy, phi / M_PI, M_PI);
                            }
                        }
                        else if (xx < 0.0 && yy < 0.0) {
                            //--
                            yy_abs = -yy;
                            xx_abs = -xx;
                            phi    = M_PI + atan(yy_abs / xx_abs);
                            if (phi < M_PI || phi > ((3.0 * M_PI) / 2.0 + 1e-6)) {
                                printf("Explosion PHI WRONG --: xx = %.3f, yy "
                                       "= %.3f, xx_abs = %.3f, yy_abs = %.3f, "
                                       "phi = %.3e, (3.0*M_PI)/2.0 = %.3e\n",
                                       xx, yy, xx_abs, yy_abs, phi, (3.0 * M_PI) / 2.0);
                            }
                        }
                        else {
                            //++
                            phi = atan(yy / xx);
                            if (phi < 0 || phi > M_PI / 2.0) {
                                printf("Explosion PHI WRONG --: xx = %.3f, yy = "
                                       "%.3f, phi = %.3e, (3.0*M_PI)/2.0 = %.3e\n",
                                       xx, yy, phi, (3.0 * M_PI) / 2.0);
                            }
                        }
                    }
                    else { // To avoid div by zero with atan
                        if (yy > 0.0) {
                            phi = M_PI / 2.0;
                        }
                        else if (yy < 0.0) {
                            phi = (3.0 * M_PI) / 2.0;
                        }
                        else {
                            phi = 0.0;
                        }
                    }

                    // Set zero for explicit safekeeping
                    if (xx == 0.0 && yy == 0.0) {
                        phi = 0.0;
                    }

                    // Gaussian velocity
                    // uu_radial = AMPL_UU*exp( -rr2 / (2.0*pow(WIDTH_UU, 2.0))
                    // ); New distribution, where that gaussion wave is not in
                    // the exact centre coordinates uu_radial = AMPL_UU*exp(
                    // -pow((rr - 4.0*WIDTH_UU),2.0) / (2.0*pow(WIDTH_UU, 2.0))
                    // ); //TODO: Parametrize the peak location.
                    uu_radial = AMPL_UU *
                                exp(-pow((rr - UU_SHELL_R), 2.0) / (2.0 * pow(WIDTH_UU, 2.0)));
                }
                else {
                    uu_radial = 0.0; // TODO: There will be a discontinuity in
                                     // the origin... Should the shape of the
                                     // distribution be different?
                }

                // Determine the carthesian velocity components and lnrho
                uu_x[idx] = AcReal(uu_radial * sin(theta) * cos(phi));
                uu_y[idx] = AcReal(uu_radial * sin(theta) * sin(phi));
                uu_z[idx] = AcReal(uu_radial * cos(theta));

                // Temporary diagnosticv output (TODO: Remove after not needed)
                // if (theta > theta_old) {
                // if (theta > M_PI || theta < 0.0 || phi < 0.0 || phi > 2*M_PI)
                // {
                /*	printf("Explosion: xx = %.3f, yy = %.3f, zz = %.3f, rr =
                   %.3f, phi = %.3e/PI, theta = %.3e/PI\n, M_PI = %.3e", xx, yy,
                   zz, rr, phi/M_PI, theta/M_PI, M_PI); printf(" uu_radial =
                   %.3e, uu_x[%i] = %.3e, uu_y[%i] = %.3e, uu_z[%i] = %.3e \n",
                                uu_radial, idx, uu_x[idx], idx, uu_y[idx], idx,
                   uu_z[idx]); theta_old = theta;
                */
            }
        }
    }
}

void
acmesh_init_to(const InitType& init_type, AcMesh* mesh)
{
    srand(123456789);

    const int n = acVertexBufferSize(mesh->info);

    const int mx = mesh->info.int_params[AC_mx];
    const int my = mesh->info.int_params[AC_my];
    const int mz = mesh->info.int_params[AC_mz];

    const int nx_min = mesh->info.int_params[AC_nx_min];
    const int nx_max = mesh->info.int_params[AC_nx_max];
    const int ny_min = mesh->info.int_params[AC_ny_min];
    const int ny_max = mesh->info.int_params[AC_ny_max];
    const int nz_min = mesh->info.int_params[AC_nz_min];
    const int nz_max = mesh->info.int_params[AC_nz_max];

    switch (init_type) {
    case INIT_TYPE_RANDOM: {
        acmesh_clear(mesh);
        const AcReal range = AcReal(0.01);
        for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w)
            for (int i = 0; i < n; ++i)
                mesh->vertex_buffer[w][i] = 2 * range * randr() - range;

        break;
    }
    case INIT_TYPE_GAUSSIAN_RADIAL_EXPL:
        acmesh_clear(mesh);
        vertex_buffer_set(VTXBUF_LNRHO, mesh->info.real_params[AC_ampl_lnrho], mesh);
        // acmesh_init_to(INIT_TYPE_RANDOM, mesh);
        gaussian_radial_explosion(mesh);

        break;
    case INIT_TYPE_KICKBALL:
        acmesh_clear(mesh);
        vertex_buffer_set(VTXBUF_LNRHO, mesh->info.real_params[AC_ampl_lnrho], mesh);
        kickball(mesh);

        break;
    case INIT_TYPE_XWAVE:
        acmesh_clear(mesh);
        acmesh_init_to(INIT_TYPE_RANDOM, mesh);
        for (int k = 0; k < mz; k++) {
            for (int j = 0; j < my; j++) {
                for (int i = 0; i < mx; i++) {
                    int idx                              = i + j * mx + k * mx * my;
                    mesh->vertex_buffer[VTXBUF_UUX][idx] = 2 * AcReal(sin(j * AcReal(M_PI) / mx)) -
                                                           1;
                }
            }
        }
        break;
    case INIT_TYPE_SIMPLE_CORE:
        acmesh_clear(mesh);
        simple_uniform_core(mesh);
        break;
    case INIT_TYPE_VEDGE:
        acmesh_clear(mesh);
        inflow_vedge_freefall(mesh);
        break;
    case INIT_TYPE_VEDGEX:
        acmesh_clear(mesh);
        inflow_freefall_x(mesh);
        break;
    case INIT_TYPE_RAYLEIGH_TAYLOR:
        acmesh_clear(mesh);
        inflow_freefall_x(mesh);
        lnrho_step(mesh);
        break;
    case INIT_TYPE_ABC_FLOW: {
        acmesh_clear(mesh);
        acmesh_init_to(INIT_TYPE_RANDOM, mesh);
        for (int k = nz_min; k < nz_max; k++) {
            for (int j = ny_min; j < ny_max; j++) {
                for (int i = nx_min; i < nx_max; i++) {
                    const int idx = i + j * mx + k * mx * my;

                    /*
                    const double xx = double(
                        mesh->info.real_params[AC_dsx] *
                            (i - mesh->info.int_params[AC_nx_min]) -
                        XORIG + AcReal(.5) * mesh->info.real_params[AC_dsx]);
                    const double yy = double(
                        mesh->info.real_params[AC_dsy] *
                            (j - mesh->info.int_params[AC_ny_min]) -
                        YORIG + AcReal(.5) * mesh->info.real_params[AC_dsy]);
                    const double zz = double(
                        mesh->info.real_params[AC_dsz] *
                            (k - mesh->info.int_params[AC_nz_min]) -
                        ZORIG + AcReal(.5) * mesh->info.real_params[AC_dsz]);
                    */

                    const AcReal xx = (i - nx_min) * mesh->info.real_params[AC_dsx] - XORIG;
                    const AcReal yy = (j - ny_min) * mesh->info.real_params[AC_dsy] - YORIG;
                    const AcReal zz = (k - nz_min) * mesh->info.real_params[AC_dsz] - ZORIG;

                    const AcReal ampl_uu = 0.5;
                    const AcReal ABC_A   = 1.;
                    const AcReal ABC_B   = 1.;
                    const AcReal ABC_C   = 1.;
                    const AcReal kx_uu   = 8.;
                    const AcReal ky_uu   = 8.;
                    const AcReal kz_uu   = 8.;

                    mesh->vertex_buffer[VTXBUF_UUX][idx] = ampl_uu *
                                                           (ABC_A * (AcReal)sin(kz_uu * zz) +
                                                            ABC_C * (AcReal)cos(ky_uu * yy));
                    mesh->vertex_buffer[VTXBUF_UUY][idx] = ampl_uu *
                                                           (ABC_B * (AcReal)sin(kx_uu * xx) +
                                                            ABC_A * (AcReal)cos(kz_uu * zz));
                    mesh->vertex_buffer[VTXBUF_UUZ][idx] = ampl_uu *
                                                           (ABC_C * (AcReal)sin(ky_uu * yy) +
                                                            ABC_B * (AcReal)cos(kx_uu * xx));
                }
            }
        }
        break;
    }
    case INIT_TYPE_RAYLEIGH_BENARD: {
        acmesh_init_to(INIT_TYPE_RANDOM, mesh);
#if LTEMPERATURE
        vertex_buffer_set(VTXBUF_LNRHO, 1, mesh);
        const AcReal range = AcReal(0.9);
        for (int k = nz_min; k < nz_max; k++) {
            for (int j = ny_min; j < ny_max; j++) {
                for (int i = nx_min; i < nx_max; i++) {
                    const int idx                                = i + j * mx + k * mx * my;
                    mesh->vertex_buffer[VTXBUF_TEMPERATURE][idx] = (range * (k - nz_min)) /
                                                                       mesh->info
                                                                           .int_params[AC_nz] +
                                                                   0.1;
                }
            }
        }
#else
        WARNING("INIT_TYPE_RAYLEIGH_BERNARD called even though VTXBUF_TEMPERATURE is not used");
#endif
        break;
    }
    default:
        ERROR("Unknown init_type");
    }

    AcReal max_val = AcReal(-1e-32);
    AcReal min_val = AcReal(1e32);
    // Normalize the grid
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        for (int i = 0; i < n; ++i) {
            if (mesh->vertex_buffer[w][i] < min_val)
                min_val = mesh->vertex_buffer[w][i];
            if (mesh->vertex_buffer[w][i] > max_val)
                max_val = mesh->vertex_buffer[w][i];
        }
    }
    printf("MAX: %f MIN %f\n", double(max_val), double(min_val));
    /*
    const AcReal inv_range = AcReal(1.) / fabs(max_val - min_val);
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        for (int i = 0; i < n; ++i) {
            mesh->vertex_buffer[w][i] = 2*inv_range*(mesh->vertex_buffer[w][i] - min_val) - 1;
        }
    }
    */
}

void
acmesh_destroy(AcMesh* mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        free(mesh->vertex_buffer[VertexBufferHandle(i)]);

    free(mesh);
}

ModelMesh*
modelmesh_create(const AcMeshInfo& mesh_info)
{
    ModelMesh* mesh = (ModelMesh*)malloc(sizeof(*mesh));
    mesh->info      = mesh_info;

    const size_t bytes = acVertexBufferSize(mesh->info) * sizeof(mesh->vertex_buffer[0][0]);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        mesh->vertex_buffer[VertexBufferHandle(i)] = (ModelScalar*)malloc(bytes);
        ERRCHK(mesh->vertex_buffer[VertexBufferHandle(i)] != NULL);
    }

    return mesh;
}

void
modelmesh_destroy(ModelMesh* mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        free(mesh->vertex_buffer[VertexBufferHandle(i)]);

    free(mesh);
}

#include <string.h> // memcpy
void
acmesh_to_modelmesh(const AcMesh& acmesh, ModelMesh* modelmesh)
{
    ERRCHK(sizeof(acmesh.info) == sizeof(modelmesh->info));
    memcpy(&modelmesh->info, &acmesh.info, sizeof(acmesh.info));

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        for (size_t j = 0; j < acVertexBufferSize(acmesh.info); ++j)
            modelmesh->vertex_buffer[i][j] = (ModelScalar)acmesh.vertex_buffer[i][j];
}

void
modelmesh_to_acmesh(const ModelMesh& modelmesh, AcMesh* acmesh)
{
    ERRCHK(sizeof(acmesh->info) == sizeof(modelmesh.info));
    memcpy(&acmesh->info, &modelmesh.info, sizeof(modelmesh.info));

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        for (size_t j = 0; j < acVertexBufferSize(modelmesh.info); ++j)
            acmesh->vertex_buffer[i][j] = (AcReal)modelmesh.vertex_buffer[i][j];
}
