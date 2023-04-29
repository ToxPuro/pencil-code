/*
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

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
#pragma once

/*
 * =============================================================================
 * Logical switches
 * =============================================================================
 */
#if AC_DOUBLE_PRECISION == 1
  #define DOUBLE_PRECISION
#endif

  #include "/homeappl/home/mreinhar/git/pencil-code/samples/gputest/src/cparam_c.h"
  #include "/homeappl/home/mreinhar/git/pencil-code/samples/gputest/src/cdata_c.h"
  #define STENCIL_ORDER (2*NGHOST)

  #include "/homeappl/home/mreinhar/git/pencil-code/samples/gputest/src/astaroth/PC_moduleflags.h"

  #define CONFIG_PATH
  #define LUPWD (0)
/*
 * =============================================================================
 * User-defined parameters
 * =============================================================================
 */
// clang-format off
#if LFORCING

#define AC_FOR_USER_INT_PARAM_TYPES(FUNC)\
        FUNC(AC_iforcing_zsym), 

#define AC_FORCING_USER_REAL_PARAM_TYPES(FUNC) \
        FUNC(AC_k1_ff), \
        FUNC(AC_fact), \
        FUNC(AC_phase),

#else
#define AC_FOR_USER_INT_PARAM_TYPES(FUNC)
#define AC_FORCING_USER_REAL_PARAM_TYPES(FUNC)
#endif

#define AC_FOR_USER_INT3_PARAM_TYPES(FUNC)

#define AC_BASIC_USER_REAL_PARAM_TYPES(FUNC)\
        /* cparams */\
        FUNC(AC_dt), \
        FUNC(AC_dsx), \
        FUNC(AC_dsy), \
        FUNC(AC_dsz), \
        FUNC(AC_dsmin), \
        /* physical grid*/\
        FUNC(AC_xlen), \
        FUNC(AC_ylen), \
        FUNC(AC_zlen), \
        FUNC(AC_xorig), \
        FUNC(AC_yorig), \
        FUNC(AC_zorig), \
        /*Physical units*/\
        FUNC(AC_unit_density),\
        FUNC(AC_unit_velocity),\
        FUNC(AC_unit_length),\
        /* properties of gravitating star*/\
        FUNC(AC_star_pos_x),\
        FUNC(AC_star_pos_y),\
        FUNC(AC_star_pos_z),\
        FUNC(AC_M_star),
        /* Run params */

#define AC_BASIC_USER_SCALARARRAY_HANDLES(FUNC)

#if LVISCOSITY
#define AC_VISCOSITY_USER_REAL_PARAM_TYPES(FUNC)\
        FUNC(AC_nu_visc), \
        FUNC(AC_zeta),
#else
#define AC_VISCOSITY_USER_REAL_PARAM_TYPES(FUNC)
#endif
#define AC_VISCOSITY_USER_SCALARARRAY_HANDLES(FUNC)

#if LMAGNETIC
#define AC_MAGNETIC_USER_REAL_PARAM_TYPES(FUNC)\
        FUNC(AC_eta),
#else
#define AC_MAGNETIC_USER_REAL_PARAM_TYPES(FUNC)
#endif
#define AC_MAGNETIC_USER_SCALARARRAY_HANDLES(FUNC)

#if LENTROPY
#define AC_ENTROPY_USER_REAL_PARAM_TYPES(FUNC) \
        FUNC(AC_chi), 
#else
#define AC_ENTROPY_USER_REAL_PARAM_TYPES(FUNC)
#endif
#define AC_ENTROPY_USER_SCALARARRAY_HANDLES(FUNC)

#define AC_EOS_USER_REAL_PARAM_TYPES(FUNC) \
        FUNC(AC_mu0), \
        FUNC(AC_cs_sound), \
        FUNC(AC_cp_sound), \
        FUNC(AC_gamma), \
        FUNC(AC_cv_sound), \
        FUNC(AC_lnT0), \
        FUNC(AC_lnrho0), \
        /* Additional helper params */
        /* (deduced from other params do not set these directly!) */

#define AC_EOS_USER_SCALARARRAY_HANDLES(FUNC)

#define AC_HELPER_USER_REAL_PARAM_TYPES(FUNC) \
        FUNC(AC_G_CONST),\
        FUNC(AC_GM_star),\
        FUNC(AC_sq2GM_star),\
        FUNC(AC_cs2_sound), \
        FUNC(AC_inv_dsx), \
        FUNC(AC_inv_dsy), \
        FUNC(AC_inv_dsz),

#define AC_HELPER_USER_SCALARARRAY_HANDLES(FUNC)

#define AC_FOR_BUILTIN_REAL3_PARAM_TYPES(FUNC)

#if LFORCING
#define AC_FOR_USER_REAL3_PARAM_TYPES(FUNC) \
        FUNC(AC_coef1), \
        FUNC(AC_coef2), \
        FUNC(AC_coef3), \
        FUNC(AC_fda), \
        FUNC(AC_kk), 
#define AC_FORCING_USER_SCALARARRAY_HANDLES(FUNC) \
        FUNC(AC_profx_ampl), \
        FUNC(AC_profy_ampl), \
        FUNC(AC_profz_ampl), \
        FUNC(AC_profx_hel), \
        FUNC(AC_profy_hel), \
        FUNC(AC_profz_hel),
#else
#define AC_FOR_USER_REAL3_PARAM_TYPES(FUNC) 
#define AC_FORCING_USER_SCALARARRAY_HANDLES(FUNC)
#endif

#define AC_FOR_BUILTIN_REAL_PARAM_TYPES(FUNC)

#define AC_FOR_USER_REAL_PARAM_TYPES(FUNC) \
        AC_BASIC_USER_REAL_PARAM_TYPES(FUNC) \
        AC_HELPER_USER_REAL_PARAM_TYPES(FUNC) \
        AC_VISCOSITY_USER_REAL_PARAM_TYPES(FUNC) \
        AC_EOS_USER_REAL_PARAM_TYPES(FUNC) \
        AC_ENTROPY_USER_REAL_PARAM_TYPES(FUNC) \
        AC_MAGNETIC_USER_REAL_PARAM_TYPES(FUNC) \
        AC_FORCING_USER_REAL_PARAM_TYPES(FUNC)

#define AC_FOR_SCALARARRAY_HANDLES(FUNC) \
        AC_BASIC_USER_SCALARARRAY_HANDLES(FUNC) \
        AC_HELPER_USER_SCALARARRAY_HANDLES(FUNC) \
        AC_VISCOSITY_USER_SCALARARRAY_HANDLES(FUNC) \
        AC_EOS_USER_SCALARARRAY_HANDLES(FUNC) \
        AC_ENTROPY_USER_SCALARARRAY_HANDLES(FUNC) \
        AC_MAGNETIC_USER_SCALARARRAY_HANDLES(FUNC) \
        AC_FORCING_USER_SCALARARRAY_HANDLES(FUNC)

// clang-format on

/*
 * =============================================================================
 * User-defined vertex buffers
 * =============================================================================
 */
// clang-format off
#if LHYDRO
  #define AC_HYDRO_VTXBUF_HANDLES(FUNC) \
    FUNC(VTXBUF_UUX), \
    FUNC(VTXBUF_UUY), \
    FUNC(VTXBUF_UUZ),
#else
  #define AC_HYDRO_VTXBUF_HANDLES(FUNC)
#endif

#if LDENSITY
  #define AC_DENSITY_VTXBUF_HANDLES(FUNC) \
    FUNC(VTXBUF_LNRHO),
#else
  #define AC_DENSITY_VTXBUF_HANDLES(FUNC)
#endif

#if LENTROPY
  #define AC_ENTROPY_VTXBUF_HANDLES(FUNC) \
    FUNC(VTXBUF_ENTROPY),
#else
  #define AC_ENTROPY_VTXBUF_HANDLES(FUNC)
#endif

#if LMAGNETIC
  #define AC_MAGNETIC_VTXBUF_HANDLES(FUNC) \
    FUNC(VTXBUF_AX), \
    FUNC(VTXBUF_AY), \
    FUNC(VTXBUF_AZ),
#else
  #define AC_MAGNETIC_VTXBUF_HANDLES(FUNC)
#endif

#define AC_FOR_VTXBUF_HANDLES(FUNC) \
  AC_HYDRO_VTXBUF_HANDLES(FUNC) \
  AC_DENSITY_VTXBUF_HANDLES(FUNC) \
  AC_ENTROPY_VTXBUF_HANDLES(FUNC) \
  AC_MAGNETIC_VTXBUF_HANDLES(FUNC) 

#define USER_DEFINED
// clang-format on
