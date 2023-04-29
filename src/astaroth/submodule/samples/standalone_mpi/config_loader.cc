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
#include "config_loader.h"

#include <stdint.h> // uint8_t, uint32_t
#include <stdio.h>  // print
#include <string.h> // memset

#include "errchk.h"
#include "math_utils.h"

void
set_extra_config_params(AcMeshInfo* config)
{

    // Spacing
    /*
    // %JP: AC_inv_ds[xyz] now calculated inside the mhd kernel
    config->real_params[AC_inv_dsx] = AcReal(1.) / config->real_params[AC_dsx];
    config->real_params[AC_inv_dsy] = AcReal(1.) / config->real_params[AC_dsy];
    config->real_params[AC_inv_dsz] = AcReal(1.) / config->real_params[AC_dsz];
    */
    config->real_params[AC_dsmin] = min(config->real_params[AC_dsx],
                                        min(config->real_params[AC_dsy],
                                            config->real_params[AC_dsz]));

    // Real grid coordanates (DEFINE FOR GRID WITH THE GHOST ZONES)
    config->real_params[AC_xlen] = config->real_params[AC_dsx] * config->int_params[AC_mx];
    config->real_params[AC_ylen] = config->real_params[AC_dsy] * config->int_params[AC_my];
    config->real_params[AC_zlen] = config->real_params[AC_dsz] * config->int_params[AC_mz];

    config->real_params[AC_xorig] = AcReal(.5) * config->real_params[AC_xlen];
    config->real_params[AC_yorig] = AcReal(.5) * config->real_params[AC_ylen];
    config->real_params[AC_zorig] = AcReal(.5) * config->real_params[AC_zlen];

    // Real helpers
    config->real_params[AC_cs2_sound] = config->real_params[AC_cs_sound] *
                                        config->real_params[AC_cs_sound];

    config->real_params[AC_cv_sound] = config->real_params[AC_cp_sound] /
                                       config->real_params[AC_gamma];

    AcReal G_CONST_CGS = AcReal(
        6.674e-8); // cm^3/(g*s^2) GGS definition //TODO define in a separate module
    AcReal M_sun = AcReal(1.989e33); // g solar mass

    config->real_params[AC_unit_mass] = (config->real_params[AC_unit_length] *
                                         config->real_params[AC_unit_length] *
                                         config->real_params[AC_unit_length]) *
                                        config->real_params[AC_unit_density];

    config->real_params[AC_M_sink] = config->real_params[AC_M_sink_Msun] * M_sun /
                                     config->real_params[AC_unit_mass];
    config->real_params[AC_M_sink_init] = config->real_params[AC_M_sink_Msun] * M_sun /
                                          config->real_params[AC_unit_mass];

    config->real_params[AC_G_const] = G_CONST_CGS / ((config->real_params[AC_unit_velocity] *
                                                      config->real_params[AC_unit_velocity]) /
                                                     (config->real_params[AC_unit_density] *
                                                      config->real_params[AC_unit_length] *
                                                      config->real_params[AC_unit_length]));

    config->real_params[AC_sq2GM_star] = AcReal(sqrt(AcReal(2) * config->real_params[AC_GM_star]));

#if VERBOSE_PRINTING // Defined in astaroth.h
    printf("###############################################################\n");
    printf("Config dimensions recalculated:\n");
    acPrintMeshInfo(*config);
    printf("###############################################################\n");
#endif
}
