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

#include <stdint.h> // uint8_t, uint32_t
#include <string.h>

#include "errchk.h"

/**
 \brief Find the index of the keyword in names
 \return Index in range 0...n if the keyword is in names. -1 if the keyword was
 not found.
 */
static int
find_str(const char keyword[], const char* names[], const int n)
{
    for (int i = 0; i < n; ++i)
        if (!strcmp(keyword, names[i]))
            return i;

    return -1;
}

static bool
is_bctype(const int idx)
{
    return idx == AC_bc_type_top_x || idx == AC_bc_type_bot_x || //
           idx == AC_bc_type_top_y || idx == AC_bc_type_bot_y || //
           idx == AC_bc_type_top_z || idx == AC_bc_type_bot_z;
}

static bool
is_initcondtype(const int idx)
{
    return idx == AC_init_type;
}

static int
parse_intparam(const size_t idx, const char* value)
{
    if (is_bctype(idx)) {
        int bctype = -1;
        if ((bctype = find_str(value, bctype_names, NUM_BCTYPES)) >= 0)
            return bctype;
        else {
            fprintf(stderr,
                    "ERROR PARSING CONFIG: Invalid BC type: %s, do not know what to do with it.\n",
                    value);
            fprintf(stdout, "Valid BC types:\n");
            acQueryBCtypes();
            ERROR("Invalid boundary condition type found in config");
            return 0;
        }
    }
    else if (is_initcondtype(idx)) {
        int initcondtype = -1;
        if ((initcondtype = find_str(value, initcondtype_names, NUM_INIT_TYPES)) >= 0)
            return initcondtype;
        else {
            fprintf(stderr,
                    "ERROR PARSING CONFIG: Invalid initial condition type: %s, do not know what to "
                    "do with it.\n",
                    value);
            fprintf(stdout, "Valid initial condition types:\n");
            acQueryInitcondtypes();
            ERROR("Invalid initial condition type found in config");
            return 0;
        }
    }
    else {
        return atoi(value);
    }
}

static void
parse_config(const char* path, AcMeshInfo* config)
{
    FILE* fp;
    fp = fopen(path, "r");
    // For knowing which .conf file will be used
    printf("Config file path: %s\n", path);
    ERRCHK_ALWAYS(fp != NULL);

    const size_t BUF_SIZE = 128;
    char keyword[BUF_SIZE];
    char value[BUF_SIZE];
    int items_matched;
    while ((items_matched = fscanf(fp, "%s = %s", keyword, value)) != EOF) {

        if (items_matched < 2)
            continue;

        int idx = -1;
        if ((idx = find_str(keyword, intparam_names, NUM_INT_PARAMS)) >= 0) {
            config->int_params[idx] = parse_intparam(idx, value);
        }
        else if ((idx = find_str(keyword, realparam_names, NUM_REAL_PARAMS)) >= 0) {
            AcReal real_val = atof(value);
            if (isnan(real_val)) {
                fprintf(stderr,
                        "ERROR PARSING CONFIG: parameter \"%s\" value \"%s\" parsed as NAN\n",
                        keyword, value);
            }
            // OL: should we fail here? Could be dangerous to continue
            config->real_params[idx] = real_val;
        }
    }

    fclose(fp);
}

/**
\brief Loads data from astaroth.conf into a config struct.
\return AC_SUCCESS on success, AC_FAILURE if there are potentially uninitialized values.
*/
AcResult
acLoadConfig(const char* config_path, AcMeshInfo* config)
{
    ERRCHK_ALWAYS(config_path);

    // memset reads the second parameter as a byte even though it says int in
    // the function declaration
    memset(config, (uint8_t)0xFF, sizeof(*config));

    parse_config(config_path, config);
    acHostUpdateBuiltinParams(config);
#if AC_VERBOSE
    printf("###############################################################\n");
    printf("Config dimensions loaded:\n");
    acPrintMeshInfo(*config);
    printf("###############################################################\n");
#endif

    // sizeof(config) must be a multiple of 4 bytes for this to work
    ERRCHK_ALWAYS(sizeof(*config) % sizeof(uint32_t) == 0);

    // Check for uninitialized config values
    bool uninitialized_config_val = false;
    for (size_t i = 0; i < sizeof(*config) / sizeof(uint32_t); ++i) {
        uninitialized_config_val |= ((uint32_t*)config)[i] == (uint32_t)0xFFFFFFFF;
    }

#if AC_VERBOSE
    if (uninitialized_config_val) {
        fprintf(stderr, "Some config values may be uninitialized. "
                        "See that all are defined in astaroth.conf\n");
    }
#endif

    return uninitialized_config_val ? AC_FAILURE : AC_SUCCESS;
}
