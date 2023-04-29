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
#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"

int
main(void)
{
    ERRCHK_ALWAYS(acCheckDeviceAvailability() == AC_SUCCESS);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    // Alloc
    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);

    // Init
    acHostMeshRandomize(&model);
    acHostMeshApplyPeriodicBounds(&model);

    // Verify that the mesh was loaded and stored correctly
    acInit(info);
    acLoad(model);
    acStore(&candidate);
    acVerifyMesh("Load/Store", model, candidate);

    // Attempt to integrate and check max and min
    printf("Integrating... ");
    acIntegrate((AcReal)FLT_EPSILON);

    printf("Done.\nVTXBUF ranges after one integration step:\n");
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        printf("\t%-15s... [%.3g, %.3g]\n", vtxbuf_names[i], //
               (double)acReduceScal(RTYPE_MIN, i),           //
               (double)acReduceScal(RTYPE_MAX, i));

    // Destroy
    acQuit();
    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);

    puts("ctest complete.");
    return EXIT_SUCCESS;
}
