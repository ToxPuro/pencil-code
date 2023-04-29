#pragma once

#include "astaroth.h"
#include "astaroth_utils.h"

#define DER1_3 (AcReal)(1. / 60.)
#define DER1_2 (AcReal)(-3. / 20.)
#define DER1_1 (AcReal)(3. / 4.)
#define DER1_0 (AcReal)(0)

#define DER2_3 (AcReal)(1. / 90.)
#define DER2_2 (AcReal)(-3. / 20.)
#define DER2_1 (AcReal)(3. / 2.)
#define DER2_0 (AcReal)(-49. / 18.)

#define DERX_3 (AcReal)(2. / 720.)
#define DERX_2 (AcReal)(-27. / 720.)
#define DERX_1 (AcReal)(270. / 720.)
#define DERX_0 (AcReal)(0)

#define DER6UPWD_3 (1. / 60.)
#define DER6UPWD_2 (-6. / 60.)
#define DER6UPWD_1 (15. / 60.)
#define DER6UPWD_0 (-20. / 60.)

#define MID (STENCIL_ORDER / 2)

void
load_stencil_from_config(const AcMeshInfo info)
{
    AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH] = {{{{0}}}};

    // Midpoint
    stencils[stencil_value][MID][MID][MID] = 1;

    AcReal der1[]     = {-DER1_3, -DER1_2, -DER1_1, DER1_0, DER1_1, DER1_2, DER1_3};
    AcReal der2[]     = {DER2_3, DER2_2, DER2_1, DER2_0, DER2_1, DER2_2, DER2_3};
    AcReal derx[]     = {DERX_3, DERX_2, DERX_1, DERX_0, DERX_1, DERX_2, DERX_3};
    AcReal der6upwd[] = {DER6UPWD_3, DER6UPWD_2, DER6UPWD_1, DER6UPWD_0,
                         DER6UPWD_1, DER6UPWD_2, DER6UPWD_3};

    const AcReal inv_dsx = AcReal(1.0) / info.real_params[AC_dsx];
    const AcReal inv_dsy = AcReal(1.0) / info.real_params[AC_dsy];
    const AcReal inv_dsz = AcReal(1.0) / info.real_params[AC_dsz];

    // 1st and 2nd order derivatives
    for (size_t i = 0; i < STENCIL_ORDER + 1; ++i) {
        stencils[stencil_derx][MID][MID][i]  = inv_dsx * der1[i];
        stencils[stencil_derxx][MID][MID][i] = inv_dsx * inv_dsx * der2[i];
    }

    for (size_t i = 0; i < STENCIL_ORDER + 1; ++i) {
        stencils[stencil_dery][MID][i][MID]  = inv_dsy * der1[i];
        stencils[stencil_deryy][MID][i][MID] = inv_dsy * inv_dsy * der2[i];
    }

    for (size_t i = 0; i < STENCIL_ORDER + 1; ++i) {
        stencils[stencil_derz][i][MID][MID]  = inv_dsz * der1[i];
        stencils[stencil_derzz][i][MID][MID] = inv_dsz * inv_dsz * der2[i];
    }

    // Cross derivatives
    for (size_t i = 0; i < STENCIL_ORDER + 1; ++i) {
        stencils[stencil_derxy][MID][i][i] = inv_dsx * inv_dsy * derx[i];
        if (i != MID)
            stencils[stencil_derxy][MID][i][STENCIL_ORDER - i] = -inv_dsx * inv_dsy * derx[i];
    }

    for (size_t i = 0; i < STENCIL_ORDER + 1; ++i) {
        stencils[stencil_deryz][i][i][MID] = inv_dsy * inv_dsz * derx[i];
        if (i != MID)
            stencils[stencil_deryz][i][STENCIL_ORDER - i][MID] = -inv_dsy * inv_dsz * derx[i];
    }

    for (size_t i = 0; i < STENCIL_ORDER + 1; ++i) {
        stencils[stencil_derxz][i][MID][i] = inv_dsx * inv_dsz * derx[i];
        if (i != MID)
            stencils[stencil_derxz][i][MID][STENCIL_ORDER - i] = -inv_dsx * inv_dsz * derx[i];
    }

    // Upwinding
    for (size_t i = 0; i < STENCIL_ORDER + 1; ++i) {
        stencils[stencil_der6x_upwd][MID][MID][i] = inv_dsx * der6upwd[i];
        stencils[stencil_der6y_upwd][MID][i][MID] = inv_dsy * der6upwd[i];
        stencils[stencil_der6z_upwd][i][MID][MID] = inv_dsz * der6upwd[i];
    }

    // Load all stencils at once
    acGridLoadStencils(STREAM_DEFAULT, stencils);
}
