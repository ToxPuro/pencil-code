#include "grid.h"

#define DER1_2 (-3. / 20.)
#define DER1_1 ( 3. / 4.)

#define DER2_3 ( 1. / 90.)
#define DER2_2 (-3. / 20.)
#define DER2_1 ( 3. / 2.)
#define DER2_0 (-49./ 18.)

#define DERX_3 (  2. / 720.)
#define DERX_2 (-27. / 720.)
#define DERX_1 (270. / 720.)
#define DERX_0 (0)

Stencil value {
    [0][0][0] = 1
}

Stencil derx {
    [0][0][-3] = -AC_inv_dsx * DER1_3,
    [0][0][-2] = -AC_inv_dsx * DER1_2,
    [0][0][-1] = -AC_inv_dsx * DER1_1,
    [0][0][1]  = AC_inv_dsx * DER1_1,
    [0][0][2]  = AC_inv_dsx * DER1_2,
    [0][0][3]  = AC_inv_dsx * DER1_3
}

Stencil dery {
    [0][-3][0] = -AC_inv_dsy * DER1_3,
    [0][-2][0] = -AC_inv_dsy * DER1_2,
    [0][-1][0] = -AC_inv_dsy * DER1_1,
    [0][1][0]  = AC_inv_dsy * DER1_1,
    [0][2][0]  = AC_inv_dsy * DER1_2,
    [0][3][0]  = AC_inv_dsy * DER1_3
}

Stencil derz {
    [-3][0][0] = -AC_inv_dsz * DER1_3,
    [-2][0][0] = -AC_inv_dsz * DER1_2,
    [-1][0][0] = -AC_inv_dsz * DER1_1,
    [1][0][0]  = AC_inv_dsz * DER1_1,
    [2][0][0]  = AC_inv_dsz * DER1_2,
    [3][0][0]  = AC_inv_dsz * DER1_3
}

Stencil derxx {
    [0][0][-3] = AC_inv_dsx * AC_inv_dsx * DER2_3,
    [0][0][-2] = AC_inv_dsx * AC_inv_dsx * DER2_2,
    [0][0][-1] = AC_inv_dsx * AC_inv_dsx * DER2_1,
    [0][0][0]  = AC_inv_dsx * AC_inv_dsx * DER2_0,
    [0][0][1]  = AC_inv_dsx * AC_inv_dsx * DER2_1,
    [0][0][2]  = AC_inv_dsx * AC_inv_dsx * DER2_2,
    [0][0][3]  = AC_inv_dsx * AC_inv_dsx * DER2_3
}

Stencil deryy {
    [0][-3][0] = AC_inv_dsy * AC_inv_dsy * DER2_3,
    [0][-2][0] = AC_inv_dsy * AC_inv_dsy * DER2_2,
    [0][-1][0] = AC_inv_dsy * AC_inv_dsy * DER2_1,
    [0][0][0]  = AC_inv_dsy * AC_inv_dsy * DER2_0,
    [0][1][0]  = AC_inv_dsy * AC_inv_dsy * DER2_1,
    [0][2][0]  = AC_inv_dsy * AC_inv_dsy * DER2_2,
    [0][3][0]  = AC_inv_dsy * AC_inv_dsy * DER2_3
}

Stencil derzz {
    [-3][0][0] = AC_inv_dsz * AC_inv_dsz * DER2_3,
    [-2][0][0] = AC_inv_dsz * AC_inv_dsz * DER2_2,
    [-1][0][0] = AC_inv_dsz * AC_inv_dsz * DER2_1,
    [0][0][0]  = AC_inv_dsz * AC_inv_dsz * DER2_0,
    [1][0][0]  = AC_inv_dsz * AC_inv_dsz * DER2_1,
    [2][0][0]  = AC_inv_dsz * AC_inv_dsz * DER2_2,
    [3][0][0]  = AC_inv_dsz * AC_inv_dsz * DER2_3
}

Stencil derxy {
    [0][-3][-3] = AC_inv_dsx * AC_inv_dsy * DERX_3,
    [0][-2][-2] = AC_inv_dsx * AC_inv_dsy * DERX_2,
    [0][-1][-1] = AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][0][0]  = AC_inv_dsx * AC_inv_dsy * DERX_0,
    [0][1][1]  = AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][2][2]  = AC_inv_dsx * AC_inv_dsy * DERX_2,
    [0][3][3]  = AC_inv_dsx * AC_inv_dsy * DERX_3,
    [0][-3][3] = -AC_inv_dsx * AC_inv_dsy * DERX_3,
    [0][-2][2] = -AC_inv_dsx * AC_inv_dsy * DERX_2,
    [0][-1][1] = -AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][1][-1] = -AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][2][-2] = -AC_inv_dsx * AC_inv_dsy * DERX_2,
    [0][3][-3] = -AC_inv_dsx * AC_inv_dsy * DERX_3
}

Stencil derxz {
    [-3][0][-3] = AC_inv_dsx * AC_inv_dsz * DERX_3,
    [-2][0][-2] = AC_inv_dsx * AC_inv_dsz * DERX_2,
    [-1][0][-1] = AC_inv_dsx * AC_inv_dsz * DERX_1,
    [0][0][0]  = AC_inv_dsx * AC_inv_dsz * DERX_0,
    [1][0][1]  = AC_inv_dsx * AC_inv_dsz * DERX_1,
    [2][0][2]  = AC_inv_dsx * AC_inv_dsz * DERX_2,
    [3][0][3]  = AC_inv_dsx * AC_inv_dsz * DERX_3,
    [-3][0][3] = -AC_inv_dsx * AC_inv_dsz * DERX_3,
    [-2][0][2] = -AC_inv_dsx * AC_inv_dsz * DERX_2,
    [-1][0][1] = -AC_inv_dsx * AC_inv_dsz * DERX_1,
    [1][0][-1] = -AC_inv_dsx * AC_inv_dsz * DERX_1,
    [2][0][-2] = -AC_inv_dsx * AC_inv_dsz * DERX_2,
    [3][0][-3] = -AC_inv_dsx * AC_inv_dsz * DERX_3
}

Stencil deryz {
    [-3][-3][0] = AC_inv_dsy * AC_inv_dsz * DERX_3,
    [-2][-2][0] = AC_inv_dsy * AC_inv_dsz * DERX_2,
    [-1][-1][0] = AC_inv_dsy * AC_inv_dsz * DERX_1,
    [0][0][0]  = AC_inv_dsy * AC_inv_dsz * DERX_0,
    [1][1][0]  = AC_inv_dsy * AC_inv_dsz * DERX_1,
    [2][2][0]  = AC_inv_dsy * AC_inv_dsz * DERX_2,
    [3][3][0]  = AC_inv_dsy * AC_inv_dsz * DERX_3,
    [-3][3][0] = -AC_inv_dsy * AC_inv_dsz * DERX_3,
    [-2][2][0] = -AC_inv_dsy * AC_inv_dsz * DERX_2,
    [-1][1][0] = -AC_inv_dsy * AC_inv_dsz * DERX_1,
    [1][-1][0] = -AC_inv_dsy * AC_inv_dsz * DERX_1,
    [2][-2][0] = -AC_inv_dsy * AC_inv_dsz * DERX_2,
    [3][-3][0] = -AC_inv_dsy * AC_inv_dsz * DERX_3
}

