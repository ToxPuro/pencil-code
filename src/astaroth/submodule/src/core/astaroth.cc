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
#include "astaroth.h"
#include <string.h> // strcmp

#include "math_utils.h"
#include "../../acc-runtime/api/math_utils.h"

static const int max_num_nodes   = 1;
static Node nodes[max_num_nodes] = {0};
static int num_nodes             = 0;

AcResult
acInit(const AcMeshInfo mesh_info, int rank)
{
    num_nodes = 1;
    return acNodeCreate(0, mesh_info, &nodes[0], rank);
}

AcResult
acQuit(void)
{
    ERRCHK_ALWAYS(num_nodes);
    num_nodes = 0;
    return acNodeDestroy(nodes[0]);
}
#if PACKED_DATA_TRANSFERS 
AcResult
acLoadPlate(const int3& start, const int3& end, AcMesh* host_mesh, AcReal* plateBuffer, PlateType plate)
{
    return acNodeLoadPlate(nodes[0], STREAM_DEFAULT, start, end, host_mesh, plateBuffer, plate);
}
#endif 
AcResult
acCheckDeviceAvailability(void)
{
    int device_count; // Separate from num_devices to avoid side effects
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(&device_count));
    if (device_count > 0)
        return AC_SUCCESS;
    else
        return AC_FAILURE;
}

AcResult
acSynchronize(void)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeStream(nodes[0], STREAM_ALL);
}

AcResult
acSynchronizeStream(const Stream stream)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeStream(nodes[0], stream);
}

AcResult
acLoadDeviceConstant(const AcRealParam param, const AcReal value)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadConstant(nodes[0], STREAM_DEFAULT, param, value);
}
/*
AcResult
acLoadVectorConstant(const AcReal3Param param, const AcReal3 value)
{   
    return acNodeLoadVectorConstant(nodes[0], STREAM_DEFAULT, param, value);
}       
*/  
AcResult
acLoad(const AcMesh host_mesh)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadMesh(nodes[0], STREAM_DEFAULT, host_mesh);
}

AcResult
acSetVertexBuffer(const VertexBufferHandle handle, const AcReal value)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSetVertexBuffer(nodes[0], STREAM_DEFAULT, handle, value);
}

AcResult
acStore(AcMesh* host_mesh)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeStoreMesh(nodes[0], STREAM_DEFAULT, host_mesh);
}

AcResult
acIntegrate(const AcReal dt)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrate(nodes[0], dt);
}

AcResult
acIntegrateGBC(const AcMeshInfo config, const AcReal dt)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrateGBC(nodes[0], config, dt);
}

AcResult
acIntegrateStep(const int isubstep, const AcReal dt)
{
    ERRCHK_ALWAYS(num_nodes);
    DeviceConfiguration config;
    acNodeQueryDeviceConfiguration(nodes[0], &config);

    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = start + config.grid.n;
    return acNodeIntegrateSubstep(nodes[0], STREAM_DEFAULT, isubstep, start, end, dt);
}

AcResult
acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const int3 start, const int3 end)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrateSubstep(nodes[0], STREAM_DEFAULT, isubstep, start, end, dt);
}

AcResult
acBoundcondStep(void)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodePeriodicBoundconds(nodes[0], STREAM_DEFAULT);
}

AcResult
acBoundcondStepGBC(const AcMeshInfo config)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeGeneralBoundconds(nodes[0], STREAM_DEFAULT, config);
}

AcReal
acReduceScal(const ReductionType rtype, const VertexBufferHandle vtxbuf_handle)
{
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceScal(nodes[0], STREAM_DEFAULT, rtype, vtxbuf_handle, &result);
    return result;
}

AcReal
acReduceVec(const ReductionType rtype, const VertexBufferHandle a, const VertexBufferHandle b,
            const VertexBufferHandle c)
{
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceVec(nodes[0], STREAM_DEFAULT, rtype, a, b, c, &result);
    return result;
}

AcReal
acReduceVecScal(const ReductionType rtype, const VertexBufferHandle a, const VertexBufferHandle b,
                const VertexBufferHandle c, const VertexBufferHandle d)
{
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceVecScal(nodes[0], STREAM_DEFAULT, rtype, a, b, c, d, &result);
    return result;
}

AcResult
acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeStoreMeshWithOffset(nodes[0], STREAM_DEFAULT, dst, dst, num_vertices, host_mesh);
}

AcResult
acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadMeshWithOffset(nodes[0], STREAM_DEFAULT, host_mesh, src, src, num_vertices);
}

AcResult
acSynchronizeMesh(void)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeMesh(nodes[0], STREAM_DEFAULT);
}

int
acGetNumDevicesPerNode(void)
{
    int num_devices;
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(&num_devices));
    return num_devices;
}

size_t
acGetNumFields(void)
{
    return NUM_VTXBUF_HANDLES;
}

AcResult
acGetFieldHandle(const char* field, size_t* handle)
{
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        if (!strcmp(field, field_names[i])) {
            *handle = i;
            return AC_SUCCESS;
        }
    }

    *handle = SIZE_MAX;
    return AC_FAILURE;
}

Node
acGetNode(void)
{
    ERRCHK_ALWAYS(num_nodes > 0);
    return nodes[0];
}

AcResult
acHostUpdateBuiltinParams(AcMeshInfo* config)
{
    ERRCHK_ALWAYS(config->int_params[AC_nx] > 0);
    ERRCHK_ALWAYS(config->int_params[AC_ny] > 0);
    ERRCHK_ALWAYS(config->int_params[AC_nz] > 0);

    config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER;
    ///////////// PAD TEST
    // config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER + PAD_SIZE;
    ///////////// PAD TEST
    config->int_params[AC_my] = config->int_params[AC_ny] + STENCIL_ORDER;
    config->int_params[AC_mz] = config->int_params[AC_nz] + STENCIL_ORDER;

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    config->int_params[AC_nx_min] = STENCIL_ORDER / 2;
    config->int_params[AC_ny_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nz_min] = STENCIL_ORDER / 2;

    config->int_params[AC_nx_max] = config->int_params[AC_nx_min] + config->int_params[AC_nx];
    config->int_params[AC_ny_max] = config->int_params[AC_ny_min] + config->int_params[AC_ny];
    config->int_params[AC_nz_max] = config->int_params[AC_nz_min] + config->int_params[AC_nz];

    /*
    #ifdef AC_dsx
        printf("HELLO!\n");
        ERRCHK_ALWAYS(config->real_params[AC_dsx] > 0);
        config->real_params[AC_inv_dsx] = (AcReal)(1.) / config->real_params[AC_dsx];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsx]));
    #endif
    #ifdef AC_dsy
        ERRCHK_ALWAYS(config->real_params[AC_dsy] > 0);
        config->real_params[AC_inv_dsy] = (AcReal)(1.) / config->real_params[AC_dsy];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsy]));
    #endif
    #ifdef AC_dsz
        ERRCHK_ALWAYS(config->real_params[AC_dsz] > 0);
        config->real_params[AC_inv_dsz] = (AcReal)(1.) / config->real_params[AC_dsz];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsz]));
    #endif
    */

    /* Additional helper params */
    // Int helpers
    config->int_params[AC_mxy]  = config->int_params[AC_mx] * config->int_params[AC_my];
    config->int_params[AC_nxy]  = config->int_params[AC_nx] * config->int_params[AC_ny];
    config->int_params[AC_nxyz] = config->int_params[AC_nxy] * config->int_params[AC_nz];

    return AC_SUCCESS;
}

AcResult
acSetMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info)
{
    info->int_params[AC_nx] = nx;
    info->int_params[AC_ny] = ny;
    info->int_params[AC_nz] = nz;
    return acHostUpdateBuiltinParams(info);
}

AcResult
acHostMeshCreate(const AcMeshInfo info, AcMesh* mesh)
{
    mesh->info = info;

    const size_t n_cells = acVertexBufferSize(mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        mesh->vertex_buffer[w] = (AcReal*)calloc(n_cells, sizeof(AcReal));
        ERRCHK_ALWAYS(mesh->vertex_buffer[w]);
    }

    return AC_SUCCESS;
}

static AcReal
randf(void)
{
    // TODO: rand() considered harmful, replace
    return (AcReal)rand() / (AcReal)RAND_MAX;
}

AcResult
acHostMeshRandomize(AcMesh* mesh)
{
    const size_t n = acVertexBufferSize(mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        for (size_t i = 0; i < n; ++i) {
            mesh->vertex_buffer[w][i] = randf();
        }
    }

    return AC_SUCCESS;
}

AcResult
acHostMeshDestroy(AcMesh* mesh)
{
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        free(mesh->vertex_buffer[w]);

    return AC_SUCCESS;
}
