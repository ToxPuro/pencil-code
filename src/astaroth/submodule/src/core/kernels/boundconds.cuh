#pragma once

// TODO remove clang-format on/off
// clang-format off

extern "C" {
/**************************
 *                        *
 *   Generic boundconds   *
 *      (Any vtxbuf)      *
 *                        *
 **************************/

static __global__ void
kernel_symmetric_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{
    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + DCONST(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + DCONST(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
                              (region_id.z == 1 ? NGHOST + DCONST(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};

    const int3 boundary = int3{normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
                               normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vtxbuf[ghost_idx] = vtxbuf[domain_idx];
    }
}

AcResult
acKernelSymmetricBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                            const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_symmetric_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}

static __global__ void
kernel_antisymmetric_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                                AcReal* vtxbuf)
{
    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + DCONST(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + DCONST(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
                              (region_id.z == 1 ? NGHOST + DCONST(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};

    const int3 boundary = int3{normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
                               normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vtxbuf[ghost_idx] = -vtxbuf[domain_idx];
    }
}

AcResult
acKernelAntiSymmetricBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                                const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_antisymmetric_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}

// Boundcond "a2"
// Does not set the boundary value itself, mainly used for density

static __global__ void
kernel_a2_boundconds(const int3 region_id, const int3 normal, const int3 dims, AcReal* vtxbuf)
{

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + DCONST(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + DCONST(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
                              (region_id.z == 1 ? NGHOST + DCONST(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};

    const int3 boundary = int3{normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
                               normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};

    int boundary_idx = DEVICE_VTXBUF_IDX(boundary.x, boundary.y, boundary.z);

    AcReal boundary_val = vtxbuf[boundary_idx];

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vtxbuf[ghost_idx] = 2 * boundary_val - vtxbuf[domain_idx];
    }
}

AcResult
acKernelA2Boundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                     const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_a2_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}

#ifdef AC_INTEGRATION_ENABLED

// Constant derivative at boundary
// Sets the normal derivative at the boundary to a value

static __global__ void
kernel_prescribed_derivative_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                                        AcReal* vtxbuf, AcRealParam der_val_param)
{

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + DCONST(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + DCONST(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
                              (region_id.z == 1 ? NGHOST + DCONST(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};

    const int3 boundary = int3{normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
                               normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};

    int3 domain = boundary;
    int3 ghost  = boundary;

    AcReal d;
    AcReal direction;
    if (normal.x != 0) {
        d = DCONST(AC_dsx);
        direction = normal.x;
    }
    else if (normal.y != 0) {
        d = DCONST(AC_dsy);
        direction = normal.y;
    }
    else if (normal.z != 0) {
        d = DCONST(AC_dsz);
        direction = normal.z;
    }

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        AcReal distance = AcReal(2 * (i + 1)) * d;
        // Otherwise resulting derivatives are of different sign and opposite edges.
        if (direction < 0.0) {
            distance = -distance;
        }

        vtxbuf[ghost_idx] = vtxbuf[domain_idx] + distance * DCONST(der_val_param);
    }
}

AcResult
acKernelPrescribedDerivativeBoundconds(const cudaStream_t stream, const int3 region_id,
                                       const int3 normal, const int3 dims, AcReal* vtxbuf,
                                       AcRealParam der_val_param)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_prescribed_derivative_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims,
                                                                     vtxbuf, der_val_param);
    return AC_SUCCESS;
}

/************************
 *                      *
 *  Entropy boundconds  *
 *                      *
 ************************/

#if LENTROPY
static __global__ void
kernel_entropy_const_temperature_boundconds(const int3 region_id, const int3 normal,
                                            const int3 dims, VertexBufferArray vba)
{

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + DCONST(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + DCONST(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
                              (region_id.z == 1 ? NGHOST + DCONST(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};

    const int3 boundary = int3{normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
                               normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};

    int boundary_idx = DEVICE_VTXBUF_IDX(boundary.x, boundary.y, boundary.z);

    AcReal lnrho_diff   = vba.in[VTXBUF_LNRHO][boundary_idx] - DCONST(AC_lnrho0);
    AcReal gas_constant = DCONST(AC_cp) - DCONST(AC_cv);

    // Same as lnT(), except we are reading the values from the boundary
    AcReal lnT_boundary = DCONST(AC_lnT0) +
                          DCONST(AC_gamma) * vba.in[VTXBUF_ENTROPY][boundary_idx] /
                              DCONST(AC_cp) +
                          (DCONST(AC_gamma) - AcReal(1.)) * lnrho_diff;

    AcReal tmp = AcReal(2.0) * DCONST(AC_cv) * (lnT_boundary - DCONST(AC_lnT0));

    vba.in[VTXBUF_ENTROPY][boundary_idx] = AcReal(0.5) * tmp - gas_constant * lnrho_diff;

    // Set the values in the halo
    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vba.in[VTXBUF_ENTROPY][ghost_idx] = -vba.in[VTXBUF_ENTROPY][domain_idx] + tmp -
                                            gas_constant * (vba.in[VTXBUF_LNRHO][domain_idx] +
                                                            vba.in[VTXBUF_LNRHO][ghost_idx] -
                                                            2 * DCONST(AC_lnrho0));
    }
}

AcResult
acKernelEntropyConstantTemperatureBoundconds(const cudaStream_t stream, const int3 region_id,
                                             const int3 normal, const int3 dims,
                                             VertexBufferArray vba)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_entropy_const_temperature_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims,
                                                                         vba);
    return AC_SUCCESS;
}

static __global__ void
kernel_entropy_blackbody_radiation_kramer_conductivity_boundconds(const int3 region_id,
                                                                  const int3 normal,
                                                                  const int3 dims,
                                                                  VertexBufferArray vba)
{

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + DCONST(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + DCONST(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
                              (region_id.z == 1 ? NGHOST + DCONST(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};

    const int3 boundary = int3{normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
                               normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};

    int boundary_idx = DEVICE_VTXBUF_IDX(boundary.x, boundary.y, boundary.z);

    AcReal rho_boundary = exp(vba.in[VTXBUF_LNRHO][boundary_idx]);

    AcReal gamma_m1 = DCONST(AC_gamma) - AcReal(1.0);
    AcReal cv1      = DCONST(AC_gamma) / DCONST(AC_cp);

    // cs20*exp(gamma_m1*(f(l1,:,:,ilnrho)-lnrho0)+cv1*f(l1,:,:,iss))/(gamma_m1*cp)
    AcReal T_boundary = DCONST(AC_cs2) *
                        exp(gamma_m1 * (vba.in[VTXBUF_LNRHO][boundary_idx] - DCONST(AC_lnrho0)) +
                            cv1 * vba.in[VTXBUF_ENTROPY][boundary_idx]) /
                        gamma_m1 * DCONST(AC_cp);

    // dlnrhodx_yz= coeffs_1_x(1)*(f(l1+1,:,:,ilnrho)-f(l1-1,:,:,ilnrho)) &
    //            +coeffs_1_x(2)*(f(l1+2,:,:,ilnrho)-f(l1-2,:,:,ilnrho)) &
    //            +coeffs_1_x(3)*(f(l1+3,:,:,ilnrho)-f(l1-3,:,:,ilnrho))

    AcReal c[3] = {(AcReal(1.) / (AcReal(0.04908738521))) * (AcReal(3.) / AcReal(4.)),
                   (AcReal(1.) / (AcReal(0.04908738521))) * (-AcReal(3.) / AcReal(20.)),
                   (AcReal(1.) / (AcReal(0.04908738521))) * (AcReal(1.) / AcReal(60.))};

    AcReal der_lnrho_boundary = 0;

    int3 left       = boundary;
    int3 right      = boundary;
    int3 abs_normal = int3{abs(normal.x), abs(normal.y), abs(normal.z)};

    for (int i = 0; i < 3; i++) {
        left          = left - abs_normal;
        right         = right - abs_normal;
        int left_idx  = DEVICE_VTXBUF_IDX(left.x, left.y, left.z);
        int right_idx = DEVICE_VTXBUF_IDX(right.x, right.y, right.z);
        der_lnrho_boundary += c[i] *
                              (vba.in[VTXBUF_LNRHO][right_idx] - vba.in[VTXBUF_LNRHO][left_idx]);
    }

    // dsdx_yz=-cv*((sigmaSBt/hcond0_kramers)*TT_yz**(3-6.5*nkramers)*rho_yz**(2.*nkramers) &
    //        +gamma_m1*dlnrhodx_yz)

    AcReal der_ss_boundary = -DCONST(AC_cv) *
                                 (DCONST(AC_sigma_SBt) / DCONST(AC_hcond0_kramers)) *
                                 pow(T_boundary, AcReal(3.0) - AcReal(6.5) * DCONST(AC_n_kramers)) *
                                 pow(rho_boundary, AcReal(2.0) * DCONST(AC_n_kramers)) +
                             gamma_m1 * der_lnrho_boundary;

    AcReal d;
    if (normal.x != 0) {
        d = DCONST(AC_dsx);
    }
    else if (normal.y != 0) {
        d = DCONST(AC_dsy);
    }
    else if (normal.z != 0) {
        d = DCONST(AC_dsz);
    }

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        AcReal distance = AcReal(2 * (i + 1)) * d;

        vba.in[VTXBUF_ENTROPY][ghost_idx] = vba.in[VTXBUF_ENTROPY][domain_idx] -
                                            distance * der_ss_boundary;
    }
}

AcResult
acKernelEntropyBlackbodyRadiationKramerConductivityBoundconds(const cudaStream_t stream,
                                                              const int3 region_id,
                                                              const int3 normal, const int3 dims,
                                                              VertexBufferArray vba)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_entropy_blackbody_radiation_kramer_conductivity_boundconds<<<bpg, tpb, 0,
                                                                        stream>>>(region_id, normal,
                                                                                  dims, vba);
    return AC_SUCCESS;
}

// Prescribed heat flux

static __global__ void
kernel_entropy_prescribed_heat_flux_boundconds(const int3 region_id, const int3 normal,
                                               const int3 dims, VertexBufferArray vba,
                                               AcRealParam F_param)
{

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + DCONST(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + DCONST(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
                              (region_id.z == 1 ? NGHOST + DCONST(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};

    const int3 boundary = int3{normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
                               normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};

    int boundary_idx = DEVICE_VTXBUF_IDX(boundary.x, boundary.y, boundary.z);

#if (L_HEAT_CONDUCTION_CHICONST) || (L_HEAT_CONDUCTION_KRAMERS)
    AcReal rho_boundary = exp(vba.in[VTXBUF_LNRHO][boundary_idx]);
#endif

    AcReal cp = DCONST(AC_cp);
    AcReal cv = DCONST(AC_cv);

    AcReal gamma_m1 = DCONST(AC_gamma) - AcReal(1.0);
    AcReal cv1      = DCONST(AC_gamma) / cp;

    // cs20*exp(gamma_m1*(f(l1,:,:,ilnrho)-lnrho0)+cv1*f(l1,:,:,iss))
    AcReal cs2_boundary = DCONST(AC_cs2) *
                          exp(gamma_m1 * (vba.in[VTXBUF_LNRHO][boundary_idx] - DCONST(AC_lnrho0)) +
                              cv1 * vba.in[VTXBUF_ENTROPY][boundary_idx]);

    AcReal F_boundary = DCONST(F_param);
#if (L_HEAT_CONDUCTION_CHICONST)
    // TODO: use chi in the calculation
    AcReal chi = DCONST(AC_chi);
    AcReal tmp = F_boundary / (rho_boundary * chi * cs2_boundary);
#elif (L_HEAT_CONDUCTION_KRAMERS)
    AcReal n_kramers      = DCONST(AC_n_kramers);
    AcReal hcond0_kramers = DCONST(AC_hcond0_kramers);
    AcReal tmp            = F_boundary * pow(rho_boundary, AcReal(2.0) * n_kramers) *
                 pow(cp * gamma_m1, AcReal(6.5) * n_kramers) /
                 (hcond0_kramers * pow(cs2_boundary, AcReal(6.5) * n_kramers + AcReal(1.0)));
#else
    // NOTE: FbotKbot, FtopKtop, ... = F_param, just like Fbot, Ftop, ... = F_param
    // If both are needed, it would be preferable if they were separate boundary conditions
    // and that the switch would be between them in the main program that creates the task graph
    AcReal tmp            = F_boundary / cs2_boundary;
#endif

    AcReal d;
    if (normal.x != 0) {
        d = DCONST(AC_dsx);
    }
    else if (normal.y != 0) {
        d = DCONST(AC_dsy);
    }
    else if (normal.z != 0) {
        d = DCONST(AC_dsz);
    }

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        AcReal distance = AcReal(2 * (i + 1)) * d;

        AcReal rho_diff = vba.in[VTXBUF_LNRHO][ghost_idx] - vba.in[VTXBUF_LNRHO][domain_idx];
        vba.in[VTXBUF_ENTROPY][ghost_idx] = vba.in[VTXBUF_ENTROPY][domain_idx] +
                                            cp * (cp - cv) * (rho_diff + distance * tmp);
    }
}

AcResult
acKernelEntropyPrescribedHeatFluxBoundconds(const cudaStream_t stream, const int3 region_id,
                                            const int3 normal, const int3 dims,
                                            VertexBufferArray vba, AcRealParam F_param)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_entropy_prescribed_heat_flux_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims,
                                                                            vba, F_param);
    return AC_SUCCESS;
}

// Prescribed normal + turbulent heat flux

static __global__ void
kernel_entropy_prescribed_normal_and_turbulent_heat_flux_boundconds(
    const int3 region_id, const int3 normal, const int3 dims, VertexBufferArray vba,
    AcRealParam hcond_param, AcRealParam F_param)
{

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + DCONST(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + DCONST(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
                              (region_id.z == 1 ? NGHOST + DCONST(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};

    const int3 boundary = int3{normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
                               normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};

    int boundary_idx = DEVICE_VTXBUF_IDX(boundary.x, boundary.y, boundary.z);

    AcReal gamma_m1 = DCONST(AC_gamma) - AcReal(1.0);
    AcReal cv1      = DCONST(AC_gamma) / DCONST(AC_cp);

    // cs20*exp(gamma_m1*(f(l1,:,:,ilnrho)-lnrho0)+cv1*f(l1,:,:,iss))/(gamma_m1*cp)
    AcReal T_boundary = DCONST(AC_cs2) *
                        exp(gamma_m1 * (vba.in[VTXBUF_LNRHO][boundary_idx] - DCONST(AC_lnrho0)) +
                            cv1 * vba.in[VTXBUF_ENTROPY][boundary_idx]) /
                        gamma_m1 * DCONST(AC_cp);

    AcReal rho_boundary = exp(vba.in[VTXBUF_LNRHO][boundary_idx]);
#if (L_HEAT_CONDUCTION_CHICONST) || (L_HEAT_CONDUCTION_KRAMERS)
    AcReal cv           = DCONST(AC_cv_sound);
#endif

#if (L_HEAT_CONDUCTION_CHICONST)
    // TODO: use chi in the calculation
    AcReal chi = DCONST(AC_chi);
    AcReal K   = chi * rho_boundary * cv;
#elif (L_HEAT_CONDUCTION_KRAMERS)
    AcReal n_kramers      = DCONST(AC_n_kramers);
    AcReal hcond0_kramers = DCONST(AC_hcond0_kramers);
    AcReal K              = hcond0_kramers * pow(T_boundary, AcReal(6.5) * n_kramers) /
               pow(rho_boundary, AcReal(2.0) * n_kramers);
#else
    AcReal hcond_boundary = DCONST(hcond_param);
    AcReal K              = hcond_boundary;
#endif

    AcReal F_boundary  = DCONST(F_param);
    AcReal chi_t_prof1 = DCONST(AC_chi_t_prof1);
    AcReal chi_t       = DCONST(AC_chi_t);

    AcReal der_s_boundary = (F_boundary / T_boundary) /
                            (chi_t_prof1 * chi_t * rho_boundary + K * cv1);

    AcReal d;
    if (normal.x != 0) {
        d = DCONST(AC_dsx);
    }
    else if (normal.y != 0) {
        d = DCONST(AC_dsy);
    }
    else if (normal.z != 0) {
        d = DCONST(AC_dsz);
    }

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        AcReal der_lnrho = vba.in[VTXBUF_LNRHO][domain_idx] - vba.in[VTXBUF_LNRHO][ghost_idx];

        AcReal distance = AcReal(2 * (i + 1)) * d;

        vba.in[VTXBUF_ENTROPY][ghost_idx] = vba.in[VTXBUF_ENTROPY][domain_idx] +
                                            K * gamma_m1 * der_lnrho /
                                                (K * cv1 + chi_t_prof1 * chi_t * rho_boundary) +
                                            distance * der_s_boundary;
    }
}

AcResult
acKernelEntropyPrescribedNormalAndTurbulentHeatFluxBoundconds(
    const cudaStream_t stream, const int3 region_id, const int3 normal, const int3 dims,
    VertexBufferArray vba, AcRealParam hcond_param, AcRealParam F_param)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_entropy_prescribed_normal_and_turbulent_heat_flux_boundconds<<<
        bpg, tpb, 0, stream>>>(region_id, normal, dims, vba, hcond_param, F_param);
    return AC_SUCCESS;
}
#endif

#else
AcResult
acKernelPrescribedDerivativeBoundconds(const cudaStream_t stream, const int3 region_id,
                                       const int3 normal, const int3 dims, AcReal* vtxbuf,
                                       AcRealParam der_val_param)
{
    fprintf(stderr, "acKernelPrescribedDerivativeBoundconds() called but AC_INTEGRATION_ENABLED "
                    "was false\n");
    return AC_FAILURE;
}

#endif // AC_INTEGRATION_ENABLED
} // extern "C"

// clang-format on