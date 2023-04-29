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
 * \brief Functions for loading and updating AcMeshInfo.
 *
 */
#pragma once
#include "astaroth.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdbool.h>

typedef struct {
    AcReal model;
    AcReal candidate;
    long double abs_error;
    long double ulp_error;
    long double rel_error;
    AcReal maximum_magnitude;
    AcReal minimum_magnitude;
} Error;

/** Loads data from the config file */
AcResult acLoadConfig(const char* config_path, AcMeshInfo* config);

/** */
AcResult acHostVertexBufferSet(const VertexBufferHandle handle, const AcReal value, AcMesh* mesh);

/** */
AcResult acHostMeshSet(const AcReal value, AcMesh* mesh);

/** */
AcResult acHostMeshApplyPeriodicBounds(AcMesh* mesh);

/** */
AcResult acHostMeshClear(AcMesh* mesh);

/** Applies a full integration step on host mesh using the compact 2N RK3 scheme. The boundaries are
 * not updated after the final substep. A call to acHostMeshApplyPeriodicBounds is required if this
 * is not desired. */
AcResult acHostIntegrateStep(AcMesh mesh, const AcReal dt);

/** */
AcReal acHostReduceScal(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a);

/** */
AcReal acHostReduceVec(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a,
                       const VertexBufferHandle b, const VertexBufferHandle c);
/** */
AcReal acHostReduceVecScal(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a,
                           const VertexBufferHandle b, const VertexBufferHandle c,
                           const VertexBufferHandle d);

Error acGetError(const AcReal model, const AcReal candidate);

bool acEvalError(const char* label, const Error error);

AcResult acVerifyMesh(const char* label, const AcMesh model, const AcMesh candidate);

AcResult acMeshDiffWriteSliceZ(const char* path, const AcMesh model, const AcMesh candidate,
                               const size_t z);

AcResult acMeshDiffWrite(const char* path, const AcMesh model, const AcMesh candidate);

AcResult acHostMeshWriteToFile(const AcMesh mesh, const size_t id);

AcResult acHostMeshReadFromFile(const size_t id, AcMesh* mesh);

// Logging utils

/* Log a message with a timestamp from the root proc (if pid == 0) */
void acLogFromRootProc(const int pid, const char* msg, ...);
void acVA_LogFromRootProc(const int pid, const char* msg, va_list args);

/* Log a message with a timestamp from the root proc (if pid == 0) if the build flag VERBOSE is on
 */
void acVerboseLogFromRootProc(const int pid, const char* msg, ...);
void acVA_VerboseLogFromRootProc(const int pid, const char* msg, va_list args);

/* Log a message with a timestamp from the root proc (if pid == 0) in a debug build */
void acDebugFromRootProc(const int pid, const char* msg, ...);
void acVA_DebugFromRootProc(const int pid, const char* msg, va_list arg);

#ifdef __cplusplus
} // extern "C"
#endif
