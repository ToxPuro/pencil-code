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
#pragma once
#include "astaroth.h"
#include "modelmesh.h"

/*
// clang-format off
#define AC_FOR_INIT_TYPES(FUNC)\
        FUNC(INIT_TYPE_RANDOM), \
        FUNC(INIT_TYPE_XWAVE), \
        FUNC(INIT_TYPE_GAUSSIAN_RADIAL_EXPL), \
        FUNC(INIT_TYPE_KICKBALL), \
        FUNC(INIT_TYPE_ABC_FLOW) , \
        FUNC(INIT_TYPE_SIMPLE_CORE), \
        FUNC(INIT_TYPE_VEDGE), \
        FUNC(INIT_TYPE_VEDGEX), \
        FUNC(INIT_TYPE_RAYLEIGH_TAYLOR), \
        FUNC(INIT_TYPE_RAYLEIGH_BENARD)
// clang-format on

#define AC_GEN_ID(X) X
typedef enum { AC_FOR_INIT_TYPES(AC_GEN_ID), NUM_INIT_TYPES } InitType;
#undef AC_GEN_ID
*/

extern const char* init_type_names[]; // Defined in host_memory.cc

AcMesh* acmesh_create(const AcMeshInfo& mesh_info);

void vertex_buffer_set(const VertexBufferHandle& key, const AcReal& val, AcMesh* mesh);

void acmesh_clear(AcMesh* mesh);

void acmesh_init_to(const InitType& type, AcMesh* mesh);

void acmesh_destroy(AcMesh* mesh);

ModelMesh* modelmesh_create(const AcMeshInfo& mesh_info);
void modelmesh_destroy(ModelMesh* mesh);
void acmesh_to_modelmesh(const AcMesh& acmesh, ModelMesh* modelmesh);
void modelmesh_to_acmesh(const ModelMesh& model, AcMesh* acmesh);
