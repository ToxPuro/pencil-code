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

ModelScalar model_reduce_scal(const ModelMesh& mesh, const ReductionType& rtype,
                              const VertexBufferHandle& a);

ModelScalar model_reduce_vec(const ModelMesh& mesh, const ReductionType& rtype,
                             const VertexBufferHandle& a, const VertexBufferHandle& b,
                             const VertexBufferHandle& c);
