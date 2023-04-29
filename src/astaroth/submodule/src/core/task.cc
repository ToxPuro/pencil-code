/*
    Copyright (C) 2020, Oskar Lappi

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
#if AC_MPI_ENABLED
/**
 * Quick overview of tasks
 *
 * Each halo segment is assigned a HaloExchangeTask.
 * A HaloExchangeTask sends local data as a halo to a neighbor
 * and receives halo data from a (possibly different) neighbor.
 *
 * Each shell segment is assigned a ComputeTask.
 * ComputeTasks integrate their segment of the domain.
 *
 * After a task has been completed, its dependent tasks can be started with notifyDependents()
 * E.g. ComputeTasks may depend on HaloExchangeTasks because they're waiting to receive data.
 * Vv.  HaloExchangeTasks may depend on ComputeTasks because they're waiting for data to send.
 *
 * This all happens in grid.cc:GridIntegrate
 */

#include "task.h"
#include "astaroth.h"
#include "astaroth_utils.h"

#include <cassert>
#include <memory>
#include <mpi.h>
#include <stdlib.h>
#include <vector>

#include "decomposition.h" //getPid and friends
#include "errchk.h"
#include "kernels/kernels.h" //AcRealPacked, ComputeKernel

#define HALO_TAG_OFFSET (100) //"Namespacing" the MPI tag space to avoid collisions

#if AC_USE_HIP
template <typename T, typename... Args>
std::unique_ptr<T>
make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#else
using std::make_unique;
#endif

AcTaskDefinition
acCompute(const AcKernel kernel, Field fields_in[], const size_t num_fields_in, Field fields_out[],
          const size_t num_fields_out, int shell_num, int step_number)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_COMPUTE;
    task_def.kernel         = kernel;
    task_def.fields_in      = fields_in;
    task_def.num_fields_in  = num_fields_in;
    task_def.fields_out     = fields_out;
    task_def.num_fields_out = num_fields_out;
    task_def.shell_num      = shell_num;
    task_def.step_number    = step_number;
    return task_def;
}

AcTaskDefinition
acHaloExchange(Field fields[], const size_t num_fields)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_HALOEXCHANGE;
    task_def.fields_in      = fields;
    task_def.num_fields_in  = num_fields;
    task_def.fields_out     = fields;
    task_def.num_fields_out = num_fields;
    task_def.shell_num      = 0;
    task_def.step_number    = -1;
    return task_def;
}

AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, const AcBoundcond bound_cond, Field fields[],
                    const size_t num_fields, AcRealParam parameters[], const size_t num_parameters)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_BOUNDCOND;
    task_def.boundary       = boundary;
    task_def.bound_cond     = bound_cond;
    task_def.fields_in      = fields;
    task_def.num_fields_in  = num_fields;
    task_def.fields_out     = fields;
    task_def.num_fields_out = num_fields;
    task_def.parameters     = parameters;
    task_def.num_parameters = num_parameters;
    task_def.shell_num      = 0;
    task_def.step_number    = -1;
    return task_def;
}

#ifdef AC_INTEGRATION_ENABLED
AcTaskDefinition
acSpecialMHDBoundaryCondition(const AcBoundary boundary, const AcSpecialMHDBoundcond bound_cond,
                              AcRealParam parameters[], const size_t num_parameters)
{
    AcTaskDefinition task_def{};
    task_def.task_type              = TASKTYPE_SPECIAL_MHD_BOUNDCOND;
    task_def.boundary               = boundary;
    task_def.special_mhd_bound_cond = bound_cond;
    // TODO: look these up from a table
    task_def.fields_in      = nullptr;
    task_def.num_fields_in  = 0;
    task_def.fields_out     = nullptr;
    task_def.num_fields_out = 0;
    task_def.parameters     = parameters;
    task_def.num_parameters = num_parameters;
    task_def.shell_num      = 0;
    task_def.step_number    = -1;
    return task_def;
}

AcTaskDefinition
acSpecialMHDBoundaryCondition(const AcBoundary boundary, const AcSpecialMHDBoundcond bound_cond)
{
    return acSpecialMHDBoundaryCondition(boundary, bound_cond, nullptr, 0);
}

#endif

Region::Region(RegionFamily family_, int tag_, int3 nn, Field fields_[], size_t num_fields)
    : family(family_), tag(tag_), fields(fields_, fields_ + num_fields)
{
    id = tag_to_id(tag);
    // facet class 0 = inner core
    // facet class 1 = face
    // facet class 2 = edge
    // facet class 3 = corner
    facet_class = (id.x == 0 ? 0 : 1) + (id.y == 0 ? 0 : 1) + (id.z == 0 ? 0 : 1);
    ERRCHK_ALWAYS(facet_class <= 3);

    switch (family) {
    case RegionFamily::Compute_output: {
        // clang-format off
        position = (int3){
                    id.x == -1  ? NGHOST : id.x == 1 ? nn.x : NGHOST * 2,
                    id.y == -1  ? NGHOST : id.y == 1 ? nn.y : NGHOST * 2,
                    id.z == -1  ? NGHOST : id.z == 1 ? nn.z : NGHOST * 2};
        // clang-format on
        dims = (int3){id.x == 0 ? nn.x - NGHOST * 2 : NGHOST,
                      id.y == 0 ? nn.y - NGHOST * 2 : NGHOST,
                      id.z == 0 ? nn.z - NGHOST * 2 : NGHOST};
        break;
    }
    case RegionFamily::Compute_input: {
        // clang-format off
        position = (int3){
                    id.x == -1  ? 0 : id.x == 1 ? nn.x - NGHOST : NGHOST ,
                    id.y == -1  ? 0 : id.y == 1 ? nn.y - NGHOST : NGHOST ,
                    id.z == -1  ? 0 : id.z == 1 ? nn.z - NGHOST : NGHOST };
        // clang-format on
        dims = (int3){id.x == 0 ? nn.x : NGHOST * 3, id.y == 0 ? nn.y : NGHOST * 3,
                      id.z == 0 ? nn.z : NGHOST * 3};
        break;
    }
    case RegionFamily::Exchange_output: {
        // clang-format off
        position = (int3){
                    id.x == -1  ? 0 : id.x == 1 ? NGHOST + nn.x : NGHOST,
                    id.y == -1  ? 0 : id.y == 1 ? NGHOST + nn.y : NGHOST,
                    id.z == -1  ? 0 : id.z == 1 ? NGHOST + nn.z : NGHOST};
        // clang-format on
        dims = (int3){id.x == 0 ? nn.x : NGHOST, id.y == 0 ? nn.y : NGHOST,
                      id.z == 0 ? nn.z : NGHOST};
        break;
    }
    case RegionFamily::Exchange_input: {
        position = (int3){id.x == 1 ? nn.x : NGHOST, id.y == 1 ? nn.y : NGHOST,
                          id.z == 1 ? nn.z : NGHOST};
        dims = (int3){id.x == 0 ? nn.x : NGHOST, id.y == 0 ? nn.y : NGHOST,
                      id.z == 0 ? nn.z : NGHOST};
        break;
    }
    default: {
        ERROR("Unknown region family.");
    }
    }
    volume = dims.x * dims.y * dims.z;
}
Region::Region(RegionFamily family_, RegionId region_id, int3 nn, Field fields_[], size_t num_fields)
    : family(family_), fields(fields_, fields_ + num_fields)
{
    id = region_id.shell_id; 
    tag = region_id_to_tag(region_id);
    int shell_num = region_id.shell_num;
    // facet class 0 = inner core
    // facet class 1 = face
    // facet class 2 = edge
    // facet class 3 = corner
    facet_class = (id.x == 0 ? 0 : 1) + (id.y == 0 ? 0 : 1) + (id.z == 0 ? 0 : 1);
    facet_class = shell_num == 0 ? facet_class : 0;
    ERRCHK_ALWAYS(facet_class <= 3);
    switch (family) {
    case RegionFamily::Compute_output: {
        // clang-format off
        position = (int3){
                    id.x == -1  ? NGHOST*(shell_num+1): id.x == 1 ? nn.x - NGHOST*shell_num : NGHOST*(2+shell_num),
                    id.y == -1  ? NGHOST*(shell_num+1) : id.y == 1 ? nn.y -NGHOST*shell_num : NGHOST*(2+shell_num),
                    id.z == -1  ? NGHOST*(shell_num+1): id.z == 1 ? nn.z - NGHOST*shell_num: NGHOST*(2+shell_num)};
        // clang-format on
        // position = (int3){
        //             id.x == -1  ? NGHOST : id.x == 1 ? nn.x : NGHOST * 2,
        //             id.y == -1  ? NGHOST : id.y == 1 ? nn.y : NGHOST * 2,
        //             id.z == -1  ? NGHOST : id.z == 1 ? nn.z : NGHOST * 2};
      dims = (int3){id.x == 0 ? nn.x - NGHOST *2*(1+shell_num) : NGHOST,
                      id.y == 0 ? nn.y - NGHOST *2*(1+shell_num) : NGHOST,
                      id.z == 0 ? nn.z - NGHOST *2*(1+shell_num) : NGHOST};
        // clang-format on
        // dims = (int3){id.x == 0 ? nn.x - NGHOST * 2 : NGHOST,
        //               id.y == 0 ? nn.y - NGHOST * 2 : NGHOST,
        //               id.z == 0 ? nn.z - NGHOST * 2 : NGHOST};
        break;
    }
    case RegionFamily::Compute_input: {
        // clang-format off
        position = (int3){
                    id.x == -1  ? NGHOST*shell_num : id.x == 1 ? nn.x - NGHOST*(1+shell_num) : NGHOST*(1+shell_num) ,
                    id.y == -1  ? NGHOST*shell_num : id.y == 1 ? nn.y - NGHOST*(1+shell_num) : NGHOST*(1+shell_num) ,
                    id.z == -1  ? NGHOST*shell_num : id.z == 1 ? nn.z - NGHOST*(1+shell_num) : NGHOST*(1+shell_num) };
        dims = (int3){id.x == 0 ? nn.x-2*NGHOST*shell_num : NGHOST * 3, id.y == 0 ? nn.y-2*NGHOST*shell_num : NGHOST * 3,
                      id.z == 0 ? nn.z-2*NGHOST*shell_num : NGHOST * 3};
        break;
    }
    case RegionFamily::Exchange_output: {
        ERROR("Not supported for exhange regions at the moment\n");
        break;
    }
    case RegionFamily::Exchange_input: {
        ERROR("Not supported for exhange regions at the moment\n");
        break;
    }
    default: {
        ERROR("Unknown region family.");
    }
    }
    volume = dims.x * dims.y * dims.z;
}
Region::Region(RegionFamily family_, int3 id_, int3 nn, Field fields_[], size_t num_fields)
    : Region{family_, id_to_tag(id_), nn, fields_, num_fields}
{
    ERRCHK_ALWAYS(id_.x == id.x && id_.y == id.y && id_.z == id.z);
}

Region::Region(int3 position_, int3 dims_, int tag_, std::vector<Field> fields_)
    : position(position_), dims(dims_), family(RegionFamily::None), tag(tag_), fields(fields_)
{
    id          = tag_to_id(tag);
    facet_class = (id.x == 0 ? 0 : 1) + (id.y == 0 ? 0 : 1) + (id.z == 0 ? 0 : 1);
}

Region
Region::translate(int3 translation)
{
    return Region(this->position + translation, this->dims, this->tag, this->fields);
}

bool
Region::overlaps(const Region* other)
{
    return (this->position.x < other->position.x + other->dims.x) &&
           (other->position.x < this->position.x + this->dims.x) &&
           (this->position.y < other->position.y + other->dims.y) &&
           (other->position.y < this->position.y + this->dims.y) &&
           (this->position.z < other->position.z + other->dims.z) &&
           (other->position.z < this->position.z + this->dims.z);
}

AcBoundary
Region::boundary(uint3_64 decomp, int pid)
{
    int3 pid3d = getPid3D(pid, decomp);
    return boundary(decomp, pid3d, id);
}

bool
Region::is_on_boundary(uint3_64 decomp, int pid, AcBoundary boundary)
{
    int3 pid3d = getPid3D(pid, decomp);
    return is_on_boundary(decomp, pid3d, id, boundary);
}

// Static functions
int
Region::id_to_tag(int3 id)
{
    return ((3 + id.x) % 3) * 9 + ((3 + id.y) % 3) * 3 + (3 + id.z) % 3;
}
int
Region::region_id_to_tag(RegionId id)
{
    return id_to_tag(id.shell_id)+27*id.shell_num;
}

int3
Region::tag_to_id(int _tag)
{
    int3 _id = (int3){(_tag) / 9, ((_tag) % 9) / 3, (_tag) % 3};
    _id.x    = _id.x == 2 ? -1 : _id.x;
    _id.y    = _id.y == 2 ? -1 : _id.y;
    _id.z    = _id.z == 2 ? -1 : _id.z;
    ERRCHK_ALWAYS(id_to_tag(_id) == _tag);
    return _id;
}

AcBoundary
Region::boundary(uint3_64 decomp, int pid, int tag)
{
    int3 pid3d = getPid3D(pid, decomp);
    int3 id    = tag_to_id(tag);
    return boundary(decomp, pid3d, id);
}

AcBoundary
Region::boundary(uint3_64 decomp, int3 pid3d, int3 id)
{
    int3 neighbor = pid3d + id;
    return (AcBoundary)((neighbor.x == -1 ? BOUNDARY_X_BOT : 0) |
                        (neighbor.x == (int)decomp.x ? BOUNDARY_X_TOP : 0) |
                        (neighbor.y == -1 ? BOUNDARY_Y_TOP : 0) |
                        (neighbor.y == (int)decomp.y ? BOUNDARY_Y_TOP : 0) |
                        (neighbor.z == -1 ? BOUNDARY_Z_TOP : 0) |
                        (neighbor.z == (int)decomp.z ? BOUNDARY_Z_TOP : 0));
}

bool
Region::is_on_boundary(uint3_64 decomp, int pid, int tag, AcBoundary boundary)
{
    int3 pid3d     = getPid3D(pid, decomp);
    int3 region_id = tag_to_id(tag);
    return is_on_boundary(decomp, pid3d, region_id, boundary);
}

bool
Region::is_on_boundary(uint3_64 decomp, int3 pid3d, int3 id, AcBoundary boundary)
{
    AcBoundary b = Region::boundary(decomp, pid3d, id);
    return b & boundary ? true : false;
}

/* Task interface */
Task::Task(int order_, Region input_region_, Region output_region_, AcTaskDefinition op,
           Device device_, std::array<bool, NUM_VTXBUF_HANDLES> swap_offset_)
    : device(device_), swap_offset(swap_offset_), state(wait_state), dep_cntr(), loop_cntr(),
      order(order_), active(true), boundary(BOUNDARY_NONE), input_region(input_region_),
      output_region(output_region_),
      input_parameters(op.parameters, op.parameters + op.num_parameters)
{
    MPI_Comm_rank(acGridMPIComm(), &rank);
}

void
Task::registerDependent(std::shared_ptr<Task> t, size_t offset)
{
    dependents.emplace_back(t, offset);
    t->registerPrerequisite(offset);
}

void
Task::registerPrerequisite(size_t offset)
{
    // Ensure targets exist
    if (offset >= dep_cntr.targets.size()) {
        size_t initial_val = dep_cntr.targets.empty() ? 0 : dep_cntr.targets.back();
        dep_cntr.targets.resize(offset + 1, initial_val);
    }
    for (; offset < dep_cntr.targets.size(); offset++) {
        dep_cntr.targets[offset]++;
    }
}

bool
Task::isPrerequisiteTo(std::shared_ptr<Task> other)
{
    for (auto dep : dependents) {
        if (dep.first.lock() == other) {
            return true;
        }
    }
    return false;
}

void
Task::setIterationParams(size_t begin, size_t end)
{
    loop_cntr.i   = begin;
    loop_cntr.end = end;

    // Reset dependency counter, and ensure it has enough space
    dep_cntr.counts.resize(0);
    dep_cntr.counts.resize(end, 0);
}

bool
Task::isFinished()
{
    return loop_cntr.i == loop_cntr.end;
}

void
Task::update(std::array<bool, NUM_VTXBUF_HANDLES> vtxbuf_swaps, const TraceFile* trace_file)
{
    if (isFinished())
        return;

    bool ready;
    if (state == wait_state) {
        // dep_cntr.targets contains a rising series of targets e.g. {5,10}. The reason that earlier
        // iterations of a task might have fewer prerequisites in the task graph because the
        // prerequisites would have been satisfied by work that was performed before the beginning
        // of the task graph execution.
        //
        // Therefore, in the example, dep_cntr.targets = {5,10}:
        // if the loop counter is 0 or 1, we choose targets[0] (5) and targets[1] (10) respecively
        // if the loop counter is greater than that (e.g. 3) we select the final target count (10).

        if (dep_cntr.targets.size() == 0) {
            ready = true;
        }
        else if (loop_cntr.i >= dep_cntr.targets.size()) {
            ready = (dep_cntr.counts[loop_cntr.i] == dep_cntr.targets.back());
        }
        else {
            ready = (dep_cntr.counts[loop_cntr.i] == dep_cntr.targets[loop_cntr.i]);
        }
    }
    else {
        ready = test();
    }

    if (ready) {
        advance(trace_file);
        if (state == wait_state) {
            swapVBA(vtxbuf_swaps);
            notifyDependents();
            loop_cntr.i++;
        }
    }
}

void
Task::notifyDependents()
{
    for (auto& dep : dependents) {
        std::shared_ptr<Task> dependent = dep.first.lock();
        dependent->satisfyDependency(loop_cntr.i + dep.second);
    }
}

void
Task::satisfyDependency(size_t iteration)
{
    if (iteration < loop_cntr.end) {
        dep_cntr.counts[iteration]++;
    }
}

void
Task::syncVBA()
{
    cudaSetDevice(device->id);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        if (swap_offset[i]) {
            vba.in[i]  = device->vba.out[i];
            vba.out[i] = device->vba.in[i];
        }
        else {
            vba.in[i]  = device->vba.in[i];
            vba.out[i] = device->vba.out[i];
        }
    }
}

void
Task::swapVBA(std::array<bool, NUM_VTXBUF_HANDLES> vtxbuf_swaps)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        // printf("vtxbuf_swaps %i VTXBUF number %i (VTXBUF_SHOCK %i) \n", vtxbuf_swaps[i], i,
        // VTXBUF_SHOCK);

        if (vtxbuf_swaps[i]) {
            AcReal* tmp = vba.in[i];
            vba.in[i]   = vba.out[i];
            vba.out[i]  = tmp;
        }
    }
}

bool
Task::poll_stream()
{
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaSuccess) {
        return true;
    }
    if (err == cudaErrorNotReady) {
        return false;
    }
    fprintf(stderr,
            "CUDA error in task %s while polling CUDA stream"
            " (probably occured in the CUDA kernel):\n\t%s\n",
            name.c_str(), cudaGetErrorString(err));
    fflush(stderr);
    exit(EXIT_FAILURE);
    return false;
}

/* Computation */
ComputeTask::ComputeTask(AcTaskDefinition op, int order_, int region_tag, int3 nn, Device device_,
                         std::array<bool, NUM_VTXBUF_HANDLES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Compute_input, region_tag, nn, op.fields_in, op.num_fields_in),
           Region(RegionFamily::Compute_output, region_tag, nn, op.fields_out, op.num_fields_out),
           op, device_, swap_offset_)
{
    // stream = device->streams[STREAM_DEFAULT + region_tag];
    {
        cudaSetDevice(device->id);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }

    syncVBA();

    // compute_func = compute_func_;
    step_number_set = false;
    params = KernelParameters{kernels[(int)op.kernel], stream, 0, output_region.position,
                              output_region.position + output_region.dims};
    name   = "Compute " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) + "," +
           std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) + ")";
    task_type = TASKTYPE_COMPUTE;
}
/* Computation */
ComputeTask::ComputeTask(AcTaskDefinition op, int order_, RegionId region_id, int3 nn, Device device_,
                         std::array<bool, NUM_VTXBUF_HANDLES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Compute_input, region_id, nn, op.fields_in, op.num_fields_in),
           Region(RegionFamily::Compute_output, region_id, nn, op.fields_out, op.num_fields_out),
           op, device_, swap_offset_)
{
    // stream = device->streams[STREAM_DEFAULT + region_tag];
    {
        cudaSetDevice(device->id);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }

    syncVBA();

    // compute_func = compute_func_;
    step_number_set = op.step_number > -1;
    params = KernelParameters{kernels[(int)op.kernel], stream, op.step_number, output_region.position,
                              output_region.position + output_region.dims};
    name   = "Compute " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) + "," +
           std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) + ")";
    task_type = TASKTYPE_COMPUTE;
}
void
ComputeTask::compute()
{
    // IDEA: we could make loop_cntr.i point at params.step_number
    // if(!step_number_set){
    //     printf("Step number not set\n");
    // }
    // else{
    //     printf("Step number set\n");
    // }
    if(!step_number_set){
        params.step_number = (int)(loop_cntr.i % 3);
    }
    acKernel(params, vba);
}

bool
ComputeTask::test()
{
    switch (static_cast<ComputeState>(state)) {
    case ComputeState::Running: {
        return poll_stream();
    }
    default: {
        ERROR("ComputeTask in an invalid state.");
        return false;
    }
    }
}

void
ComputeTask::advance(const TraceFile* trace_file)
{
    switch (static_cast<ComputeState>(state)) {
    case ComputeState::Waiting: {
        trace_file->trace(this, "waiting", "running");
        compute();
        state = static_cast<int>(ComputeState::Running);
        break;
    }
    case ComputeState::Running: {
        trace_file->trace(this, "running", "waiting");
        state = static_cast<int>(ComputeState::Waiting);
        break;
    }
    default:
        ERROR("ComputeTask in an invalid state.");
    }
}

/*  Communication   */

// HaloMessage contains all information needed to send or receive a single message
HaloMessage::HaloMessage(int3 dims, size_t num_vars)
{
    length       = dims.x * dims.y * dims.z * num_vars;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data, bytes));
#if !(USE_CUDA_AWARE_MPI)
    ERRCHK_CUDA_ALWAYS(cudaMallocHost((void**)&data_pinned, bytes));
#endif
    request = MPI_REQUEST_NULL;
}

HaloMessage::~HaloMessage()
{
    if (request != MPI_REQUEST_NULL) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    length = -1;
    cudaFree(data);
#if !(USE_CUDA_AWARE_MPI)
    cudaFree(data_pinned);
#endif
    data = NULL;
}

#if !(USE_CUDA_AWARE_MPI)
void
HaloMessage::pin(const Device device, const cudaStream_t stream)
{
    cudaSetDevice(device->id);
    pinned       = true;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA(cudaMemcpyAsync(data_pinned, data, bytes, cudaMemcpyDefault, stream));
}

void
HaloMessage::unpin(const Device device, const cudaStream_t stream)
{
    if (!pinned)
        return;

    cudaSetDevice(device->id);
    pinned       = false;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA(cudaMemcpyAsync(data, data_pinned, bytes, cudaMemcpyDefault, stream));
}
#endif

// HaloMessageSwapChain
HaloMessageSwapChain::HaloMessageSwapChain() {}

HaloMessageSwapChain::HaloMessageSwapChain(int3 dims, size_t num_vars)
    : buf_idx(SWAP_CHAIN_LENGTH - 1)
{
    buffers.reserve(SWAP_CHAIN_LENGTH);
    for (int i = 0; i < SWAP_CHAIN_LENGTH; i++) {
        buffers.emplace_back(dims, num_vars);
    }
}

HaloMessage*
HaloMessageSwapChain::get_current_buffer()
{
    return &buffers[buf_idx];
}

HaloMessage*
HaloMessageSwapChain::get_fresh_buffer()
{
    buf_idx         = (buf_idx + 1) % SWAP_CHAIN_LENGTH;
    MPI_Request req = buffers[buf_idx].request;
    if (req != MPI_REQUEST_NULL) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    return &buffers[buf_idx];
}

// HaloExchangeTask
HaloExchangeTask::HaloExchangeTask(AcTaskDefinition op, int order_, int tag_0, int halo_region_tag,
                                   int3 nn, uint3_64 decomp, Device device_,
                                   std::array<bool, NUM_VTXBUF_HANDLES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Exchange_input, halo_region_tag, nn, op.fields_in,
                  op.num_fields_in),
           Region(RegionFamily::Exchange_output, halo_region_tag, nn, op.fields_out,
                  op.num_fields_out),
           op, device_, swap_offset_),
      recv_buffers(output_region.dims, NUM_VTXBUF_HANDLES),
      send_buffers(input_region.dims, NUM_VTXBUF_HANDLES)
// Below are for partial halo exchanges.
// TODO: enable partial halo exchanges when
// vtxbuf_dependencies_->num_vars < NUM_VTXBUF_HANDLES (see performance first)
// recv_buffers(output_region.dims, vtxbuf_dependencies_->num_vars),
// send_buffers(input_region.dims, vtxbuf_dependencies_->num_vars)
{
    // Create stream for packing/unpacking
    acVerboseLogFromRootProc(rank, "Halo exchange task ctor: creating CUDA stream\n");
    {
        cudaSetDevice(device->id);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    acVerboseLogFromRootProc(rank, "Halo exchange task ctor: done creating CUDA stream\n");

    acVerboseLogFromRootProc(rank, "Halo exchange task ctor: syncing VBA\n");
    syncVBA();
    acVerboseLogFromRootProc(rank, "Halo exchange task ctor: done syncing VBA\n");

    counterpart_rank = getPid(getPid3D(rank, decomp) + output_region.id, decomp);
    // MPI tags are namespaced to avoid collisions with other MPI tasks
    send_tag = tag_0 + input_region.tag;
    recv_tag = tag_0 + Region::id_to_tag(-output_region.id);

    // Post receive immediately, this avoids unexpected messages
    active = ((MPI_INCL_CORNERS) || output_region.facet_class != 3) ? true : false;
    if (active) {
        acVerboseLogFromRootProc(rank, "Halo exchange task ctor: posting early receive\n");
        receive();
        acVerboseLogFromRootProc(rank, "Halo exchange task ctor: done posting early receive\n");
    }
    name = "Halo exchange " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) +
           "," + std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) +
           ")";
    task_type = TASKTYPE_HALOEXCHANGE;
}

HaloExchangeTask::~HaloExchangeTask()
{
    // Cancel last eager request
    auto msg = recv_buffers.get_current_buffer();
    if (msg->request != MPI_REQUEST_NULL) {
        MPI_Cancel(&msg->request);
    }

    cudaSetDevice(device->id);
    // dependents.clear();
    cudaStreamDestroy(stream);
}

void
HaloExchangeTask::pack()
{
    auto msg = send_buffers.get_fresh_buffer();
    // acKernelPartialPackData(stream, vba, input_region.position, input_region.dims,
    //                         msg->data, vtxbuf_dependencies->variables,
    //                         vtxbuf_dependencies->num_vars);
    acKernelPackData(stream, vba, input_region.position, input_region.dims, msg->data);
}

void
HaloExchangeTask::unpack()
{

    auto msg = recv_buffers.get_current_buffer();
#if !(USE_CUDA_AWARE_MPI)
    msg->unpin(device, stream);
#endif
    // acKernelPartialUnpackData(stream, msg->data, output_region.position, output_region.dims,
    //                           vba, vtxbuf_dependencies->variables,
    //                           vtxbuf_dependencies->num_vars);
    acKernelUnpackData(stream, msg->data, output_region.position, output_region.dims, vba);
}

void
HaloExchangeTask::sync()
{
    cudaStreamSynchronize(stream);
}

void
HaloExchangeTask::wait_recv()
{
    auto msg = recv_buffers.get_current_buffer();
    MPI_Wait(&msg->request, MPI_STATUS_IGNORE);
}

void
HaloExchangeTask::wait_send()
{
    auto msg = send_buffers.get_current_buffer();
    MPI_Wait(&msg->request, MPI_STATUS_IGNORE);
}

void
HaloExchangeTask::receiveDevice()
{
    // TODO: change these to debug log statements at high verbosity (there will be very many of
    // these outputs)
    if (rank == 0) {
        // fprintf(stderr, "receiveDevice, getting buffer\n");
    }
    auto msg = recv_buffers.get_fresh_buffer();
    if (rank == 0) {
        // fprintf(stderr, "calling MPI_Irecv\n");
    }
    MPI_Irecv(msg->data, msg->length, AC_REAL_MPI_TYPE, counterpart_rank,
              recv_tag + HALO_TAG_OFFSET, acGridMPIComm(), &msg->request);
    if (rank == 0) {
        // fprintf(stderr, "Returned from MPI_Irecv\n");
    }
}

void
HaloExchangeTask::sendDevice()
{
    auto msg = send_buffers.get_current_buffer();
    sync();
    MPI_Isend(msg->data, msg->length, AC_REAL_MPI_TYPE, counterpart_rank,
              send_tag + HALO_TAG_OFFSET, acGridMPIComm(), &msg->request);
}

void
HaloExchangeTask::exchangeDevice()
{
    // cudaSetDevice(device->id);
    receiveDevice();
    sendDevice();
}

#if !(USE_CUDA_AWARE_MPI)
void
HaloExchangeTask::receiveHost()
{
    // TODO: change these to debug log statements at high verbosity (there will be very many of
    // these outputs)
    if (rank == 0) {
        // fprintf("receiveHost, getting buffer\n");
    }
    auto msg = recv_buffers.get_fresh_buffer();
    if (rank == 0) {
        // fprintf("Called MPI_Irecv\n");
    }
    MPI_Irecv(msg->data_pinned, msg->length, AC_REAL_MPI_TYPE, counterpart_rank,
              recv_tag + HALO_TAG_OFFSET, acGridMPIComm(), &msg->request);
    if (rank == 0) {
        // fprintf("Returned from MPI_Irecv\n");
    }
    msg->pinned = true;
}

void
HaloExchangeTask::sendHost()
{
    auto msg = send_buffers.get_current_buffer();
    msg->pin(device, stream);
    sync();
    MPI_Isend(msg->data_pinned, msg->length, AC_REAL_MPI_TYPE, counterpart_rank,
              send_tag + HALO_TAG_OFFSET, acGridMPIComm(), &msg->request);
}
void
HaloExchangeTask::exchangeHost()
{
    // cudaSetDevice(device->id);
    receiveHost();
    sendHost();
}
#endif

void
HaloExchangeTask::receive()
{
    // TODO: change these fprintfs to debug log statements at high verbosity (there will be very
    // many of these outputs)
#if USE_CUDA_AWARE_MPI
    if (rank == 0) {
        // fprintf(stderr, "receiveDevice()\n");
    }
    receiveDevice();
    if (rank == 0) {
        // fprintf(stderr, "returned from receiveDevice()\n");
    }
#else
    if (rank == 0) {
        // fprintf(stderr, "receiveHost()\n");
    }
    receiveHost();
    if (rank == 0) {
        // fprintf(stderr, "returned from receiveHost()\n");
    }
#endif
}

void
HaloExchangeTask::send()
{
#if USE_CUDA_AWARE_MPI
    sendDevice();
#else
    sendHost();
#endif
}

void
HaloExchangeTask::exchange()
{
#if USE_CUDA_AWARE_MPI
    exchangeDevice();
#else
    exchangeHost();
#endif
}

bool
HaloExchangeTask::test()
{
    switch (static_cast<HaloExchangeState>(state)) {
    case HaloExchangeState::Packing: {
        return poll_stream();
    }
    case HaloExchangeState::Unpacking: {
        return poll_stream();
    }
    case HaloExchangeState::Exchanging: {
        auto msg = recv_buffers.get_current_buffer();
        int request_complete;
        MPI_Test(&msg->request, &request_complete, MPI_STATUS_IGNORE);
        return request_complete ? true : false;
    }
    default: {
        ERROR("HaloExchangeTask in an invalid state.");
        return false;
    }
    }
}

void
HaloExchangeTask::advance(const TraceFile* trace_file)
{
    switch (static_cast<HaloExchangeState>(state)) {
    case HaloExchangeState::Waiting:
        trace_file->trace(this, "waiting", "packing");
        pack();
        state = static_cast<int>(HaloExchangeState::Packing);
        break;
    case HaloExchangeState::Packing:
        trace_file->trace(this, "packing", "receiving");
        sync();
        send();
        state = static_cast<int>(HaloExchangeState::Exchanging);
        break;
    case HaloExchangeState::Exchanging:
        trace_file->trace(this, "receiving", "unpacking");
        sync();
        unpack();
        state = static_cast<int>(HaloExchangeState::Unpacking);
        break;
    case HaloExchangeState::Unpacking:
        trace_file->trace(this, "unpacking", "waiting");
        receive();
        sync();
        state = static_cast<int>(HaloExchangeState::Waiting);
        break;
    default:
        ERROR("HaloExchangeTask in an invalid state.");
    }
}

BoundaryConditionTask::BoundaryConditionTask(AcTaskDefinition op, int3 boundary_normal_, int order_,
                                             int region_tag, int3 nn, Device device_,
                                             std::array<bool, NUM_VTXBUF_HANDLES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Exchange_input, region_tag, nn, op.fields_in, op.num_fields_in),
           Region(RegionFamily::Exchange_output, region_tag, nn, op.fields_out, op.num_fields_out),
           op, device_, swap_offset_),
      boundcond(op.bound_cond), boundary_normal(boundary_normal_)
{
    // Create stream for boundary condition task
    {
        cudaSetDevice(device->id);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    syncVBA();

    int3 translation = int3{(output_region.dims.x + 1) * (-boundary_normal.x),
                            (output_region.dims.y + 1) * (-boundary_normal.y),
                            (output_region.dims.z + 1) * (-boundary_normal.z)};

    // TODO: input_region is now set twice, overwritten here
    input_region = Region(output_region.translate(translation));

    boundary_dims = int3{
        boundary_normal.x == 0 ? output_region.dims.x : 1,
        boundary_normal.y == 0 ? output_region.dims.y : 1,
        boundary_normal.z == 0 ? output_region.dims.z : 1,
    };

    name = "Boundary condition " + std::to_string(order_) + ".(" +
           std::to_string(output_region.id.x) + "," + std::to_string(output_region.id.y) + "," +
           std::to_string(output_region.id.z) + ")" + ".(" + std::to_string(boundary_normal.x) +
           "," + std::to_string(boundary_normal.y) + "," + std::to_string(boundary_normal.z) + ")";
    task_type = TASKTYPE_BOUNDCOND;
}

void
BoundaryConditionTask::populate_boundary_region()
{
    // TODO: could assign a separate stream to each launch
    //       currently they are on a single stream
    for (auto variable : output_region.fields) {
        switch (boundcond) {
        case BOUNDCOND_SYMMETRIC: {
            acKernelSymmetricBoundconds(stream, output_region.id, boundary_normal, boundary_dims,
                                        vba.in[variable]);
            break;
        }
        case BOUNDCOND_ANTISYMMETRIC: {
            acKernelAntiSymmetricBoundconds(stream, output_region.id, boundary_normal,
                                            boundary_dims, vba.in[variable]);
            break;
        }
        case BOUNDCOND_A2: {
            acKernelA2Boundconds(stream, output_region.id, boundary_normal, boundary_dims,
                                 vba.in[variable]);
            break;
        }
        case BOUNDCOND_PRESCRIBED_DERIVATIVE: {
            assert(input_parameters.size() == 1);
            acKernelPrescribedDerivativeBoundconds(stream, output_region.id, boundary_normal,
                                                   boundary_dims, vba.in[variable],
                                                   input_parameters[0]);
            break;
        }
        default:
            ERROR("BoundaryCondition not implemented yet.");
        }
    }
}

bool
BoundaryConditionTask::test()
{
    switch (static_cast<BoundaryConditionState>(state)) {
    case BoundaryConditionState::Running: {
        return poll_stream();
    }
    default: {
        ERROR("BoundaryConditionTask in an invalid state.");
        return false;
    }
    }
}

void
BoundaryConditionTask::advance(const TraceFile* trace_file)
{
    switch (static_cast<BoundaryConditionState>(state)) {
    case BoundaryConditionState::Waiting:
        trace_file->trace(this, "waiting", "running");
        populate_boundary_region();
        state = static_cast<int>(BoundaryConditionState::Running);
        break;
    case BoundaryConditionState::Running:
        trace_file->trace(this, "running", "waiting");
        state = static_cast<int>(BoundaryConditionState::Waiting);
        break;
    default:
        ERROR("BoundaryConditionTask in an invalid state.");
    }
}

#ifdef AC_INTEGRATION_ENABLED
// SpecialMHDBoundaryConditions are tied to some specific DSL implementation (At the moment, the MHD
// implementation). They launch specially written CUDA kernels that implement the specific boundary
// condition procedure They are a stop-gap temporary solution. The sensible solution is to replace
// them with a task type that runs a boundary condition procedure written in the Astaroth DSL.
SpecialMHDBoundaryConditionTask::SpecialMHDBoundaryConditionTask(
    AcTaskDefinition op, int3 boundary_normal_, int order_, int region_tag, int3 nn, Device device_,
    std::array<bool, NUM_VTXBUF_HANDLES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Exchange_input, region_tag, nn, op.fields_in, op.num_fields_in),
           Region(RegionFamily::Exchange_output, region_tag, nn, op.fields_out, op.num_fields_out),
           op, device_, swap_offset_),
      boundcond(op.special_mhd_bound_cond), boundary_normal(boundary_normal_)
{
    // TODO: the input regions for some of these will be weird, because they will depend on the
    // ghost zone of other fields
    //  This is not currently reflected

    // Create stream for boundary condition task
    {
        cudaSetDevice(device->id);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    syncVBA();

    int3 translation = int3{(output_region.dims.x + 1) * (-boundary_normal.x),
                            (output_region.dims.y + 1) * (-boundary_normal.y),
                            (output_region.dims.z + 1) * (-boundary_normal.z)};

    // TODO: input_region is now set twice, overwritten here
    input_region = Region(output_region.translate(translation));

    boundary_dims = int3{
        boundary_normal.x == 0 ? output_region.dims.x : 1,
        boundary_normal.y == 0 ? output_region.dims.y : 1,
        boundary_normal.z == 0 ? output_region.dims.z : 1,
    };

    name = "Special MHD Boundary condition " + std::to_string(order_) + ".(" +
           std::to_string(output_region.id.x) + "," + std::to_string(output_region.id.y) + "," +
           std::to_string(output_region.id.z) + ")" + ".(" + std::to_string(boundary_normal.x) +
           "," + std::to_string(boundary_normal.y) + "," + std::to_string(boundary_normal.z) + ")";
    task_type = TASKTYPE_SPECIAL_MHD_BOUNDCOND;
}

void
SpecialMHDBoundaryConditionTask::populate_boundary_region()
{
    // TODO: could assign a separate stream to each launch of symmetric boundconds
    //       currently they are on a single stream
    switch (boundcond) {
#if LENTROPY
    case SPECIAL_MHD_BOUNDCOND_ENTROPY_CONSTANT_TEMPERATURE: {
        acKernelEntropyConstantTemperatureBoundconds(stream, output_region.id, boundary_normal,
                                                     boundary_dims, vba);
        break;
    }
    case SPECIAL_MHD_BOUNDCOND_ENTROPY_BLACKBODY_RADIATION: {
        acKernelEntropyBlackbodyRadiationKramerConductivityBoundconds(stream, output_region.id,
                                                                      boundary_normal,
                                                                      boundary_dims, vba);
        break;
    }
    case SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_HEAT_FLUX: {
        assert(input_parameters.size() == 1);
        acKernelEntropyPrescribedHeatFluxBoundconds(stream, output_region.id, boundary_normal,
                                                    boundary_dims, vba, input_parameters[0]);
        break;
    }
    case SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_NORMAL_AND_TURBULENT_HEAT_FLUX: {
        assert(input_parameters.size() == 2);
        acKernelEntropyPrescribedNormalAndTurbulentHeatFluxBoundconds(stream, output_region.id,
                                                                      boundary_normal,
                                                                      boundary_dims, vba,
                                                                      input_parameters[0],
                                                                      input_parameters[1]);
        break;
    }
#endif

    default:
        ERROR("SpecialMHDBoundaryCondition not implemented yet.");
    }
}

bool
SpecialMHDBoundaryConditionTask::test()
{
    switch (static_cast<SpecialMHDBoundaryConditionState>(state)) {
    case SpecialMHDBoundaryConditionState::Running: {
        return poll_stream();
    }
    default: {
        ERROR("SpecialMHDBoundaryConditionTask in an invalid state.");
        return false;
    }
    }
}

void
SpecialMHDBoundaryConditionTask::advance(const TraceFile* trace_file)
{
    switch (static_cast<SpecialMHDBoundaryConditionState>(state)) {
    case SpecialMHDBoundaryConditionState::Waiting:
        trace_file->trace(this, "waiting", "running");
        populate_boundary_region();
        state = static_cast<int>(SpecialMHDBoundaryConditionState::Running);
        break;
    case SpecialMHDBoundaryConditionState::Running:
        trace_file->trace(this, "running", "waiting");
        state = static_cast<int>(SpecialMHDBoundaryConditionState::Waiting);
        break;
    default:
        ERROR("SpecialMHDBoundaryConditionTask in an invalid state.");
    }
}
#endif // AC_INTEGRATION_ENABLED

AcBoundary
boundary_from_normal(int3 normal)
{
    return (
        AcBoundary)((normal.x == -1 ? BOUNDARY_X_BOT : 0) | (normal.x == 1 ? BOUNDARY_X_TOP : 0) |
                    (normal.y == -1 ? BOUNDARY_Y_BOT : 0) | (normal.y == 1 ? BOUNDARY_Y_TOP : 0) |
                    (normal.z == -1 ? BOUNDARY_Z_BOT : 0) | (normal.z == 1 ? BOUNDARY_Z_TOP : 0));
}

int3
normal_from_boundary(AcBoundary boundary)
{
    return int3{((BOUNDARY_X_TOP & boundary) != 0) - ((BOUNDARY_X_BOT & boundary) != 0),
                ((BOUNDARY_Y_TOP & boundary) != 0) - ((BOUNDARY_Y_BOT & boundary) != 0),
                ((BOUNDARY_Z_TOP & boundary) != 0) - ((BOUNDARY_Z_BOT & boundary) != 0)};
}

#endif // AC_MPI_ENABLED
