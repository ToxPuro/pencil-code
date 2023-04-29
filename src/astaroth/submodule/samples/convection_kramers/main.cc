#if !AC_MPI_ENABLED
#include <cstdio>
#include <cstdlib>
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#else

#include "astaroth.h"
#include "astaroth_debug.h"
#include "astaroth_utils.h"
#include "errchk.h"

#include <array>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define RESET "\x1B[0m"

#define debug_bc_errors 1
#define debug_bc_values 0

typedef AcReal (*boundcond_kernel_func)(AcReal boundary_val, AcReal domain_val, size_t r,
                                        AcMeshInfo info);

struct CellError {
    Field field;
    int3 dom;
    int3 ghost;
    AcReal expected;
    AcReal produced;
};

struct ErrorRatio {
    size_t total;
    size_t errors;
};

struct TestResultRegion {
    bool passed;
    int3 normal;
    std::vector<CellError> failed_cells;
    std::vector<ErrorRatio> field_errors;
    ErrorRatio error_ratio;
};

struct SimpleTestCase {
    std::string name;
    AcTaskGraph* task_graph;
    boundcond_kernel_func test_func;
};

struct MeshCompareTestCase {
    std::string name;
    AcTaskGraph* task_graph;
    AcMesh mesh_prior;
    AcMesh mesh_expected_result;
};

struct TestResult {
    std::string name;

    bool all_faces_passed;
    bool all_edges_passed;
    bool all_corners_passed;
    bool all_regions_passed;

    std::array<TestResultRegion, 6> face_regions;
    std::array<TestResultRegion, 12> edge_regions;
    std::array<TestResultRegion, 8> corner_regions;
};

template <size_t n_regions>
bool
All(const std::array<TestResultRegion, n_regions>& b)
{
    for (size_t i = 0; i < n_regions; i++) {
        if (!(b[i].passed)) {
            return false;
        }
    }
    return true;
}

// Tests for "simple", generic boundconds, that apply a function of type boundcond_kernel_func to
// produce the ghost zone values
TestResultRegion
test_simple_bc(AcMesh mesh, int3 direction, int3 dims, int3 domain_start, int3 ghost_start,
               AcMeshInfo info, boundcond_kernel_func kernel_func)
{
    TestResultRegion res{};
    res.passed      = true;
    res.normal      = direction;
    res.error_ratio = ErrorRatio{0, 0};

    AcReal epsilon = (AcReal)0.00000000000001;

    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        AcReal* field = mesh.vertex_buffer[i];
        res.field_errors.push_back(ErrorRatio{0, 0});

        for (int x = 0; x < dims.x; x++) {
            for (int y = 0; y < dims.y; y++) {
                for (int z = 0; z < dims.z; z++) {
                    res.field_errors.back().total++;

                    int3 dom = int3{domain_start.x + x, domain_start.y + y, domain_start.z + z};

                    int3 boundary = int3{(direction.x == 0)   ? dom.x
                                         : (direction.x == 1) ? domain_start.x + dims.x
                                                              : domain_start.x - 1,
                                         (direction.y == 0)   ? dom.y
                                         : (direction.y == 1) ? domain_start.y + dims.y
                                                              : domain_start.y - 1,
                                         (direction.z == 0)   ? dom.z
                                         : (direction.z == 1) ? domain_start.z + dims.z
                                                              : domain_start.z - 1};

                    int3 ghost = int3{(direction.x == 0) ? dom.x : ghost_start.x + dims.x - x - 1,
                                      (direction.y == 0) ? dom.y : ghost_start.y + dims.y - y - 1,
                                      (direction.z == 0) ? dom.z : ghost_start.z + dims.z - z - 1};

                    int idx_dom   = acVertexBufferIdx(dom.x, dom.y, dom.z, info);
                    int idx_bound = acVertexBufferIdx(boundary.x, boundary.y, boundary.z, info);
                    int idx_ghost = acVertexBufferIdx(ghost.x, ghost.y, ghost.z, info);

                    size_t r = std::abs(ghost.x - dom.x) + std::abs(ghost.y - dom.y) +
                               std::abs(ghost.z - dom.z);

                    AcReal expected_val = kernel_func(field[idx_bound], field[idx_dom], r / 2,
                                                      info);
                    if ((expected_val < field[idx_ghost] - epsilon) ||
                        (expected_val > field[idx_ghost] + epsilon)) {
                        res.field_errors.back().errors++;
                        res.failed_cells.push_back(
                            CellError{(Field)i, dom, ghost, expected_val, field[idx_ghost]});
                    }
                }
            }
        }
        res.error_ratio.total += res.field_errors.back().total;
        res.error_ratio.errors += res.field_errors.back().errors;
    }

    res.passed = (res.failed_cells.size() == 0);
    return res;
}

// Tests for any boundconds by comparing the result to an entire mesh that constitutes the expected
// result
TestResultRegion
test_bc_against_mesh(AcMesh mesh, int3 direction, int3 dims, int3 domain_start, int3 ghost_start,
                     AcMeshInfo info, AcMesh expected_mesh)
{
    TestResultRegion res{};
    res.passed      = true;
    res.normal      = direction;
    res.error_ratio = ErrorRatio{0, 0};

    AcReal epsilon = (AcReal)0.00000000000001;

    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        AcReal* field = mesh.vertex_buffer[i];
        res.field_errors.push_back(ErrorRatio{0, 0});

        for (int x = 0; x < dims.x; x++) {
            for (int y = 0; y < dims.y; y++) {
                for (int z = 0; z < dims.z; z++) {
                    res.field_errors.back().total++;

                    int3 dom = int3{domain_start.x + x, domain_start.y + y, domain_start.z + z};

                    int3 ghost = int3{(direction.x == 0) ? dom.x : ghost_start.x + dims.x - x - 1,
                                      (direction.y == 0) ? dom.y : ghost_start.y + dims.y - y - 1,
                                      (direction.z == 0) ? dom.z : ghost_start.z + dims.z - z - 1};

                    int idx_ghost = acVertexBufferIdx(ghost.x, ghost.y, ghost.z, info);

                    AcReal expected_val = expected_mesh.vertex_buffer[i][idx_ghost];
                    if ((expected_val < field[idx_ghost] - epsilon) ||
                        (expected_val > field[idx_ghost] + epsilon)) {
                        res.field_errors.back().errors++;
                        res.failed_cells.push_back(
                            CellError{(Field)i, dom, ghost, expected_val, field[idx_ghost]});
                    }
                }
            }
        }
        res.error_ratio.total += res.field_errors.back().total;
        res.error_ratio.errors += res.field_errors.back().errors;
    }

    res.passed = (res.failed_cells.size() == 0);
    return res;
}

// Runs a simple test for all ghost zone regions
TestResult
RunSimpleTest(SimpleTestCase test, AcMeshInfo info)
{
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    TestResult result{test.name,
                      true,
                      true,
                      true,
                      true,
                      std::array<TestResultRegion, 6>(),
                      std::array<TestResultRegion, 12>(),
                      std::array<TestResultRegion, 8>()};

    const int3 nn = int3{
        (int)(info.int_params[AC_nx]),
        (int)(info.int_params[AC_ny]),
        (int)(info.int_params[AC_nz]),
    };

    AcMesh mesh;

    if (pid == 0) {
        acHostMeshCreate(info, &mesh);
    }
    acGridExecuteTaskGraph(test.task_graph, 1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &mesh);

    if (pid == 0) {
        // acGraphPrintDependencies(test.task_graph);
        // faces
        result.face_regions[0] = test_simple_bc(mesh, int3{1, 0, 0}, int3{NGHOST, nn.y, nn.z},
                                                int3{nn.x - 1, NGHOST, NGHOST},
                                                int3{nn.x + NGHOST, NGHOST, NGHOST}, info,
                                                test.test_func);

        result.face_regions[1] = test_simple_bc(mesh, int3{0, 1, 0}, int3{nn.x, NGHOST, nn.z},
                                                int3{NGHOST, nn.y - 1, NGHOST},
                                                int3{NGHOST, nn.y + NGHOST, NGHOST}, info,
                                                test.test_func);

        result.face_regions[2] = test_simple_bc(mesh, int3{0, 0, 1}, int3{nn.x, nn.y, NGHOST},
                                                int3{NGHOST, NGHOST, nn.z - 1},
                                                int3{NGHOST, NGHOST, nn.z + NGHOST}, info,
                                                test.test_func);

        result.face_regions[3] = test_simple_bc(mesh, int3{-1, 0, 0}, int3{NGHOST, nn.y, nn.z},
                                                int3{NGHOST + 1, NGHOST, NGHOST},
                                                int3{0, NGHOST, NGHOST}, info, test.test_func);

        result.face_regions[4] = test_simple_bc(mesh, int3{0, -1, 0}, int3{nn.x, NGHOST, nn.z},
                                                int3{NGHOST, NGHOST + 1, NGHOST},
                                                int3{NGHOST, 0, NGHOST}, info, test.test_func);

        result.face_regions[5] = test_simple_bc(mesh, int3{0, 0, -1}, int3{nn.x, nn.y, NGHOST},
                                                int3{NGHOST, NGHOST, NGHOST + 1},
                                                int3{NGHOST, NGHOST, 0}, info, test.test_func);

        // edges
        result.edge_regions[0] = test_simple_bc(mesh, int3{0, 1, 0}, int3{NGHOST, NGHOST, nn.z},
                                                int3{nn.x - 1, nn.y - 1, NGHOST},
                                                int3{nn.x + NGHOST, nn.y + NGHOST, NGHOST}, info,
                                                test.test_func);

        result.edge_regions[1] = test_simple_bc(mesh, int3{0, 0, 1}, int3{NGHOST, nn.y, NGHOST},
                                                int3{nn.x - 1, NGHOST, nn.z - 1},
                                                int3{nn.x + NGHOST, NGHOST, nn.z + NGHOST}, info,
                                                test.test_func);

        result.edge_regions[2] = test_simple_bc(mesh, int3{0, 0, 1}, int3{nn.x, NGHOST, NGHOST},
                                                int3{NGHOST, nn.y - 1, nn.z - 1},
                                                int3{NGHOST, nn.y + NGHOST, nn.z + NGHOST}, info,
                                                test.test_func);

        result.edge_regions[3] = test_simple_bc(mesh, int3{0, -1, 0}, int3{NGHOST, NGHOST, nn.z},
                                                int3{nn.x - 1, NGHOST + 1, NGHOST},
                                                int3{nn.x + NGHOST, 0, NGHOST}, info,
                                                test.test_func);

        result.edge_regions[4] = test_simple_bc(mesh, int3{0, 0, -1}, int3{NGHOST, nn.y, NGHOST},
                                                int3{nn.x - 1, NGHOST, NGHOST + 1},
                                                int3{nn.x + NGHOST, NGHOST, 0}, info,
                                                test.test_func);

        result.edge_regions[5] = test_simple_bc(mesh, int3{0, 0, -1}, int3{nn.x, NGHOST, NGHOST},
                                                int3{NGHOST, nn.y - 1, NGHOST + 1},
                                                int3{NGHOST, nn.y + NGHOST, 0}, info,
                                                test.test_func);

        result.edge_regions[6] = test_simple_bc(mesh, int3{0, 1, 0}, int3{NGHOST, NGHOST, nn.z},
                                                int3{NGHOST + 1, nn.y - 1, NGHOST},
                                                int3{0, nn.y + NGHOST, NGHOST}, info,
                                                test.test_func);

        result.edge_regions[7] = test_simple_bc(mesh, int3{0, 0, 1}, int3{NGHOST, nn.y, NGHOST},
                                                int3{NGHOST + 1, NGHOST, nn.z - 1},
                                                int3{0, NGHOST, nn.z + NGHOST}, info,
                                                test.test_func);

        result.edge_regions[8] = test_simple_bc(mesh, int3{0, 0, 1}, int3{nn.x, NGHOST, NGHOST},
                                                int3{NGHOST, NGHOST + 1, nn.z - 1},
                                                int3{NGHOST, 0, nn.z + NGHOST}, info,
                                                test.test_func);

        result.edge_regions[9] = test_simple_bc(mesh, int3{0, -1, 0}, int3{NGHOST, NGHOST, nn.z},
                                                int3{NGHOST + 1, NGHOST + 1, NGHOST},
                                                int3{0, 0, NGHOST}, info, test.test_func);

        result.edge_regions[10] = test_simple_bc(mesh, int3{0, 0, -1}, int3{NGHOST, nn.y, NGHOST},
                                                 int3{NGHOST + 1, NGHOST, NGHOST + 1},
                                                 int3{0, NGHOST, 0}, info, test.test_func);

        result.edge_regions[11] = test_simple_bc(mesh, int3{0, 0, -1}, int3{nn.x, NGHOST, NGHOST},
                                                 int3{NGHOST, NGHOST + 1, NGHOST + 1},
                                                 int3{NGHOST, 0, 0}, info, test.test_func);

        // result.corners
        result.corner_regions[0] = test_simple_bc(mesh, int3{0, 0, 1}, int3{NGHOST, NGHOST, NGHOST},
                                                  int3{nn.x - 1, nn.y - 1, nn.z - 1},
                                                  int3{nn.x + NGHOST, nn.y + NGHOST, nn.z + NGHOST},
                                                  info, test.test_func);

        result.corner_regions[1] = test_simple_bc(mesh, int3{0, 0, -1},
                                                  int3{NGHOST, NGHOST, NGHOST},
                                                  int3{nn.x - 1, nn.y - 1, NGHOST + 1},
                                                  int3{nn.x + NGHOST, nn.y + NGHOST, 0}, info,
                                                  test.test_func);
        result.corner_regions[2] = test_simple_bc(mesh, int3{0, 0, 1}, int3{NGHOST, NGHOST, NGHOST},
                                                  int3{nn.x - 1, NGHOST + 1, nn.z - 1},
                                                  int3{nn.x + NGHOST, 0, nn.z + NGHOST}, info,
                                                  test.test_func);
        result.corner_regions[3] = test_simple_bc(mesh, int3{0, 0, 1}, int3{NGHOST, NGHOST, NGHOST},
                                                  int3{NGHOST + 1, nn.y - 1, nn.z - 1},
                                                  int3{0, nn.y + NGHOST, nn.z + NGHOST}, info,
                                                  test.test_func);

        result.corner_regions[4] = test_simple_bc(mesh, int3{0, 0, -1},
                                                  int3{NGHOST, NGHOST, NGHOST},
                                                  int3{NGHOST + 1, nn.y - 1, NGHOST + 1},
                                                  int3{0, nn.y + NGHOST, 0}, info, test.test_func);
        result.corner_regions[5] = test_simple_bc(mesh, int3{0, 0, -1},
                                                  int3{NGHOST, NGHOST, NGHOST},
                                                  int3{nn.x - 1, NGHOST + 1, NGHOST + 1},
                                                  int3{nn.x + NGHOST, 0, 0}, info, test.test_func);
        result.corner_regions[6] = test_simple_bc(mesh, int3{0, 0, 1}, int3{NGHOST, NGHOST, NGHOST},
                                                  int3{NGHOST + 1, NGHOST + 1, nn.z - 1},
                                                  int3{0, 0, nn.z + NGHOST}, info, test.test_func);

        result.corner_regions[7] = test_simple_bc(mesh, int3{0, 0, -1},
                                                  int3{NGHOST, NGHOST, NGHOST},
                                                  int3{NGHOST + 1, NGHOST + 1, NGHOST + 1},
                                                  int3{0, 0, 0}, info, test.test_func);

        result.all_faces_passed   = All(result.face_regions);
        result.all_edges_passed   = All(result.edge_regions);
        result.all_corners_passed = All(result.corner_regions);
        result.all_regions_passed = result.all_faces_passed && result.all_edges_passed &&
                                    result.all_edges_passed;

    } // if pid == 0
    return result;
}

// Runs a comparison test for all ghost zone regions
TestResult
RunMeshCompareTest(MeshCompareTestCase test)
{
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    TestResult result{test.name,
                      true,
                      true,
                      true,
                      true,
                      std::array<TestResultRegion, 6>(),
                      std::array<TestResultRegion, 12>(),
                      std::array<TestResultRegion, 8>()};

    acGridLoadMesh(STREAM_DEFAULT, test.mesh_prior);
    AcMeshInfo info = test.mesh_prior.info;
    const int3 nn   = int3{
        (int)(info.int_params[AC_nx]),
        (int)(info.int_params[AC_ny]),
        (int)(info.int_params[AC_nz]),
    };

    AcMesh mesh;

    if (pid == 0) {
        acHostMeshCreate(info, &mesh);
    }

    acGridExecuteTaskGraph(test.task_graph, 1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &mesh);

    if (pid == 0) {
        // acGraphPrintDependencies(test.task_graph);
        // faces
        result.face_regions[0] = test_bc_against_mesh(mesh, int3{1, 0, 0}, int3{NGHOST, nn.y, nn.z},
                                                      int3{nn.x - 1, NGHOST, NGHOST},
                                                      int3{nn.x + NGHOST, NGHOST, NGHOST}, info,
                                                      test.mesh_expected_result);

        result.face_regions[1] = test_bc_against_mesh(mesh, int3{0, 1, 0}, int3{nn.x, NGHOST, nn.z},
                                                      int3{NGHOST, nn.y - 1, NGHOST},
                                                      int3{NGHOST, nn.y + NGHOST, NGHOST}, info,
                                                      test.mesh_expected_result);

        result.face_regions[2] = test_bc_against_mesh(mesh, int3{0, 0, 1}, int3{nn.x, nn.y, NGHOST},
                                                      int3{NGHOST, NGHOST, nn.z - 1},
                                                      int3{NGHOST, NGHOST, nn.z + NGHOST}, info,
                                                      test.mesh_expected_result);

        result.face_regions[3] = test_bc_against_mesh(mesh, int3{-1, 0, 0},
                                                      int3{NGHOST, nn.y, nn.z},
                                                      int3{NGHOST + 1, NGHOST, NGHOST},
                                                      int3{0, NGHOST, NGHOST}, info,
                                                      test.mesh_expected_result);

        result.face_regions[4] = test_bc_against_mesh(mesh, int3{0, -1, 0},
                                                      int3{nn.x, NGHOST, nn.z},
                                                      int3{NGHOST, NGHOST + 1, NGHOST},
                                                      int3{NGHOST, 0, NGHOST}, info,
                                                      test.mesh_expected_result);

        result.face_regions[5] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                      int3{nn.x, nn.y, NGHOST},
                                                      int3{NGHOST, NGHOST, NGHOST + 1},
                                                      int3{NGHOST, NGHOST, 0}, info,
                                                      test.mesh_expected_result);

        // edges
        result.edge_regions[0] = test_bc_against_mesh(mesh, int3{0, 1, 0},
                                                      int3{NGHOST, NGHOST, nn.z},
                                                      int3{nn.x - 1, nn.y - 1, NGHOST},
                                                      int3{nn.x + NGHOST, nn.y + NGHOST, NGHOST},
                                                      info, test.mesh_expected_result);

        result.edge_regions[1] = test_bc_against_mesh(mesh, int3{0, 0, 1},
                                                      int3{NGHOST, nn.y, NGHOST},
                                                      int3{nn.x - 1, NGHOST, nn.z - 1},
                                                      int3{nn.x + NGHOST, NGHOST, nn.z + NGHOST},
                                                      info, test.mesh_expected_result);

        result.edge_regions[2] = test_bc_against_mesh(mesh, int3{0, 0, 1},
                                                      int3{nn.x, NGHOST, NGHOST},
                                                      int3{NGHOST, nn.y - 1, nn.z - 1},
                                                      int3{NGHOST, nn.y + NGHOST, nn.z + NGHOST},
                                                      info, test.mesh_expected_result);

        result.edge_regions[3] = test_bc_against_mesh(mesh, int3{0, -1, 0},
                                                      int3{NGHOST, NGHOST, nn.z},
                                                      int3{nn.x - 1, NGHOST + 1, NGHOST},
                                                      int3{nn.x + NGHOST, 0, NGHOST}, info,
                                                      test.mesh_expected_result);

        result.edge_regions[4] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                      int3{NGHOST, nn.y, NGHOST},
                                                      int3{nn.x - 1, NGHOST, NGHOST + 1},
                                                      int3{nn.x + NGHOST, NGHOST, 0}, info,
                                                      test.mesh_expected_result);

        result.edge_regions[5] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                      int3{nn.x, NGHOST, NGHOST},
                                                      int3{NGHOST, nn.y - 1, NGHOST + 1},
                                                      int3{NGHOST, nn.y + NGHOST, 0}, info,
                                                      test.mesh_expected_result);

        result.edge_regions[6] = test_bc_against_mesh(mesh, int3{0, 1, 0},
                                                      int3{NGHOST, NGHOST, nn.z},
                                                      int3{NGHOST + 1, nn.y - 1, NGHOST},
                                                      int3{0, nn.y + NGHOST, NGHOST}, info,
                                                      test.mesh_expected_result);

        result.edge_regions[7] = test_bc_against_mesh(mesh, int3{0, 0, 1},
                                                      int3{NGHOST, nn.y, NGHOST},
                                                      int3{NGHOST + 1, NGHOST, nn.z - 1},
                                                      int3{0, NGHOST, nn.z + NGHOST}, info,
                                                      test.mesh_expected_result);

        result.edge_regions[8] = test_bc_against_mesh(mesh, int3{0, 0, 1},
                                                      int3{nn.x, NGHOST, NGHOST},
                                                      int3{NGHOST, NGHOST + 1, nn.z - 1},
                                                      int3{NGHOST, 0, nn.z + NGHOST}, info,
                                                      test.mesh_expected_result);

        result.edge_regions[9] = test_bc_against_mesh(mesh, int3{0, -1, 0},
                                                      int3{NGHOST, NGHOST, nn.z},
                                                      int3{NGHOST + 1, NGHOST + 1, NGHOST},
                                                      int3{0, 0, NGHOST}, info,
                                                      test.mesh_expected_result);

        result.edge_regions[10] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                       int3{NGHOST, nn.y, NGHOST},
                                                       int3{NGHOST + 1, NGHOST, NGHOST + 1},
                                                       int3{0, NGHOST, 0}, info,
                                                       test.mesh_expected_result);

        result.edge_regions[11] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                       int3{nn.x, NGHOST, NGHOST},
                                                       int3{NGHOST, NGHOST + 1, NGHOST + 1},
                                                       int3{NGHOST, 0, 0}, info,
                                                       test.mesh_expected_result);

        // result.corners
        result.corner_regions[0] = test_bc_against_mesh(mesh, int3{0, 0, 1},
                                                        int3{NGHOST, NGHOST, NGHOST},
                                                        int3{nn.x - 1, nn.y - 1, nn.z - 1},
                                                        int3{nn.x + NGHOST, nn.y + NGHOST,
                                                             nn.z + NGHOST},
                                                        info, test.mesh_expected_result);

        result.corner_regions[1] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                        int3{NGHOST, NGHOST, NGHOST},
                                                        int3{nn.x - 1, nn.y - 1, NGHOST + 1},
                                                        int3{nn.x + NGHOST, nn.y + NGHOST, 0}, info,
                                                        test.mesh_expected_result);
        result.corner_regions[2] = test_bc_against_mesh(mesh, int3{0, 0, 1},
                                                        int3{NGHOST, NGHOST, NGHOST},
                                                        int3{nn.x - 1, NGHOST + 1, nn.z - 1},
                                                        int3{nn.x + NGHOST, 0, nn.z + NGHOST}, info,
                                                        test.mesh_expected_result);
        result.corner_regions[3] = test_bc_against_mesh(mesh, int3{0, 0, 1},
                                                        int3{NGHOST, NGHOST, NGHOST},
                                                        int3{NGHOST + 1, nn.y - 1, nn.z - 1},
                                                        int3{0, nn.y + NGHOST, nn.z + NGHOST}, info,
                                                        test.mesh_expected_result);

        result.corner_regions[4] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                        int3{NGHOST, NGHOST, NGHOST},
                                                        int3{NGHOST + 1, nn.y - 1, NGHOST + 1},
                                                        int3{0, nn.y + NGHOST, 0}, info,
                                                        test.mesh_expected_result);
        result.corner_regions[5] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                        int3{NGHOST, NGHOST, NGHOST},
                                                        int3{nn.x - 1, NGHOST + 1, NGHOST + 1},
                                                        int3{nn.x + NGHOST, 0, 0}, info,
                                                        test.mesh_expected_result);
        result.corner_regions[6] = test_bc_against_mesh(mesh, int3{0, 0, 1},
                                                        int3{NGHOST, NGHOST, NGHOST},
                                                        int3{NGHOST + 1, NGHOST + 1, nn.z - 1},
                                                        int3{0, 0, nn.z + NGHOST}, info,
                                                        test.mesh_expected_result);

        result.corner_regions[7] = test_bc_against_mesh(mesh, int3{0, 0, -1},
                                                        int3{NGHOST, NGHOST, NGHOST},
                                                        int3{NGHOST + 1, NGHOST + 1, NGHOST + 1},
                                                        int3{0, 0, 0}, info,
                                                        test.mesh_expected_result);

        result.all_faces_passed   = All(result.face_regions);
        result.all_edges_passed   = All(result.edge_regions);
        result.all_corners_passed = All(result.corner_regions);
        result.all_regions_passed = result.all_faces_passed && result.all_edges_passed &&
                                    result.all_edges_passed;

    } // if pid == 0
    return result;
}

// Runs a comparison test for all ghost zone regions

void
colored_feedback(const char* s, bool passed)
{
    if (passed) {
        printf("\"%s\": %sPASSED%s\n", s, GRN, RESET);
    }
    else {
        printf("\"%s\": %sFAILED%s\n", s, RED, RESET);
    }
}

template <size_t n_regions>
void
colored_pass_ratio(const char* s, const std::array<TestResultRegion, n_regions>& regions)
{
    size_t n_passed = 0;
    for (size_t i = 0; i < n_regions; i++) {
        if (regions[i].passed) {
            ++n_passed;
        }
    }

    if (n_passed == n_regions) {
        printf("\t%s: %s%lu%s / %s%lu%s", s, GRN, n_passed, RESET, GRN, n_regions, RESET);
    }
    else {
        printf("\t%s: %s%lu%s / %s%lu%s", s, RED, n_passed, RESET, GRN, n_regions, RESET);
    }
}

void
PrintTestResultRegion(const TestResultRegion& region)
{
    int col_width[4] = {0, 0, 0, 0};

    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        int s_len = strlen(vtxbuf_names[i]);
        if (col_width[i % 4] < s_len) {
            col_width[i % 4] = s_len;
        }
    }

    printf("\n\tRegion with normal (%2d,%2d,%d):\n\t", region.normal.x, region.normal.y,
           region.normal.z);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        size_t errors = region.field_errors[i].errors;
        size_t total  = region.field_errors[i].total;

        printf("%*s: %s%lu%s/%s%lu%s.", col_width[i % 4], vtxbuf_names[i], (errors > 0 ? RED : GRN),
               (total - errors), RESET, GRN, total, RESET);
        if ((i + 1) % 4 == 0) {
            printf("\n\t");
        }
    }

#if debug_bc_errors
    constexpr size_t n_errors_to_print = 5;
    printf("\n\terror samples:\n");
    for (size_t i = 0; i < region.failed_cells.size() && i < n_errors_to_print; i++) {
        auto& error = region.failed_cells[i];
        printf("\t\t%*s[(%3d,%3d,%3d)], domain index %*s[(%3d,%3d,%3d)]\n\t\texpected %s%f%s, got "
               "%s%f%s\n",
               col_width[error.field % 4], vtxbuf_names[error.field], error.ghost.x, error.ghost.y,
               error.ghost.z, col_width[error.field % 4], vtxbuf_names[error.field], error.dom.x,
               error.dom.y, error.dom.z, GRN, (double)error.produced, RESET, RED,
               (double)error.expected, RESET);
        fflush(stdout);
    }
#endif

#if debug_bc_values
    /*
    printf("domain[(%3d,%3d,%3d)] = %f != ghost[(%3d,%3d,%3d)] = %f\n", dom.x,
           dom.y, dom.z, field[idx_dom], ghost.x, ghost.y, ghost.z,
           field[idx_ghost]);
    */
#endif
}

void
PrintTestResult(const TestResult& result)
{
    printf("Test case ");
    colored_feedback(result.name.c_str(), result.all_regions_passed);
    if (!result.all_regions_passed) {
        colored_pass_ratio("\n\tface regions passed", result.face_regions);
        printf("\n\t──────────────────────────");
        for (auto& test_region : result.face_regions) {
            if (!test_region.passed) {
                PrintTestResultRegion(test_region);
            }
        }
        colored_pass_ratio("\n\tedge regions passed", result.edge_regions);
        printf("\n\t───────────────────────────");
        for (auto& test_region : result.edge_regions) {
            if (!test_region.passed) {
                PrintTestResultRegion(test_region);
            }
        }
        colored_pass_ratio("\n\tcorner regions passed", result.corner_regions);
        printf("\n\t────────────────────────────");
        for (auto& test_region : result.corner_regions) {
            if (!test_region.passed) {
                PrintTestResultRegion(test_region);
            }
        }
        printf("\n\n");
    }
}

int
main(void)
{
    int ret_val = 0;
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    // Astaroth setup
    AcMesh mesh;
    if (pid == 0) {
        acHostMeshCreate(info, &mesh);
        acHostMeshRandomize(&mesh);
    }

    // Specific setup for one of the tests
    constexpr AcReal prescribed_val          = 6.0;
    info.real_params[AC_boundary_derivative] = prescribed_val;

    acGridInit(info);

    acGridLoadMesh(STREAM_DEFAULT, mesh);
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, FLT_EPSILON);
    acGridSynchronizeStream(STREAM_DEFAULT);
    // End setup

    // Preparing tests
    std::vector<SimpleTestCase> test_cases;

    VertexBufferHandle all_fields[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields[i] = (VertexBufferHandle)i;
    }

    // Draft of pilot boundconds
    Field lnrho[]{VTXBUF_LNRHO};
    Field uux_uuy[]{VTXBUF_UUX, VTXBUF_UUY};
    Field uuz[]{VTXBUF_UUZ};
    Field ax_ay[]{VTXBUF_AX, VTXBUF_AY};
    Field az[]{VTXBUF_AZ};

    AcTaskGraph* pilot_bcs = acGridBuildTaskGraph(
        {acHaloExchange(all_fields),

         acBoundaryCondition(BOUNDARY_X, BOUNDCOND_PERIODIC, all_fields),
         acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_PERIODIC, all_fields),

         acSpecialMHDBoundaryCondition(BOUNDARY_Z_TOP,
                                       SPECIAL_MHD_BOUNDCOND_ENTROPY_BLACKBODY_RADIATION),
         acSpecialMHDBoundaryCondition(BOUNDARY_Z_BOT,
                                       SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_HEAT_FLUX),
         acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_A2, lnrho),
         acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, uux_uuy),
         acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_ANTISYMMETRIC, uuz),
         acBoundaryCondition(BOUNDARY_Z_TOP, BOUNDCOND_SYMMETRIC, ax_ay),
         acBoundaryCondition(BOUNDARY_Z_BOT, BOUNDCOND_ANTISYMMETRIC, ax_ay),
         acBoundaryCondition(BOUNDARY_Z_TOP, BOUNDCOND_ANTISYMMETRIC, az),
         acBoundaryCondition(BOUNDARY_Z_BOT, BOUNDCOND_SYMMETRIC, az)});

    // Symmetric bc
    AcTaskGraph* symmetric_bc_graph = acGridBuildTaskGraph(
        {acHaloExchange(all_fields),
         acBoundaryCondition(BOUNDARY_X, BOUNDCOND_SYMMETRIC, all_fields),
         acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_SYMMETRIC, all_fields),
         acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, all_fields)});

    auto mirror = [](AcReal, AcReal domain_val, size_t, AcMeshInfo) { return domain_val; };

    test_cases.push_back(SimpleTestCase{"Symmetric boundconds", symmetric_bc_graph, mirror});

    // AntiSymmetric bc
    AcTaskGraph* antisymmetric_bc_graph = acGridBuildTaskGraph(
        {acHaloExchange(all_fields),
         acBoundaryCondition(BOUNDARY_X, BOUNDCOND_ANTISYMMETRIC, all_fields),
         acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_ANTISYMMETRIC, all_fields),
         acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_ANTISYMMETRIC, all_fields)});

    auto antimirror = [](AcReal, AcReal domain_val, size_t, AcMeshInfo) { return -domain_val; };

    test_cases.push_back(
        SimpleTestCase{"AntiSymmetric boundconds", antisymmetric_bc_graph, antimirror});

    // Prescribed derivative bc
    AcRealParam bc_param[1] = {AC_boundary_derivative};

    AcTaskGraph* prescribed_der_bc_graph = acGridBuildTaskGraph(
        {acHaloExchange(all_fields),
         acBoundaryCondition(BOUNDARY_X, BOUNDCOND_PRESCRIBED_DERIVATIVE, all_fields, bc_param),
         acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_PRESCRIBED_DERIVATIVE, all_fields, bc_param),
         acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_PRESCRIBED_DERIVATIVE, all_fields, bc_param)});

    // Assumption: AC_dsx = AC_dsy = AC_dsz

    auto der_bc_func = [](AcReal, AcReal domain_val, size_t r, AcMeshInfo inf) {
        AcReal d        = inf.real_params[AC_dsx];
        AcReal der_val  = inf.real_params[AC_boundary_derivative];
        AcReal distance = AcReal(2 * r) * d;
        return domain_val + distance * der_val;
    };

    test_cases.push_back(
        SimpleTestCase{"Prescribed derivative boundconds", prescribed_der_bc_graph, der_bc_func});

    //"A2"
    AcTaskGraph* relative_antisymmetry_bc_graph = acGridBuildTaskGraph(
        {acHaloExchange(all_fields),
         acBoundaryCondition(BOUNDARY_X, BOUNDCOND_A2, all_fields, bc_param),
         acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_A2, all_fields, bc_param),
         acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_A2, all_fields, bc_param)});

    auto a2_func = [](AcReal boundary_val, AcReal domain_val, size_t, AcMeshInfo) {
        return 2 * boundary_val - domain_val;
    };

    test_cases.push_back(SimpleTestCase{"Relative antisymmetry boundconds",
                                        relative_antisymmetry_bc_graph, a2_func});

    // Running the simple tests
    std::vector<TestResult> test_results;
    for (const auto& test : test_cases) {
        test_results.push_back(RunSimpleTest(test, info));
    }

    /********************************************************************
     *                                                                   *
     * Comparison test cases (harder to test by comparing functionality) *
     *                                                                   *
     *********************************************************************/
    /*
    std::vector<MeshCompareTestCase> comparison_test_cases;

    if (pid == 0){
        acHostMeshCreate(info, &mesh);
    }
    acGridSynchronizeStream(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &mesh);
    comparison_test_cases.push_back(MeshCompareTestCase{"Testing the comparison test using an
    idempotent boundcond", relative_antisymmetry_bc_graph, mesh, mesh});

    for (const auto &test: comparison_test_cases){
        test_results.push_back(RunMeshCompareTest(test));
    }

    for (const auto &test: comparison_test_cases){
        //
        //acGridDestroyTaskGraph(test.task_graph);
    }
    */

    // Cleanup and test output
    for (const auto& test : test_cases) {
        acGridDestroyTaskGraph(test.task_graph);
    }
    if (pid == 0) {
        for (const auto& result : test_results) {
            PrintTestResult(result);
        }
    }
    // End
    acGridQuit();
    MPI_Finalize();

    return ret_val;
}
#endif // !AC_MPI_ENABLED
