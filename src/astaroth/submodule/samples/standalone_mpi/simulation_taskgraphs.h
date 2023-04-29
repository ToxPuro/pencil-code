#include <astaroth.h>

// TODO: allow selecting single our doublepass here?
enum class Simulation { Solve, Shock_Singlepass_Solve, Default = Solve };

void
log_simulation_choice(int pid, Simulation sim)
{
    const char* sim_label;
    switch (sim) {
    case Simulation::Solve:
        sim_label = "Solve";
        break;
    case Simulation::Shock_Singlepass_Solve:
        sim_label = "Shock with singlepass solve";
        break;
    default:
        sim_label = "WARNING: No label exists for simulation";
        break;
    }
    acLogFromRootProc(pid, "Simulation program: %s", sim_label);
}

static std::map<Simulation, AcTaskGraph*> task_graphs;

AcTaskGraph*
get_simulation_graph(int pid, Simulation sim)
{

    auto make_graph = [pid](Simulation sim) -> AcTaskGraph* {
        acLogFromRootProc(pid, "Creating task graph for simulation\n");
        switch (sim) {
        case Simulation::Shock_Singlepass_Solve: {
#if LSHOCK
            // This still has to be behind a preprocessor feature, because e.g., VTXBUF_SHOCK is not
            // defined in general
            VertexBufferHandle all_fields[] =
                {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                 VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ, // VTXBUF_ENTROPY,
                 VTXBUF_SHOCK, BFIELDX,    BFIELDY,    BFIELDZ};

            VertexBufferHandle shock_field[] = {VTXBUF_SHOCK};
            AcTaskDefinition shock_ops[] =
                {acHaloExchange(all_fields),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
                 acCompute(KERNEL_shock_1_divu, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
                 acCompute(KERNEL_shock_2_max, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
                 acCompute(KERNEL_shock_3_smooth, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
                 acCompute(KERNEL_singlepass_solve, all_fields)};
            acLogFromRootProc(pid, "Creating shock singlepass solve task graph\n");
            return acGridBuildTaskGraph(shock_ops);
#endif
        }
        default:
            acLogFromRootProc(pid, "ERROR: no custom task graph exists for selected simulation. "
                                   "This is probably a fatal error.\n");
            return nullptr;
        }
    };

    if (sim == Simulation::Default) {
        printf("Using default graph\n");
        return acGridGetDefaultTaskGraph();
    }

    if (task_graphs.count(sim) == 0) {
        task_graphs[sim] = make_graph(sim);
    }
    return task_graphs[sim];
}

void
free_simulation_graphs(int pid)
{
    for (auto& [sim, graph] : task_graphs) {
        acLogFromRootProc(pid, "Destroying custom task graph\n");
        acGridDestroyTaskGraph(graph);
    }
    task_graphs.clear();
}
