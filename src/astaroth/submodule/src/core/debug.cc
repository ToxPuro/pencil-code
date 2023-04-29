#if AC_MPI_ENABLED
#include "astaroth.h"
#include "astaroth_debug.h"
#include "task.h"

#include <stdio.h>
#include <string>

static void
writeTaskKey(FILE* fp, const Task* task)
{
    fprintf(fp, "{");
    fprintf(fp, "\"order\":%d,", task->order);
    fprintf(fp, "\"tag\":%d,", task->output_region.tag);
    fprintf(fp, "\"task_type\":");
    switch (task->task_type) {
    case TASKTYPE_COMPUTE:
        fprintf(fp, "\"COMPUTE\"");
        break;
    case TASKTYPE_HALOEXCHANGE:
        fprintf(fp, "\"HALOEXCHANGE\"");
        break;
    case TASKTYPE_BOUNDCOND:
        fprintf(fp, "\"BOUNDCOND\"");
        break;
    default:
        fprintf(fp, "\"UNKNOWN\"");
        break;
    }
    fprintf(fp,"\"position\":%d,%d,%d",task->output_region.position.x,task->output_region.position.y,task->output_region.position.z);
    fprintf(fp,"\"dims\":%d,%d,%d",task->output_region.dims.x,task->output_region.dims.y,task->output_region.dims.z);

    fprintf(fp, "}");
}

static void
graphWriteDependencies(FILE* fp, const AcTaskGraph* graph)
{
    fprintf(fp, "[");
    bool first_item = true;
    for (auto& prerequisite : graph->all_tasks) {
        if (prerequisite->active) {
            for (auto& dependency : prerequisite->dependents) {
                if (first_item) {
                    first_item = false;
                }
                else {
                    fprintf(fp, ",");
                }
                std::shared_ptr<Task> dependent = dependency.first.lock();
                fprintf(fp, "\n\t{\n");
                fprintf(fp, "\t\t\"prerequisite\":");
                writeTaskKey(fp, prerequisite.get());
                fprintf(fp, ",\n");
                fprintf(fp, "\t\t\"dependent\":");
                writeTaskKey(fp, dependent.get());
                fprintf(fp, ",\n");
                fprintf(fp, "\t\t\"offset\":\"%lu\"\n", dependency.second);
                fprintf(fp, "\t}");
            }
        }
    }
    fprintf(fp, "\n]\n");
}

void
acGraphPrintDependencies(const AcTaskGraph* graph)
{
    graphWriteDependencies(stdout, graph);
}

void
acGraphWriteDependencies(const char* path, const AcTaskGraph* graph)
{
    FILE* fp;
    fp = fopen(path, "w");
    if (fp == NULL) {
        WARNING("Cannot open file to write graph dependencies.");
        fprintf(stderr, "file \"%s\" could not be opened\n", path);
        return;
    }
    graphWriteDependencies(fp, graph);
    fclose(fp);
}

static void
graphWriteOrder(FILE* fp, const AcTaskGraph* graph)
{
    if ((*(graph->all_tasks.begin()))->rank == 0) {
        fprintf(fp, "Order\n");
        for (auto t : graph->all_tasks) {
            fprintf(fp, "\t%s\t%lu\n", t->name.c_str(), t->output_region.volume);
        }
    }
}

void
acGraphPrintOrder(const AcTaskGraph* graph)
{
    graphWriteOrder(stdout, graph);
}

void
acGraphWriteOrder(const char* path, const AcTaskGraph* graph)
{
    FILE* fp;
    fp = fopen(path, "w");
    if (fp == NULL) {
        WARNING("Cannot open file to write graph task order.");
        fprintf(stderr, "file \"%s\" could not be opened\n", path);
        return;
    }

    graphWriteOrder(fp, graph);
}

void
acGraphEnableTrace(const char* trace_filepath, AcTaskGraph* const graph)
{
    if (graph->trace_file.fp != NULL) {
        fclose(graph->trace_file.fp);
    }
    graph->trace_file.enabled  = true;
    graph->trace_file.filepath = trace_filepath;
    graph->trace_file.fp       = fopen(trace_filepath, "w");
}

void
acGraphDisableTrace(AcTaskGraph* const graph)
{
    if (graph->trace_file.fp != NULL) {
        fclose(graph->trace_file.fp);
    }
    graph->trace_file.enabled  = false;
    graph->trace_file.filepath = "";
    graph->trace_file.fp       = NULL;
}

void
TraceFile::trace(const Task* task, const std::string old_state, const std::string new_state) const
{
    if (enabled) {
        fprintf(fp, "{");
        fprintf(fp, "\"msg_type\":\"state_changed_event\",");
        fprintf(fp, "\"task\":");
        writeTaskKey(fp, task);
        fprintf(fp, ",");
        fprintf(fp, "\"iteration\":%lu,", task->loop_cntr.i);
        fprintf(fp, "\"timestamp\":%lu,", timer_diff_nsec(this->timer));
        fprintf(fp, "\"from\":\"%s\",", old_state.c_str());
        fprintf(fp, "\"to\":\"%s\"", new_state.c_str());
        fprintf(fp, "}\n");
    }
}
#endif
