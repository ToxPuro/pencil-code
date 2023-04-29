#if AC_MPI_ENABLED
/** */
void acGraphPrintDependencies(const AcTaskGraph* graph);

/** */
void acGraphWriteDependencies(const char* path, const AcTaskGraph* graph);

/** */
void acGraphEnableTrace(const char* trace_path, AcTaskGraph* const graph);

/** */
void acGraphDisableTrace(AcTaskGraph* const graph);
#endif
