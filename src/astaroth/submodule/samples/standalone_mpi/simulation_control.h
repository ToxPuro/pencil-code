#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <ctime>

#include <sys/stat.h>
#include <unistd.h>

#include <string.h>
#include <string>

#include "astaroth_utils.h"

// Structure for keeping track of any generic condition
struct SimulationPeriod {

    static constexpr AcIntParam NoStepParam  = static_cast<AcIntParam>(-1);
    static constexpr AcRealParam NoTimeParam = static_cast<AcRealParam>(-1);

    AcIntParam step_period_param  = NoStepParam;
    AcRealParam time_period_param = NoTimeParam;

    int step_period;
    AcReal time_period;

    AcReal current_time_period_start;

    SimulationPeriod() : step_period(0), time_period(0), current_time_period_start(0) {}

    // Generic version
    SimulationPeriod(int s, AcReal p, AcReal time_offset = 0.0)
        : step_period(s), time_period(p), current_time_period_start(time_offset)
    {
    }

    // Parametrized version
    SimulationPeriod(AcMeshInfo config, AcIntParam step_p, AcRealParam time_p,
                     AcReal time_offset = 0.0)
        : step_period_param{step_p}, time_period_param{time_p}, step_period{0}, time_period{0.0},
          current_time_period_start{time_offset}
    {
        update(config);
    }

    void update(AcMeshInfo config)
    {
        size_t step_period_param_idx = static_cast<size_t>(step_period_param);
        if (step_period_param_idx < NUM_INT_PARAMS) {
            step_period = config.int_params[step_period_param_idx];
        }

        size_t time_period_param_idx = static_cast<size_t>(time_period_param);
        if (time_period_param_idx < NUM_REAL_PARAMS) {
            time_period = config.real_params[time_period_param_idx];
        }
    }

    bool check(int time_step, AcReal time)
    {
        if ((time_period > 0 && time >= current_time_period_start + time_period) ||
            (step_period > 0 && time_step % step_period == 0)) {
            current_time_period_start += time_period;
            return true;
        }

        return false;
    }
};

// std::string to_timestamp(time_t t) {
//  char timestamp[80];
//  strftime(timestamp, 80, "%Y-%m-%d-%I:%M:%S", localtime(&t));
//  return std::string(timestamp);
//}

// Structure for keeping track of a file used by the user to signal things
struct UserSignalFile {
    std::string file_path;
    time_t mod_time;

    UserSignalFile() : file_path{""}, mod_time{0} {}

    UserSignalFile(std::string filename) : file_path(filename), mod_time{stat_file_mod_time()} {};

    bool file_exists() { return access(file_path.c_str(), F_OK) == 0; }

    time_t stat_file_mod_time()
    {
        struct stat s;
        if (stat(file_path.c_str(), &s) == 0) {
            return s.st_mtime;
        }
        return 0;
    }

    bool check()
    {
        time_t statted_mod_time = stat_file_mod_time();
        time_t prev_mod_time    = mod_time;
        mod_time                = std::max(statted_mod_time, prev_mod_time);
        return statted_mod_time > prev_mod_time;
    };
};

// Logging in specific formats
constexpr size_t sim_log_msg_len         = 512;
static size_t sim_tstamp_len             = 0;
static char sim_log_msg[sim_log_msg_len] = "";

static void
set_simulation_timestamp(int step, AcReal time)
{
    // TODO: only set step and time, and lazily create the log stamp whenever it's needed
    snprintf(sim_log_msg, sim_log_msg_len, "[i:%d, t:%.2e] ", step, time);
    sim_tstamp_len = strlen(sim_log_msg);
}

static void
log_from_root_proc_with_sim_progress(int pid, std::string msg, ...)
{
    if (pid == 0) {
        strncpy(sim_log_msg + sim_tstamp_len, msg.c_str(), sim_log_msg_len - sim_tstamp_len);
        va_list args;
        va_start(args, msg);
        acVA_LogFromRootProc(pid, sim_log_msg, args);
        va_end(args);
    }
}

static void
log_from_root_proc_with_sim_progress(std::string msg, ...)
{
    int pid = 0;
#if AC_MPI_ENABLED
    MPI_Comm_rank(acGridMPIComm(), &pid);
#endif
    if (pid == 0) {
        strncpy(sim_log_msg + sim_tstamp_len, msg.c_str(), sim_log_msg_len - sim_tstamp_len);
        va_list args;
        va_start(args, msg);
        acVA_LogFromRootProc(pid, sim_log_msg, args);
        va_end(args);
    }
}

static void
debug_log_from_root_proc_with_sim_progress(int pid, std::string msg, ...)
{
#ifndef NDEBUG
    if (pid == 0) {
        strncpy(sim_log_msg + (sim_tstamp_len), msg.c_str(), sim_log_msg_len - sim_tstamp_len);
        va_list args;
        va_start(args, msg);
        acVA_DebugFromRootProc(pid, sim_log_msg, args);
        va_end(args);
    }
#else
    (void)pid;
    (void)msg;
#endif
}
