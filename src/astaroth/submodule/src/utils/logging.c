#include "astaroth_utils.h"
#include <stdarg.h>
#include <string.h>
#include <time.h>

// Logging utils

void
acVA_LogFromRootProc(const int pid, const char* msg, va_list args)
{
    if (pid == 0) {
        time_t now       = time(NULL);
        char* timestamp  = ctime(&now);
        size_t stamp_len = strlen(timestamp);
        // Remove trailing newline
        timestamp[stamp_len - 1] = '\0';
        // We know the exact length of the timestamp (26 chars), so we could force this function to
        // take chars with a 26 prefix blank buffer
        fprintf(stderr, "%s : ", timestamp);
        vfprintf(stderr, msg, args);
        fflush(stderr);
    }
}

void
acLogFromRootProc(const int pid, const char* msg, ...)
{
    va_list args;
    va_start(args, msg);
    acVA_LogFromRootProc(pid, msg, args);
    va_end(args);
}

void
acVA_VerboseLogFromRootProc(const int pid, const char* msg, va_list args)
{
#if AC_VERBOSE
    acVA_LogFromRootProc(pid, msg, args);
#else
    (void)pid;  // Unused
    (void)msg;  // Unused
    (void)args; // Unused
#endif
}

void
acVerboseLogFromRootProc(const int pid, const char* msg, ...)
{
    va_list args;
    va_start(args, msg);
    acVA_VerboseLogFromRootProc(pid, msg, args);
    va_end(args);
}

void
acVA_DebugFromRootProc(const int pid, const char* msg, va_list args)
{
#ifndef NDEBUG
    acVA_LogFromRootProc(pid, msg, args);
#else
    (void)pid;  // Unused
    (void)msg;  // Unused
    (void)args; // Unused
#endif
}

void
acDebugFromRootProc(const int pid, const char* msg, ...)
{
    va_list args;
    va_start(args, msg);
    acVA_DebugFromRootProc(pid, msg, args);
    va_end(args);
}
