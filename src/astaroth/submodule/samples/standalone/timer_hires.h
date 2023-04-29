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
  @file
 \brief High-resolution timer.

    Usage:
        Timer t;
        timer_reset(&t);
        timer_diff_nsec(t);

    If there are issues, try compiling with -std=gnu11 -lrt
 */
#pragma once
#include <stdio.h> // perror
#include <time.h>

typedef struct timespec Timer;
// Contains at least the following members:
// time_t tv_sec;
// long tv_nsec;

static inline int
timer_reset(Timer* t)
{
    const int retval = clock_gettime(CLOCK_REALTIME, t);
    if (retval == -1)
        perror("clock_gettime failure");

    return retval;
}

static inline long
timer_diff_nsec(const Timer start)
{
    Timer end;
    timer_reset(&end);
    const long diff = (end.tv_sec - start.tv_sec) * 1000000000l + (end.tv_nsec - start.tv_nsec);
    return diff;
}

static inline void
timer_diff_print(const Timer t)
{
    printf("Time elapsed: %g ms\n", timer_diff_nsec(t) / 1e6);
}
