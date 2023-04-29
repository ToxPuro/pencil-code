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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "errchk.h"
#include "run.h"

// Write all errors from stderr to an <errorlog_name> in the current working
// directory
static const bool write_log_to_a_file = false;
static const char* errorlog_name      = "error.log";

static void
errorlog_init(void)
{
    FILE* fp = freopen(errorlog_name, "w", stderr); // Log errors to a file
    if (!fp)
        perror("Error redirecting stderr to a file");
}

static void
errorlog_quit(void)
{
    fclose(stderr);

    // Print contents of the latest errorlog to screen
    FILE* fp = fopen(errorlog_name, "r");
    if (fp) {
        for (int c = getc(fp); c != EOF; c = getc(fp))
            putchar(c);
        fclose(fp);
    }
    else {
        perror("Error opening error log");
    }
}

typedef struct {
    char* key[2];
    char* description;
} Option;

static Option
createOption(const char* key, const char* key_short, const char* description)
{
    Option option;

    option.key[0]      = strdup(key);
    option.key[1]      = strdup(key_short);
    option.description = strdup(description);

    return option;
}

static void
destroyOption(Option* option)
{
    free(option->key[0]);
    free(option->key[1]);
    free(option->description);
}

typedef enum {
    HELP,
    TEST,
    BENCHMARK,
    SIMULATE,
    RENDER,
    CONFIG,
    NUM_OPTIONS,
} OptionType;

static int
findOption(const char* str, const Option options[NUM_OPTIONS])
{
    for (int i = 0; i < NUM_OPTIONS; ++i)
        if (!strcmp(options[i].key[0], str) || !strcmp(options[i].key[1], str))
            return i;

    return -1;
}

static void
print_options(const Option options[NUM_OPTIONS])
{
    // Formatting
    int keylen[2] = {0};
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        int len0 = strlen(options[i].key[0]);
        int len1 = strlen(options[i].key[1]);
        if (keylen[0] < len0)
            keylen[0] = len0;
        if (keylen[1] < len1)
            keylen[1] = len1;
    }

    for (int i = 0; i < NUM_OPTIONS; ++i)
        printf("\t%*s | %*s: %s\n", keylen[0], options[i].key[0], keylen[1], options[i].key[1],
               options[i].description);
}

static void
print_help(const Option options[NUM_OPTIONS])
{
    puts("Usage: ./ac_run [options]");
    print_options(options);
    printf("\n");
    puts("For bug reporting, see README.md");
}

int
main(int argc, char* argv[])
{
    if (write_log_to_a_file) {
        errorlog_init();
        atexit(errorlog_quit);
    }

    // Create options
    // clang-format off
    Option options[NUM_OPTIONS];
    options[HELP]               = createOption("--help", "-h", "Prints this help.");
    options[TEST]               = createOption("--test", "-t", "Runs autotests.");
    options[BENCHMARK]          = createOption("--benchmark", "-b", "Runs benchmarks.");
    options[SIMULATE]           = createOption("--simulate", "-s", "Runs the simulation.");
    options[RENDER]             = createOption("--render", "-r", "Runs the real-time renderer.");
    options[CONFIG]             = createOption("--config", "-c", "Uses the config file given after this flag instead of the default.");
    // clang-format on

    if (argc == 1) {
        print_help(options);
    }
    else {
        char* config_path = NULL;
        for (int i = 1; i < argc; ++i) {
            const int option = findOption(argv[i], options);
            switch (option) {
            case CONFIG:
                if (i + 1 < argc) {
                    config_path = strdup(argv[i + 1]);
                }
                else {
                    printf("Syntax error. Usage: --config <config path>.\n");
                    return EXIT_FAILURE;
                }
                break;
            default:
                break; // Do nothing
            }
        }
        if (!config_path)
            config_path = strdup(AC_DEFAULT_CONFIG);

        printf("Config path: %s\n", config_path);
        ERRCHK_ALWAYS(config_path);

        for (int i = 1; i < argc; ++i) {
            const int option = findOption(argv[i], options);
            switch (option) {
            case HELP:
                print_help(options);
                break;
            case TEST:
                run_autotest(config_path);
                break;
            case BENCHMARK:
                run_benchmark(config_path);
                break;
            case SIMULATE:
                run_simulation(config_path);
                break;
            case RENDER:
                run_renderer(config_path);
                break;
            case CONFIG:
                ++i;
                break;
            default:
                printf("Invalid option %s\n", argv[i]);
                break; // Do nothing
            }
        }

        free(config_path);
    }

    for (int i = 0; i < NUM_OPTIONS; ++i)
        destroyOption(&options[i]);

    return EXIT_SUCCESS;
}
