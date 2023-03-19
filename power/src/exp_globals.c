/*
 * =====================================================================================
 *
 *       Filename:  exp_globals.c
 *
 *    Description:  Global variables used across the project
 *
 *        Version:  1.0
 *        Created:  19/03/2023
 *       Revision:  none
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include "exp_globals.h"
#include <stddef.h>

int MAX_CPUS = 0;
char *input_governor = NULL;
char *target_frequency = NULL;
int selected_case = 0;
char *algorithm = NULL;
