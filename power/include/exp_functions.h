/*
 * =====================================================================================
 *
 *       Filename:  exp_functions.h
 *
 *    Description:  Functions used inside the experiment function
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

#ifndef EXP_FUNCTIONS_H
#define EXP_FUNCTIONS_H

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <zmq.h>
#include <stdint.h>

int start_experiment();

void set_initial_governor_and_frequency();
void handle_energy_measurement(int cpu, int active_packages, uint64_t *pkg_energy_start, uint64_t *pkg_energy_finish, uint64_t *dram_energy_start, uint64_t *dram_energy_finish);
void restore_original_governor_and_frequency(char *original_governor, unsigned long original_frequency);

#endif // EXP_FUNCTIONS_H
