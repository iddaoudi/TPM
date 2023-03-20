/*
 * =====================================================================================
 *
 *       Filename:  exp_functions.h
 *
 *    Description:  Utility functions
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

#include <stdint.h>

#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

void logs(unsigned long *frequencies_vector, int frequencies_vector_size,
          char **governors_vector, char *original_governor,
          unsigned long selected_frequency);
void file_dump(char *algorithm, int matrix_size, int tile_size, int selected_case, int active_packages, uint64_t *pkg_energy_start,
               uint64_t *pkg_energy_finish, uint64_t *dram_energy_start,
               uint64_t *dram_energy_finish, double exec_time);
int frequencies_vector_size_counter(unsigned long *frequencies_vector);
unsigned long select_frequency(char *target_frequency, unsigned long *frequencies_vector, int frequencies_vector_size);

#endif // UTILITY_FUNCTIONS_H
