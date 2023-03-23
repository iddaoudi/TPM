/*
 * =====================================================================================
 *
 *       Filename:  exp_functions.c
 *
 *    Description:  Utility functions
 *
 *        Version:  1.0
 *        Created:  19/03/2023
 *       Revision:  20/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include "utility_functions.h"
#include "exp_globals.h"

void logs(unsigned long *frequencies_vector, int frequencies_vector_size,
          char **governors_vector, char *original_governor,
          unsigned long selected_frequency)
{
  printf("############################################\n");
  printf("Available frequencies: ");
  for (int i = 0; i < frequencies_vector_size; i++)
  {
    printf("%lu ", frequencies_vector[i]);
  }
  printf("\n");
  int counter = 0;
  printf("Available governors: ");
  while (strcmp(governors_vector[counter], "") != 0)
  {
    printf("%s ", governors_vector[counter]);
    counter++;
  }
  printf("\n");
  printf("Current governor: %s\n", original_governor);
  printf("Selected frequency: %lu\n", selected_frequency);
  printf("############################################\n");
}

void file_dump(char *algorithm, int matrix_size, int tile_size, int selected_case, int active_packages, uint64_t *pkg_energy_start,
               uint64_t *pkg_energy_finish, uint64_t *dram_energy_start,
               uint64_t *dram_energy_finish, double exec_time)
{
  char filename[20];
  int TPM_ITER = atoi(getenv("TPM_ITER"));
  sprintf(filename, "energy_data_%d.csv", TPM_ITER);
  
  FILE *file;
  struct stat buffer;
  int file_already_exists = (stat(filename, &buffer) == 0);

  file = fopen(filename, "a");
  if (file == NULL)
  {
    perror("fopen failed");
    exit(EXIT_FAILURE);
  }

  if (!file_already_exists)
  {
    fprintf(file, "algorithm, matrix_size, tile_size, case, PKG1, PKG2, DRAM1, DRAM2, time\n");
  }
  // FIXME: assuming there is a maximum of 2 NUMA nodes
  uint64_t pkg_energy[2] = {0, 0};
  uint64_t dram_energy[2] = {0, 0};

  for (int counter = 0; counter < active_packages; counter++)
  {
    pkg_energy[counter] = pkg_energy_finish[counter] - pkg_energy_start[counter];
    dram_energy[counter] = dram_energy_finish[counter] - dram_energy_start[counter];
  }

  fprintf(file, "%s, %d, %d, %d, %" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %f\n",
          algorithm, matrix_size, tile_size, selected_case, pkg_energy[0], pkg_energy[1], dram_energy[0], dram_energy[1], exec_time);

  fclose(file);
}

int frequencies_vector_size_counter(unsigned long *frequencies_vector)
{
  int frequencies_vector_size = 0;
  while (frequencies_vector[frequencies_vector_size] != 0)
  {
    frequencies_vector_size++;
  }
  return frequencies_vector_size;
}

unsigned long select_frequency(char *target_frequency,
                               unsigned long *frequencies_vector,
                               int frequencies_vector_size)
{
  if (strcmp(target_frequency, "mid") == 0)
  {
    return frequencies_vector[frequencies_vector_size / 2];
  }
  else if (strcmp(target_frequency, "max") == 0)
  {
    return frequencies_vector[0]; // first element is the last value inserted in the array
  }
  else if (strcmp(target_frequency, "min") == 0)
  {
    return frequencies_vector[frequencies_vector_size - 1];
  }
  return -1;
}
