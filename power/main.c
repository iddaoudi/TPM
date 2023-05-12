/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  Main file
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "utility_functions.h"
#include "exp_functions.h"
#include "exp_globals.h"
#include "tpm_functions.h"

void change_governor()
{
  for (int i = 0; i < MAX_CPUS; i++)
    tpm_set_governor_policy(i, input_governor);
}

void restore_original_governor_and_frequency(char *original_governor, unsigned long original_frequency)
{
  for (int i = 0; i < MAX_CPUS; i++)
  {
    tpm_set_governor_policy(i, original_governor);
    tpm_set_max_frequency(i, original_frequency);
  }
}

void handle_energy_measurement(int cpu, int active_packages, uint64_t *pkg_energy_start, uint64_t *pkg_energy_finish, uint64_t *dram_energy_start, uint64_t *dram_energy_finish)
{
  if (cpu == 0)
  {
    for (int i = 0; i < active_packages; i++)
    {
      pkg_energy_start[i] = rapl_readenergy_uj(i, "pkg");
      dram_energy_start[i] = rapl_readenergy_uj(i, "dram");
      if (pkg_energy_start[i] >= check_max(i, "pkg") || dram_energy_start[i] >= check_max(i, "dram"))
      {
        fprintf(stderr, "Energy measurements exceeded the maximum value.\n");
        exit(EXIT_FAILURE);
      }
    }
  }
  else if (cpu == 1)
  {
    for (int i = 0; i < active_packages; i++)
    {
      pkg_energy_finish[i] = rapl_readenergy_uj(i, "pkg");
      dram_energy_finish[i] = rapl_readenergy_uj(i, "dram");
      if (pkg_energy_finish[i] < pkg_energy_start[i])
        pkg_energy_finish[i] += check_max(i, "pkg");
      if (dram_energy_finish[i] < dram_energy_start[i])
        dram_energy_finish[i] += check_max(i, "dram");
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc != 6)
  {
    printf("Incorrect number of arguments.\n");
    exit(EXIT_FAILURE);
  }

  MAX_CPUS = atoi(argv[1]);      // number of CPUs to use
  input_governor = argv[2];      // wanted governor
  target_frequency = argv[3];    // mid or max for now FIXME
  selected_case = atoi(argv[4]); // combination to select
  algorithm = argv[5];

  // Based on the experiences made, it appears that the userspace governor isn't allowing any frequency scaling in real time, when the ondemand governor does.
  // This requires further investigation that we are postponing for now. (something related to the ACPI driver configuration?)
  if (strcmp(input_governor, "ondemand") != 0)
  {
  	printf("*** TPM Power: Warning: the selected governor is %s\n", input_governor);
  }

  int ret = start_experiment();
  if (ret != 0)
  {
    fprintf(stderr, "Failed to start experiment.\n");
    exit(EXIT_FAILURE);
  }

  return 0;
}
