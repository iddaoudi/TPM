/*
 * =====================================================================================
 *
 *       Filename:  exp_functions.c
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

#include "exp_functions.h"
#include "exp_globals.h"
#include "utility_functions.h"
#include "tpm_functions.h"

#include "rapl.h"

#include "cholesky.h"
#include "qr.h"
#include "lu.h"
#include "sparselu.h"

int start_experiment()
{
  void *context = zmq_ctx_new();
  void *server = zmq_socket(context, ZMQ_PULL);

  int ret = tpm_zmq_connect_server(server);
  if (ret != 0)
  {
    fprintf(stderr, "Failed to connect to the server.\n");
    exit(EXIT_FAILURE);
  }

  /* Initialization */
  int active_packages = rapl_init();

  /* Get current governor policy */
  char *original_governor = tpm_query_current_governor_policy(0);

  /* Get all possible frequencies (in KHz) and all available governors */
  unsigned long *frequencies_vector = tpm_query_available_frequencies(0);
  char **governors_vector = tpm_query_available_governors(0);

  int frequencies_vector_size =
      frequencies_vector_size_counter(frequencies_vector);
  /* By default, the original frequency of every CPU is set to its maximum */
  unsigned long original_frequency = frequencies_vector[0];
  unsigned long selected_frequency = select_frequency(
      target_frequency, frequencies_vector, frequencies_vector_size);
  if (frequencies_vector_size == 0 || original_frequency == 0 || selected_frequency == -1)
  {
    fprintf(stderr, "Invalid frequency values.\n");
    exit(EXIT_FAILURE);
  }

#ifdef LOG
  logs(frequencies_vector, frequencies_vector_size, governors_vector,
       original_governor, selected_frequency);
#endif

  uint64_t pkg_energy_start[active_packages];
  uint64_t pkg_energy_finish[active_packages];
  uint64_t dram_energy_start[active_packages];
  uint64_t dram_energy_finish[active_packages];

  /* Change the governor */
  set_initial_governor_and_frequency();

  while (1)
  {
    char task_and_cpu[TPM_STRING_SIZE];
    char task[TPM_TASK_STRING_SIZE];
    int cpu = 0;

    /* Receive current task name and its CPU */
    zmq_recv(server, task_and_cpu, TPM_STRING_SIZE, 0);
    sscanf(task_and_cpu, "%s %d", task, &cpu);

    /* Frequency control */
    if (!strcmp(algorithm, "cholesky"))
      cholesky_control(selected_case, task, cpu, selected_frequency,
                       original_frequency);
    else if (!strcmp(algorithm, "qr"))
      qr_control(selected_case, task, cpu, selected_frequency,
                 original_frequency);
    else if (!strcmp(algorithm, "lu"))
      lu_control(selected_case, task, cpu, selected_frequency,
                 original_frequency);
    else if (!strcmp(algorithm, "sparselu"))
      sparselu_control(selected_case, task, cpu, selected_frequency,
                       original_frequency);

    if (strcmp(task, "energy") == 0)
    {
      /* Handle energy measurement */
      handle_energy_measurement(cpu, active_packages, pkg_energy_start, pkg_energy_finish, dram_energy_start, dram_energy_finish);
    }

    /* End measurements */
    if (strcmp(task_and_cpu, "end") == 0)
      break;
  }

  /* Set back the original governor policy and frequency (max by default) */
  restore_original_governor_and_frequency(original_governor, original_frequency);

  file_dump(active_packages, pkg_energy_start, pkg_energy_finish,
            dram_energy_start, dram_energy_finish);
  
  //printf("%lu\n", selected_frequency);

  return 0;
}