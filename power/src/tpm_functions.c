/*
 * =====================================================================================
 *
 *       Filename:  tpm_functions.c
 *
 *    Description:  TPM functions for managing frequencies and ZMQ
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

#include "tpm_functions.h"

struct cpufreq_available_governors *governor;
struct cpufreq_available_frequencies *frequency;
struct cpufreq_policy *policy_governor;
struct cpufreq_policy *new_policy_governor;

void tpm_set_governor_policy(unsigned int cpu, char *new_governor)
{
  int ret = cpufreq_modify_policy_governor(cpu, new_governor);
  if (ret != 0)
  {
    printf("No root access on CPU %d\n", cpu);
    exit(EXIT_FAILURE);
  }
}

void tpm_set_max_frequency(unsigned int cpu, unsigned long max_freq)
{
  int ret = cpufreq_modify_policy_max(cpu, max_freq);
  if (ret != 0)
  {
    printf("No root access on CPU %d\n", cpu);
    exit(EXIT_FAILURE);
  }
}

unsigned long *tpm_query_available_frequencies(unsigned int cpu)
{
  frequency = cpufreq_get_available_frequencies(cpu);
  static unsigned long frequencies_vector[MAX_FREQUENCIES];
  int counter = 0;
  if (frequency)
  {
    do
    {
      frequencies_vector[counter] = frequency->frequency;
      frequency = frequency->next;
      counter++;
    } while (frequency);
  }
  cpufreq_put_available_frequencies(frequency);
  return frequencies_vector;
}

char **tpm_query_available_governors(unsigned int cpu)
{
  governor = cpufreq_get_available_governors(cpu);
  char **governors_vector = malloc(MAX_GOVERNORS * sizeof(char *));
  for (int i = 0; i < MAX_GOVERNORS; i++)
  {
    governors_vector[i] = malloc(MAX_CHARACTERS * sizeof(char));
    governors_vector[i][0] = '\0';
  }
  int counter = 0;
  if (governor)
  {
    do
    {
      strcpy(governors_vector[counter], governor->governor);
      governor = governor->next;
      counter++;
    } while (governor);
  }
  cpufreq_put_available_governors(governor);
  return governors_vector;
}

char *tpm_query_current_governor_policy(unsigned int cpu)
{
  policy_governor = cpufreq_get_policy(cpu);
  char *tmp = NULL;
  if (policy_governor)
  {
    tmp = malloc(strlen(policy_governor->governor) + 1);
    strcpy(tmp, policy_governor->governor);
  }
  cpufreq_put_policy(policy_governor);
  return tmp;
}

double tpm_query_current_frequency_hardware(unsigned int cpu)
{
  return cpufreq_get_freq_hardware(cpu) * 1e-3; // Mhz
}

double tpm_query_current_frequency_kernel(unsigned int cpu)
{
  return cpufreq_get_freq_kernel(cpu) * 1e-3; // Mhz
}

int tpm_zmq_connect_server(void *server)
{
  int num = 0;
  zmq_setsockopt(server, ZMQ_LINGER, &num, sizeof(int));
  num = -1;
  zmq_setsockopt(server, ZMQ_SNDHWM, &num, sizeof(int));
  zmq_setsockopt(server, ZMQ_RCVHWM, &num, sizeof(int));
  int ret = zmq_bind(server, "tcp://127.0.0.1:5555");

  return ret;
}
