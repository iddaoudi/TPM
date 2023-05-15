/*
 * =====================================================================================
 *
 *       Filename:  common.h
 *
 *    Description:  Common tracing functions for power management
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  14/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#define _GNU_SOURCE

#include <getopt.h>
#include <sched.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>

#include <papi.h>

#define TPM_STRING_SIZE 128

// External functions for tracing
void __attribute__((weak))
tpm_start(char *algorithm, int matrix_size, int tile_size, int nthreads);
void __attribute__((weak)) tpm_finalize(double exec_time);
void __attribute__((weak)) tpm_set_task_name(const char *name);
void __attribute__((weak)) tpm_set_task_cpu_node(int cpu, int node, char *name);
void __attribute__((weak)) tpm_get_task_time(struct timeval start, struct timeval end, char *name);

extern inline void tpm_downstream_start(char *algorithm, int matrix_size,
                                        int tile_size, int nthreads)
{
  tpm_start(algorithm, matrix_size, tile_size, nthreads);
}

extern inline void tpm_upstream_finalize(double exec_time)
{
  tpm_finalize(exec_time);
}

extern inline void tpm_upstream_set_task_name(const char *name)
{
  tpm_set_task_name(name);
}

extern inline void tpm_upstream_set_task_cpu_node(int cpu, int node,
                                                  char *name)
{
  tpm_set_task_cpu_node(cpu, node, name);
}

extern inline void tpm_upstream_get_task_time(struct timeval start,
                                              struct timeval end, char *name)
{
  tpm_get_task_time(start, end, name);
}

// Give a task name a unique identification according to iterations
char *tpm_unique_task_identifier(char *name, int a, int b, int c)
{
  char *tmp_name = (char *)malloc(TPM_STRING_SIZE * sizeof(char));
  memset(tmp_name, '\0', TPM_STRING_SIZE);
  strcpy(tmp_name, name);

  // Room for up to 10 digits, a sign, and a null terminator
  char string_a[12];
  char string_b[12];
  char string_c[12];

  sprintf(string_a, "%d", a);
  sprintf(string_b, "%d", b);
  sprintf(string_c, "%d", c);

  strcat(tmp_name, string_a);
  strcat(tmp_name, string_b);
  strcat(tmp_name, string_c);

  return tmp_name;
}