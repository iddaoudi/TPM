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

double time_start, time_finish;

/* TPM modifications start */
/* Weak attributes */
void __attribute__((weak)) TPM_middle_man_start();
void __attribute__((weak)) TPM_middle_man_finalize(double total_execution_time);
void __attribute__((weak)) TPM_middle_man_task_start(const char *task_name);
void __attribute__((weak)) TPM_middle_man_task_finish(const char *task_name);

/* Application tracing functions */
static inline void TPM_application_start()
{
  TPM_middle_man_start();
}
static inline void TPM_application_finalize(double total_execution_time)
{
  TPM_middle_man_finalize(total_execution_time);
}
static inline void TPM_application_task_start(const char *task_name)
{
  TPM_middle_man_task_start(task_name);
}
static inline void TPM_application_task_finish(const char *task_name)
{
  TPM_middle_man_task_finish(task_name);
}
/* TPM modifications end */

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