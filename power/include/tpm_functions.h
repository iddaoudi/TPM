/*
 * =====================================================================================
 *
 *       Filename:  tpm_functions.h
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

#ifndef TPM_FUNCTIONS_H
#define TPM_FUNCTIONS_H

#include <stdint.h>
#include <unistd.h>
#include <cpufreq.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <zmq.h>

#define MAX_GOVERNORS 8
#define MAX_FREQUENCIES 20
#define MAX_CHARACTERS 10

unsigned long *tpm_query_available_frequencies(unsigned int cpu);
char **tpm_query_available_governors(unsigned int cpu);
char *tpm_query_current_governor_policy(unsigned int cpu);
double tpm_query_current_frequency_hardware(unsigned int cpu);
double tpm_query_current_frequency_kernel(unsigned int cpu);
void tpm_set_governor_policy(unsigned int cpu, char *new_governor);
void tpm_set_max_frequency(unsigned int cpu, unsigned long max_freq);

uint64_t rapl_readenergy_uj(int pkgid, char *domain);
uint64_t check_max(int pkgid, char *domain);

int tpm_zmq_connect_server(void *server);

#endif // TPM_FUNCTIONS_H