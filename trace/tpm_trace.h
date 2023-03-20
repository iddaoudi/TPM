/*
 * =====================================================================================
 *
 *       Filename:  tpm_trace.h
 *
 *    Description:  Tracing library functions without OMPT
 *
 *        Version:  1.0
 *        Created:  31/12/2022
 *       Revision:  19/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include <bits/types/struct_timeval.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define TPM_FILENAME_SIZE 64
#define TPM_STRING_SIZE 64

const char *new_name = NULL;
char *algorithm = NULL;
int matrix_size = 0;
int tile_size = 0;
int n_threads = 0;
int TPM_POWER = 0;
int order = 0;

#include "include/cvector.h"
#include "include/tpm_task.h"
#include "include/hashmap.h"
#include "include/common.h"
#include "include/client.h"

// Hash map functions
bool tpm_map_iter(const void *item, void *udata);
int tpm_task_name_compare(const void *a, const void *b, void *udata);
uint64_t tpm_map_hash(const void *item, uint64_t seed0, uint64_t seed1);

// Hash map
struct hashmap *map;

cvector_vector_type(tpm_task_t *) tpm_tasks = NULL;
pthread_mutex_t mutex;

void tpm_task_create(const char *name);

extern void tpm_trace_start(char *upstream_algorithm, int upstream_matrix_size,
                            int upstream_tile_size, int upstream_n_threads);
extern void tpm_trace_finalize(double time);
extern void tpm_trace_set_task_name(const char *name);
extern void tpm_trace_set_task_cpu_node(int cpu, int node, char *name);
extern void tpm_trace_get_task_time(struct timeval start, struct timeval end,
                                    char *name);

extern void tpm_start(char *upstream_algorithm, int upstream_matrix_size,
                      int upstream_tile_size, int upstream_n_threads)
{
  tpm_trace_start(upstream_algorithm, upstream_matrix_size, upstream_tile_size,
                  upstream_n_threads);
}
extern void tpm_finalize(double exec_time) { tpm_trace_finalize(exec_time); }
extern void tpm_set_task_name(const char *name)
{
  tpm_trace_set_task_name(name);
}
extern void tpm_set_task_cpu_node(int cpu, int node, char *name)
{
  tpm_trace_set_task_cpu_node(cpu, node, name);
}
extern void tpm_get_task_time(struct timeval start, struct timeval end,
                              char *name)
{
  tpm_trace_get_task_time(start, end, name);
}
