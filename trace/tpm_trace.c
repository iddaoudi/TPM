/*
 * =====================================================================================
 *
 *       Filename:  tpm_trace.c
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

#include <pthread.h>
#include <stddef.h>
#include <sys/time.h>

#include "tpm_trace.h"
#include "include/cvector.h"
#include "include/hashmap.h"

FILE *file;

// Write task information to the file
bool tpm_map_iter(const void *item, void *udata)
{
  tpm_task_t *user = (tpm_task_t *)item;
  long double tmp = user->end_time - user->start_time; // in us
  fprintf(file, "%s, %Lf\n", user->name, tmp * 1e-3);  // in ms
  return true;
}

// Compare two tasks by their name
int tpm_task_name_compare(const void *a, const void *b, void *udata)
{
  const tpm_task_t *ua = a;
  const tpm_task_t *ub = b;
  return strcmp(ua->name, ub->name);
}

// Hash function for task
uint64_t tpm_map_hash(const void *item, uint64_t seed0, uint64_t seed1)
{
  const tpm_task_t *task = item;
  return hashmap_sip(task->name, strlen(task->name), seed0, seed1);
}

// Create a new task with the given name
void tpm_task_create(const char *name)
{
  tpm_task_t *task = malloc(sizeof(*task));
  task->name = malloc(TPM_STRING_SIZE * sizeof(char));
  strncpy(task->name, name, TPM_STRING_SIZE);
  task->cpu = 0;
  task->node = 0;
  task->start_time = 0.0;
  task->end_time = 0.0;
  pthread_mutex_lock(&mutex);
  hashmap_set(map, task);
  pthread_mutex_unlock(&mutex);
}

// Start the tracing with the given parameters
extern void tpm_trace_start(char *upstream_algorithm, int upstream_matrix_size,
                            int upstream_tile_size, int upstream_n_threads)
{
  TPM_POWER = atoi(getenv("TPM_POWER"));
  if (TPM_POWER)
  {
    // Initialize ZMQ context and request
    context = zmq_ctx_new();
    request = zmq_socket(context, ZMQ_PUSH);
    // Connect to client and send request to start energy measurements
    tpm_zmq_connect_client(request);
    tpm_zmq_send_signal(request, "energy 0");
  }
  algorithm = upstream_algorithm;
  matrix_size = upstream_matrix_size;
  tile_size = upstream_tile_size;
  n_threads = upstream_n_threads;
  map = hashmap_new(sizeof(tpm_task_t), 0, 0, 0, tpm_map_hash, tpm_task_name_compare, NULL, NULL);
}

// Set the new task's name before creation
extern void tpm_trace_set_task_name(const char *name)
{
  new_name = name;
  tpm_task_create(new_name);
}

// Set the task's CPU and node information
extern void tpm_trace_set_task_cpu_node(int cpu, int node, char *input_name)
{
  pthread_mutex_lock(&mutex);
  char *task_and_cpu = tpm_task_and_cpu_string(input_name, cpu);
  tpm_zmq_send_signal(request, task_and_cpu);
  tpm_task_t *task = hashmap_get(map, &(tpm_task_t){.name = input_name});
  if (task != NULL)
  {
    task->cpu = cpu;
    task->node = node;
  }
  pthread_mutex_unlock(&mutex);
}

// Get task's execution time
extern void tpm_trace_get_task_time(struct timeval start, struct timeval end, char *name)
{
  pthread_mutex_lock(&mutex);
  tpm_task_t *task = hashmap_get(map, &(tpm_task_t){.name = name});
  if (task != NULL)
  {
    task->start_time = (start.tv_sec) * 1000000 + start.tv_usec;
    task->end_time = (end.tv_sec) * 1000000 + end.tv_usec;
  }
  pthread_mutex_unlock(&mutex);
}

// Finalize the tracing and clean up resources
extern void tpm_trace_finalize(double exec_time)
{
  int file_dump = 0;
  if (file_dump)
  {
    char file_name[TPM_FILENAME_SIZE];
    snprintf(file_name, TPM_FILENAME_SIZE, "tasks_%s_%d_%d_%d.dat", algorithm,
             matrix_size, tile_size, n_threads);
    file = fopen(file_name, "a");
    if (file == NULL)
    {
      printf("fopen error\n");
    }
    hashmap_scan(map, tpm_map_iter, file);
    fclose(file);
  }

  pthread_mutex_destroy(&mutex);
  hashmap_free(map);

  // Send request to end energy measurements
  tpm_zmq_send_signal(request, "energy 1");

  char *matrix = tpm_task_and_cpu_string("matrix", matrix_size);
  tpm_zmq_send_signal(request, matrix);
  char *tile = tpm_task_and_cpu_string("tile", tile_size);
  tpm_zmq_send_signal(request, tile);
  char* time = tpm_str_and_double_to_str("time", exec_time);
  tpm_zmq_send_signal(request, time);
  tpm_zmq_close(request, context);

  free(matrix);
  free(tile);
  free(time);
}
