/*
 * =====================================================================================
 *
 *       Filename:  jacobi.h
 *
 *    Description:	Jacobi iteration
 *
 *        Version:  1.0
 *        Created:  18/01/2023
 *       Revision:  13/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include "functions.h"
#include "copy.h"
#include "estimate.h"

void jacobi(int matrix_size, double dx, double *falloc,
			int niter, double *ualloc, double *unewalloc, int tile_size)
{
	double(*f)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])falloc;
	double(*u)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])ualloc;
	double(*unew)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])unewalloc;

	char *name_with_id_char = NULL;
	struct timeval start = (struct timeval){0};
	struct timeval end = (struct timeval){0};

	int eventset = PAPI_NULL;
	long long values[NEVENTS];
	const int available_threads = omp_get_max_threads();

	// NEVENTS + 1 for the task weights
	CounterData *cblock = (CounterData *)malloc(available_threads * sizeof(CounterData));
	CounterData *cestimate = (CounterData *)malloc(available_threads * sizeof(CounterData));

	if (TPM_PAPI)
	{
		int ret = PAPI_create_eventset(&eventset);
		PAPI_add_events(eventset, events, NEVENTS);

		for (int i = 0; i < available_threads; i++)
		{
			memset(cblock[i].values, 0, (NEVENTS + 1) * sizeof(long long));
			memset(cestimate[i].values, 0, (NEVENTS + 1) * sizeof(long long));
		}
	}

	int i, j, it;

	// TPM library: initialization
	if (TPM_TRACE)
		tpm_downstream_start("poisson", matrix_size, tile_size, NTH);

#pragma omp parallel \
shared(ualloc, unewalloc, f, matrix_size, dx, niter, tile_size)
#pragma omp master
	{
		for (it = 1; it <= niter; it++)
		{
			for (j = 0; j < matrix_size; j += tile_size)
			{
				for (i = 0; i < matrix_size; i += tile_size)
				{
					if (TPM_TRACE)
					{
						// TPM library: create a unique task name
						name_with_id_char = tpm_unique_task_identifier("cblock", it, j, i);
						tpm_upstream_set_task_name(name_with_id_char);
					}
#pragma omp task shared(ualloc, unewalloc) firstprivate(i, j, tile_size, matrix_size, name_with_id_char) \
	depend(in : unew[i : tile_size][j : tile_size])                                                      \
	depend(out : u[i : tile_size][j : tile_size])
					{
						if (TPM_PAPI)
						{
							memset(values, 0, sizeof(values));
							// Start PAPI counters
							int ret_start = PAPI_start(eventset);
							if (ret_start != PAPI_OK)
							{
								printf("PAPI_start CBLOCK error %d: %s\n", ret_start, PAPI_strerror(ret_start));
								exit(EXIT_FAILURE);
							}
						}
						else if (TPM_TRACE)
						{
							// TPM library: send CPU and name
							unsigned int cpu, node;
							getcpu(&cpu, &node);
							tpm_upstream_set_task_cpu_node(cpu, node, name_with_id_char);
							gettimeofday(&start, NULL);
						}

						// Kernel
						copy_block(matrix_size, i / tile_size, j / tile_size, ualloc, unewalloc, tile_size);

						if (TPM_PAPI)
						{
							// Stop PAPI counters
							int ret_stop = PAPI_stop(eventset, values);
							if (ret_stop != PAPI_OK)
							{
								printf("PAPI_stop CBLOCK error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
								exit(EXIT_FAILURE);
							}
							// Accumulate events values
							for (int i = 0; i < NEVENTS; i++)
							{
								cblock[omp_get_thread_num()].values[i] += values[i];
							}
							cblock[omp_get_thread_num()].values[NEVENTS]++;
						}
						else if (TPM_TRACE)
						{
							gettimeofday(&end, NULL);
							// TPM library: send time and name
							tpm_upstream_get_task_time(start, end, name_with_id_char);
						}
					}
				}
			}

			for (j = 0; j < matrix_size; j += tile_size)
			{
				for (i = 0; i < matrix_size; i += tile_size)
				{
					int xdm1 = i == 0 ? 0 : tile_size;
					int xdp1 = i == matrix_size - tile_size ? 0 : tile_size;
					int ydp1 = j == matrix_size - tile_size ? 0 : tile_size;
					int ydm1 = j == 0 ? 0 : tile_size;

					if (TPM_TRACE)
					{
						// TPM library: create a unique task name
						name_with_id_char = tpm_unique_task_identifier("cestimate", it, j, i);
						tpm_upstream_set_task_name(name_with_id_char);
					}

#pragma omp task shared(ualloc, unewalloc)                                                    \
	firstprivate(dx, matrix_size, tile_size, i, j, xdm1, xdp1, ydp1, ydm1, name_with_id_char) \
	depend(out : unew[i : tile_size][j : tile_size])                                          \
	depend(in : f[i : tile_size][j : tile_size],                                              \
			   u[i : tile_size][j : tile_size],                                               \
			   u[(i - xdm1) : tile_size][j : tile_size],                                      \
			   u[i : tile_size][(j + ydp1) : tile_size],                                      \
			   u[i : tile_size][(j - ydm1) : tile_size],                                      \
			   u[(i + xdp1) : tile_size][j : tile_size])
					{
						if (TPM_PAPI)
						{
							memset(values, 0, sizeof(values));
							// Start PAPI counters
							int ret_start = PAPI_start(eventset);
							if (ret_start != PAPI_OK)
							{
								printf("PAPI_start CESTIMATE error %d: %s\n", ret_start, PAPI_strerror(ret_start));
								exit(EXIT_FAILURE);
							}
						}
						else if (TPM_TRACE)
						{
							// TPM library: send CPU and name
							unsigned int cpu, node;
							getcpu(&cpu, &node);
							tpm_upstream_set_task_cpu_node(cpu, node, name_with_id_char);
							gettimeofday(&start, NULL);
						}

						// Kernel
						compute_estimate(i / tile_size, j / tile_size, ualloc, unewalloc, falloc, dx,
										 matrix_size, tile_size);

						if (TPM_PAPI)
						{
							// Stop PAPI counters
							int ret_stop = PAPI_stop(eventset, values);
							if (ret_stop != PAPI_OK)
							{
								printf("PAPI_stop CESTIMATE error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
								exit(EXIT_FAILURE);
							}
							// Accumulate events values
							for (int i = 0; i < NEVENTS; i++)
							{
								cestimate[omp_get_thread_num()].values[i] += values[i];
							}
							cestimate[omp_get_thread_num()].values[NEVENTS]++;
						}
						else if (TPM_TRACE)
						{
							gettimeofday(&end, NULL);
							// TPM library: send time and name
							tpm_upstream_get_task_time(start, end, name_with_id_char);
						}
					}
				}
			}
		}
	}
	if (TPM_PAPI)
	{
#pragma omp taskwait
		PAPI_destroy_eventset(&eventset);
		PAPI_shutdown();

		const char *task_names[] = {"cblock", "cestimate"};
		CounterData *counters[] = {cblock, cestimate};
		int num_tasks = sizeof(task_names) / sizeof(task_names[0]); // This gives the length of the tasks array
		dump_counters("poisson", task_names, counters, num_tasks, matrix_size, tile_size, l3_cache_size, available_threads);
	}
}
