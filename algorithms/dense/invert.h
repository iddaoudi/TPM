/*
 * =====================================================================================
 *
 *       Filename:  invert.h
 *
 *    Description:  Task-based matrix invert algorithm using an LU decomposition with
 *                  partial pivoting
 *
 *        Version:  1.0
 *        Created:  14/05/2023
 *       Revision:  21/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void invert(double *A, int *ipiv, int matrix_size, int tile_size)
{
    char *name_with_id_char = NULL;
    struct timeval start = (struct timeval){0};
    struct timeval end = (struct timeval){0};

    int eventset = PAPI_NULL;
    long long values[NEVENTS];
    const int available_threads = omp_get_max_threads();

    // NEVENTS + 1 for the task weights
    CounterData *getrf = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *trsm = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *gemm = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *getri = (CounterData *)malloc(available_threads * sizeof(CounterData));

    if (TPM_PAPI)
    {
        int ret = PAPI_create_eventset(&eventset);
        PAPI_add_events(eventset, events, NEVENTS);

        for (int i = 0; i < available_threads; i++)
        {
            memset(getrf[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(trsm[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(gemm[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(getri[i].values, 0, (NEVENTS + 1) * sizeof(long long));
        }
    }

    // TPM library: initialization
    if (TPM_TRACE)
        tpm_downstream_start("invert", matrix_size, tile_size, NTH);

    int lda = matrix_size;

    for (int i = 0; i < matrix_size; i += tile_size)
    {
        if (TPM_TRACE)
        {
            // TPM library: create a unique task name
            name_with_id_char = tpm_unique_task_identifier("getrf", i, 0, 0);
            tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(name_with_id_char) depend(inout : A[i * tile_size + i])
        {
            if (TPM_PAPI)
            {
                memset(values, 0, sizeof(values));
                // Start PAPI counters
                int ret_start = PAPI_start(eventset);
                if (ret_start != PAPI_OK)
                {
                    printf("PAPI_start GETRF error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, tile_size, tile_size, &A[i * matrix_size + i], lda, &ipiv[i]);

            if (TPM_PAPI)
            {
                // Start PAPI counters
                int ret_stop = PAPI_stop(eventset, values);
                if (ret_stop != PAPI_OK)
                {
                    printf("PAPI_stop GETRF error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                    exit(EXIT_FAILURE);
                }
                // Accumulate events values
                for (int i = 0; i < NEVENTS; i++)
                {
                    getrf[omp_get_thread_num()].values[i] += values[i];
                }
                getrf[omp_get_thread_num()].values[NEVENTS]++;
            }
            else if (TPM_TRACE)
            {
                gettimeofday(&end, NULL);
                // TPM library: send time and name
                tpm_upstream_get_task_time(start, end, name_with_id_char);
            }
        }
        for (int j = i + tile_size; j < matrix_size; j += tile_size)
        {
            if (TPM_TRACE)
            {
                // TPM library: create a unique task name
                name_with_id_char = tpm_unique_task_identifier("trsm", i, j, 0);
                tpm_upstream_set_task_name(name_with_id_char);
            }
#pragma omp task firstprivate(name_with_id_char) depend(in : A[i * tile_size + i]) \
    depend(inout : A[i * tile_size + j])
            {
                if (TPM_PAPI)
                {
                    memset(values, 0, sizeof(values));
                    // Start PAPI counters
                    int ret_start = PAPI_start(eventset);
                    if (ret_start != PAPI_OK)
                    {
                        printf("PAPI_start TRSM error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
                cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, tile_size,
                            tile_size, 1.0, &A[i * matrix_size + i], lda, &A[i * matrix_size + j], lda);

                if (TPM_PAPI)
                {
                    // Start PAPI counters
                    int ret_stop = PAPI_stop(eventset, values);
                    if (ret_stop != PAPI_OK)
                    {
                        printf("PAPI_stop TRSM error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                        exit(EXIT_FAILURE);
                    }
                    // Accumulate events values
                    for (int i = 0; i < NEVENTS; i++)
                    {
                        trsm[omp_get_thread_num()].values[i] += values[i];
                    }
                    trsm[omp_get_thread_num()].values[NEVENTS]++;
                }
                else if (TPM_TRACE)
                {
                    gettimeofday(&end, NULL);
                    // TPM library: send time and name
                    tpm_upstream_get_task_time(start, end, name_with_id_char);
                }
            }
            if (TPM_TRACE)
            {
                // TPM library: create a unique task name
                name_with_id_char = tpm_unique_task_identifier("trsm", i, j, 0);
                tpm_upstream_set_task_name(name_with_id_char);
            }
#pragma omp task firstprivate(name_with_id_char) depend(in : A[i * tile_size + i]) \
    depend(inout : A[j * tile_size + i])
            {
                if (TPM_TRACE)
                {
                    // TPM library: send CPU and name
                    unsigned int cpu, node;
                    getcpu(&cpu, &node);
                    tpm_upstream_set_task_cpu_node(cpu, node, name_with_id_char);
                    gettimeofday(&start, NULL);
                }

                // Kernel
                cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, tile_size,
                            tile_size, 1.0, &A[i * matrix_size + i], lda, &A[j * matrix_size + i], lda);

                if (TPM_TRACE)
                {
                    gettimeofday(&end, NULL);
                    // TPM library: send time and name
                    tpm_upstream_get_task_time(start, end, name_with_id_char);
                }
            }
        }

        for (int j = i + tile_size; j < matrix_size; j += tile_size)
        {
            for (int k = i + tile_size; k < matrix_size; k += tile_size)
            {
                if (TPM_TRACE)
                {
                    // TPM library: create a unique task name
                    name_with_id_char = tpm_unique_task_identifier("gemm", i, j, k);
                    tpm_upstream_set_task_name(name_with_id_char);
                }
#pragma omp task firstprivate(name_with_id_char)            \
    depend(in : A[i * tile_size + j], A[k * tile_size + i]) \
    depend(inout : A[k * tile_size + j])
                {
                    if (TPM_PAPI)
                    {
                        memset(values, 0, sizeof(values));
                        // Start PAPI counters
                        int ret_start = PAPI_start(eventset);
                        if (ret_start != PAPI_OK)
                        {
                            printf("PAPI_start GEMM error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tile_size, tile_size, tile_size,
                                -1.0, &A[k * matrix_size + i], lda, &A[i * matrix_size + j], lda,
                                1.0, &A[k * matrix_size + j], lda);

                    if (TPM_PAPI)
                    {
                        // Start PAPI counters
                        int ret_stop = PAPI_stop(eventset, values);
                        if (ret_stop != PAPI_OK)
                        {
                            printf("PAPI_stop GEMM error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                            exit(EXIT_FAILURE);
                        }
                        // Accumulate events values
                        for (int i = 0; i < NEVENTS; i++)
                        {
                            gemm[omp_get_thread_num()].values[i] += values[i];
                        }
                        gemm[omp_get_thread_num()].values[NEVENTS]++;
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
// This pragma is optional
#pragma omp taskwait

    for (int i = 0; i < matrix_size; i += tile_size)
    {
        if (TPM_TRACE)
        {
            // TPM library: create a unique task name
            name_with_id_char = tpm_unique_task_identifier("getri", i, 0, 0);
            tpm_upstream_set_task_name(name_with_id_char);
        }
#pragma omp task firstprivate(name_with_id_char) depend(inout : A[i * tile_size + i])
        {
            if (TPM_PAPI)
            {
                memset(values, 0, sizeof(values));
                // Start PAPI counters
                int ret_start = PAPI_start(eventset);
                if (ret_start != PAPI_OK)
                {
                    printf("PAPI_start GETRI error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, tile_size, &A[i * matrix_size + i], lda, &ipiv[i]);

            if (TPM_PAPI)
            {
                // Start PAPI counters
                int ret_stop = PAPI_stop(eventset, values);
                if (ret_stop != PAPI_OK)
                {
                    printf("PAPI_stop GETRI error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                    exit(EXIT_FAILURE);
                }
                // Accumulate events values
                for (int i = 0; i < NEVENTS; i++)
                {
                    getri[omp_get_thread_num()].values[i] += values[i];
                }
                getri[omp_get_thread_num()].values[NEVENTS]++;
            }
            else if (TPM_TRACE)
            {
                gettimeofday(&end, NULL);
                // TPM library: send time and name
                tpm_upstream_get_task_time(start, end, name_with_id_char);
            }
        }
    }

    if (TPM_PAPI)
    {
#pragma omp taskwait
        PAPI_destroy_eventset(&eventset);
        PAPI_shutdown();

        const char *task_names[] = {"getrf", "trsm", "gemm", "getri"};
        CounterData *counters[] = {getrf, trsm, gemm, getri};
        int num_tasks = sizeof(task_names) / sizeof(task_names[0]); // This gives the length of the tasks array
        dump_counters("invert", task_names, counters, num_tasks, matrix_size, tile_size, l3_cache_size, available_threads);
    }
}
