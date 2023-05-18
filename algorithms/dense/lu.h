/*
 * =====================================================================================
 *
 *       Filename:  lu.h
 *
 *    Description:  Task-based LU algorithm
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  18/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void lu(int matrix_size, int tile_size, double *pA, int *ipiv, double *A)
{
    char *name_with_id_char = NULL;
    struct timeval start = (struct timeval){0};
    struct timeval end = (struct timeval){0};

    int eventset = PAPI_NULL;
    long long values[NEVENTS];
    const int available_threads = omp_get_max_threads();

    // NEVENTS + 1 for the task weights
    CounterData *getrfpiv = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *trsmswp = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *gemm = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *geswp = (CounterData *)malloc(available_threads * sizeof(CounterData));

    if (TPM_PAPI)
    {
        int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
        int ret = PAPI_create_eventset(&eventset);
        PAPI_add_events(eventset, events, NEVENTS);

        for (int i = 0; i < available_threads; i++)
        {
            memset(getrfpiv[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(trsmswp[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(gemm[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(geswp[i].values, 0, (NEVENTS + 1) * sizeof(long long));
        }
    }

    // TPM library: initialization
    if (TPM_TRACE)
        tpm_downstream_start("lu", matrix_size, tile_size, NTH);

    double alpha = 1., neg = -1.;

    for (int k = 0; k < matrix_size / tile_size; k++)
    {
        int m = matrix_size - k * tile_size;
        double *akk = A + k * tile_size * matrix_size + k * tile_size * tile_size;

        if (TPM_TRACE)
        {
            // TPM library: create a unique task name
            name_with_id_char = tpm_unique_task_identifier("getrfpiv", k, 0, 0);
            tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(name_with_id_char, akk, m) depend(inout : akk[0 : m * tile_size]) \
    depend(out : ipiv[k * tile_size : tile_size])
        {
            if (TPM_PAPI)
            {
                memset(values, 0, sizeof(values));
                // Start PAPI counters
                int ret_start = PAPI_start(eventset);
                if (ret_start != PAPI_OK)
                {
                    printf("PAPI_start GETRFPIV error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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

            // Kernels
            tpm_tile_to_matrix(A + k * tile_size * matrix_size + k * tile_size * tile_size,
                               pA + k * tile_size * matrix_size + k * tile_size, m, tile_size,
                               tile_size, matrix_size);
            LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, tile_size, pA + k * tile_size * matrix_size + k * tile_size,
                           matrix_size, ipiv + k * tile_size);
            // Update the ipiv
            for (int i = k * tile_size; i < k * tile_size + tile_size; i++)
            {
                ipiv[i] += k * tile_size;
            }
            tpm_matrix_to_tile(A + k * tile_size * matrix_size + k * tile_size * tile_size,
                               pA + k * tile_size * matrix_size + k * tile_size, m, tile_size,
                               tile_size, matrix_size);

            if (TPM_PAPI)
            {
                // Start PAPI counters
                int ret_stop = PAPI_stop(eventset, values);
                if (ret_stop != PAPI_OK)
                {
                    printf("PAPI_stop GETRFPIV error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                    exit(EXIT_FAILURE);
                }
                // Accumulate events values
                for (int i = 0; i < NEVENTS; i++)
                {
                    getrfpiv[omp_get_thread_num()].values[i] += values[i];
                }
                getrfpiv[omp_get_thread_num()].values[NEVENTS]++;
            }
            else if (TPM_TRACE)
            {
                gettimeofday(&end, NULL);
                // TPM library: send time and name
                tpm_upstream_get_task_time(start, end, name_with_id_char);
            }
        }

        // Update trailing submatrix
        for (int j = k + 1; j < matrix_size / tile_size; j++)
        {
            double *akj = A + j * tile_size * matrix_size + k * tile_size * tile_size;

            if (TPM_TRACE)
            {
                // TPM library: create a unique task name
                name_with_id_char = tpm_unique_task_identifier("trsmswp", k, j, 0);
                tpm_upstream_set_task_name(name_with_id_char);
            }

#pragma omp task firstprivate(name_with_id_char, akk, akj, m) depend(in : akk[0 : m * tile_size]) \
    depend(in : ipiv[k * tile_size : tile_size]) depend(inout : akj[0 : tile_size * tile_size])
            {
                if (TPM_PAPI)
                {
                    memset(values, 0, sizeof(values));
                    // Start PAPI counters
                    int ret_start = PAPI_start(eventset);
                    if (ret_start != PAPI_OK)
                    {
                        printf("PAPI_start TRSMSWP error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
                int k1 = k * tile_size;
                int k2 = k * tile_size + tile_size;
                tpm_geswp(A + j * tile_size * matrix_size, tile_size, k1, k2, ipiv);

                cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, tile_size,
                            tile_size, alpha, akk, tile_size, akj, tile_size);

                if (TPM_PAPI)
                {
                    // Start PAPI counters
                    int ret_stop = PAPI_stop(eventset, values);
                    if (ret_stop != PAPI_OK)
                    {
                        printf("PAPI_stop TRSMSWP error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                        exit(EXIT_FAILURE);
                    }
                    // Accumulate events values
                    for (int i = 0; i < NEVENTS; i++)
                    {
                        trsmswp[omp_get_thread_num()].values[i] += values[i];
                    }
                    trsmswp[omp_get_thread_num()].values[NEVENTS]++;
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
                name_with_id_char = tpm_unique_task_identifier("gemm", k, j, 0);
                tpm_upstream_set_task_name(name_with_id_char);
            }
#pragma omp task firstprivate(name_with_id_char, akk, akj, m) depend(in : akk[0 : m * tile_size]) \
    depend(inout : akj[0 : tile_size * tile_size], ipiv[k * tile_size : tile_size])
            {
                for (int i = k + 1; i < matrix_size / tile_size; i++)
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
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, tile_size, tile_size,
                                tile_size, neg,
                                A + k * tile_size * matrix_size + i * tile_size * tile_size,
                                tile_size, akj, tile_size, alpha,
                                A + j * tile_size * matrix_size + i * tile_size * tile_size, tile_size);

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
    // Pivoting to the left
    for (int t = 1; t < matrix_size / tile_size; t++)
    {
        if (TPM_TRACE)
        {
            // TPM library: create a unique task name
            name_with_id_char = tpm_unique_task_identifier("geswp", t, 0, 0);
            tpm_upstream_set_task_name(name_with_id_char);
        }
#pragma omp task firstprivate(name_with_id_char)                               \
    depend(in : ipiv[((matrix_size / tile_size) - 1) * tile_size : tile_size]) \
    depend(inout : A[t * tile_size * matrix_size : matrix_size * tile_size])
        {
            if (TPM_PAPI)
            {
                memset(values, 0, sizeof(values));
                // Start PAPI counters
                int ret_start = PAPI_start(eventset);
                if (ret_start != PAPI_OK)
                {
                    printf("PAPI_start GESWP error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
            tpm_geswp(A + (t - 1) * tile_size * matrix_size, tile_size, t * tile_size, matrix_size, ipiv);

            if (TPM_PAPI)
            {
                // Start PAPI counters
                int ret_stop = PAPI_stop(eventset, values);
                if (ret_stop != PAPI_OK)
                {
                    printf("PAPI_stop GESWP error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                    exit(EXIT_FAILURE);
                }
                // Accumulate events values
                for (int i = 0; i < NEVENTS; i++)
                {
                    geswp[omp_get_thread_num()].values[i] += values[i];
                }
                geswp[omp_get_thread_num()].values[NEVENTS]++;
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

        const char *task_names[] = {"getrfpiv", "trsmswp", "gemm", "geswp"};
        CounterData *counters[] = {getrfpiv, trsmswp, gemm, geswp};
        int num_tasks = sizeof(task_names) / sizeof(task_names[0]); // This gives the length of the tasks array
        dump_counters("lu", task_names, counters, num_tasks, matrix_size, tile_size, l3_cache_size, available_threads);
    }
}