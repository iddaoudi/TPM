/*
 * =====================================================================================
 *
 *       Filename:  sylsvd.h
 *
 *    Description:  Task-based combined Sylvester-SVD algorithms
 *
 *        Version:  1.0
 *        Created:  14/05/2023
 *       Revision:  15/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

// #define LOG 1

void sylsvd(double *As[], double *Bs[], double *Xs[], double *Us[], double *Ss[], double *VTs[],
            double *EVs[], double *Ms[], int matrix_size, int iter)
{
    int info;

    char *name_with_id_char = NULL;
    struct timeval start = (struct timeval){0};
    struct timeval end = (struct timeval){0};

    int eventset = PAPI_NULL;
    long long values[NEVENTS];
    const int available_threads = omp_get_max_threads();

    // NEVENTS + 1 for the task weights
    CounterData *trsyl = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *gesvd = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *geev = (CounterData *)malloc(available_threads * sizeof(CounterData));
    CounterData *gemm = (CounterData *)malloc(available_threads * sizeof(CounterData));

    if (TPM_PAPI)
    {
        int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
        int ret = PAPI_create_eventset(&eventset);
        PAPI_add_events(eventset, events, NEVENTS);

        for (int i = 0; i < available_threads; i++)
        {
            memset(trsyl[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(gesvd[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(geev[i].values, 0, (NEVENTS + 1) * sizeof(long long));
            memset(gemm[i].values, 0, (NEVENTS + 1) * sizeof(long long));
        }
    }

    // TPM library: initialization
    if (TPM_TRACE)
        tpm_downstream_start("sylsvd", matrix_size * iter, matrix_size, NTH);

    for (int i = 0; i < iter; i++)
    {
        if (TPM_TRACE)
        {
            // TPM library: create a unique task name
            name_with_id_char = tpm_unique_task_identifier("trsyl", i, 0, 0);
            tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(name_with_id_char) depend(in : Xs[i]) depend(out : Xs[i])
        {
            double scale;
            if (TPM_PAPI)
            {
                memset(values, 0, sizeof(values));
                // Start PAPI counters
                int ret_start = PAPI_start(eventset);
                if (ret_start != PAPI_OK)
                {
                    printf("PAPI_start TRSYL error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
            info = LAPACKE_dtrsyl(LAPACK_ROW_MAJOR, 'N', 'N', 1, matrix_size, matrix_size, As[i], matrix_size, Bs[i], matrix_size, Xs[i], matrix_size, &scale);

            if (TPM_PAPI)
            {
                // Start PAPI counters
                int ret_stop = PAPI_stop(eventset, values);
                if (ret_stop != PAPI_OK)
                {
                    printf("PAPI_stop TRSYL error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                    exit(EXIT_FAILURE);
                }
                // Accumulate events values
                for (int i = 0; i < NEVENTS; i++)
                {
                    trsyl[omp_get_thread_num()].values[i] += values[i];
                }
                trsyl[omp_get_thread_num()].values[NEVENTS]++;
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
            name_with_id_char = tpm_unique_task_identifier("gesvd", i, 0, 0);
            tpm_upstream_set_task_name(name_with_id_char);
        }
#pragma omp task firstprivate(name_with_id_char) depend(in : Xs[i]) depend(out : Us[i], Ss[i], VTs[i])
        {
#ifdef LOG
            tpm_default_print_matrix("X", Xs[i], matrix_size);
#endif
            if (TPM_PAPI)
            {
                memset(values, 0, sizeof(values));
                // Start PAPI counters
                int ret_start = PAPI_start(eventset);
                if (ret_start != PAPI_OK)
                {
                    printf("PAPI_start GESVD error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
            double *superb = (double *)malloc((matrix_size - 1) * sizeof(double));
            info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', matrix_size, matrix_size, Xs[i], matrix_size, Ss[i], Us[i], matrix_size, VTs[i], matrix_size, superb);
            free(superb);

            if (TPM_PAPI)
            {
                // Start PAPI counters
                int ret_stop = PAPI_stop(eventset, values);
                if (ret_stop != PAPI_OK)
                {
                    printf("PAPI_stop GESVD error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                    exit(EXIT_FAILURE);
                }
                // Accumulate events values
                for (int i = 0; i < NEVENTS; i++)
                {
                    gesvd[omp_get_thread_num()].values[i] += values[i];
                }
                gesvd[omp_get_thread_num()].values[NEVENTS]++;
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
            name_with_id_char = tpm_unique_task_identifier("geev", i, 0, 0);
            tpm_upstream_set_task_name(name_with_id_char);
        }
#pragma omp task firstprivate(name_with_id_char) depend(in : Xs[i]) depend(out : EVs[i])
        {
            if (TPM_PAPI)
            {
                memset(values, 0, sizeof(values));
                // Start PAPI counters
                int ret_start = PAPI_start(eventset);
                if (ret_start != PAPI_OK)
                {
                    printf("PAPI_start GEEV error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
            double *EVI = (double *)malloc(matrix_size * sizeof(double)); // Imaginary part of the eigenvalues
            int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', matrix_size, Xs[i], matrix_size, EVs[i], EVI, NULL, matrix_size, Xs[i], matrix_size);
            free(EVI);

            if (TPM_PAPI)
            {
                // Start PAPI counters
                int ret_stop = PAPI_stop(eventset, values);
                if (ret_stop != PAPI_OK)
                {
                    printf("PAPI_stop GEEV error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                    exit(EXIT_FAILURE);
                }
                // Accumulate events values
                for (int i = 0; i < NEVENTS; i++)
                {
                    geev[omp_get_thread_num()].values[i] += values[i];
                }
                geev[omp_get_thread_num()].values[NEVENTS]++;
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
            name_with_id_char = tpm_unique_task_identifier("gemm", i, 0, 0);
            tpm_upstream_set_task_name(name_with_id_char);
        }
#pragma omp task firstprivate(name_with_id_char) depend(in : Us[i], VTs[i]) depend(out : Ms[i])
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
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        matrix_size, matrix_size, matrix_size,
                        1.0, Us[i], matrix_size, VTs[i], matrix_size,
                        0.0, Ms[i], matrix_size);

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

#pragma omp taskwait

    if (info != 0)
    {
        printf("Info error code: %d\n", info);
    }
#ifdef LOG
    else
    {
        for (int i = 0; i < iter; i++)
        {

            tpm_default_print_matrix("U", Us[i], matrix_size);
            tpm_default_print_matrix("S", Ss[i], matrix_size);
            tpm_default_print_matrix("VT", VTs[i], matrix_size);
            tpm_default_print_matrix("EV", EVs[i], matrix_size);
            tpm_default_print_matrix("M", Ms[i], matrix_size);
        }
    }
#endif

    if (TPM_PAPI)
    {
#pragma omp taskwait
        PAPI_destroy_eventset(&eventset);
        PAPI_shutdown();

        const char *task_names[] = {"trsyl", "gesvd", "geev", "gemm"};
        CounterData *counters[] = {trsyl, gesvd, geev, gemm};
        int num_tasks = sizeof(task_names) / sizeof(task_names[0]); // This gives the length of the tasks array
        dump_counters("sylsvd", task_names, counters, num_tasks, A.matrix_size, A.tile_size, l3_cache_size, available_threads);
    }
}