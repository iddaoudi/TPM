/*
 * =====================================================================================
 *
 *       Filename:  cholesky.h
 *
 *    Description:  Task-based Choklesky algorithm
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  21/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void cholesky(tpm_desc A)
{
  char *name_with_id_char = NULL;
  struct timeval start = (struct timeval){0};
  struct timeval end = (struct timeval){0};

  int eventset = PAPI_NULL;
  long long values[NEVENTS];
  const int available_threads = omp_get_max_threads();

  // NEVENTS + 1 for the task weights
  CounterData *potrf = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *trsm = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *syrk = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *gemm = (CounterData *)malloc(available_threads * sizeof(CounterData));

  if (TPM_PAPI)
  {
    int ret = PAPI_create_eventset(&eventset);
    PAPI_add_events(eventset, events, NEVENTS);

    for (int i = 0; i < available_threads; i++)
    {
      memset(potrf[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(trsm[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(syrk[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(gemm[i].values, 0, (NEVENTS + 1) * sizeof(long long));
    }
  }

  // TPM library: initialization
  if (TPM_TRACE)
    tpm_downstream_start("cholesky", A.matrix_size, A.tile_size, NTH);

  int k = 0, m = 0, n = 0;

  for (k = 0; k < A.matrix_size / A.tile_size; k++)
  {
    double *tileA = A(k, k);

    if (TPM_TRACE)
    {
      // TPM library: create a unique task name
      name_with_id_char = tpm_unique_task_identifier("potrf", k, m, n);
      tpm_upstream_set_task_name(name_with_id_char);
    }

#pragma omp task firstprivate(name_with_id_char) \
    depend(inout : tileA[0 : A.tile_size * A.tile_size])
    {
      if (TPM_PAPI)
      {
        memset(values, 0, sizeof(values));
        // Start PAPI counters
        int ret_start = PAPI_start(eventset);
        if (ret_start != PAPI_OK)
        {
          printf("PAPI_start POTRF error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', A.tile_size, tileA, A.tile_size);

      if (TPM_PAPI)
      {
        // Stop PAPI counters
        int ret_stop = PAPI_stop(eventset, values);
        if (ret_stop != PAPI_OK)
        {
          printf("PAPI_stop POTRF error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
          exit(EXIT_FAILURE);
        }
        // Accumulate events values
        for (int i = 0; i < NEVENTS; i++)
        {
          potrf[omp_get_thread_num()].values[i] += values[i];
        }
        potrf[omp_get_thread_num()].values[NEVENTS]++;
      }
      else if (TPM_TRACE)
      {
        gettimeofday(&end, NULL);
        // TPM library: send time and name
        tpm_upstream_get_task_time(start, end, name_with_id_char);
      }

      for (m = k + 1; m < A.matrix_size / A.tile_size; m++)
      {
        double *tileA = A(k, k);
        double *tileB = A(k, m);

        if (TPM_TRACE)
        {
          // TPM library: create a unique task name
          name_with_id_char = tpm_unique_task_identifier("trsm", k, m, n);
          tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(name_with_id_char)      \
    depend(in : tileA[0 : A.tile_size * A.tile_size]) \
    depend(inout : tileB[0 : A.tile_size * A.tile_size])
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
          cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                      CblasNonUnit, A.tile_size, A.tile_size, 1.0, tileA,
                      A.tile_size, tileB, A.tile_size);

          if (TPM_PAPI)
          {
            // Stop PAPI counters
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
      }

      for (m = k + 1; m < A.matrix_size / A.tile_size; m++)
      {
        double *tileA = A(k, m);
        double *tileB = A(m, m);

        if (TPM_TRACE)
        {
          // TPM library: create a unique task name
          name_with_id_char = tpm_unique_task_identifier("syrk", k, m, n);
          tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(name_with_id_char)      \
    depend(in : tileA[0 : A.tile_size * A.tile_size]) \
    depend(inout : tileB[0 : A.tile_size * A.tile_size])
        {
          if (TPM_PAPI)
          {
            memset(values, 0, sizeof(values));
            // Start PAPI counters
            int ret_start = PAPI_start(eventset);
            if (ret_start != PAPI_OK)
            {
              printf("PAPI_start SYRK error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
          cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, A.tile_size,
                      A.tile_size, -1.0, tileA, A.tile_size, 1.0, tileB,
                      A.tile_size);

          if (TPM_PAPI)
          {
            // Stop PAPI counters
            int ret_stop = PAPI_stop(eventset, values);
            if (ret_stop != PAPI_OK)
            {
              printf("PAPI_stop SYRK error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
              exit(EXIT_FAILURE);
            }
            // Accumulate events values
            for (int i = 0; i < NEVENTS; i++)
            {
              syrk[omp_get_thread_num()].values[i] += values[i];
            }
            syrk[omp_get_thread_num()].values[NEVENTS]++;
          }
          else if (TPM_TRACE)
          {
            gettimeofday(&end, NULL);
            // TPM library: send time and name
            tpm_upstream_get_task_time(start, end, name_with_id_char);
          }
        }

        for (n = k + 1; n < m; n++)
        {
          double *tileA = A(k, n);
          double *tileB = A(k, m);
          double *tileC = A(n, m);

          if (TPM_TRACE)
          {
            // TPM library: create a unique task name
            name_with_id_char = tpm_unique_task_identifier("gemm", k, m, n);
            tpm_upstream_set_task_name(name_with_id_char);
          }

#pragma omp task firstprivate(name_with_id_char)      \
    depend(in : tileA[0 : A.tile_size * A.tile_size], \
               tileB[0 : A.tile_size * A.tile_size])  \
    depend(inout : tileC[0 : A.tile_size * A.tile_size])
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
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A.tile_size,
                        A.tile_size, A.tile_size, -1.0, tileA, A.tile_size, tileB,
                        A.tile_size, 1.0, tileC, A.tile_size);

            if (TPM_PAPI)
            {
              // Stop PAPI counters
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
  }

  if (TPM_PAPI)
  {
#pragma omp taskwait
    PAPI_destroy_eventset(&eventset);
    PAPI_shutdown();

    const char *task_names[] = {"potrf", "trsm", "syrk", "gemm"};
    CounterData *counters[] = {potrf, trsm, syrk, gemm};
    int num_tasks = sizeof(task_names) / sizeof(task_names[0]); // This gives the length of the tasks array
    dump_counters("cholesky", task_names, counters, num_tasks, A.matrix_size, A.tile_size, l3_cache_size, available_threads);
  }
}
