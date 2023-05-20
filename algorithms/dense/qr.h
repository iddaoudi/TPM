/*
 * =====================================================================================
 *
 *       Filename:  qr.h
 *
 *    Description:  Task-based QR algorithm
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

void qr(tpm_desc A, tpm_desc S)
{
  char *name_with_id_char = NULL;
  struct timeval start = (struct timeval){0};
  struct timeval end = (struct timeval){0};

  int eventset = PAPI_NULL;
  long long values[NEVENTS];
  const int available_threads = omp_get_max_threads();

  // NEVENTS + 1 for the task weights
  CounterData *geqrt = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *ormqr = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *tsqrt = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *tsmqr = (CounterData *)malloc(available_threads * sizeof(CounterData));

  if (TPM_PAPI)
  {
    int ret = PAPI_create_eventset(&eventset);
    PAPI_add_events(eventset, events, NEVENTS);

    for (int i = 0; i < available_threads; i++)
    {
      memset(geqrt[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(ormqr[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(tsqrt[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(tsmqr[i].values, 0, (NEVENTS + 1) * sizeof(long long));
    }
  }

  // TPM library: initialization
  if (TPM_TRACE)
    tpm_downstream_start("qr", A.matrix_size, A.tile_size, NTH);

  int k = 0, m = 0, n = 0;

  for (k = 0; k < A.matrix_size / A.tile_size; k++)
  {
    double *tileA = A(k, k);
    double *tileS = S(k, k);

    if (TPM_TRACE)
    {
      // TPM library: create a unique task name
      name_with_id_char = tpm_unique_task_identifier("geqrt", k, m, n);
      tpm_upstream_set_task_name(name_with_id_char);
    }

#pragma omp task firstprivate(name_with_id_char) \
    depend(inout : tileA[0 : S.tile_size * S.tile_size]) depend(out : tileS[0 : A.tile_size * S.tile_size])
    {
      double tho[S.tile_size];
      double work[S.tile_size * S.tile_size];

      if (TPM_PAPI)
      {
        memset(values, 0, sizeof(values));
        // Start PAPI counters
        int ret_start = PAPI_start(eventset);
        if (ret_start != PAPI_OK)
        {
          printf("PAPI_start GEQRT error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
      tpm_dgeqrt(A.tile_size, S.tile_size, tileA, A.tile_size, tileS,
                 S.tile_size, &tho[0], &work[0]);

      if (TPM_PAPI)
      {
        // Start PAPI counters
        int ret_stop = PAPI_stop(eventset, values);
        if (ret_stop != PAPI_OK)
        {
          printf("PAPI_stop GEQRT error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
          exit(EXIT_FAILURE);
        }
        // Accumulate events values
        for (int i = 0; i < NEVENTS; i++)
        {
          geqrt[omp_get_thread_num()].values[i] += values[i];
        }
        geqrt[omp_get_thread_num()].values[NEVENTS]++;
      }
      else if (TPM_TRACE)
      {
        gettimeofday(&end, NULL);
        // TPM library: send time and name
        tpm_upstream_get_task_time(start, end, name_with_id_char);
      }

      for (n = k + 1; n < A.matrix_size / A.tile_size; n++)
      {
        double *tileA = A(k, k);
        double *tileS = S(k, k);
        double *tileB = A(k, n);

        if (TPM_TRACE)
        {
          // TPM library: create a unique task name
          name_with_id_char = tpm_unique_task_identifier("ormqr", k, m, n);
          tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(name_with_id_char) depend(in : tileA[0 : S.tile_size * S.tile_size], tileS[0 : A.tile_size * S.tile_size]) depend(inout : tileB[0 : S.tile_size * S.tile_size])
        {
          double work[S.tile_size * S.tile_size];

          if (TPM_PAPI)
          {
            memset(values, 0, sizeof(values));
            // Start PAPI counters
            int ret_start = PAPI_start(eventset);
            if (ret_start != PAPI_OK)
            {
              printf("PAPI_start ORMQR error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
          tpm_dormqr(tpm_left, tpm_transpose, A.tile_size, tileA, A.tile_size,
                     tileS, S.tile_size, tileB, A.tile_size, &work[0],
                     S.tile_size);

          if (TPM_PAPI)
          {
            // Start PAPI counters
            int ret_stop = PAPI_stop(eventset, values);
            if (ret_stop != PAPI_OK)
            {
              printf("PAPI_stop ORMQR error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
              exit(EXIT_FAILURE);
            }
            // Accumulate events values
            for (int i = 0; i < NEVENTS; i++)
            {
              ormqr[omp_get_thread_num()].values[i] += values[i];
            }
            ormqr[omp_get_thread_num()].values[NEVENTS]++;
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
        double *tileA = A(k, k);
        double *tileS = S(m, k);
        double *tileB = A(m, k);

        if (TPM_TRACE)
        {
          // TPM library: create a unique task name
          name_with_id_char = tpm_unique_task_identifier("tsqrt", k, m, n);
          tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(name_with_id_char) depend(inout : tileA[0 : S.tile_size * S.tile_size], tileB[0 : S.tile_size * S.tile_size]) depend(out : tileS[0 : S.tile_size * A.tile_size])
        {
          double work[S.tile_size * S.tile_size];
          double tho[S.tile_size];

          if (TPM_PAPI)
          {
            memset(values, 0, sizeof(values));
            // Start PAPI counters
            int ret_start = PAPI_start(eventset);
            if (ret_start != PAPI_OK)
            {
              printf("PAPI_start TSQRT error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
          tpm_dtsqrt(A.tile_size, tileA, A.tile_size, tileB, A.tile_size, tileS,
                     S.tile_size, &tho[0], &work[0]);

          if (TPM_PAPI)
          {
            // Start PAPI counters
            int ret_stop = PAPI_stop(eventset, values);
            if (ret_stop != PAPI_OK)
            {
              printf("PAPI_stop TSQRT error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
              exit(EXIT_FAILURE);
            }
            // Accumulate events values
            for (int i = 0; i < NEVENTS; i++)
            {
              tsqrt[omp_get_thread_num()].values[i] += values[i];
            }
            tsqrt[omp_get_thread_num()].values[NEVENTS]++;
          }
          else if (TPM_TRACE)
          {
            gettimeofday(&end, NULL);
            // TPM library: send time and name
            tpm_upstream_get_task_time(start, end, name_with_id_char);
          }
        }

        for (n = k + 1; n < A.matrix_size / A.tile_size; n++)
        {
          double *tileA = A(k, n);
          double *tileS = S(m, k);
          double *tileB = A(m, n);
          double *tileC = A(m, k);

          if (TPM_TRACE)
          {
            // TPM library: create a unique task name
            name_with_id_char = tpm_unique_task_identifier("tsmqr", k, m, n);
            tpm_upstream_set_task_name(name_with_id_char);
          }

#pragma omp task firstprivate(name_with_id_char) depend(inout : tileA[0 : S.tile_size * S.tile_size], tileB[0 : S.tile_size * S.tile_size]) depend(in : tileC[0 : S.tile_size * S.tile_size], tileS[0 : A.tile_size * S.tile_size])
          {
            double work[S.tile_size * S.tile_size];

            if (TPM_PAPI)
            {
              memset(values, 0, sizeof(values));
              // Start PAPI counters
              int ret_start = PAPI_start(eventset);
              if (ret_start != PAPI_OK)
              {
                printf("PAPI_start TSMQR error %d: %s\n", ret_start, PAPI_strerror(ret_start));
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
            tpm_dtsmqr(tpm_left, tpm_transpose, A.tile_size, A.tile_size,
                       A.tile_size, A.tile_size, tileA, A.tile_size, tileB,
                       A.tile_size, tileC, A.tile_size, tileS, S.tile_size,
                       &work[0], A.tile_size);

            if (TPM_PAPI)
            {
              // Start PAPI counters
              int ret_stop = PAPI_stop(eventset, values);
              if (ret_stop != PAPI_OK)
              {
                printf("PAPI_stop TSMQR error %d: %s\n", ret_stop, PAPI_strerror(ret_stop));
                exit(EXIT_FAILURE);
              }
              // Accumulate events values
              for (int i = 0; i < NEVENTS; i++)
              {
                tsmqr[omp_get_thread_num()].values[i] += values[i];
              }
              tsmqr[omp_get_thread_num()].values[NEVENTS]++;
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

    const char *task_names[] = {"geqrt", "ormqr", "tsqrt", "tsmqr"};
    CounterData *counters[] = {geqrt, ormqr, tsqrt, tsmqr};
    int num_tasks = sizeof(task_names) / sizeof(task_names[0]); // This gives the length of the tasks array
    dump_counters("qr", task_names, counters, num_tasks, A.matrix_size, A.tile_size, l3_cache_size, available_threads);
  }
}