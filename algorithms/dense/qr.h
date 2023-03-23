/*
 * =====================================================================================
 *
 *       Filename:  qr.h
 *
 *    Description:  Task-based QR algorithm
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  21/03/2023
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

  long long(*values_by_thread_geqrt)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_ormqr)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_tsqrt)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_tsmqr)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));

  if (TPM_PAPI)
  {
    int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
    int ret = PAPI_create_eventset(&eventset);
    PAPI_add_events(eventset, events, NEVENTS);
    memset(values_by_thread_geqrt, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_ormqr, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_tsqrt, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_tsmqr, 0, available_threads * sizeof(long long[NEVENTS]));
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

#pragma omp task firstprivate(eventset) depend(inout                                             \
                                               : tileA [0:S.tile_size * S.tile_size]) depend(out \
                                                                                             : tileS [0:A.tile_size * S.tile_size])
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
          values_by_thread_geqrt[omp_get_thread_num()][i] += values[i];
        }
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
          name_with_id_char = tpm_unique_task_identifier("geqrt", k, m, n);
          tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(eventset) depend(in                                                                                       \
                                               : tileA [0:S.tile_size * S.tile_size], tileS [0:A.tile_size * S.tile_size]) depend(inout \
                                                                                                                                  : tileB [0:S.tile_size * S.tile_size])
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
              values_by_thread_ormqr[omp_get_thread_num()][i] += values[i];
            }
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
          name_with_id_char = tpm_unique_task_identifier("geqrt", k, m, n);
          tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task firstprivate(eventset) depend(inout                                                                                  \
                                               : tileA [0:S.tile_size * S.tile_size], tileB [0:S.tile_size * S.tile_size]) depend(out \
                                                                                                                                  : tileS [0:S.tile_size * A.tile_size])
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
              values_by_thread_tsqrt[omp_get_thread_num()][i] += values[i];
            }
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
            name_with_id_char = tpm_unique_task_identifier("geqrt", k, m, n);
            tpm_upstream_set_task_name(name_with_id_char);
          }

#pragma omp task firstprivate(eventset) depend(inout                                                                                 \
                                               : tileA [0:S.tile_size * S.tile_size], tileB [0:S.tile_size * S.tile_size]) depend(in \
                                                                                                                                  : tileC [0:S.tile_size * S.tile_size], tileS [0:A.tile_size * S.tile_size])
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
                values_by_thread_tsmqr[omp_get_thread_num()][i] += values[i];
              }
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

    CounterData geqrt, ormqr, tsqrt, tsmqr;

    accumulate_counters(geqrt.values, values_by_thread_geqrt, available_threads);
    accumulate_counters(ormqr.values, values_by_thread_ormqr, available_threads);
    accumulate_counters(tsqrt.values, values_by_thread_tsqrt, available_threads);
    accumulate_counters(tsmqr.values, values_by_thread_tsmqr, available_threads);

    compute_derived_metrics(&geqrt);
    compute_derived_metrics(&ormqr);
    compute_derived_metrics(&tsqrt);
    compute_derived_metrics(&tsmqr);

    // PAPI opens too much file descriptors without closing them
    int file_desc;
    for (file_desc = 3; file_desc < 1024; ++file_desc)
    {
      close(file_desc);
    }

    FILE *file;
    if ((file = fopen("counters_qr.csv", "a+")) == NULL)
    {
      perror("fopen failed");
      exit(EXIT_FAILURE);
    }
    else
    {
      fseek(file, 0, SEEK_SET);
      int first_char = fgetc(file);
      if (first_char == EOF)
      {
        fprintf(file, "algorithm, task, matrix_size, tile_size, mem_boundness, arithm_intensity, bmr, ilp, l3_cache_ratio\n");
      }

      fprintf(file, "qr, geqrt, %d, %d, %f, %f, %f, %f, %f\n", A.matrix_size, A.tile_size,
              geqrt.mem_boundness, geqrt.arithm_intensity, geqrt.bmr, geqrt.ilp, (double)geqrt.values[0] / (double)l3_cache_size);
      fprintf(file, "qr, ormqr, %d, %d, %f, %f, %f, %f, %f\n", A.matrix_size, A.tile_size,
              ormqr.mem_boundness, ormqr.arithm_intensity, ormqr.bmr, ormqr.ilp, (double)ormqr.values[0] / (double)l3_cache_size);
      fprintf(file, "qr, tsqrt, %d, %d, %f, %f, %f, %f, %f\n", A.matrix_size, A.tile_size,
              tsqrt.mem_boundness, tsqrt.arithm_intensity, tsqrt.bmr, tsqrt.ilp, (double)tsqrt.values[0] / (double)l3_cache_size);
      fprintf(file, "qr, tsmqr, %d, %d, %f, %f, %f, %f, %f\n", A.matrix_size, A.tile_size,
              tsmqr.mem_boundness, tsmqr.arithm_intensity, tsmqr.bmr, tsmqr.ilp, (double)tsmqr.values[0] / (double)l3_cache_size);

      fclose(file);
    }
  }

  free(values_by_thread_geqrt);
  free(values_by_thread_ormqr);
  free(values_by_thread_tsqrt);
  free(values_by_thread_tsmqr);
}
