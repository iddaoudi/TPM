/*
 * =====================================================================================
 *
 *       Filename:  lu.h
 *
 *    Description:  Task-based LU algorithm
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  19/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void lu(tpm_desc A)
{
  char *name_with_id_char = NULL;
  struct timeval start, end;

  int eventset = PAPI_NULL;
  long long values[NEVENTS];
  const int available_threads = omp_get_max_threads();

  long long(*values_by_thread_getrf)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_trsm1)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_trsm2)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_gemm)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));

  if (TPM_PAPI)
  {
    int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};

    memset(values_by_thread_getrf, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_trsm1, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_trsm2, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_gemm, 0, available_threads * sizeof(long long[NEVENTS]));
  }

  // TPM library: initialization
  if (TPM_TRACE)
    tpm_downstream_start("lu", A.matrix_size, A.tile_size, NTH);

  int k = 0, m = 0, n = 0;

  for (k = 0; k < A.matrix_size / A.tile_size; k++)
  {
    double *tileA = A(k, k);

    if (TPM_TRACE)
    {
      // TPM library: create a unique task name
      name_with_id_char = tpm_unique_task_identifier("getrf", k, m, n);
      tpm_upstream_set_task_name(name_with_id_char);
    }

#pragma omp task depend(inout \
                        : tileA[0])
    {
      if (TPM_PAPI)
      {
        memset(values, 0, sizeof(values));
        eventset = PAPI_NULL;
        int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
        int ret = PAPI_create_eventset(&eventset);
        if (ret != PAPI_OK)
        {
          printf("GETRF task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
          exit(1);
        }
        PAPI_add_events(eventset, events, NEVENTS);

        // Start PAPI counters
        PAPI_start(eventset);
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
      tpm_dgetrf(A.tile_size, tileA, A.tile_size);

      if (TPM_PAPI)
      {
        // Start PAPI counters
        PAPI_stop(eventset, values);

        // Accumulate events values
        for (int i = 0; i < NEVENTS; i++)
        {
#pragma omp atomic update
          values_by_thread_getrf[omp_get_thread_num()][i] += values[i];
        }
        PAPI_unregister_thread();
      }
      else if (TPM_TRACE)
      {
        gettimeofday(&end, NULL);

        // TPM library: send time and name
        tpm_upstream_get_task_time(start, end, name_with_id_char);
      }
    }

    for (m = k + 1; m < A.matrix_size / A.tile_size; m++)
    {
      double *tileA = A(k, k);
      double *tileB = A(m, k);

      if (TPM_TRACE)
      {
        // TPM library: create a unique task name
        name_with_id_char = tpm_unique_task_identifier("trsm1", k, m, n);
        tpm_upstream_set_task_name(name_with_id_char);
      }

#pragma omp task depend(in                                     \
                        : tileA [0:A.tile_size * A.tile_size]) \
    depend(inout                                               \
           : tileB[A.tile_size * A.tile_size])
      {
        if (TPM_PAPI)
        {
          memset(values, 0, sizeof(values));
          eventset = PAPI_NULL;
          int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
          int ret = PAPI_create_eventset(&eventset);
          if (ret != PAPI_OK)
          {
            printf("TRSM1 task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
            exit(1);
          }
          PAPI_add_events(eventset, events, NEVENTS);

          // Start PAPI counters
          PAPI_start(eventset);
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
        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, A.tile_size, A.tile_size, 1.0, tileA,
                    A.tile_size, tileB, A.tile_size);

        if (TPM_PAPI)
        {
          // Start PAPI counters
          PAPI_stop(eventset, values);

          // Accumulate events values
          for (int i = 0; i < NEVENTS; i++)
          {
#pragma omp atomic update
            values_by_thread_trsm1[omp_get_thread_num()][i] += values[i];
          }
          PAPI_unregister_thread();
        }
        else if (TPM_TRACE)
        {
          gettimeofday(&end, NULL);

          // TPM library: send time and name
          tpm_upstream_get_task_time(start, end, name_with_id_char);
        }
      }
    }

    for (n = k + 1; n < A.matrix_size / A.tile_size; n++)
    {
      double *tileA = A(k, k);
      double *tileB = A(k, n);

      if (TPM_TRACE)
      {
        // TPM library: create a unique task name
        name_with_id_char = tpm_unique_task_identifier("trsm2", k, m, n);
        tpm_upstream_set_task_name(name_with_id_char);
      }

#pragma omp task depend(in                                     \
                        : tileA [0:A.tile_size * A.tile_size]) \
    depend(inout                                               \
           : tileB[A.tile_size * A.tile_size])
      {
        if (TPM_PAPI)
        {
          memset(values, 0, sizeof(values));
          eventset = PAPI_NULL;
          int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
          int ret = PAPI_create_eventset(&eventset);
          if (ret != PAPI_OK)
          {
            printf("TRSM2 task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
            exit(1);
          }
          PAPI_add_events(eventset, events, NEVENTS);

          // Start PAPI counters
          PAPI_start(eventset);
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
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasUnit, A.tile_size, A.tile_size, 1.0, tileA,
                    A.tile_size, tileB, A.tile_size);

        if (TPM_PAPI)
        {
          // Start PAPI counters
          PAPI_stop(eventset, values);

          // Accumulate events values
          for (int i = 0; i < NEVENTS; i++)
          {
#pragma omp atomic update
            values_by_thread_trsm2[omp_get_thread_num()][i] += values[i];
          }
          PAPI_unregister_thread();
        }
        else if (TPM_TRACE)
        {
          gettimeofday(&end, NULL);

          // TPM library: send time and name
          tpm_upstream_get_task_time(start, end, name_with_id_char);
        }
      }

      for (m = k + 1; m < A.matrix_size / A.tile_size; m++)
      {
        double *tileA = A(m, k);
        double *tileB = A(k, n);
        double *tileC = A(m, n);

        if (TPM_TRACE)
        {
          // TPM library: create a unique task name
          name_with_id_char = tpm_unique_task_identifier("gemm", k, m, n);
          tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task depend(in                                     \
                        : tileA [0:A.tile_size * A.tile_size], \
                          tileB [0:A.tile_size * A.tile_size]) \
    depend(inout                                               \
           : tileC [0:A.tile_size * A.tile_size])
        {
          if (TPM_PAPI)
          {
            memset(values, 0, sizeof(values));
            eventset = PAPI_NULL;
            int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
            int ret = PAPI_create_eventset(&eventset);
            if (ret != PAPI_OK)
            {
              printf("GEMM task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
              exit(1);
            }
            PAPI_add_events(eventset, events, NEVENTS);

            // Start PAPI counters
            PAPI_start(eventset);
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
          cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.tile_size,
                      A.tile_size, A.tile_size, -1.0, tileA, A.tile_size, tileB,
                      A.tile_size, 1.0, tileC, A.tile_size);

          if (TPM_PAPI)
          {
            // Start PAPI counters
            PAPI_stop(eventset, values);

            // Accumulate events values
            for (int i = 0; i < NEVENTS; i++)
            {
#pragma omp atomic update
              values_by_thread_gemm[omp_get_thread_num()][i] += values[i];
            }
            PAPI_unregister_thread();
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

  if (TPM_PAPI)
  {
#pragma omp taskwait
    PAPI_shutdown();

    CounterData getrf, trsm1, trsm2, gemm;

    accumulate_counters(getrf.values, values_by_thread_getrf, available_threads);
    accumulate_counters(trsm1.values, values_by_thread_trsm1, available_threads);
    accumulate_counters(trsm2.values, values_by_thread_trsm2, available_threads);
    accumulate_counters(gemm.values, values_by_thread_gemm, available_threads);

    compute_derived_metrics(&getrf);
    compute_derived_metrics(&trsm1);
    compute_derived_metrics(&trsm2);
    compute_derived_metrics(&gemm);

    // PAPI opens too much file descriptors without closing them
    int file_desc;
    for (file_desc = 3; file_desc < 1024; ++file_desc)
    {
      close(file_desc);
    }

    FILE *file;
    if ((file = fopen("counters.dat", "w")) == NULL)
    {
      perror("fopen failed");
      exit(1);
    }
    else
    {
      fprintf(file, "lu, getrf, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              getrf.mem_boundness, getrf.arithm_intensity, getrf.bmr, getrf.ilp, getrf.values[0]);
      fprintf(file, "lu, trsm1, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              trsm1.mem_boundness, trsm1.arithm_intensity, trsm1.bmr, trsm1.ilp, trsm1.values[0]);
      fprintf(file, "lu, trsm2, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              trsm2.mem_boundness, trsm2.arithm_intensity, trsm2.bmr, trsm2.ilp, trsm2.values[0]);
      fprintf(file, "lu, gemm, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              gemm.mem_boundness, gemm.arithm_intensity, gemm.bmr, gemm.ilp, gemm.values[0]);

      fclose(file);
    }
  }

  free(values_by_thread_getrf);
  free(values_by_thread_trsm1);
  free(values_by_thread_trsm2);
  free(values_by_thread_gemm);
}
