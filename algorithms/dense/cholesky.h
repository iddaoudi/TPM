/*
 * =====================================================================================
 *
 *       Filename:  cholesky.h
 *
 *    Description:  Task-based Choklesky algorithm
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

void cholesky(tpm_desc A)
{
  char *name_with_id_char = NULL;
  struct timeval start, end;

  int eventset = PAPI_NULL;
  long long values[NEVENTS];
  const int available_threads = omp_get_max_threads();

  long long(*values_by_thread_potrf)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_trsm)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_syrk)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));
  long long(*values_by_thread_gemm)[NEVENTS] = malloc(available_threads * sizeof(long long[NEVENTS]));

  if (TPM_PAPI)
  {
    int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};

    memset(values_by_thread_potrf, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_trsm, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_syrk, 0, available_threads * sizeof(long long[NEVENTS]));
    memset(values_by_thread_gemm, 0, available_threads * sizeof(long long[NEVENTS]));
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
    depend(inout                                 \
           : tileA [0:A.tile_size * A.tile_size])
    {
      if (TPM_PAPI)
      {
        memset(values, 0, sizeof(values));
        eventset = PAPI_NULL;
        int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
        int ret = PAPI_create_eventset(&eventset);
        if (ret != PAPI_OK)
        {
          printf("POTRF task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
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
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', A.tile_size, tileA, A.tile_size);

      if (TPM_PAPI)
      {
        // Start PAPI counters
        PAPI_stop(eventset, values);

        // Accumulate events values
        for (int i = 0; i < NEVENTS; i++)
        {
#pragma omp atomic update
          values_by_thread_potrf[omp_get_thread_num()][i] += values[i];
        }
        PAPI_unregister_thread();
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

#pragma omp task firstprivate(name_with_id_char)  \
    depend(in                                     \
           : tileA [0:A.tile_size * A.tile_size]) \
        depend(inout                              \
               : tileB [0:A.tile_size * A.tile_size])
        {
          if (TPM_PAPI)
          {
            memset(values, 0, sizeof(values));
            eventset = PAPI_NULL;
            int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
            int ret = PAPI_create_eventset(&eventset);
            if (ret != PAPI_OK)
            {
              printf("TRSM task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
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
          cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
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
              values_by_thread_trsm[omp_get_thread_num()][i] += values[i];
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

#pragma omp task firstprivate(name_with_id_char)  \
    depend(in                                     \
           : tileA [0:A.tile_size * A.tile_size]) \
        depend(inout                              \
               : tileB [0:A.tile_size * A.tile_size])
        {
          if (TPM_PAPI)
          {
            memset(values, 0, sizeof(values));
            eventset = PAPI_NULL;
            int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
            int ret = PAPI_create_eventset(&eventset);
            if (ret != PAPI_OK)
            {
              printf("SYRK task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
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
          cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, A.tile_size,
                      A.tile_size, -1.0, tileA, A.tile_size, 1.0, tileB,
                      A.tile_size);

          if (TPM_PAPI)
          {
            // Start PAPI counters
            PAPI_stop(eventset, values);

            // Accumulate events values
            for (int i = 0; i < NEVENTS; i++)
            {
#pragma omp atomic update
              values_by_thread_syrk[omp_get_thread_num()][i] += values[i];
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

#pragma omp task firstprivate(name_with_id_char)  \
    depend(in                                     \
           : tileA [0:A.tile_size * A.tile_size], \
             tileB [0:A.tile_size * A.tile_size]) \
        depend(inout                              \
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
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A.tile_size,
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
  }

  if (TPM_PAPI)
  {
#pragma omp taskwait
    PAPI_shutdown();

    CounterData potrf, trsm, syrk, gemm;

    accumulate_counters(potrf.values, values_by_thread_potrf, available_threads);
    accumulate_counters(trsm.values, values_by_thread_trsm, available_threads);
    accumulate_counters(syrk.values, values_by_thread_syrk, available_threads);
    accumulate_counters(gemm.values, values_by_thread_gemm, available_threads);

    compute_derived_metrics(&potrf);
    compute_derived_metrics(&trsm);
    compute_derived_metrics(&syrk);
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
      fprintf(file, "cholesky, potrf, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              potrf.mem_boundness, potrf.arithm_intensity, potrf.bmr, potrf.ilp, potrf.values[0]);
      fprintf(file, "cholesky, trsm, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              trsm.mem_boundness, trsm.arithm_intensity, trsm.bmr, trsm.ilp, trsm.values[0]);
      fprintf(file, "cholesky, syrk, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              syrk.mem_boundness, syrk.arithm_intensity, syrk.bmr, syrk.ilp, syrk.values[0]);
      fprintf(file, "cholesky, gemm, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              gemm.mem_boundness, gemm.arithm_intensity, gemm.bmr, gemm.ilp, gemm.values[0]);

      fclose(file);
    }
  }

  free(values_by_thread_potrf);
  free(values_by_thread_trsm);
  free(values_by_thread_syrk);
  free(values_by_thread_gemm);
}