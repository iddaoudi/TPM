/*
 * =====================================================================================
 *
 *       Filename:  cholesky.h
 *
 *    Description:  Task-based Choklesky algorithm
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  26/03/2023
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
  CounterData *potrf2 = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *trsm2 = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *syrk2 = (CounterData *)malloc(available_threads * sizeof(CounterData));
  CounterData *gemm2 = (CounterData *)malloc(available_threads * sizeof(CounterData));

  long long(*values_by_thread_potrf)[NEVENTS + 1] = malloc(available_threads * sizeof(long long[NEVENTS + 1]));
  long long(*values_by_thread_trsm)[NEVENTS + 1] = malloc(available_threads * sizeof(long long[NEVENTS + 1]));
  long long(*values_by_thread_syrk)[NEVENTS + 1] = malloc(available_threads * sizeof(long long[NEVENTS + 1]));
  long long(*values_by_thread_gemm)[NEVENTS + 1] = malloc(available_threads * sizeof(long long[NEVENTS + 1]));

  if (TPM_PAPI)
  {
    int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
    int ret = PAPI_create_eventset(&eventset);
    PAPI_add_events(eventset, events, NEVENTS);

    for (int i = 0; i < available_threads; i++)
    {
      memset(potrf2[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(trsm2[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(syrk2[i].values, 0, (NEVENTS + 1) * sizeof(long long));
      memset(gemm2[i].values, 0, (NEVENTS + 1) * sizeof(long long));
    }
    memset(values_by_thread_potrf, 0, available_threads * sizeof(long long[NEVENTS + 1]));
    memset(values_by_thread_trsm, 0, available_threads * sizeof(long long[NEVENTS + 1]));
    memset(values_by_thread_syrk, 0, available_threads * sizeof(long long[NEVENTS + 1]));
    memset(values_by_thread_gemm, 0, available_threads * sizeof(long long[NEVENTS + 1]));
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
          potrf2[omp_get_thread_num()].values[i] += values[i];
          values_by_thread_potrf[omp_get_thread_num()][i] += values[i];
        }
        potrf2[omp_get_thread_num()].values[NEVENTS]++;
        values_by_thread_potrf[omp_get_thread_num()][NEVENTS]++;
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
              trsm2[omp_get_thread_num()].values[i] += values[i];
              values_by_thread_trsm[omp_get_thread_num()][i] += values[i];
            }
            trsm2[omp_get_thread_num()].values[NEVENTS]++;
            values_by_thread_trsm[omp_get_thread_num()][NEVENTS]++;
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
              syrk2[omp_get_thread_num()].values[i] += values[i];
              values_by_thread_syrk[omp_get_thread_num()][i] += values[i];
            }
            syrk2[omp_get_thread_num()].values[NEVENTS]++;
            values_by_thread_syrk[omp_get_thread_num()][NEVENTS]++;
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
                gemm2[omp_get_thread_num()].values[i] += values[i];
                values_by_thread_gemm[omp_get_thread_num()][i] += values[i];
              }
              gemm2[omp_get_thread_num()].values[NEVENTS]++;
              values_by_thread_gemm[omp_get_thread_num()][NEVENTS]++;
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

    CounterData final_potrf, final_trsm, final_syrk, final_gemm;

    accumulate_counters2(&final_potrf, potrf2, available_threads);
    accumulate_counters2(&final_trsm, trsm2, available_threads);
    accumulate_counters2(&final_syrk, syrk2, available_threads);
    accumulate_counters2(&final_gemm, gemm2, available_threads);

    CounterData potrf, trsm, syrk, gemm;

    accumulate_counters(potrf.values, values_by_thread_potrf, available_threads);
    accumulate_counters(trsm.values, values_by_thread_trsm, available_threads);
    accumulate_counters(syrk.values, values_by_thread_syrk, available_threads);
    accumulate_counters(gemm.values, values_by_thread_gemm, available_threads);

    compute_derived_metrics(&final_potrf);
    compute_derived_metrics(&final_trsm);
    compute_derived_metrics(&final_syrk);
    compute_derived_metrics(&final_gemm);

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

    FILE *file2;
    if ((file2 = fopen("counters_cholesky2.csv", "a+")) == NULL)
    {
      perror("fopen failed");
      exit(EXIT_FAILURE);
    }
    else
    {
      fseek(file2, 0, SEEK_SET);
      int first_char = fgetc(file2);
      if (first_char == EOF)
      {
        fprintf(file2, "algorithm,task,matrix_size,tile_size,mem_boundness,arithm_intensity,bmr,ilp,l3_cache_ratio,weight\n");
      }

      fprintf(file2, "cholesky,potrf,%d,%d,%f,%f,%f,%f,%f,%d\n", A.matrix_size, A.tile_size,
              final_potrf.mem_boundness, final_potrf.arithm_intensity, final_potrf.bmr, final_potrf.ilp, (double)final_potrf.values[0] / (double)l3_cache_size, final_potrf.weight);
      fprintf(file2, "cholesky,trsm,%d,%d,%f,%f,%f,%f,%f,%d\n", A.matrix_size, A.tile_size,
              final_trsm.mem_boundness, final_trsm.arithm_intensity, final_trsm.bmr, final_trsm.ilp, (double)final_trsm.values[0] / (double)l3_cache_size, final_trsm.weight);
      fprintf(file2, "cholesky,syrk,%d,%d,%f,%f,%f,%f,%f,%d\n", A.matrix_size, A.tile_size,
              final_syrk.mem_boundness, final_syrk.arithm_intensity, final_syrk.bmr, final_syrk.ilp, (double)final_syrk.values[0] / (double)l3_cache_size, final_syrk.weight);
      fprintf(file2, "cholesky,gemm,%d,%d,%f,%f,%f,%f,%f,%d\n", A.matrix_size, A.tile_size,
              final_gemm.mem_boundness, final_gemm.arithm_intensity, final_gemm.bmr, final_gemm.ilp, (double)final_gemm.values[0] / (double)l3_cache_size, final_gemm.weight);

      fclose(file2);
    }

    FILE *file;
    if ((file = fopen("counters_cholesky.csv", "a+")) == NULL)
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
        fprintf(file, "algorithm,task,matrix_size,tile_size,mem_boundness,arithm_intensity,bmr,ilp,l3_cache_ratio,weight\n");
      }

      fprintf(file, "cholesky,potrf,%d,%d,%f,%f,%f,%f,%f,%d\n", A.matrix_size, A.tile_size,
              potrf.mem_boundness, potrf.arithm_intensity, potrf.bmr, potrf.ilp, (double)potrf.values[0] / (double)l3_cache_size, potrf.weight);
      fprintf(file, "cholesky,trsm,%d,%d,%f,%f,%f,%f,%f,%d\n", A.matrix_size, A.tile_size,
              trsm.mem_boundness, trsm.arithm_intensity, trsm.bmr, trsm.ilp, (double)trsm.values[0] / (double)l3_cache_size, trsm.weight);
      fprintf(file, "cholesky,syrk,%d,%d,%f,%f,%f,%f,%f,%d\n", A.matrix_size, A.tile_size,
              syrk.mem_boundness, syrk.arithm_intensity, syrk.bmr, syrk.ilp, (double)syrk.values[0] / (double)l3_cache_size, syrk.weight);
      fprintf(file, "cholesky,gemm,%d,%d,%f,%f,%f,%f,%f,%d\n", A.matrix_size, A.tile_size,
              gemm.mem_boundness, gemm.arithm_intensity, gemm.bmr, gemm.ilp, (double)gemm.values[0] / (double)l3_cache_size, gemm.weight);

      fclose(file);
    }
  }

  free(values_by_thread_potrf);
  free(values_by_thread_trsm);
  free(values_by_thread_syrk);
  free(values_by_thread_gemm);
}
