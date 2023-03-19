/*
 * =====================================================================================
 *
 *       Filename:  qr.h
 *
 *    Description:  Task-based QR algorithm
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

#define NEVENTS 6

void qr(tpm_desc A, tpm_desc S)
{
  char *name_with_id_char = NULL;
  struct timeval start, end;

  int eventset = PAPI_NULL;
  long long values[NEVENTS];
  const int available_threads = omp_get_max_threads();
  long long(*values_by_thread_geqrt)[NEVENTS] = malloc(available_threads * sizeof(*values_by_thread_geqrt));
  long long(*values_by_thread_ormqr)[NEVENTS] = malloc(available_threads * sizeof(*values_by_thread_ormqr));
  long long(*values_by_thread_tsqrt)[NEVENTS] = malloc(available_threads * sizeof(*values_by_thread_tsqrt));
  long long(*values_by_thread_tsmqr)[NEVENTS] = malloc(available_threads * sizeof(*values_by_thread_tsmqr));

  if (TPM_PAPI)
  {
    int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};

    memset(values_by_thread_geqrt, 0, sizeof(*values_by_thread_geqrt) * NEVENTS);
    memset(values_by_thread_ormqr, 0, sizeof(*values_by_thread_ormqr) * NEVENTS);
    memset(values_by_thread_tsqrt, 0, sizeof(*values_by_thread_tsqrt) * NEVENTS);
    memset(values_by_thread_tsmqr, 0, sizeof(*values_by_thread_tsmqr) * NEVENTS);
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

#pragma omp task depend(inout                                             \
                        : tileA [0:S.tile_size * S.tile_size]) depend(out \
                                                                      : tileS [0:A.tile_size * S.tile_size])
    {
      double tho[S.tile_size];
      double work[S.tile_size * S.tile_size];

      if (TPM_PAPI)
      {
        memset(values, 0, sizeof(values));
        eventset = PAPI_NULL;
        int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
        int ret = PAPI_create_eventset(&eventset);
        if (ret != PAPI_OK)
        {
          printf("GEQRT task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
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
      tpm_dgeqrt(A.tile_size, S.tile_size, tileA, A.tile_size, tileS,
                 S.tile_size, &tho[0], &work[0]);

      if (TPM_PAPI)
      {
        // Start PAPI counters
        PAPI_stop(eventset, values);

        // Accumulate events values
        for (int i = 0; i < NEVENTS; i++)
        {
#pragma omp atomic update
          values_by_thread_geqrt[omp_get_thread_num()][i] += values[i];
        }
        PAPI_unregister_thread();
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

#pragma omp task depend(in                                                                                       \
                        : tileA [0:S.tile_size * S.tile_size], tileS [0:A.tile_size * S.tile_size]) depend(inout \
                                                                                                           : tileB [0:S.tile_size * S.tile_size])
        {
          double work[S.tile_size * S.tile_size];

          if (TPM_PAPI)
          {
            memset(values, 0, sizeof(values));
            eventset = PAPI_NULL;
            int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
            int ret = PAPI_create_eventset(&eventset);
            if (ret != PAPI_OK)
            {
              printf("ORMQR task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
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
          tpm_dormqr(tpm_left, tpm_transpose, A.tile_size, tileA, A.tile_size,
                     tileS, S.tile_size, tileB, A.tile_size, &work[0],
                     S.tile_size);

          if (TPM_PAPI)
          {
            // Start PAPI counters
            PAPI_stop(eventset, values);

            // Accumulate events values
            for (int i = 0; i < NEVENTS; i++)
            {
#pragma omp atomic update
              values_by_thread_geqrt[omp_get_thread_num()][i] += values[i];
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
        double *tileA = A(k, k);
        double *tileS = S(m, k);
        double *tileB = A(m, k);

        if (TPM_TRACE)
        {
          // TPM library: create a unique task name
          name_with_id_char = tpm_unique_task_identifier("geqrt", k, m, n);
          tpm_upstream_set_task_name(name_with_id_char);
        }

#pragma omp task depend(inout                                                                                  \
                        : tileA [0:S.tile_size * S.tile_size], tileB [0:S.tile_size * S.tile_size]) depend(out \
                                                                                                           : tileS [0:S.tile_size * A.tile_size])
        {
          double work[S.tile_size * S.tile_size];
          double tho[S.tile_size];

          if (TPM_PAPI)
          {
            memset(values, 0, sizeof(values));
            eventset = PAPI_NULL;
            int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
            int ret = PAPI_create_eventset(&eventset);
            if (ret != PAPI_OK)
            {
              printf("TSQRT task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
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
          tpm_dtsqrt(A.tile_size, tileA, A.tile_size, tileB, A.tile_size, tileS,
                     S.tile_size, &tho[0], &work[0]);

          if (TPM_PAPI)
          {
            // Start PAPI counters
            PAPI_stop(eventset, values);

            // Accumulate events values
            for (int i = 0; i < NEVENTS; i++)
            {
#pragma omp atomic update
              values_by_thread_geqrt[omp_get_thread_num()][i] += values[i];
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

#pragma omp task depend(inout                                                                                 \
                        : tileA [0:S.tile_size * S.tile_size], tileB [0:S.tile_size * S.tile_size]) depend(in \
                                                                                                           : tileC [0:S.tile_size * S.tile_size], tileS [0:A.tile_size * S.tile_size])
          {
            double work[S.tile_size * S.tile_size];

            if (TPM_PAPI)
            {
              memset(values, 0, sizeof(values));
              eventset = PAPI_NULL;
              int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS};
              int ret = PAPI_create_eventset(&eventset);
              if (ret != PAPI_OK)
              {
                printf("TSMQR task - PAPI_create_eventset error %d: %s\n", ret, PAPI_strerror(ret));
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
            tpm_dtsmqr(tpm_left, tpm_transpose, A.tile_size, A.tile_size,
                       A.tile_size, A.tile_size, tileA, A.tile_size, tileB,
                       A.tile_size, tileC, A.tile_size, tileS, S.tile_size,
                       &work[0], A.tile_size);

            if (TPM_PAPI)
            {
              // Start PAPI counters
              PAPI_stop(eventset, values);

              // Accumulate events values
              for (int i = 0; i < NEVENTS; i++)
              {
#pragma omp atomic update
                values_by_thread_geqrt[omp_get_thread_num()][i] += values[i];
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
    
#pragma omp taskwait
  PAPI_shutdown();

  long long total_values_geqrt[NEVENTS];
  long long total_values_ormqr[NEVENTS];
  long long total_values_tsqrt[NEVENTS];
  long long total_values_tsmqr[NEVENTS];

  memset(total_values_geqrt, 0, sizeof(total_values_geqrt));
  memset(total_values_ormqr, 0, sizeof(total_values_ormqr));
  memset(total_values_tsqrt, 0, sizeof(total_values_tsqrt));
  memset(total_values_tsmqr, 0, sizeof(total_values_tsmqr));

  for (int i = 0; i < NEVENTS; i++)
  {
    for (int j = 0; j < available_threads; j++)
    {
      total_values_geqrt[i] += values_by_thread_geqrt[j][i];
      total_values_ormqr[i] += values_by_thread_ormqr[j][i];
      total_values_tsqrt[i] += values_by_thread_tsqrt[j][i];
      total_values_tsmqr[i] += values_by_thread_tsmqr[j][i];
    }
  }

  if (TPM_PAPI)
  {
    // PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS
    // Arithmetic intensity
    double arithm_int_geqrt = (double)(total_values_geqrt[0]) / (double)(total_values_geqrt[1]);
    double arithm_int_ormqr = (double)(total_values_ormqr[0]) / (double)(total_values_ormqr[1]);
    double arithm_int_tsqrt = (double)(total_values_tsqrt[0]) / (double)(total_values_tsqrt[1]);
    double arithm_int_tsmqr = (double)(total_values_tsmqr[0]) / (double)(total_values_tsmqr[1]);

    // Memory boundness
    double mem_boundness_geqrt = (double)(total_values_geqrt[2]) / (double)(total_values_geqrt[3]);
    double mem_boundness_ormqr = (double)(total_values_ormqr[2]) / (double)(total_values_ormqr[3]);
    double mem_boundness_tsqrt = (double)(total_values_tsqrt[2]) / (double)(total_values_tsqrt[3]);
    double mem_boundness_tsmqr = (double)(total_values_tsmqr[2]) / (double)(total_values_tsmqr[3]);

    // BMR
    double bmr_geqrt = (double)(total_values_geqrt[4]) / (double)(total_values_geqrt[5]);
    double bmr_ormqr = (double)(total_values_ormqr[4]) / (double)(total_values_ormqr[5]);
    double bmr_tsqrt = (double)(total_values_tsqrt[4]) / (double)(total_values_tsqrt[5]);
    double bmr_tsmqr = (double)(total_values_tsmqr[4]) / (double)(total_values_tsmqr[5]);

    // ILP
    double ilp_geqrt = (double)(total_values_geqrt[1]) / (double)(total_values_geqrt[3]);
    double ilp_ormqr = (double)(total_values_ormqr[1]) / (double)(total_values_ormqr[3]);
    double ilp_tsqrt = (double)(total_values_tsqrt[1]) / (double)(total_values_tsqrt[3]);
    double ilp_tsmqr = (double)(total_values_tsmqr[1]) / (double)(total_values_tsmqr[3]);

    FILE *file;
    if ((file = fopen("counters.dat", "w")) == NULL)
    {
      perror("fopen failed");
      printf("error\n");
      exit(1);
    }
    else
    {
      fprintf(file, "geqrt, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size, mem_boundness_geqrt, arithm_int_geqrt, bmr_geqrt, ilp_geqrt, total_values_geqrt[0]);
      fprintf(file, "ormqr, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size, mem_boundness_ormqr, arithm_int_ormqr, bmr_ormqr, ilp_ormqr, total_values_ormqr[0]);
      fprintf(file, "tsqrt, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size, mem_boundness_tsqrt, arithm_int_tsqrt, bmr_tsqrt, ilp_tsqrt, total_values_tsqrt[0]);
      fprintf(file, "tsmqr, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size, mem_boundness_tsmqr, arithm_int_tsmqr, bmr_tsmqr, ilp_tsmqr, total_values_tsmqr[0]);

      fclose(file);
    }
  }

  free(values_by_thread_geqrt);
  free(values_by_thread_ormqr);
  free(values_by_thread_tsqrt);
  free(values_by_thread_tsmqr);
}