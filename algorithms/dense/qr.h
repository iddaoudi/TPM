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

#include "counters.h"

void qr(tpm_desc A, tpm_desc S)
{
  char *name_with_id_char = NULL;
  struct timeval start, end;

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
              values_by_thread_ormqr[omp_get_thread_num()][i] += values[i];
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
              values_by_thread_tsqrt[omp_get_thread_num()][i] += values[i];
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
                values_by_thread_tsmqr[omp_get_thread_num()][i] += values[i];
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

    CounterData geqrt, ormqr, tsqrt, tsmqr;

    accumulate_counters(geqrt.values, values_by_thread_geqrt, available_threads);
    accumulate_counters(ormqr.values, values_by_thread_ormqr, available_threads);
    accumulate_counters(tsqrt.values, values_by_thread_tsqrt, available_threads);
    accumulate_counters(tsmqr.values, values_by_thread_tsmqr, available_threads);

    compute_derived_metrics(&geqrt);
    compute_derived_metrics(&ormqr);
    compute_derived_metrics(&tsqrt);
    compute_derived_metrics(&tsmqr);

    FILE *file;
    if ((file = fopen("counters.dat", "w")) == NULL)
    {
      perror("fopen failed");
      exit(1);
    }
    else
    {
      fprintf(file, "geqrt, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              geqrt.mem_boundness, geqrt.arithm_intensity, geqrt.bmr, geqrt.ilp, geqrt.values[0]);
      fprintf(file, "ormqr, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              ormqr.mem_boundness, ormqr.arithm_intensity, ormqr.bmr, ormqr.ilp, ormqr.values[0]);
      fprintf(file, "tsqrt, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              tsqrt.mem_boundness, tsqrt.arithm_intensity, tsqrt.bmr, tsqrt.ilp, tsqrt.values[0]);
      fprintf(file, "tsmqr, %d, %d, %f, %f, %f, %f, %lld\n", A.matrix_size, A.tile_size,
              tsmqr.mem_boundness, tsmqr.arithm_intensity, tsmqr.bmr, tsmqr.ilp, tsmqr.values[0]);

      fclose(file);
    }
  }

  free(values_by_thread_geqrt);
  free(values_by_thread_ormqr);
  free(values_by_thread_tsqrt);
  free(values_by_thread_tsmqr);
}
