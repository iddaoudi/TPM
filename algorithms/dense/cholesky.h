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

#define NEVENTS 6

void cholesky(tpm_desc A)
{
  // TPM library: initialization
  if (TPM_TRACE)
    tpm_downstream_start("cholesky", A.matrix_size, A.tile_size, NTH);

  int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC};
  int available_threads = omp_get_max_threads();

  long long values_by_thread_potrf[available_threads][NEVENTS];
  long long values_by_thread_trsm[available_threads][NEVENTS];
  long long values_by_thread_syrk[available_threads][NEVENTS];
  long long values_by_thread_gemm[available_threads][NEVENTS];
  memset(values_by_thread_potrf, 0, sizeof(values_by_thread_potrf));
  memset(values_by_thread_trsm, 0, sizeof(values_by_thread_trsm));
  memset(values_by_thread_syrk, 0, sizeof(values_by_thread_syrk));
  memset(values_by_thread_gemm, 0, sizeof(values_by_thread_gemm));

  int k = 0, m = 0, n = 0;

  for (k = 0; k < A.matrix_size / A.tile_size; k++)
  {
    double *tileA = A(k, k);

    // TPM library: create a unique task name
    char *name_with_id_char = tpm_unique_task_identifier("potrf", k, m, n);
    if (TPM_TRACE)
      tpm_upstream_set_task_name(name_with_id_char);

#pragma omp task firstprivate(name_with_id_char) \
    depend(inout                                 \
           : tileA [0:A.tile_size * A.tile_size])
    {
      // TPM library: send CPU and name
      struct timeval start, end;
      unsigned int cpu, node;
      getcpu(&cpu, &node);
      if (TPM_TRACE)
        tpm_upstream_set_task_cpu_node(cpu, node, name_with_id_char);

      // Kernel
      gettimeofday(&start, NULL);
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', A.tile_size, tileA, A.tile_size);
      gettimeofday(&end, NULL);

      // TPM library: send time and name
      if (TPM_TRACE)
        tpm_upstream_get_task_time(start, end, name_with_id_char);
    }

    for (m = k + 1; m < A.matrix_size / A.tile_size; m++)
    {
      double *tileA = A(k, k);
      double *tileB = A(k, m);

      // TPM library: create a unique task name
      char *name_with_id_char = tpm_unique_task_identifier("trsm", k, m, n);
      if (TPM_TRACE)
        tpm_upstream_set_task_name(name_with_id_char);

#pragma omp task firstprivate(name_with_id_char)  \
    depend(in                                     \
           : tileA [0:A.tile_size * A.tile_size]) \
        depend(inout                              \
               : tileB [0:A.tile_size * A.tile_size])
      {
        // TPM library: send CPU and name
        struct timeval start, end;
        unsigned int cpu, node;
        getcpu(&cpu, &node);
        if (TPM_TRACE)
          tpm_upstream_set_task_cpu_node(cpu, node, name_with_id_char);

        // Kernel
        gettimeofday(&start, NULL);
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                    CblasNonUnit, A.tile_size, A.tile_size, 1.0, tileA,
                    A.tile_size, tileB, A.tile_size);
        gettimeofday(&end, NULL);

        // TPM library: send time and name
        if (TPM_TRACE)
          tpm_upstream_get_task_time(start, end, name_with_id_char);
      }
    }

    for (m = k + 1; m < A.matrix_size / A.tile_size; m++)
    {
      double *tileA = A(k, m);
      double *tileB = A(m, m);

      // TPM library: create a unique task name
      char *name_with_id_char = tpm_unique_task_identifier("syrk", k, m, n);
      if (TPM_TRACE)
        tpm_upstream_set_task_name(name_with_id_char);

#pragma omp task firstprivate(name_with_id_char)  \
    depend(in                                     \
           : tileA [0:A.tile_size * A.tile_size]) \
        depend(inout                              \
               : tileB [0:A.tile_size * A.tile_size])
      {
        // TPM library: send CPU and name
        struct timeval start, end;
        unsigned int cpu, node;
        getcpu(&cpu, &node);
        if (TPM_TRACE)
          tpm_upstream_set_task_cpu_node(cpu, node, name_with_id_char);

        // Kernel
        gettimeofday(&start, NULL);
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, A.tile_size,
                    A.tile_size, -1.0, tileA, A.tile_size, 1.0, tileB,
                    A.tile_size);
        gettimeofday(&end, NULL);

        // TPM library: send time and name
        if (TPM_TRACE)
          tpm_upstream_get_task_time(start, end, name_with_id_char);
      }

      for (n = k + 1; n < m; n++)
      {
        double *tileA = A(k, n);
        double *tileB = A(k, m);
        double *tileC = A(n, m);

        // TPM library: create a unique task name
        char *name_with_id_char = tpm_unique_task_identifier("gemm", k, m, n);
        if (TPM_TRACE)
          tpm_upstream_set_task_name(name_with_id_char);

#pragma omp task firstprivate(name_with_id_char)  \
    depend(in                                     \
           : tileA [0:A.tile_size * A.tile_size], \
             tileB [0:A.tile_size * A.tile_size]) \
        depend(inout                              \
               : tileC [0:A.tile_size * A.tile_size])
        {
          // TPM library: send CPU and name
          struct timeval start, end;
          unsigned int cpu, node;
          getcpu(&cpu, &node);
          if (TPM_TRACE)
            tpm_upstream_set_task_cpu_node(cpu, node, name_with_id_char);

          // Kernel
          gettimeofday(&start, NULL);
          cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A.tile_size,
                      A.tile_size, A.tile_size, -1.0, tileA, A.tile_size, tileB,
                      A.tile_size, 1.0, tileC, A.tile_size);
          gettimeofday(&end, NULL);

          // TPM library: send time and name
          if (TPM_TRACE)
            tpm_upstream_get_task_time(start, end, name_with_id_char);
        }
      }
    }
  }
}
