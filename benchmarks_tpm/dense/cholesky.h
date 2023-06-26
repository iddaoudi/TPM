
void cholesky(tpm_desc A)
{

  TPM_application_start();
  double time_start = omp_get_wtime();
  printf("Time start %f\n", time_start);

  int k = 0, m = 0, n = 0;

  for (k = 0; k < A.matrix_size / A.tile_size; k++)
  {
    double *tileA = A(k, k);

#pragma omp task \
depend(inout : tileA[0 : A.tile_size * A.tile_size])
    {
      TPM_application_task_start("potrf");

      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', A.tile_size, tileA, A.tile_size);

      TPM_application_task_finish("potrf");
    }

    for (m = k + 1; m < A.matrix_size / A.tile_size; m++)
    {
      double *tileA = A(k, k);
      double *tileB = A(k, m);

#pragma omp task                                  \
depend(in : tileA[0 : A.tile_size * A.tile_size]) \
    depend(inout : tileB[0 : A.tile_size * A.tile_size])
      {
        TPM_application_task_start("trsm");

        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                    CblasNonUnit, A.tile_size, A.tile_size, 1.0, tileA,
                    A.tile_size, tileB, A.tile_size);

        TPM_application_task_finish("trsm");
      }
    }

    for (m = k + 1; m < A.matrix_size / A.tile_size; m++)
    {
      double *tileA = A(k, m);
      double *tileB = A(m, m);

#pragma omp task                                  \
depend(in : tileA[0 : A.tile_size * A.tile_size]) \
    depend(inout : tileB[0 : A.tile_size * A.tile_size])
      {
        TPM_application_task_start("syrk");
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, A.tile_size,
                    A.tile_size, -1.0, tileA, A.tile_size, 1.0, tileB,
                    A.tile_size);
        TPM_application_task_finish("syrk");
      }

      for (n = k + 1; n < m; n++)
      {
        double *tileA = A(k, n);
        double *tileB = A(k, m);
        double *tileC = A(n, m);

#pragma omp task                                  \
depend(in : tileA[0 : A.tile_size * A.tile_size], \
           tileB[0 : A.tile_size * A.tile_size])  \
    depend(inout : tileC[0 : A.tile_size * A.tile_size])
        {
          TPM_application_task_start("gemm");

          cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A.tile_size,
                      A.tile_size, A.tile_size, -1.0, tileA, A.tile_size, tileB,
                      A.tile_size, 1.0, tileC, A.tile_size);

          TPM_application_task_finish("gemm");
        }
      }
    }
  }

#pragma omp taskwait

  double time_finish = omp_get_wtime();
  printf("Time finish %f\n", time_finish);
  printf("Time total %f\n", time_finish - time_start);
  TPM_application_finalize(time_finish - time_start);
}
