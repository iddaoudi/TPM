void invert(double *A, int *ipiv, int matrix_size, int tile_size)
{
    int lda = matrix_size;

    for (int i = 0; i < matrix_size; i += tile_size)
    {

#pragma omp task depend(inout : A[i * tile_size + i])
        {
            TPM_application_task_start("getrf");

            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, tile_size, tile_size, &A[i * matrix_size + i], lda, &ipiv[i]);

            TPM_application_task_finish("getrf");
        }
        for (int j = i + tile_size; j < matrix_size; j += tile_size)
        {

#pragma omp task depend(in : A[i * tile_size + i]) \
    depend(inout : A[i * tile_size + j])
            {
                TPM_application_task_start("trsm");

                cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, tile_size,
                            tile_size, 1.0, &A[i * matrix_size + i], lda, &A[i * matrix_size + j], lda);

                TPM_application_task_finish("trsm");
            }

#pragma omp task depend(in : A[i * tile_size + i]) \
    depend(inout : A[j * tile_size + i])
            {
                TPM_application_task_start("trsm");

                cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, tile_size,
                            tile_size, 1.0, &A[i * matrix_size + i], lda, &A[j * matrix_size + i], lda);

                TPM_application_task_finish("trsm");
            }
        }

        for (int j = i + tile_size; j < matrix_size; j += tile_size)
        {
            for (int k = i + tile_size; k < matrix_size; k += tile_size)
            {

#pragma omp task                                        \
depend(in : A[i * tile_size + j], A[k * tile_size + i]) \
    depend(inout : A[k * tile_size + j])
                {
                    TPM_application_task_start("gemm");

                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tile_size, tile_size, tile_size,
                                -1.0, &A[k * matrix_size + i], lda, &A[i * matrix_size + j], lda,
                                1.0, &A[k * matrix_size + j], lda);

                    TPM_application_task_finish("gemm");
                }
            }
        }
    }

#pragma omp taskwait

    for (int i = 0; i < matrix_size; i += tile_size)
    {

#pragma omp task depend(inout : A[i * tile_size + i])
        {
            TPM_application_task_start("getri");

            LAPACKE_dgetri(LAPACK_ROW_MAJOR, tile_size, &A[i * matrix_size + i], lda, &ipiv[i]);

            TPM_application_task_finish("getri");
        }
    }
}
