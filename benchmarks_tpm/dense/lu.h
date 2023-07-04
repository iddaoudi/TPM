void lu(int matrix_size, int tile_size, double *pA, int *ipiv, double *A)
{
    double alpha = 1., neg = -1.;
    for (int k = 0; k < matrix_size / tile_size; k++)
    {
        int m = matrix_size - k * tile_size;
        double *akk = A + k * tile_size * matrix_size + k * tile_size * tile_size;

#pragma omp task firstprivate(akk, m) depend(inout : akk[0 : m * tile_size]) \
    depend(out : ipiv[k * tile_size : tile_size])
        {
            TPM_application_task_start("getrfpiv");

            tpm_tile_to_matrix(A + k * tile_size * matrix_size + k * tile_size * tile_size,
                               pA + k * tile_size * matrix_size + k * tile_size, m, tile_size,
                               tile_size, matrix_size);
            LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, tile_size, pA + k * tile_size * matrix_size + k * tile_size,
                           matrix_size, ipiv + k * tile_size);
            // Update the ipiv
            for (int i = k * tile_size; i < k * tile_size + tile_size; i++)
            {
                ipiv[i] += k * tile_size;
            }
            tpm_matrix_to_tile(A + k * tile_size * matrix_size + k * tile_size * tile_size,
                               pA + k * tile_size * matrix_size + k * tile_size, m, tile_size,
                               tile_size, matrix_size);

            TPM_application_task_finish("getrfpiv");
        }

        // Update trailing submatrix
        for (int j = k + 1; j < matrix_size / tile_size; j++)
        {
            double *akj = A + j * tile_size * matrix_size + k * tile_size * tile_size;

#pragma omp task firstprivate(akk, akj, m) depend(in : akk[0 : m * tile_size]) \
    depend(in : ipiv[k * tile_size : tile_size]) depend(inout : akj[0 : tile_size * tile_size])
            {
                TPM_application_task_start("trsmswp");

                int k1 = k * tile_size;
                int k2 = k * tile_size + tile_size;
                tpm_geswp(A + j * tile_size * matrix_size, tile_size, k1, k2, ipiv);

                cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, tile_size,
                            tile_size, alpha, akk, tile_size, akj, tile_size);

                TPM_application_task_finish("trsmswp");
            }

#pragma omp task firstprivate(akk, akj, m) depend(in : akk[0 : m * tile_size]) \
    depend(inout : akj[0 : tile_size * tile_size], ipiv[k * tile_size : tile_size])
            {
                for (int i = k + 1; i < matrix_size / tile_size; i++)
                {
                    TPM_application_task_start("gemm");

                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, tile_size, tile_size,
                                tile_size, neg,
                                A + k * tile_size * matrix_size + i * tile_size * tile_size,
                                tile_size, akj, tile_size, alpha,
                                A + j * tile_size * matrix_size + i * tile_size * tile_size, tile_size);

                    TPM_application_task_finish("gemm");
                }
            }
        }
    }
    // Pivoting to the left
    for (int t = 1; t < matrix_size / tile_size; t++)
    {

#pragma omp task                                                           \
depend(in : ipiv[((matrix_size / tile_size) - 1) * tile_size : tile_size]) \
    depend(inout : A[t * tile_size * matrix_size : matrix_size * tile_size])
        {
            TPM_application_task_start("geswp");

            tpm_geswp(A + (t - 1) * tile_size * matrix_size, tile_size, t * tile_size, matrix_size, ipiv);

            TPM_application_task_finish("geswp");
        }
    }
}