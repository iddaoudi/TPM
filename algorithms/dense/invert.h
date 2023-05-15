/*
 * =====================================================================================
 *
 *       Filename:  invert.h
 *
 *    Description:  Task-based matrix invert algorithm using an LU decomposition with
 *                  partial pivoting
 *
 *        Version:  1.0
 *        Created:  14/05/2023
 *       Revision:  none
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void invert(double *A, int *ipiv, int matrix_size, int tile_size)
{
    int lda = matrix_size;

    for (int i = 0; i < matrix_size; i += tile_size)
    {
#pragma omp task depend(inout : A[i * tile_size + i])
        {
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, tile_size, tile_size, &A[i * matrix_size + i], lda, &ipiv[i]);
        }

        for (int j = i + tile_size; j < matrix_size; j += tile_size)
        {
#pragma omp task depend(in : A[i * tile_size + i]) depend(inout : A[i * tile_size + j])
            {
                cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, tile_size,
                            tile_size, 1.0, &A[i * matrix_size + i], lda, &A[i * matrix_size + j], lda);
            }
#pragma omp task depend(in : A[i * tile_size + i]) depend(inout : A[j * tile_size + i])
            {
                cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, tile_size,
                            tile_size, 1.0, &A[i * matrix_size + i], lda, &A[j * matrix_size + i], lda);
            }
        }

        for (int j = i + tile_size; j < matrix_size; j += tile_size)
        {
            for (int k = i + tile_size; k < matrix_size; k += tile_size)
            {
#pragma omp task depend(in : A[i * tile_size + j], A[k * tile_size + i]) depend(inout : A[k * tile_size + j])
                {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tile_size, tile_size, tile_size,
                                -1.0, &A[k * matrix_size + i], lda, &A[i * matrix_size + j], lda,
                                1.0, &A[k * matrix_size + j], lda);
                }
            }
        }
    }
// This pragma is optional
#pragma omp taskwait

    for (int i = 0; i < matrix_size; i += tile_size)
    {
#pragma omp task depend(inout : A[i * tile_size + i])
        {
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, tile_size, &A[i * matrix_size + i], lda, &ipiv[i]);
        }
    }
}
