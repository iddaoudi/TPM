/*
 * =====================================================================================
 *
 *       Filename:  svd.h
 *
 *    Description:  Task-based SVD algorithm
 *
 *        Version:  1.0
 *        Created:  17/03/2023
 *       Revision:  20/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void print_matrix(double *A, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%lf\t", A[i * cols + j]);
        }
        printf("\n");
    }
}

int svd()
{
    // Matrix dimensions
    int m = 8; // Number of rows
    int n = 6; // Number of columns
    int tile_size = 4;

    // Allocate memory for the input matrix A
    double *A = (double *)malloc(m * n * sizeof(double));

    // Initialize the matrix A
    for (int i = 0; i < m * n; i++)
    {
        A[i] = (double)(i + 1);
    }

    // Allocate memory for the matrices U, S, and VT
    double *U = (double *)malloc(m * m * sizeof(double));
    double *S = (double *)malloc(n * sizeof(double));
    double *VT = (double *)malloc(n * n * sizeof(double));

    // Calculate the number of tiles
    int num_tiles_m = m / tile_size;
    int num_tiles_n = n / tile_size;

    // Initialize the bidiagonal matrix B
    double *B = (double *)malloc(m * n * sizeof(double));
    double *tauU = (double *)malloc(m * sizeof(double));
    double *tauVT = (double *)malloc(n * sizeof(double));

    // Reduction to bidiagonal form using LAPACKE_dgebrd
    for (int i = 0; i < num_tiles_m; i++)
    {
        for (int j = 0; j < num_tiles_n; j++)
        {
#pragma omp task depend(inout \
                        : A, tauU, tauVT)
            {
                int offset_i = i * tile_size;
                int offset_j = j * tile_size;
                double *A_tile = A + offset_i * n + offset_j;
                LAPACKE_dgebrd(LAPACK_ROW_MAJOR, tile_size, tile_size, A_tile, n, S + offset_j, B + offset_j, tauU + offset_i, tauVT + offset_j);
            }
        }
    }

    // Initialize the temporary U and VT matrices
    double *Utmp = (double *)malloc(m * n * sizeof(double));
    double *VTtmp = (double *)malloc(n * n * sizeof(double));
    double *work = (double *)malloc((4 * n - 4) * sizeof(double));

    // Compute the SVD of the bidiagonal matrix B using LAPACKE_dbdsqr
    for (int i = 0; i < num_tiles_n; i++)
    {
#pragma omp task depend(in                                 \
                        : A, S, B, tauU, tauVT) depend(out \
                                                       : Utmp, VTtmp, work)
        {
            int offset = i * tile_size;
            int info = LAPACKE_dbdsqr(LAPACK_ROW_MAJOR, 'U', tile_size, tile_size, m, 0, S + offset, B + offset, VTtmp + offset * n, n, Utmp + offset * m, m, work, 0);
            if (info > 0)
            {
                printf("SVD computation did not converge.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    // Multiply the orthogonal matrices to obtain the final U and VT matrices
    for (int i = 0; i < num_tiles_m; i++)
    {
        for (int j = 0; j < num_tiles_n; j++)
        {
#pragma omp task depend(in                                        \
                        : A, tauU, tauVT, Utmp, VTtmp) depend(out \
                                                              : U, VT)
            {
                int offset_i = i * tile_size;
                int offset_j = j * tile_size;
                double *A_tile = A + offset_i * n + offset_j;
                double *U_tile = U + offset_i * m + offset_j;
                double *VT_tile = VT + offset_j * n + offset_j;
                LAPACKE_dormbr(LAPACK_ROW_MAJOR, 'Q', 'L', 'N', tile_size, tile_size, tile_size, A_tile, n, tauU + offset_i, U_tile, m);
                LAPACKE_dormbr(LAPACK_ROW_MAJOR, 'P', 'R', 'T', tile_size, tile_size, tile_size, A_tile, n, tauVT + offset_j, VT_tile, n);
            }
        }
    }

// Wait for all tasks to finish
#pragma omp taskwait

    // Free allocated memory
    free(A);
    free(U);
    free(S);
    free(VT);
    free(B);
    free(tauU);
    free(tauVT);
    free(Utmp);
    free(VTtmp);
    free(work);

    return 0;
}
