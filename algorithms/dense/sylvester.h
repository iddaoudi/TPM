/*
 * =====================================================================================
 *
 *       Filename:  sylverster.h
 *
 *    Description:  Task-based Sylverster algorithm
 *
 *        Version:  1.0
 *        Created:  17/03/2023
 *       Revision:  19/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void random_matrix(double *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = (double)rand() / RAND_MAX;
        }
    }
}

int sylverster()
{
    int m = 2000, n = 2000;
    int tile_size = 500;
    int num_tiles_m = (m + tile_size - 1) / tile_size;
    int num_tiles_n = (n + tile_size - 1) / tile_size;

    double *A = (double *)malloc(m * m * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(m * n * sizeof(double));
    double *U = (double *)malloc(m * m * sizeof(double));
    double *V = (double *)malloc(n * n * sizeof(double));
    double *s = (double *)malloc(m * sizeof(double));
    double *t = (double *)malloc(n * sizeof(double));

    random_matrix(A, m, m);
    random_matrix(B, n, n);
    random_matrix(C, m, n);

    // Schur decomposition
    int sdim = 0;
    double *wr = (double *)malloc(m * sizeof(double));
    double *wi = (double *)malloc(m * sizeof(double));
    double *wr2 = (double *)malloc(n * sizeof(double));
    double *wi2 = (double *)malloc(n * sizeof(double));

#pragma omp task depend(out               \
                        : U, s) depend(in \
                                       : A)
    LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'N', NULL, m, A, m, &sdim, wr, wi, U, m);

#pragma omp task depend(out               \
                        : V, t) depend(in \
                                       : B)
    LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'N', NULL, n, B, n, &sdim, wr2, wi2, V, n);

#pragma omp taskwait

    free(wr);
    free(wi);
    free(wr2);
    free(wi2);

// Solve the Sylvester equation
#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_tiles_m; i++)
    {
        for (int j = 0; j < num_tiles_n; j++)
        {
            int M_tile = fmin(tile_size, m - i * tile_size);
            int N_tile = fmin(tile_size, n - j * tile_size);
            double *D = (double *)malloc(M_tile * N_tile * sizeof(double));

            int offset_i = i * tile_size;
            int offset_j = j * tile_size;

#pragma omp task depend(in                    \
                        : U, V, C) depend(out \
                                          : D)
            {
                double *D = (double *)malloc(M_tile * M_tile * sizeof(double));
                double *E = (double *)malloc(N_tile * N_tile * sizeof(double));
                double *F = (double *)malloc(M_tile * N_tile * sizeof(double));
                double scale;
                double dif;
                int info = LAPACKE_dtgsyl(LAPACK_ROW_MAJOR, 'N', 1, M_tile, N_tile, A + offset_i * m, m, B + offset_j * n, n, C + offset_i * n + offset_j, n, D, M_tile, E, N_tile, F, M_tile, &scale, &dif);

                if (info)
                {
                    printf("Error in LAPACKE_dtgsyl: %d\n", info);
                }
            }

#pragma omp task depend(in              \
                        : D) depend(out \
                                    : C)
            {
                cblas_daxpy(M_tile * N_tile, 1.0, D, 1, C + offset_i * n + offset_j, 1);
                free(D);
            }
        }
    }

#pragma omp taskwait

    free(A);
    free(B);
    free(C);
    free(U);
    free(V);
    free(s);
    free(t);

    return 0;
}
