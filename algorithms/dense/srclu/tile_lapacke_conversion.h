/*
 * =====================================================================================
 *
 *       Filename:  tile_lapacke_conversion.h
 *
 *    Description:  Functions that perform the conversion from/to LAPACKE layout
 *                  to/from tile layout
 *
 *        Version:  1.0
 *        Created:  15/05/2023
 *       Revision:  none
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void tpm_matrix_to_tile(double *A, double *pA, int M, int N, int tile_size, int LDA)
{
    int mt = M / tile_size, nt = N / tile_size;
    int n, m;
    double *tptr;
    double *cptr;
    for (n = 0; n < nt; n++)
    {
        for (m = 0; m < mt; m++)
        {

            tptr = A + n * tile_size * LDA + m * tile_size * tile_size;
            cptr = pA + n * tile_size * LDA + m * tile_size;
            tpm_lacpy('t', cptr, tptr, LDA, tile_size);
        }
    }
}

void tpm_tile_to_matrix(double *A, double *pA, int M, int N, int tile_size, int LDA)
{
    int mt = M / tile_size, nt = N / tile_size;
    int n, m;
    double *tptr;
    double *cptr;
    for (n = 0; n < nt; n++)
    {
        for (m = 0; m < mt; m++)
        {

            tptr = A + n * tile_size * LDA + m * tile_size * tile_size;
            cptr = pA + n * tile_size * LDA + m * tile_size;
            tpm_lacpy('c', tptr, cptr, LDA, tile_size);
        }
    }
}