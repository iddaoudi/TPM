/*
 * =====================================================================================
 *
 *       Filename:  geswp.h
 *
 *    Description:  Swap according to the permutations array
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

void tpm_geswp(double *A, int tile_size, int k1, int k2, int *ipiv)
{
    int m, m1, m2;
    for (m = k1; m < k2; m++)
    {
        if (ipiv[m] - 1 != m)
        {
            m1 = m;
            m2 = ipiv[m] - 1;

            cblas_dswap(tile_size, A + (m1 / tile_size) * (tile_size * tile_size) + m1 % tile_size, tile_size,
                        A + (m2 / tile_size) * (tile_size * tile_size) + m2 % tile_size, tile_size);
        }
    }
}