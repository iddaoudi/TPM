/*
 * =====================================================================================
 *
 *       Filename:  svd.h
 *
 *    Description:  Single Value Decomposition (SVD) task
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

int tpm_svd(double *X, double *U, double *S, double *VT, int m, int n)
{
    double *superb = (double *)malloc((m < n ? m : n - 1) * sizeof(double));
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, X, m, S, U, m, VT, n, superb);
    free(superb);
    return info;
}