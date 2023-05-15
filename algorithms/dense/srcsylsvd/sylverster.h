/*
 * =====================================================================================
 *
 *       Filename:  sylvester.h
 *
 *    Description:  Sylverster problem task
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

int tpm_sylvester(double *A, double *B, double *C, int m, int n)
{
    double scale;
    int info = LAPACKE_dtrsyl(LAPACK_ROW_MAJOR, 'N', 'N', 1, m, n, A, m, B, n, C, m, &scale);
    return info;
}