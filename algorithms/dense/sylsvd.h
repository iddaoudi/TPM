/*
 * =====================================================================================
 *
 *       Filename:  sylsvd.h
 *
 *    Description:  Task-based combined Sylvester-SVD algorithms
 *
 *        Version:  1.0
 *        Created:  14/05/2023
 *       Revision:  15/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

// #define LOG 1

void sylsvd(double *As[], double *Bs[], double *Xs[], double *Us[], double *Ss[], double *VTs[],
            int matrix_size, int iter)
{
    // LAPACKE error code
    int info;

    for (int i = 0; i < iter; i++)
    {
#pragma omp task depend(in : Xs[i]) depend(out : Xs[i])
        {
            info = tpm_sylvester(As[i], Bs[i], Xs[i], matrix_size, matrix_size);
        }

#pragma omp task depend(in : Xs[i]) depend(out : Us[i], Ss[i], VTs[i])
        {
#ifdef LOG
            tpm_default_print_matrix("X", Xs[i], matrix_size);
#endif
            info = tpm_svd(Xs[i], Us[i], Ss[i], VTs[i], matrix_size, matrix_size);
        }
    }

#pragma omp taskwait

    if (info != 0)
    {
        printf("Info error code: %d\n", info);
    }
#ifdef LOG
    else
    {
        for (int i = 0; i < iter; i++)
        {

            tpm_default_print_matrix("U", Us[i], matrix_size);
            tpm_default_print_matrix("S", Ss[i], matrix_size);
            tpm_default_print_matrix("VT", VTs[i], matrix_size);
        }
    }
#endif
}