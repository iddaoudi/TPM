
// #define LOG 1

void sylsvd(double *As[], double *Bs[], double *Xs[], double *Us[], double *Ss[], double *VTs[],
            double *EVs[], double *Ms[], int matrix_size, int iter)
{

    TPM_application_start();
    double time_start = omp_get_wtime();

    int info;

    for (int i = 0; i < iter; i++)
    {

#pragma omp task depend(in : Xs[i]) depend(out : Xs[i])
        {
            TPM_application_task_start("trsyl");

            double scale;
            info = LAPACKE_dtrsyl(LAPACK_ROW_MAJOR, 'N', 'N', 1, matrix_size, matrix_size, As[i], matrix_size, Bs[i], matrix_size, Xs[i], matrix_size, &scale);

            TPM_application_task_finish("trsyl");
        }

#pragma omp task depend(in : Xs[i]) depend(out : Us[i], Ss[i], VTs[i])
        {
#ifdef LOG
            tpm_default_print_matrix("X", Xs[i], matrix_size);
#endif
            TPM_application_task_start("gesvd");

            double *superb = (double *)malloc((matrix_size - 1) * sizeof(double));
            info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', matrix_size, matrix_size, Xs[i], matrix_size, Ss[i], Us[i], matrix_size, VTs[i], matrix_size, superb);
            free(superb);

            TPM_application_task_finish("gesvd");
        }

#pragma omp task depend(in : Xs[i]) depend(out : EVs[i])
        {
            TPM_application_task_start("geev");

            double *EVI = (double *)malloc(matrix_size * sizeof(double)); // Imaginary part of the eigenvalues
            int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', matrix_size, Xs[i], matrix_size, EVs[i], EVI, NULL, matrix_size, Xs[i], matrix_size);
            free(EVI);

            TPM_application_task_finish("geev");
        }

#pragma omp task depend(in : Us[i], VTs[i]) depend(out : Ms[i])
        {
            TPM_application_task_start("gemm");

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        matrix_size, matrix_size, matrix_size,
                        1.0, Us[i], matrix_size, VTs[i], matrix_size,
                        0.0, Ms[i], matrix_size);

            TPM_application_task_finish("gemm");
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
            tpm_default_print_matrix("EV", EVs[i], matrix_size);
            tpm_default_print_matrix("M", Ms[i], matrix_size);
        }
    }
#endif

    double time_finish = omp_get_wtime();
    TPM_application_finalize(time_finish - time_start);
}