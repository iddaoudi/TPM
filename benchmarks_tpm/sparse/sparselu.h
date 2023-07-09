/*
 * =====================================================================================
 *
 *       Filename:  sparselu.h
 *
 *    Description:  Task-based sparse LU algorithm
 *
 *        Version:  1.0
 *        Created:  28/12/2022
 *       Revision:  21/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void sparselu(double **M, int matrix_size, int tile_size)
{

  int i, j, k;

#pragma omp parallel private(i, j, k) shared(M)
#pragma omp master
  {
    for (k = 0; k < matrix_size; k++)
    {
#pragma omp task firstprivate(k) shared(M) \
    depend(inout : M[k * matrix_size + k : tile_size * tile_size])
      {
        TPM_application_task_start("lu0");

        lu0(M[k * matrix_size + k], tile_size);

        TPM_application_task_finish("lu0");
      }

      for (j = k + 1; j < matrix_size; j++)
        if (M[k * matrix_size + j] != NULL)
        {
#pragma omp task firstprivate(j, k) shared(M)                   \
    depend(in : M[k * matrix_size + k : tile_size * tile_size]) \
    depend(inout : M[k * matrix_size + j : tile_size * tile_size])
          {
            TPM_application_task_start("fwd");

            fwd(M[k * matrix_size + k], M[k * matrix_size + j], tile_size);

            TPM_application_task_finish("fwd");
          }
        }

      for (i = k + 1; i < matrix_size; i++)
        if (M[i * matrix_size + k] != NULL)
        {
#pragma omp task firstprivate(i, k) shared(M)                   \
    depend(in : M[k * matrix_size + k : tile_size * tile_size]) \
    depend(inout : M[i * matrix_size + k : tile_size * tile_size])
          {
            TPM_application_task_start("bdiv");

            bdiv(M[k * matrix_size + k], M[i * matrix_size + k], tile_size);

            TPM_application_task_finish("bdiv");
          }
        }

      for (i = k + 1; i < matrix_size; i++)
        if (M[i * matrix_size + k] != NULL)
          for (j = k + 1; j < matrix_size; j++)
            if (M[k * matrix_size + j] != NULL)
            {
              if (M[i * matrix_size + j] == NULL)
                M[i * matrix_size + j] = tpm_allocate_empty_block(tile_size);

#pragma omp task firstprivate(k, j, i) shared(M)                \
    depend(in : M[i * matrix_size + k : tile_size * tile_size], \
               M[k * matrix_size + j : tile_size * tile_size])  \
    depend(inout : M[i * matrix_size + j : tile_size * tile_size])
              {
                TPM_application_task_start("bmod");

                bmod(M[i * matrix_size + k], M[k * matrix_size + j],
                     M[i * matrix_size + j], tile_size);

                TPM_application_task_finish("bmod");
              }
            }
    }
#pragma omp taskwait
  }
}
