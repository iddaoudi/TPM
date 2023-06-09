/*
 * =====================================================================================
 *
 *       Filename:  populate.h
 *
 *    Description:  Dense hermitian positive matrix generator
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  20/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <time.h>

// #define SPECIAL4x4 1
// #define IDENTITY 1

void tpm_hermitian_positive_generator(tpm_desc A)
{
#ifdef SPECIAL4x4
  double *dA = A(0, 0);
  dA[0] = 4.5;
  dA[1] = -0.095026;
  dA[2] = -0.095026;
  dA[3] = 3.719688;
  dA = A(1, 0);
  dA[0] = 0.361857;
  dA[1] = 0.388551;
  dA[2] = -0.447549;
  dA[3] = 0.155058;
  dA = A(0, 1);
  dA[0] = 0.361857;
  dA[1] = -0.447549;
  dA[2] = 0.388551;
  dA[3] = 0.155058;
  dA = A(1, 1);
  dA[0] = 4.484953;
  dA[1] = 0.342457;
  dA[2] = 0.342457;
  dA[3] = 3.982519;
#else
  srand((unsigned int)time(NULL));
  for (int i = 0; i < A.matrix_size / A.tile_size; i++)
  {
    for (int j = 0; j < A.matrix_size / A.tile_size; j++)
    {
      double *dA = A(i, j);
      for (int k = 0; k < A.tile_size; k++)
      {
        for (int l = 0; l < A.tile_size; l++)
        {
          // Random diagonal elements on the diagonal tiles of matrix
          if (i == j && k == l)
          {
#ifdef IDENTITY
            dA[k * A.tile_size + l] = 1.0;
#else
            double seed = 173.0;
            dA[k * A.tile_size + l] =
                ((double)rand() / (double)(RAND_MAX)) * seed + seed;
#endif
          }
          // Fixed small value for all the rest
          else
          {
#ifdef IDENTITY
            dA[k * A.tile_size + l] = 0.0;
#else
            dA[k * A.tile_size + l] = 0.5;
#endif
          }
        }
      }
    }
  }
#endif
}

static void tpm_sparse_generator(double *M[], int matrix_size, int tile_size)
{
  int init_val = 1325;
  int zero_element = 0;
  int i, j, m, n;

  // Generating the structure
  for (m = 0; m < matrix_size; m++)
  {
    for (n = 0; n < matrix_size; n++)
    {
      double *p;
      // Computing null entries
      zero_element = 0;
      if ((m < n) && (m % 3 != 0))
        zero_element = 1;
      if ((m > n) && (n % 3 != 0))
        zero_element = 1;
      if (m % 2 == 1)
        zero_element = 1;
      if (n % 2 == 1)
        zero_element = 1;
      if (m == n)
        zero_element = 0;
      if (m == n - 1)
        zero_element = 0;
      if (m - 1 == n)
        zero_element = 0;
      // Allocating matrix
      if (zero_element == 0)
      {
        M[m * matrix_size + n] =
            (double *)malloc(tile_size * tile_size * sizeof(double));
        if (M[m * matrix_size + n] == NULL)
        {
          printf("Not enough memory\n");
          exit(EXIT_FAILURE);
        }
        // Initializing matrix
        p = M[m * matrix_size + n];
        for (i = 0; i < tile_size; i++)
        {
          for (j = 0; j < tile_size; j++)
          {
            init_val = (3125 * init_val) % 65536;
            (*p) = (double)((init_val - 32768.0) / 16384.0);
            p++;
          }
        }
      }
      else
      {
        M[m * matrix_size + n] = NULL;
      }
    }
  }
}

static void tpm_sparse_allocate(double ***M, int matrix_size, int tile_size)
{
  *M = (double **)malloc(matrix_size * matrix_size * sizeof(double *));
  tpm_sparse_generator(*M, matrix_size, tile_size);
}

void tpm_dense_generator(double *mat, int elements)
{
  for (int i = 0; i < elements; i++)
  {
    for (int j = 0; j < elements; j++)
    {
      mat[i * elements + j] = ((double)rand() / (RAND_MAX)) * 10.;
    }
  }
}