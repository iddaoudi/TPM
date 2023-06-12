/*
 * =====================================================================================
 *
 *       Filename:  empty_block.h
 *
 *    Description:  Allocate empty block for bmod task in sparse LU
 * factorization
 *
 *        Version:  1.0
 *        Created:  29/12/2022
 *       Revision:  20/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

double *tpm_allocate_empty_block(int tile_size)
{
  int i, j;
  double *p, *q;

  p = (double *)malloc(tile_size * tile_size * sizeof(double));
  q = p;
  if (p != NULL)
  {
    for (i = 0; i < tile_size; i++)
    {
      for (j = 0; j < tile_size; j++)
      {
        (*p) = 0.0;
        p++;
      }
    }
  }
  else
  {
    printf("Not enough memory\n");
    exit(EXIT_FAILURE);
  }
  return (q);
}
