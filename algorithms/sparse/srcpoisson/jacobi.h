/*
 * =====================================================================================
 *
 *       Filename:  jacobi.h
 *
 *    Description:	Jacobi iteration
 *
 *        Version:  1.0
 *        Created:  18/01/2023
 *       Revision:  13/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include "functions.h"
#include "copy.h"
#include "estimate.h"

void jacobi(int matrix_size, double dx, double *falloc,
			int niter, double *ualloc, double *unewalloc, int tile_size)
{
	double(*f)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])falloc;
	double(*u)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])ualloc;
	double(*unew)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])unewalloc;

#pragma omp parallel \
shared(ualloc, unewalloc, f, matrix_size, dx, niter, tile_size)
#pragma omp master
	{
		for (int it = 1; it <= niter; it++)
		{
			for (int j = 0; j < matrix_size; j += tile_size)
			{
				for (int i = 0; i < matrix_size; i += tile_size)
				{
#pragma omp task shared(ualloc, unewalloc) firstprivate(i, j, tile_size, matrix_size) \
	depend(in : unew[i : tile_size][j : tile_size])                                   \
	depend(out : u[i : tile_size][j : tile_size])
					copy_block(matrix_size, i / tile_size, j / tile_size, ualloc, unewalloc, tile_size);
				}
			}

			for (int j = 0; j < matrix_size; j += tile_size)
			{
				for (int i = 0; i < matrix_size; i += tile_size)
				{
					int xdm1 = i == 0 ? 0 : tile_size;
					int xdp1 = i == matrix_size - tile_size ? 0 : tile_size;
					int ydp1 = j == matrix_size - tile_size ? 0 : tile_size;
					int ydm1 = j == 0 ? 0 : tile_size;
#pragma omp task shared(ualloc, unewalloc) firstprivate(dx, matrix_size, tile_size, i, j, xdm1, xdp1, ydp1, ydm1) \
	depend(out : unew[i : tile_size][j : tile_size])                                                              \
	depend(in : f[i : tile_size][j : tile_size],                                                                  \
			   u[i : tile_size][j : tile_size],                                                                   \
			   u[(i - xdm1) : tile_size][j : tile_size],                                                          \
			   u[i : tile_size][(j + ydp1) : tile_size],                                                          \
			   u[i : tile_size][(j - ydm1) : tile_size],                                                          \
			   u[(i + xdp1) : tile_size][j : tile_size])
					compute_estimate(i / tile_size, j / tile_size, ualloc, unewalloc, falloc, dx,
									 matrix_size, tile_size);
				}
			}
		}
	}
}
