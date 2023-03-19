/*
 * =====================================================================================
 *
 *       Filename:  sweep.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  18/01/2023
 *       Revision:  none
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include "copy.h"
#include "estimate.h"

void sweep(int matrix_size, int dx, double *f_, int n_iter, double *u_, double *u_new_, int tile_size)
{
	double(*f)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])f_;
	double(*u)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])u_;
	double(*unew)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])u_new_;

#pragma omp parallel shared(u_, u_new_, f_, matrix_size, dx, n_iter, tile_size) // FIXME f?
#pragma omp master
	{
		for (int it = 1; it <= n_iter; it++)
		{
			for (int j = 0; j < matrix_size; j += tile_size)
			{
				for (int i = 0; i < matrix_size; i += tile_size)
				{
#pragma omp task shared(u_, u_new_) firstprivate(i, j, tile_size, matrix_size) \
	depend(in                                                                  \
		   : unew [i:tile_size] [j:tile_size])                                 \
		depend(out                                                             \
			   : u [i:tile_size] [j:tile_size])
					copy_block(matrix_size, i / tile_size, j / tile_size, u_, u_new_, tile_size);
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
#pragma omp task shared(u_, u_new_) firstprivate(dx, matrix_size, tile_size, i, j, xdm1, xdp1, ydp1, ydm1) \
	depend(out                                                                                             \
		   : unew [i:tile_size] [j:tile_size])                                                             \
		depend(in                                                                                          \
			   : f [i:tile_size] [j:tile_size],                                                            \
				 u [i:tile_size] [j:tile_size],                                                            \
				 u [(i - xdm1):tile_size] [j:tile_size],                                                   \
				 u [i:tile_size] [(j + ydp1):tile_size],                                                   \
				 u [i:tile_size] [(j - ydm1):tile_size],                                                   \
				 u [(i + xdp1):tile_size] [j:tile_size])
					compute_estimate(i / tile_size, j / tile_size, u_, u_new_, f_, dx, matrix_size, tile_size);
				}
			}
		}
	}
}
