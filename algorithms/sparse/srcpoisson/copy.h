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

#include <assert.h>

static inline void copy_block(int matrix_size, int block_x, int block_y, double *u_, double *u_new_, int tile_size)
{
	int i, j;

	double(*u)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])u_;
	double(*unew)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])u_new_;

	int start_i = block_x * tile_size;
	int start_j = block_y * tile_size;

	for (i = start_i; i < start_i + tile_size; i++)
	{
		for (j = start_j; j < start_j + tile_size; j++)
		{
			assert((i < matrix_size) && (j < matrix_size));
			(*u)[i][j] = (*unew)[i][j];
		}
	}
}
