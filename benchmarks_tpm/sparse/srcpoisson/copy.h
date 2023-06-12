/*
 * =====================================================================================
 *
 *       Filename:  copy_block.h
 *
 *    Description:  Save the estimate
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

#include <assert.h>

static inline void copy_block(int matrix_size, int block_x, int block_y, double *ualloc, double *unewalloc, int tile_size)
{
	int i, j;

	double(*u)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])ualloc;
	double(*unew)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])unewalloc;

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
