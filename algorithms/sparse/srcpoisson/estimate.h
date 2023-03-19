/*
 * =====================================================================================
 *
 *       Filename:  estimate.h
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

static inline void compute_estimate(int block_x, int block_y, double *u_, double *u_new_, double *f_, double dx, int matrix_size, int tile_size)
{
	int i, j;

	double(*f)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])f_;
	double(*u)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])u_;
	double(*unew)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])u_new_;

	int start_i = block_x * tile_size;
	int start_j = block_y * tile_size;

	for (i = start_i; i < start_i + tile_size; i++)
	{
		for (j = start_j; j < start_j + tile_size; j++)
		{
			if (i == 0 || j == 0 || i == matrix_size - 1 || j == matrix_size - 1)
			{
				(*unew)[i][j] = (*f)[i][j];
			}
			else
			{
				(*unew)[i][j] = 0.25 * ((*u)[i - 1][j] + (*u)[i][j + 1] + (*u)[i][j - 1] + (*u)[i + 1][j] + (*f)[i][j] * dx * dx);
			}
		}
	}
}
