/*
 * =====================================================================================
 *
 *       Filename:  poisson.h
 *
 *    Description:  Task-based 2 dimensions Poisson algorithm
 *
 *        Version:  1.0
 *        Created:  11/01/2023
 *       Revision:  14/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void poisson(int matrix_size, int tile_size, double *time_start, double *time_finish)
{
	// Check the convergence
	int check = 0;
	// Number of iterations
	int niter = 100;

	double dx;
	int k, i, l, j;

	double *falloc = malloc(matrix_size * matrix_size * sizeof(double));
	double(*f)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])falloc;

	double *ualloc = malloc(matrix_size * matrix_size * sizeof(double));
	double *unewalloc = malloc(matrix_size * matrix_size * sizeof(double));
	double(*unew)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])unewalloc;

	dx = 1.0 / (double)(matrix_size - 1);

	rhs(matrix_size, falloc, tile_size);

#pragma omp parallel
#pragma omp master
	for (j = 0; j < matrix_size; j += tile_size)
		for (i = 0; i < matrix_size; i += tile_size)
		{
#pragma omp task firstprivate(i, j) private(k, l)
			{
				for (l = j; l < j + tile_size; ++l)
					for (k = i; k < i + tile_size; ++k)
					{
						if (k == 0 || k == matrix_size - 1 || l == 0 || l == matrix_size - 1)
						{
							(*unew)[k][l] = (*f)[k][l];
						}
						else
						{
							(*unew)[k][l] = 0.0;
						}
					}
			}
		}

	*time_start = omp_get_wtime();
	jacobi(matrix_size, dx, falloc, niter, ualloc, unewalloc, tile_size);
	*time_finish = omp_get_wtime();

	if (check)
	{
		double x;
		double y;

		double *udiffalloc = malloc(matrix_size * matrix_size * sizeof(double));
		double(*udiff)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])udiffalloc;

		for (j = 0; j < matrix_size; j++)
		{
			y = (double)(j) / (double)(matrix_size - 1);
			for (i = 0; i < matrix_size; i++)
			{
				x = (double)(i) / (double)(matrix_size - 1);
				(*udiff)[i][j] = (*unew)[i][j] - ux(x, y);
			}
		}
		double error1 = r8mat_rms(matrix_size, udiffalloc);

		rhs(matrix_size, falloc, tile_size);

		for (j = 0; j < matrix_size; j++)
		{
			for (i = 0; i < matrix_size; i++)
			{
				if (i == 0 || i == matrix_size - 1 || j == 0 || j == matrix_size - 1)
				{
					(*unew)[i][j] = (*f)[i][j];
				}
				else
				{
					(*unew)[i][j] = 0.0;
				}
			}
		}

		jacobi(matrix_size, dx, falloc, niter, ualloc, unewalloc, tile_size);

		// Check convergence
		for (j = 0; j < matrix_size; j++)
		{
			y = (double)(j) / (double)(matrix_size - 1);
			for (i = 0; i < matrix_size; i++)
			{
				x = (double)(i) / (double)(matrix_size - 1);
				(*udiff)[i][j] = (*unew)[i][j] - ux(x, y);
			}
		}
		double error2 = r8mat_rms(matrix_size, udiffalloc);

		bool convergence = fabs(error1 - error2) < 1.0E-6;
		// In case of convergence, print True, otherwise print False
		printf("Convergence: %s\n", formatBool(convergence));

		free(udiffalloc);
	}
	free(falloc);
	free(ualloc);
	free(unewalloc);
}