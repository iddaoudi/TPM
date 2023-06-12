/*
 * =====================================================================================
 *
 *       Filename:  functions.h
 *
 *    Description:  Various functions needed for Poisson problem
 *
 *        Version:  1.0
 *        Created:  14/05/2023
 *       Revision:  none
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#define formatBool(b) ((b) ? "True" : "False")

double ux(double x, double y)
{
    double pi = 3.141592653589793;
    return sin(pi * x * y);
}

double ux_derivative(double x, double y)
{
    double pi = 3.141592653589793;
    return -pi * pi * (x * x + y * y) * sin(pi * x * y);
}

// Right hand side vector initialization
void rhs(int matrix_size, double *falloc, int tile_size)
{
    double(*f)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])falloc;
    double x, y;

    int i, j, k, l;
#pragma omp parallel
#pragma omp master
    for (j = 0; j < matrix_size; j += tile_size)
    {
        for (i = 0; i < matrix_size; i += tile_size)
        {
#pragma omp task firstprivate(tile_size, i, j, matrix_size) private(l, k, x, y) shared(f)
            for (k = j; k < j + tile_size; k++)
            {
                y = (double)(k) / (double)(matrix_size - 1);
                for (l = i; l < i + tile_size; l++)
                {
                    x = (double)(l) / (double)(matrix_size - 1);
                    if (l == 0 || l == matrix_size - 1 || k == 0 || k == matrix_size - 1)
                        (*f)[l][k] = ux(x, y);
                    else
                        (*f)[l][k] = -ux_derivative(x, y);
                }
            }
        }
    }
}

// Root mean squared norm
double r8mat_rms(int matrix_size, double *udiffalloc)
{
    // a tmp
    double(*a)[matrix_size][matrix_size] = (double(*)[matrix_size][matrix_size])udiffalloc;

    double norm = 0.0;

    for (int j = 0; j < matrix_size; j++)
    {
        for (int i = 0; i < matrix_size; i++)
        {
            norm = norm + (*a)[i][j] * (*a)[i][j];
        }
    }
    norm = sqrt(norm / (double)(matrix_size * matrix_size));

    return norm;
}