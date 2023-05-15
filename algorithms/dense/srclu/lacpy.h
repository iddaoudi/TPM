/*
 * =====================================================================================
 *
 *       Filename:  lacpy.h
 *
 *    Description:  Tile copy
 *
 *        Version:  1.0
 *        Created:  15/05/2023
 *       Revision:  none
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void tpm_lacpy(char flag, double *source, double *dest, int matrix_size, int tile_size)
{
    // Transpose
    if (flag == 't')
    {
        for (int i = 0; i < tile_size; i++)
        {
            memcpy(dest, source, sizeof(double) * tile_size);
            source += matrix_size;
            dest += tile_size;
        }
    }
    // Column
    else if (flag == 'c')
    {
        for (int i = 0; i < tile_size; i++)
        {
            memcpy(dest, source, sizeof(double) * tile_size);
            source += tile_size;
            dest += matrix_size;
        }
    }
}