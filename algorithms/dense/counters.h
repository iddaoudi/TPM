/*
 * =====================================================================================
 *
 *       Filename:  counters.h
 *
 *    Description:  PAPI counters struct
 *
 *        Version:  1.0
 *        Created:  19/03/2023
 *       Revision:  none
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#define NEVENTS 6

typedef struct
{
    long long values[NEVENTS];
    double arithm_intensity;
    double mem_boundness;
    double bmr;
    double ilp;
} CounterData;

void compute_derived_metrics(CounterData *data)
{
    // PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS
    data->arithm_intensity = (double)data->values[0] / (double)data->values[1];
    data->mem_boundness = (double)data->values[2] / (double)data->values[3];
    data->bmr = (double)data->values[4] / (double)data->values[5];
    data->ilp = (double)data->values[1] / (double)data->values[3];
}

void accumulate_counters(long long dst[], long long src[][NEVENTS], int available_threads)
{
    memset(dst, 0, NEVENTS * sizeof(long long));
    for (int i = 0; i < NEVENTS; i++)
    {
        for (int j = 0; j < available_threads; j++)
        {
            dst[i] += src[j][i];
        }
    }
}