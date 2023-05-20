/*
 * =====================================================================================
 *
 *       Filename:  counters.h
 *
 *    Description:  PAPI counters struct
 *
 *        Version:  1.0
 *        Created:  19/03/2023
 *       Revision:  13/05/2023
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
    // NEVENTS + 1 for the task weight
    long long values[NEVENTS + 1];
    long long papi_l3_tcm;
    long long papi_tot_ins;
    long long papi_res_stl;
    long long papi_tot_cyc;
    long long papi_br_msp;
    long long papi_br_ins;
    int weight;
} CounterData;

void compute_derived_metrics(CounterData *data)
{
    // PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS
    data->papi_l3_tcm = data->values[0];
    data->papi_tot_ins = data->values[1];
    data->papi_res_stl = data->values[2];
    data->papi_tot_cyc = data->values[3];
    data->papi_br_msp = data->values[4];
    data->papi_br_ins = data->values[5];
    data->weight = (int)data->values[6];
}

void accumulate_counters(CounterData *dst, CounterData *src, int available_threads)
{
    memset(dst->values, 0, (NEVENTS + 1) * sizeof(long long));
    for (int i = 0; i < NEVENTS + 1; i++)
    {
        for (int j = 0; j < available_threads; j++)
        {
            dst->values[i] += src[j].values[i];
        }
    }
}

void dump_counters(const char *algorithm, const char *task_names[], CounterData *counters[], int num_tasks, int matrix_size, int tile_size, long l3_cache_size, int available_threads)
{
    CounterData *total_counters = malloc(num_tasks * sizeof(CounterData));
    for (int i = 0; i < num_tasks; i++)
    {
        accumulate_counters(&total_counters[i], counters[i], available_threads);
        compute_derived_metrics(&total_counters[i]);
    }

    int TPM_ITER = atoi(getenv("TPM_ITER"));
    int TPM_PAPI_FREQ = atoi(getenv("TPM_PAPI_FREQ"));

    // PAPI opens too much file descriptors without closing them
    int file_desc;
    for (file_desc = 3; file_desc < 1024; ++file_desc)
    {
        close(file_desc);
    }

    char filename[TPM_STRING_SIZE];
    sprintf(filename, "counters_%s_%d.csv", algorithm, TPM_ITER);

    FILE *file;
    if ((file = fopen(filename, "a+")) == NULL)
    {
        perror("fopen failed");
        exit(EXIT_FAILURE);
    }
    else
    {
        fseek(file, 0, SEEK_SET);
        int first_char = fgetc(file);
        if (first_char == EOF)
        {
            fprintf(file, "algorithm,task,matrix_size,tile_size,frequency,papi_l3_tcm,papi_tot_ins,papi_res_stl,papi_tot_cyc,papi_br_msp,papi_br_ins,l3_cache_size,weight\n");
        }

        for (int i = 0; i < num_tasks; i++)
        {
            fprintf(file, "%s,%s,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%ld,%d\n", algorithm, task_names[i], matrix_size, tile_size, TPM_PAPI_FREQ,
                    total_counters[i].papi_l3_tcm, total_counters[i].papi_tot_ins, total_counters[i].papi_res_stl, total_counters[i].papi_tot_cyc, total_counters[i].papi_br_msp, total_counters[i].papi_br_ins,
                    l3_cache_size, total_counters[i].weight);
        }

        fclose(file);
    }
    free(total_counters);
}
