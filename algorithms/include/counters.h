/*
 * =====================================================================================
 *
 *       Filename:  counters.h
 *
 *    Description:  PAPI counters struct
 *
 *        Version:  1.0
 *        Created:  19/03/2023
 *       Revision:  21/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#define NEVENTS 5

int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP}; //, PAPI_BR_INS};

typedef struct
{
    long long values[NEVENTS + 1];
} CounterData;

void accumulate_counters(CounterData *dst, CounterData *src, int available_threads)
{
    memset(dst->values, 0, (NEVENTS + 1) * sizeof(long long));
    for (int i = 0; i < (NEVENTS + 1); i++)
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
            // fprintf(file, "algorithm,task,matrix_size,tile_size,frequency,papi_l3_tcm,papi_tot_ins,papi_res_stl,papi_tot_cyc,papi_br_msp,papi_br_ins,weight,l3_cache_size\n");
            fprintf(file, "algorithm,task,matrix_size,tile_size,frequency,papi_l3_tcm,papi_tot_ins,papi_res_stl,papi_tot_cyc,papi_br_msp,weight,l3_cache_size\n");
        }
        for (int i = 0; i < num_tasks; i++)
        {
            fprintf(file, "%s,%s,%d,%d,%d", algorithm, task_names[i], matrix_size, tile_size, TPM_PAPI_FREQ);
            for (int j = 0; j < (NEVENTS + 1); j++)
            {
                fprintf(file, ",%lld", total_counters[i].values[j]);
            }
            fprintf(file, ",%ld\n", l3_cache_size);
        }
        fclose(file);
    }
    free(total_counters);
}
