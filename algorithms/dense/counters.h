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
    // NEVENTS + 1 for the task weight
    long long values[NEVENTS + 1];
    double arithm_intensity;
    double mem_boundness;
    double bmr;
    double ilp;
    int weight;
} CounterData;

void compute_derived_metrics(CounterData *data)
{
    // PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_INS
    data->arithm_intensity = (double)data->values[0] / (double)data->values[1];
    data->mem_boundness = (double)data->values[2] / (double)data->values[3];
    data->bmr = (double)data->values[4] / (double)data->values[5];
    data->ilp = (double)data->values[1] / (double)data->values[3];
    data->weight = (int)data->values[6];
}

void accumulate_counters(CounterData *dst, CounterData src[], int available_threads)
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

void dump_counters(const char *algorithm, const char *task_names[], CounterData *counters[], int num_tasks, int matrix_size, int tile_size, double l3_cache_size, int available_threads)
{
    CounterData *total_counters = malloc(num_tasks * sizeof(CounterData));
    for (int i = 0; i < num_tasks; i++)
    {
        accumulate_counters(&total_counters[i], counters[i], available_threads);
        compute_derived_metrics(&total_counters[i]);
    }

    int TPM_PAPI_FREQ = atoi(getenv("TPM_PAPI_FREQ"));

    // PAPI opens too much file descriptors without closing them
    int file_desc;
    for (file_desc = 3; file_desc < 1024; ++file_desc)
    {
        close(file_desc);
    }

    char filename[TPM_STRING_SIZE];
    sprintf(filename, "counters_%s.csv", algorithm);

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
            fprintf(file, "algorithm,task,matrix_size,tile_size,frequency,mem_boundness,arithm_intensity,bmr,ilp,l3_cache_ratio,weight\n");
        }

        for (int i = 0; i < num_tasks; i++)
        {
            fprintf(file, "%s,%s,%d,%d,%d,%f,%f,%f,%f,%f,%d\n", algorithm, task_names[i], matrix_size, tile_size, TPM_PAPI_FREQ,
                    total_counters[i].mem_boundness, total_counters[i].arithm_intensity, total_counters[i].bmr, total_counters[i].ilp,
                    (double)total_counters[i].values[0] / (double)l3_cache_size, total_counters[i].weight);
        }

        fclose(file);
    }
    free(total_counters);
}
