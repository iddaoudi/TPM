void dump(long l3_cache_size)
{
    int TPM_ITER = atoi(getenv("TPM_ITER"));
    int TPM_FREQUENCY = atoi(getenv("TPM_FREQUENCY"));
    int TPM_MATRIX = atoi(getenv("TPM_MATRIX"));
    int TPM_TILE = atoi(getenv("TPM_TILE"));

    if (strcmp(TPM_ALGORITHM, "sylsvd") == 0)
    {
        if (TPM_TILE == 1024)
        {
            TPM_MATRIX = TPM_TILE * TPM_MATRIX / 2;
        }
        else if (TPM_TILE == 2048)
        {
            TPM_MATRIX = TPM_TILE * TPM_MATRIX / 4;
        }
        else if (TPM_TILE == 512)
        {
            TPM_MATRIX = TPM_TILE * TPM_MATRIX;
        }
        else
        {
            fprintf(stderr, "SylSVD parameters problem in trace\n");
            exit(EXIT_FAILURE);
        }
    }

    int file_desc;
    for (file_desc = 3; file_desc < 1024; ++file_desc)
    {
        close(file_desc);
    }

    char filename[TPM_FILENAME_SIZE];
    sprintf(filename, "counters_%s_%d_%d.csv", TPM_ALGORITHM, TPM_ITER, TPM_PAPI_COUNTERS);

    FILE *file;
    if ((file = fopen(filename, "a+")) == NULL)
    {
        fprintf(stderr, "fopen failed");
        exit(EXIT_FAILURE);
    }
    else
    {
        fseek(file, 0, SEEK_SET);
        int first_char = fgetc(file);
        if (first_char == EOF)
        {
            fprintf(file, "algorithm,task,matrix_size,tile_size,l3_cache_size,frequency,weight,");
            for (int i = 0; i < NEVENTS; i++)
            {
                fprintf(file, "%s,", events_strings[i]);
            }
            fprintf(file, "\n");
        }
        for (int i = 0; i < algorithm->num_tasks; i++)
        {
            fprintf(file, "%s,%s,%d,%d,%ld,%d,%lld,", TPM_ALGORITHM,
                    algorithm->task_index[i].task_name,
                    TPM_MATRIX, TPM_TILE, l3_cache_size, TPM_FREQUENCY,
                    algorithm->counters[i]->values[NEVENTS]);
            for (int j = 0; j < NEVENTS; j++)
            {
                fprintf(file, "%lld,", algorithm->counters[i]->values[j]);
            }
            fprintf(file, "\n");
        }
    }
    fclose(file);
}