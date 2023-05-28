void dump(long l3_cache_size)
{
    int TPM_ITER = atoi(getenv("TPM_ITER"));
    int TPM_FREQUENCY = atoi(getenv("TPM_FREQUENCY"));
    int TPM_MATRIX = atoi(getenv("TPM_MATRIX"));
    int TPM_TILE = atoi(getenv("TPM_TILE"));

    int file_desc;
    for (file_desc = 3; file_desc < 1024; ++file_desc)
    {
        close(file_desc);
    }

    char filename[TPM_FILENAME_SIZE];
    sprintf(filename, "counters_%s_%d.csv", TPM_ALGORITHM, TPM_ITER);

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