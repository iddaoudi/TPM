void dump(int active_packages,
          uint64_t *pkg_energy_start,
          uint64_t *pkg_energy_finish,
          uint64_t *dram_energy_start,
          uint64_t *dram_energy_finish,
          double exec_time)
{

    char filename[TPM_FILENAME_SIZE];
    int TPM_ITER = atoi(getenv("TPM_ITER"));
    sprintf(filename, "energy_data_%s_%d_%d.csv", ALGORITHM, MATRIX, TPM_ITER);

    FILE *file;
    struct stat buffer;
    int file_already_exists = (stat(filename, &buffer) == 0);

    file = fopen(filename, "a");
    if (file == NULL)
    {
        fprintf(stderr, "fopen failed\n");
        exit(EXIT_FAILURE);
    }

    if (!file_already_exists)
    {
        fprintf(file, "algorithm,matrix_size,tile_size,case,PKG1,PKG2,DRAM1,DRAM2,time\n");
    }

    uint64_t *pkg_energy = (uint64_t *)calloc(active_packages, sizeof(uint64_t));
    uint64_t *dram_energy = (uint64_t *)calloc(active_packages, sizeof(uint64_t));

    for (int i = 0; i < active_packages; i++)
    {
        pkg_energy[i] = pkg_energy_finish[i] - pkg_energy_start[i];
        dram_energy[i] = dram_energy_finish[i] - dram_energy_start[i];
    }

    if (active_packages > 2)
    {
        fprintf(stderr, "More packages that what dump can handle\n");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%s,%d,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%f\n",
            ALGORITHM, MATRIX, TILE, combination_of_tasks,
            pkg_energy[0], pkg_energy[1], dram_energy[0], dram_energy[1],
            exec_time);

    fclose(file);

    free(pkg_energy);
    free(dram_energy);
}