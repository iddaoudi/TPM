#include "tpm_power.h"

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Incorrect number of arguments\n");
        exit(EXIT_FAILURE);
    }

    combination_of_tasks = atoi(argv[1]);
    frequency_to_set = atoi(argv[2]);
    default_frequency = atoi(argv[3]);

    algorithm = getenv("TPM_ALGORITHM");
    number_of_threads = atoi(getenv("TPM_THREADS"));
    matrix_size = atoi(getenv("TPM_MATRIX"));
    tile_size = atoi(getenv("TPM_TILE"));

    /* Check that the current governor is ondemand */
    TPM_power_check_current_governor();

    /* Control power */
    TPM_power_monitor(combination_of_tasks, frequency_to_set, default_frequency);

    return 0;
}