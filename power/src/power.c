#include "tpm_power.h"

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        fprintf(stderr, "Incorrect number of arguments\n");
        exit(EXIT_FAILURE);
    }

    algorithm = argv[1];
    number_of_threads = atoi(argv[2]);
    combination_of_tasks = atoi(argv[3]);
    frequency_to_set = atoi(argv[4]);
    default_frequency = atoi(argv[5]);

    /* Check that the current governor is ondemand */
    TPM_power_check_current_governor();

    /* Control power */
    TPM_power_monitor(combination_of_tasks, frequency_to_set, default_frequency);

    return 0;
}