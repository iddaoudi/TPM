
void TPM_power_monitor(int combination_of_tasks,
                       int frequency_to_set,
                       int default_frequency)
{
    TPM_power_start_zmq_server();

    int active_packages = TPM_rapl_init();

    uint64_t *pkg_energy_start = (uint64_t *)calloc(active_packages, sizeof(uint64_t));
    uint64_t *pkg_energy_finish = (uint64_t *)calloc(active_packages, sizeof(uint64_t));
    uint64_t *dram_energy_start = (uint64_t *)calloc(active_packages, sizeof(uint64_t));
    uint64_t *dram_energy_finish = (uint64_t *)calloc(active_packages, sizeof(uint64_t));

    double exec_time = 0.0;
    const char **list_of_tasks;

    while (1)
    {
        char received_message[TPM_MESSAGE_SIZE];
        char key[TPM_STRING_SIZE];
        double value = 0.0;

        zmq_recv(zmq_server, received_message, TPM_MESSAGE_SIZE, 0);
        sscanf(received_message, "%s %lf", key, &value);

        if (strcmp(key, "energy") == 0)
        {
            if ((unsigned int)value == 0)
            {
                TPM_power_start_measuring_uj(active_packages,
                                             pkg_energy_start,
                                             dram_energy_start);
            }
            else if ((unsigned int)value == 1)
            {
                TPM_power_finish_measuring_uj(active_packages,
                                              pkg_energy_finish,
                                              dram_energy_finish,
                                              pkg_energy_start,
                                              dram_energy_start);
            }
        }
        else if (strcmp(key, "time") == 0)
        {
            exec_time = value;
            break;
        }
        else
        {
            list_of_tasks = TPM_power_control(combination_of_tasks, key,
                                              (unsigned int)value, frequency_to_set,
                                              default_frequency);
        }
    }
    TPM_power_close_zmq_server();
    dump(active_packages, pkg_energy_start, pkg_energy_finish,
         dram_energy_start, dram_energy_finish,
         exec_time, list_of_tasks);

    free(pkg_energy_start);
    free(pkg_energy_finish);
    free(dram_energy_start);
    free(dram_energy_finish);
}