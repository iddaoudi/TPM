void TPM_power_start_measuring_uj(int active_packages,
                                  uint64_t *pkg_energy_start,
                                  uint64_t *dram_energy_start)
{
    for (int i = 0; i < active_packages; i++)
    {
        pkg_energy_start[i] = TPM_rapl_get_uj(i, "pkg");
        dram_energy_start[i] = TPM_rapl_get_uj(i, "dram");
        if (pkg_energy_start[i] >= TPM_rapl_get_maxuj(i, "pkg") ||
            dram_energy_start[i] >= TPM_rapl_get_maxuj(i, "dram"))
        {
            fprintf(stderr, "Energy measured larger than max?\n");
            exit(EXIT_FAILURE);
        }
    }
}

void TPM_power_finish_measuring_uj(int active_packages,
                                   uint64_t *pkg_energy_finish,
                                   uint64_t *dram_energy_finish,
                                   uint64_t *pkg_energy_start,
                                   uint64_t *dram_energy_start)
{
    for (int i = 0; i < active_packages; i++)
    {
        pkg_energy_finish[i] = TPM_rapl_get_uj(i, "pkg");
        dram_energy_finish[i] = TPM_rapl_get_uj(i, "dram");
        if (pkg_energy_finish[i] < pkg_energy_start[i])
        {
            pkg_energy_finish[i] += TPM_rapl_get_maxuj(i, "pkg");
        }
        if (dram_energy_finish[i] < dram_energy_start[i])
        {
            dram_energy_finish[i] += TPM_rapl_get_maxuj(i, "dram");
        }
    }
}

void TPM_power_set_frequency(unsigned int cpu, unsigned long frequency)
{
    int ret = cpufreq_modify_policy_max(cpu, frequency);
    if (ret != 0)
    {
        fprintf(stderr, "Couldn't set frequency, check root access\n");
        exit(EXIT_FAILURE);
    }
}