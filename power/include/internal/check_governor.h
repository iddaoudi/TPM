struct cpufreq_policy *current_governor;

void TPM_power_check_current_governor()
{
    /* Checking for CPU 0 */
    current_governor = cpufreq_get_policy(0);
    int ret = strcmp(current_governor->governor, "ondemand");
    if (ret != 0)
    {
        fprintf(stderr, "Current governor is not ondemand\n");
        exit(EXIT_FAILURE);
    }
}