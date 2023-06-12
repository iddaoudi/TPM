typedef struct
{
    const char *algorithm;
    const char **task_names;
    int num_tasks;
} AlgorithmTasks;

const char **TPM_power_control(int selected_case, const char *task, unsigned int cpu,
                               unsigned long frequency_to_set,
                               unsigned long original_frequency)
{
    static AlgorithmTasks algorithms[] = {
        {"cholesky", cholesky_tasks, sizeof(cholesky_tasks) / sizeof(cholesky_tasks[0])},
        {"qr", qr_tasks, sizeof(qr_tasks) / sizeof(qr_tasks[0])},
        {"lu", lu_tasks, sizeof(lu_tasks) / sizeof(lu_tasks[0])},
        {"invert", invert_tasks, sizeof(invert_tasks) / sizeof(invert_tasks[0])},
        {"sylsvd", sylsvd_tasks, sizeof(sylsvd_tasks) / sizeof(sylsvd_tasks[0])},
        {"dgram", dgram_tasks, sizeof(dgram_tasks) / sizeof(dgram_tasks[0])},
        {"dcesca", dcesca_tasks, sizeof(dcesca_tasks) / sizeof(dcesca_tasks[0])},
        {"dgetrs_nopiv", dgetrs_nopiv_tasks, sizeof(dgetrs_nopiv_tasks) / sizeof(dgetrs_nopiv_tasks[0])},
        {"dgetrf_nopiv", dgetrf_nopiv_tasks, sizeof(dgetrf_nopiv_tasks) / sizeof(dgetrf_nopiv_tasks[0])},
        {"dgesv_nopiv", dgesv_nopiv_tasks, sizeof(dgesv_nopiv_tasks) / sizeof(dgesv_nopiv_tasks[0])},
        {"dgenm2", dgenm2_tasks, sizeof(dgenm2_tasks) / sizeof(dgenm2_tasks[0])},
        {"dlauum", dlauum_tasks, sizeof(dlauum_tasks) / sizeof(dlauum_tasks[0])},
        {"dtrtri", dtrtri_tasks, sizeof(dtrtri_tasks) / sizeof(dtrtri_tasks[0])},
        {"dtradd", dtradd_tasks, sizeof(dtradd_tasks) / sizeof(dtradd_tasks[0])},
        {"dpoinv", dpoinv_tasks, sizeof(dpoinv_tasks) / sizeof(dpoinv_tasks[0])},
        {"dpotri", dpotri_tasks, sizeof(dpotri_tasks) / sizeof(dpotri_tasks[0])},
        {"dposv", dposv_tasks, sizeof(dposv_tasks) / sizeof(dposv_tasks[0])},
        {"dpotrs", dpotrs_tasks, sizeof(dpotrs_tasks) / sizeof(dpotrs_tasks[0])},
        {"dpotrf", dpotrf_tasks, sizeof(dpotrf_tasks) / sizeof(dpotrf_tasks[0])},
        {"dtrsm", dtrsm_tasks, sizeof(dtrsm_tasks) / sizeof(dtrsm_tasks[0])},
        {"dtrmm", dtrmm_tasks, sizeof(dtrmm_tasks) / sizeof(dtrmm_tasks[0])},
        {"dsyr2k", dsyr2k_tasks, sizeof(dsyr2k_tasks) / sizeof(dsyr2k_tasks[0])},
        {"dsyrk", dsyrk_tasks, sizeof(dsyrk_tasks) / sizeof(dsyrk_tasks[0])},
        {"dsymm", dsymm_tasks, sizeof(dsymm_tasks) / sizeof(dsymm_tasks[0])},
        {"dlantr", dlantr_tasks, sizeof(dlantr_tasks) / sizeof(dlantr_tasks[0])},
        {"dlansy", dlansy_tasks, sizeof(dlansy_tasks) / sizeof(dlansy_tasks[0])},
        {"dlange", dlange_tasks, sizeof(dlange_tasks) / sizeof(dlange_tasks[0])},
    };

    const char **task_names = NULL;
    int num_tasks = 0;
    for (int i = 0; i < sizeof(algorithms) / sizeof(algorithms[0]); ++i)
    {
        if (!strcmp(ALGORITHM, algorithms[i].algorithm))
        {
            num_tasks = algorithms[i].num_tasks;
            task_names = malloc(sizeof(char *) * num_tasks);
            if (!task_names)
            {
                fprintf(stderr, "Failed to allocate memory for task_names\n");
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < num_tasks; j++)
            {
                task_names[j] = algorithms[i].task_names[j];
            }
            break;
        }
    }

    if (task_names == NULL)
    {
        fprintf(stderr, "Algorithm for power control not found\n");
        exit(EXIT_FAILURE);
    }

    if (selected_case >= 1 && selected_case <= ((1 << num_tasks) - 1))
    {
        int task_mask = selected_case - 1;
        int task_found = 0;
        for (int i = 0; i < num_tasks; ++i)
        {

            if (!strcmp(task, task_names[i]) && (task_mask & (1 << i)))
            {
                TPM_power_set_frequency(cpu, frequency_to_set);
                task_found = 1;
                break;
            }
        }
        if (!task_found)
        {
            TPM_power_set_frequency(cpu, original_frequency);
        }
    }
    else if (selected_case == (1 << num_tasks))
    {
        TPM_power_set_frequency(cpu, frequency_to_set);
    }

    return task_names;
}
