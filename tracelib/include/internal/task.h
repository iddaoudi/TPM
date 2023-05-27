typedef struct
{
    char *name;
    double start_time; // in us
    double end_time;
    int cpu;
    int node;
    int order;
} TPM_task_t;