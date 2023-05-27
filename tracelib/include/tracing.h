#include "tpm.h"

/* Library tracing functions */

// Start the tracing/power tool: when launching the application
extern void TPM_trace_start();

// Capture the task name: will only be used right before the OpenMP task region in the
// warmup phase
extern void TPM_trace_task_name(const char *task_name);

// Capture the task and the CPU its running on for power control: inside the
// OpenMP task region
extern void TPM_trace_task_cpu_and_node(const char *task_name,
                                        int cpu,
                                        int node);

// End the power control and send the captured application metrics: when the
// application ends
extern void TPM_trace_finalize(double total_execution_time);

/* Middle man tracing functions */

extern void TPM_middle_man_start(char *algorithm,
                                 int matrix_size,
                                 int tile_size,
                                 int number_of_threads)
{
    TPM_trace_start();
}

extern void TPM_middle_man_task_name(const char *task_name)
{
    TPM_trace_task_name(task_name);
}

extern void TPM_middle_man_task_cpu_and_node(const char *task_name,
                                             int cpu,
                                             int node)
{
    TPM_trace_task_cpu_and_node(task_name,
                                cpu,
                                node);
}

extern void TPM_middle_man_finalize(double total_execution_time)
{
    TPM_trace_finalize(total_execution_time);
}