#include "tpm.h"

/* Library tracing functions */

// Start the tracing/power tool: when launching the application
extern void TPM_trace_start();

// Capture the beggining of an OpenMP task region
extern void TPM_trace_task_start(const char *task_name);

// Capture the end of an OpenMP task region
extern void TPM_trace_task_finish(const char *task_name);

// End the power control and send the captured application metrics: when the
// application ends
extern void TPM_trace_finalize(double total_execution_time);

/* Middle man tracing functions */

extern void TPM_middle_man_start()
{
    TPM_trace_start();
}

extern void TPM_middle_man_task_start(const char *task_name)
{
    TPM_trace_task_start(task_name);
}

extern void TPM_middle_man_task_finish(const char *task_name)
{
    TPM_trace_task_finish(task_name);
}

extern void TPM_middle_man_finalize(double total_execution_time)
{
    TPM_trace_finalize(total_execution_time);
}