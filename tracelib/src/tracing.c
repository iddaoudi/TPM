#include "tracing.h"

extern void TPM_trace_start()
{
    if (activate_zmq)
    {
        zmq_context = zmq_ctx_new();
        zmq_request = zmq_socket(zmq_context, ZMQ_PUSH);

        TPM_zmq_connect_client(zmq_request);
        TPM_zmq_send_signal(zmq_request, "energy 0"); // FIXME: warmup run is to capture
        // tasks names, second is to control power
    }
    recorded_tasks_vector = NULL;
}

extern void TPM_trace_task_name(const char *task_name)
{
    recorded_task_name = task_name;
    int task_is_present = 0;
    for (int i = 0; i < cvector_size(recorded_tasks_vector); i++)
    {
        if (strcmp(recorded_tasks_vector[i], recorded_task_name) == 0)
        {
            task_is_present++;
            break;
        }
    }
    if (task_is_present == 0)
    {
        cvector_push_back(recorded_tasks_vector, (char *)recorded_task_name);
    }
    // FIXME send signal if warmup phase
}

extern void TPM_trace_task_cpu_and_node(const char *task_name,
                                        int cpu,
                                        int node)
{
    pthread_mutex_lock(&mutex);

    char *signal_control_task_on_cpu = TPM_str_and_int_to_str(task_name, cpu);
    TPM_zmq_send_signal(zmq_request, signal_control_task_on_cpu);
    free(signal_control_task_on_cpu);

    pthread_mutex_unlock(&mutex);
}

extern void TPM_trace_finalize(double total_execution_time)
{
    if (activate_zmq)
    {
        TPM_zmq_send_signal(zmq_request, "energy 1");

        char *signal_matrix_size = TPM_str_and_int_to_str("matrix", recorded_matrix_size);
        TPM_zmq_send_signal(zmq_request, signal_matrix_size);
        free(signal_matrix_size);

        char *signal_tile_size = TPM_str_and_int_to_str("tile", recorded_tile_size);
        TPM_zmq_send_signal(zmq_request, signal_tile_size);
        free(signal_tile_size);

        char *signal_execution_time = TPM_str_and_double_to_str("time", total_execution_time);
        TPM_zmq_send_signal(zmq_request, signal_execution_time);
        free(signal_execution_time);

        TPM_zmq_close(zmq_request, zmq_context);
    }
    for (int i = 0; i < cvector_size(recorded_tasks_vector); i++)
    {
        printf("%s ", recorded_tasks_vector[i]);
    }
    printf("\n");
}