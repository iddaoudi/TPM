#include "tracing.h"

extern void TPM_trace_start()
{
    printf("Library trace weak ok start\n");
    if (activate_zmq)
    {
        zmq_context = zmq_ctx_new();
        zmq_request = zmq_socket(zmq_context, ZMQ_PUSH);

        TPM_zmq_connect_client(zmq_request);
        TPM_zmq_send_signal(zmq_request, "energy 0");
        printf("Library connected to client\n");
    }
    recorded_tasks_vector = NULL;
}

extern void TPM_trace_task_start(const char *task_name)
{
    pthread_mutex_lock(&mutex);

    printf("tracelib %s start\n", task_name);
    unsigned int cpu, node;
    getcpu(&cpu, &node);
    char *signal_control_task_on_cpu = TPM_str_and_int_to_str(task_name, cpu);
    TPM_zmq_send_signal(zmq_request, signal_control_task_on_cpu);
    free(signal_control_task_on_cpu);

    pthread_mutex_unlock(&mutex);
}

extern void TPM_trace_task_finish(const char *task_name)
{
    pthread_mutex_lock(&mutex);

    printf("tracelib %s finish\n", task_name);

    pthread_mutex_unlock(&mutex);
}

extern void TPM_trace_finalize(double total_execution_time)
{
    if (activate_zmq)
    {
        TPM_zmq_send_signal(zmq_request, "energy 1");

        char *signal_execution_time = TPM_str_and_double_to_str("time", total_execution_time);
        TPM_zmq_send_signal(zmq_request, signal_execution_time);
        free(signal_execution_time);

        TPM_zmq_close(zmq_request, zmq_context);
    }
}