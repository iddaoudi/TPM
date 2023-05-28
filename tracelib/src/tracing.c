#include "tracing.h"

extern void TPM_trace_start()
{
    TPM_ALGORITHM = getenv("TPM_ALGORITHM");
    TPM_PAPI = atoi(getenv("TPM_PAPI_SET"));
    TPM_POWER = atoi(getenv("TPM_POWER_SET"));

    /* ZMQ initialization */
    if (TPM_POWER)
    {
        zmq_context = zmq_ctx_new();
        zmq_request = zmq_socket(zmq_context, ZMQ_PUSH);

        TPM_zmq_connect_client(zmq_request);
        TPM_zmq_send_signal(zmq_request, "energy 0");
    }

    /* PAPI initialization */
    if (TPM_PAPI)
    {
        int papi_version = PAPI_library_init(PAPI_VER_CURRENT);
        if (papi_version != PAPI_VER_CURRENT && papi_version > 0)
        {
            fprintf(stderr, "PAPI library version mismatch: %s\n", PAPI_strerror(papi_version));
            exit(EXIT_FAILURE);
        }
        else if (papi_version < 0)
        {
            fprintf(stderr, "PAPI library init error: %s\n", PAPI_strerror(papi_version));
            exit(EXIT_FAILURE);
        }
        /* Threaded PAPI initialization */
        int ret;
        if ((ret = PAPI_thread_init(pthread_self)) != PAPI_OK)
        {
            fprintf(stderr, "PAPI_thread_init error: %s\n", PAPI_strerror(ret));
            exit(EXIT_FAILURE);
        }

        /* Add PAPI events */
        eventset = PAPI_NULL;
        ret = PAPI_create_eventset(&eventset);
        if (ret != PAPI_OK)
        {
            fprintf(stderr, "PAPI_create_eventset error: %s\n", PAPI_strerror(ret));
            exit(EXIT_FAILURE);
        }
        ret = PAPI_add_events(eventset, events, NEVENTS);
        if (ret != PAPI_OK)
        {
            fprintf(stderr, "PAPI_add_events error: %s\n", PAPI_strerror(ret));
            exit(EXIT_FAILURE);
        }

        /* Find the algorithm corresponding tasks */
        for (int i = 0; i < sizeof(algorithms) / sizeof(Algorithm); i++)
        {
            if (strcmp(algorithms[i].algorithm_name, TPM_ALGORITHM) == 0)
            {
                algorithm = &algorithms[i];
                printf("Found algorithm: %s\n", algorithm->algorithm_name);
                break;
            }
        }
        if (!algorithm)
        {
            fprintf(stderr, "Algorithm not found\n");
            exit(EXIT_FAILURE);
        }
        algorithm->counters = (CounterData **)malloc(algorithm->num_tasks * sizeof(CounterData *));
        algorithm->task_index = (TaskIndex *)malloc(algorithm->num_tasks * sizeof(TaskIndex));
        for (int i = 0; i < algorithm->num_tasks; i++)
        {
            algorithm->counters[i] = (CounterData *)calloc(1, sizeof(CounterData));
            algorithm->task_index[i].task_name = algorithm->task_names[i];
            algorithm->task_index[i].index = i;
        }
    }
}

extern void TPM_trace_task_start(const char *task_name)
{
    pthread_mutex_lock(&mutex);

    if (TPM_POWER)
    {
        unsigned int cpu, node;
        getcpu(&cpu, &node);
        char *signal_control_task_on_cpu = TPM_str_and_int_to_str(task_name, cpu);
        TPM_zmq_send_signal(zmq_request, signal_control_task_on_cpu);
        free(signal_control_task_on_cpu);
    }
    if (TPM_PAPI)
    {
        /* Start PAPI counters */
        memset(values, 0, sizeof(values));
        int ret = PAPI_start(eventset);
        if (ret != PAPI_OK)
        {
            fprintf(stderr, "PAPI_start %s error: %s\n", task_name, PAPI_strerror(ret));
            exit(EXIT_FAILURE);
        }
        printf("Started PAPI counting for %s\n", task_name);
    }
    pthread_mutex_unlock(&mutex);
}

extern void TPM_trace_task_finish(const char *task_name)
{
    pthread_mutex_lock(&mutex);

    if (TPM_PAPI == 1)
    {
        /* Stop PAPI counters */
        int ret = PAPI_stop(eventset, values);
        if (ret != PAPI_OK)
        {
            fprintf(stderr, "PAPI_stop %s error: %s\n", task_name, PAPI_strerror(ret));
            exit(EXIT_FAILURE);
        }
        /* Who is this task? */
        int task_index = -1;
        for (int i = 0; i < algorithm->num_tasks; i++)
        {
            if (strcmp(algorithm->task_index[i].task_name, task_name) == 0)
            {
                task_index = algorithm->task_index[i].index;
                break;
            }
        }
        if (task_index == -1)
        {
            fprintf(stderr, "Task not found\n");
            exit(EXIT_FAILURE);
        }
        /* Accumulate events for this task */
        for (int i = 0; i < NEVENTS; i++)
        {
            algorithm->counters[task_index]->values[i] += values[i];
        }
        algorithm->counters[task_index]->values[NEVENTS]++;
        printf("Finished PAPI counting for %s\n", task_name);
    }

    pthread_mutex_unlock(&mutex);
}

extern void TPM_trace_finalize(double total_execution_time)
{
    if (TPM_POWER)
    {
        TPM_zmq_send_signal(zmq_request, "energy 1");

        char *signal_execution_time = TPM_str_and_double_to_str("time", total_execution_time);
        TPM_zmq_send_signal(zmq_request, signal_execution_time);
        free(signal_execution_time);

        TPM_zmq_close(zmq_request, zmq_context);
    }
    if (TPM_PAPI)
    {
        // #pragma omp taskwait
        PAPI_destroy_eventset(&eventset);
        PAPI_shutdown();
        for (int i = 0; i < algorithm->num_tasks; i++)
        {
            free(algorithm->counters[i]);
        }
        free(algorithm->counters);
        free(algorithm->task_index);
    }
}