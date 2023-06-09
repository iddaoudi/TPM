#include "tracing.h"

extern void TPM_trace_start()
{
    TPM_ALGORITHM = getenv("TPM_ALGORITHM");
    TPM_PAPI = atoi(getenv("TPM_PAPI_SET"));
    TPM_POWER = atoi(getenv("TPM_POWER_SET"));
    TPM_TASK_TIME = atoi(getenv("TPM_TASK_TIME"));
    TPM_TASK_TIME_TASK = getenv("TPM_TASK_TIME_TASK");

    /* Measure task times */
    if (TPM_TASK_TIME)
    {
        total_task_time = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &total_start);
    }

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
        TPM_PAPI_COUNTERS = atoi(getenv("TPM_PAPI_COUNTERS"));
        if (TPM_PAPI_COUNTERS == 1)
        {
            events[0] = PAPI_L3_TCM;
            events[1] = PAPI_TOT_INS;
            events[2] = PAPI_TOT_CYC;
            events[3] = PAPI_RES_STL;
            events_strings[0] = "PAPI_L3_TCM";
            events_strings[1] = "PAPI_TOT_INS";
            events_strings[2] = "PAPI_TOT_CYC";
            events_strings[3] = "PAPI_RES_STL";
            NEVENTS = 4;
        }
        else if (TPM_PAPI_COUNTERS == 2)
        {
            events[0] = PAPI_L2_TCR;
            events[1] = PAPI_L2_TCW;
            events_strings[0] = "PAPI_L2_TCR";
            events_strings[1] = "PAPI_L2_TCW";
            NEVENTS = 2;
        }
        else if (TPM_PAPI_COUNTERS == 3)
        {
            events[0] = PAPI_L3_TCR;
            events[1] = PAPI_L3_TCW;
            events_strings[0] = "PAPI_L3_TCR";
            events_strings[1] = "PAPI_L3_TCW";
            NEVENTS = 2;
        }
        else if (TPM_PAPI_COUNTERS == 4)
        {
            events[0] = PAPI_VEC_DP;
            events_strings[0] = "PAPI_VEC_DP";
            NEVENTS = 1;
        }
        ret = PAPI_add_events(eventset, events, NEVENTS);
        if (ret != PAPI_OK)
        {
            fprintf(stderr, "PAPI_add_events error: %s\n", PAPI_strerror(ret));
            exit(EXIT_FAILURE);
        }
    }
    /* Find the algorithm corresponding tasks */
    for (int i = 0; i < sizeof(algorithms) / sizeof(Algorithm); i++)
    {
        if (strcmp(algorithms[i].algorithm_name, TPM_ALGORITHM) == 0)
        {
            algorithm = &algorithms[i];
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

extern void TPM_trace_task_start(const char *task_name)
{
    pthread_mutex_lock(&mutex);

    if (TPM_TASK_TIME)
    {
        if (strcmp(task_name, TPM_TASK_TIME_TASK) == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &start);
            task_counter++;
        }
    }

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
    }

    pthread_mutex_unlock(&mutex);
}

extern void TPM_trace_task_finish(const char *task_name)
{
    pthread_mutex_lock(&mutex);

    if (TPM_TASK_TIME)
    {
        if (strcmp(task_name, TPM_TASK_TIME_TASK) == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &end);

            volatile double elapsed = (end.tv_sec - start.tv_sec);
            elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
            total_task_time += elapsed;
        }
    }

    if (TPM_PAPI)
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
#pragma omp taskwait
        PAPI_destroy_eventset(&eventset);
        PAPI_shutdown();

        /* Get L3 cache size */
        long l3_cache_size;
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3_cache_size = sysconf(_SC_LEVEL3_CACHE_SIZE);
#else
        fprintf(stderr, "_SC_LEVEL3_CACHE_SIZE is not available\n");
        exit(EXIT_FAILURE);
#endif
        if (l3_cache_size == -1)
        {
            fprintf(stderr, "L3 cache size sysconf failed\n");
            exit(EXIT_FAILURE);
        }

        dump(l3_cache_size);

        for (int i = 0; i < algorithm->num_tasks; i++)
        {
            free(algorithm->counters[i]);
        }
        free(algorithm->counters);
        free(algorithm->task_index);
    }

    if (TPM_TASK_TIME)
    {
        clock_gettime(CLOCK_MONOTONIC, &total_end);
        volatile double elapsed = (total_end.tv_sec - total_start.tv_sec);
        elapsed += (total_end.tv_nsec - total_start.tv_nsec) / 1000000000.0;
        int TPM_MATRIX = atoi(getenv("TPM_MATRIX"));
        int TPM_TILE = atoi(getenv("TPM_TILE"));
        printf("%d,%d,%f,%s,%f,%d\n", TPM_MATRIX, TPM_TILE, elapsed, TPM_TASK_TIME_TASK, total_task_time, task_counter);
    }
}