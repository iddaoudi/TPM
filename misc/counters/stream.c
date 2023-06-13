#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <papi.h>

#define NEVENTS 4

static double *vecA1, *vecB1, *vecC1;
static double *vecA2, *vecB2, *vecC2;

int main(int argc, char *argv[])
{
    int SIZE = atoi(argv[1]);

    vecA1 = calloc(SIZE, sizeof(double));
    vecA2 = calloc(SIZE, sizeof(double));
    vecB1 = calloc(SIZE, sizeof(double));
    vecB2 = calloc(SIZE, sizeof(double));
    vecC1 = calloc(SIZE, sizeof(double));
    vecC2 = calloc(SIZE, sizeof(double));

    for (int i = 0; i < SIZE; i++)
    {
        vecA1[i] = 3.2;
        vecB1[i] = 4.3;
        vecC1[i] = 5.4;
    }

    int ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT)
        printf("PAPI init problem\n");

    int eventset = PAPI_NULL;
    ret = PAPI_create_eventset(&eventset);
    if (ret != PAPI_OK)
        printf("PAPI evetnset problem\n");

    long long values[NEVENTS];

    int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_RES_STL};
    char *events_str[NEVENTS] = {"PAPI_L3_TCM", "PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_RES_STL"};
    // int events[NEVENTS] = {PAPI_L2_TCR, PAPI_L2_TCW, PAPI_VEC_DP};
    // char *events_str[NEVENTS] = {"PAPI_L2_TCR", "PAPI_L2_TCW", "PAPI_L3_TCR"};

    ret = PAPI_add_events(eventset, events, NEVENTS);
    if (ret != PAPI_OK)
    {
        fprintf(stderr, "PAPI_create_eventset error: %s\n", PAPI_strerror(ret));
        printf("PAPI add events problem %d\n", ret);
    }

    // Cache clearance
    for (int i = 0; i < SIZE; i++)
    {
        vecC2[i] = vecC1[i] * 4.2 + 3.1;
    }

    memset(values, 0, sizeof(values));
    ret = PAPI_start(eventset);
    if (ret != PAPI_OK)
        printf("PAPI start problem\n");

    // Compute bound
    for (int i = 0; i < SIZE; i++)
    {
        vecA2[i] = 2.3 * vecA1[i] + 7.8 * i + i;
    }

    ret = PAPI_stop(eventset, values);
    if (ret != PAPI_OK)
        printf("PAPI stop problem\n");

    for (int i = 0; i < NEVENTS; i++)
    {
        printf("Compute bound : %s = %llu\n", events_str[i], values[i]);
    }

    // Reset PAPI
    PAPI_reset(eventset);

    ret = PAPI_start(eventset);
    if (ret != PAPI_OK)
        printf("PAPI start problem\n");

    // Memory bound
    for (int i = 0; i < SIZE; i++)
    {
        vecB2[i] = vecB1[i];
    }

    ret = PAPI_stop(eventset, values);
    if (ret != PAPI_OK)
        printf("PAPI stop problem\n");

    for (int i = 0; i < NEVENTS; i++)
    {
        printf("Memory bound: %s = %llu\n", events_str[i], values[i]);
    }

    PAPI_destroy_eventset(&eventset);
    PAPI_shutdown();

    return 0;
}
