#define NEVENTS 5

int events[NEVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP}; //, PAPI_BR_INS};
int eventset;
long long values[NEVENTS];

typedef struct
{
    long long values[NEVENTS + 1];
} CounterData;

typedef struct
{
    const char *task_name;
    int index;
} TaskIndex;

typedef struct
{
    const char *algorithm_name;
    const char **task_names;
    int num_tasks;
    CounterData **counters;
    TaskIndex *task_index;
} Algorithm;

static const char *dgram_tasks[] = {"laset", "syssq", "gessq", "gram", "plssq", "plssq2"};
static const char *dcesca_tasks[] = {"laset", "gesum", "gessq", "geadd", "cesca", "plssq", "plssq2"};
static const char *dgetrs_nopiv_tasks[] = {"trsm", "gemm"};
static const char *dgetrf_nopiv_tasks[] = {"getrfnpiv", "trsm", "gemm"};
static const char *dgesvd_tasks[] = {"geqrt", "lacpy", "lacpyx", "unmqr", "tpqrt", "tpmqrt", "gelqt", "unmlq", "tplqt", "tpmlqt", "laset"};
static const char *dgesv_nopiv_tasks[] = {"getrfnpiv", "trsm", "gemm"};
static const char *dgenm2_tasks[] = {"laset", "zasum", "gessq", "lacpy", "lascal", "gemv", "plssq2"};
static const char *dlauum_tasks[] = {"lauum", "syrk", "trmm", "gemm"};
static const char *dtrtri_tasks[] = {"trsm", "trtri", "gemm"};
static const char *dtradd_tasks[] = {"tradd", "geadd"};
static const char *dpoinv_tasks[] = {"potrf", "trsm", "syrk", "gemm", "trtri", "lauum", "trmm"};
static const char *dpotri_tasks[] = {"trsm", "trtri", "gemm", "lauum", "syrk", "trmm"};
static const char *dposv_tasks[] = {"potrf", "trsm", "syrk", "gemm"};
static const char *dpotrs_tasks[] = {"trsm", "gemm"};
static const char *dpotrf_tasks[] = {"potrf", "trsm", "syrk", "gemm"};
static const char *dtrsm_tasks[] = {"trsm", "gemm"};
static const char *dtrmm_tasks[] = {"trmm", "gemm"};
static const char *dsyr2k_tasks[] = {"syr2k", "gemm"};
static const char *dsyrk_tasks[] = {"syrk", "gemm"};
static const char *dsymm_tasks[] = {"symm", "gemm"};
static const char *dlantr_tasks[] = {"laset", "lantr", "lange", "langemax"};
static const char *dlansy_tasks[] = {"laset", "lansy", "lange", "langemax"};
static const char *dlange_tasks[] = {"laset", "lange", "langemax"};

Algorithm algorithms[] = {
    {"dgram", dgram_tasks, sizeof(dgram_tasks) / sizeof(dgram_tasks[0]), NULL, NULL},
    {"dcesca", dcesca_tasks, sizeof(dcesca_tasks) / sizeof(dcesca_tasks[0]), NULL, NULL},
    {"dgetrs_nopiv", dgetrs_nopiv_tasks, sizeof(dgetrs_nopiv_tasks) / sizeof(dgetrs_nopiv_tasks[0]), NULL, NULL},
    {"dgetrf_nopiv", dgetrf_nopiv_tasks, sizeof(dgetrf_nopiv_tasks) / sizeof(dgetrf_nopiv_tasks[0]), NULL, NULL},
    {"dgesvd", dgesvd_tasks, sizeof(dgesvd_tasks) / sizeof(dgesvd_tasks[0]), NULL, NULL},
    {"dgesv_nopiv", dgesv_nopiv_tasks, sizeof(dgesv_nopiv_tasks) / sizeof(dgesv_nopiv_tasks[0]), NULL, NULL},
    {"dgenm2", dgenm2_tasks, sizeof(dgenm2_tasks) / sizeof(dgenm2_tasks[0]), NULL, NULL},
    {"dlauum", dlauum_tasks, sizeof(dlauum_tasks) / sizeof(dlauum_tasks[0]), NULL, NULL},
    {"dtrtri", dtrtri_tasks, sizeof(dtrtri_tasks) / sizeof(dtrtri_tasks[0]), NULL, NULL},
    {"dtradd", dtradd_tasks, sizeof(dtradd_tasks) / sizeof(dtradd_tasks[0]), NULL, NULL},
    {"dpoinv", dpoinv_tasks, sizeof(dpoinv_tasks) / sizeof(dpoinv_tasks[0]), NULL, NULL},
    {"dpotri", dpotri_tasks, sizeof(dpotri_tasks) / sizeof(dpotri_tasks[0]), NULL, NULL},
    {"dposv", dposv_tasks, sizeof(dposv_tasks) / sizeof(dposv_tasks[0]), NULL, NULL},
    {"dpotrs", dpotrs_tasks, sizeof(dpotrs_tasks) / sizeof(dpotrs_tasks[0]), NULL, NULL},
    {"dpotrf", dpotrf_tasks, sizeof(dpotrf_tasks) / sizeof(dpotrf_tasks[0]), NULL, NULL},
    {"dtrsm", dtrsm_tasks, sizeof(dtrsm_tasks) / sizeof(dtrsm_tasks[0]), NULL, NULL},
    {"dtrmm", dtrmm_tasks, sizeof(dtrmm_tasks) / sizeof(dtrmm_tasks[0]), NULL, NULL},
    {"dsyr2k", dsyr2k_tasks, sizeof(dsyr2k_tasks) / sizeof(dsyr2k_tasks[0]), NULL, NULL},
    {"dsyrk", dsyrk_tasks, sizeof(dsyrk_tasks) / sizeof(dsyrk_tasks[0]), NULL, NULL},
    {"dsymm", dsymm_tasks, sizeof(dsymm_tasks) / sizeof(dsymm_tasks[0]), NULL, NULL},
    {"dlantr", dlantr_tasks, sizeof(dlantr_tasks) / sizeof(dlantr_tasks[0]), NULL, NULL},
    {"dlansy", dlansy_tasks, sizeof(dlansy_tasks) / sizeof(dlansy_tasks[0]), NULL, NULL},
    {"dlange", dlange_tasks, sizeof(dlange_tasks) / sizeof(dlange_tasks[0]), NULL, NULL},
};

Algorithm *algorithm = NULL;