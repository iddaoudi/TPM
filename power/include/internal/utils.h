char *ALGORITHM;
int NTHREADS;
int MATRIX;
int TILE;

int frequency_to_set;
int default_frequency;
int combination_of_tasks;

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
