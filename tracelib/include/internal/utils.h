#define TPM_MESSAGE_SIZE 26
#define TPM_STRING_SIZE 10
#define TPM_FILENAME_SIZE 64

int TPM_PAPI = 0;
int TPM_POWER = 0;
int TPM_TASK_TIME = 0;

char *TPM_ALGORITHM = NULL;
char *TPM_TASK_TIME_TASK = NULL;

volatile double total_task_time;
struct timespec start, end;
struct timespec total_start, total_end;

pthread_mutex_t mutex;