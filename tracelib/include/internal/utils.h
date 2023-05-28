#define TPM_MESSAGE_SIZE 26
#define TPM_STRING_SIZE 10

int TPM_PAPI = 0;
int TPM_POWER = 0;

char *TPM_ALGORITHM;

pthread_mutex_t mutex;

cvector_vector_type(char *) recorded_tasks_vector;