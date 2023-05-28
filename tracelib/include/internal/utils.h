#define TPM_MESSAGE_SIZE 26
#define TPM_STRING_SIZE 10

int activate_zmq = 1;

int TPM_PAPI = 0;
char *TPM_ALGORITHM;

pthread_mutex_t mutex;

cvector_vector_type(char *) recorded_tasks_vector;