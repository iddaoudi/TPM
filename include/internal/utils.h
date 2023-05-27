#define TPM_MESSAGE_SIZE 26
#define TPM_STRING_SIZE 10

int activate_zmq = 0;

char *recorded_algorithm;
const char *recorded_task_name;

int recorded_matrix_size;
int recorded_tile_size;
int recorded_number_of_threads;

pthread_mutex_t mutex;

cvector_vector_type(char *) recorded_tasks_vector;