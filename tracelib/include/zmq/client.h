void TPM_zmq_connect_client(void *request)
{
    int num = 0;
    zmq_setsockopt(request, ZMQ_LINGER, &num, sizeof(int));
    num = -1;
    zmq_setsockopt(request, ZMQ_SNDHWM, &num, sizeof(int));
    zmq_setsockopt(request, ZMQ_RCVHWM, &num, sizeof(int));

    zmq_connect(request, "tcp://127.0.0.1:5555");
}

int TPM_zmq_send_signal(void *request, char *task_and_cpu)
{
    int ret = zmq_send(request, task_and_cpu, TPM_MESSAGE_SIZE, 0);
    return ret;
}

void TPM_zmq_close(void *request, void *context)
{
    zmq_close(request);
    zmq_ctx_destroy(context);
}