void TPM_power_start_zmq_server()
{
    zmq_context = zmq_ctx_new();
    zmq_server = zmq_socket(zmq_context, ZMQ_PULL);

    int number = 0;
    zmq_setsockopt(zmq_server, ZMQ_LINGER, &number, sizeof(int));
    number = -1;
    zmq_setsockopt(zmq_server, ZMQ_SNDHWM, &number, sizeof(int));
    zmq_setsockopt(zmq_server, ZMQ_RCVHWM, &number, sizeof(int));
    int ret = zmq_bind(zmq_server, "tcp://127.0.0.1:5555");
    if (ret != 0)
    {
        fprintf(stderr, "Failed to launch ZMQ server\n");
        exit(EXIT_FAILURE);
    }
}

void TPM_power_close_zmq_server()
{
    int ret = zmq_unbind(zmq_server, "tcp://127.0.0.1:5555");
    if (ret != 0)
    {
        fprintf(stderr, "Failed to shut down ZMQ server\n");
        exit(EXIT_FAILURE);
    }
}