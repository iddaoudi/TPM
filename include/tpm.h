#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zmq.h"
#include "pthread.h"

#include "cvector.h"
#include "utils.h"
#include "common.h"
// #include "internal/task.h"
// FIXME Useless for now ---> add task map and collect individual task data

#include "zutils.h"
#include "client.h"