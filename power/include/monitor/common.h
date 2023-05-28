#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <regex.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <inttypes.h>

#define SYSFS_RAPL_DIR "/sys/devices/virtual/powercap/intel-rapl"
#define MAX_PKGS 4 // FIXME considering a maximum of 4 packages

#define TPM_MESSAGE_SIZE 26
#define TPM_STRING_SIZE 10
#define TPM_FILENAME_SIZE 64

static char *pkg_energy_uj[MAX_PKGS];
static char *pkg_energy_maxuj[MAX_PKGS];
static char *dram_energy_uj[MAX_PKGS];
static char *dram_energy_maxuj[MAX_PKGS];
void *zmq_server;
void *zmq_context;