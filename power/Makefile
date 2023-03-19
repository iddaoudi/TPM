CC=clang
CXX=clang++

CFLAGS+=-std=gnu99 -O2 -Iinclude -Icontrol

ZMQ_LIB=-lzmq
CPUFREQ_LIB=-lcpufreq

SRCDIR=src
INCDIR=include

SOURCES=$(wildcard $(SRCDIR)/*.c)
OBJECTS=$(SOURCES:.c=.o)
MAIN_OBJECT=main.o

all: tpm_server

tpm_server: $(MAIN_OBJECT) $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(ZMQ_LIB) $(CPUFREQ_LIB)

$(MAIN_OBJECT): main.c
	$(CC) -c $(CFLAGS) $< -o $@

%.o: $(SRCDIR)/%.c
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: all tpm_server clean

clean:
	$(RM) $(SRCDIR)/*.o *.o tpm_server *~
