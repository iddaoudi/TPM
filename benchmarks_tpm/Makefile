# OMPT doesn't seem to be working properly with clang-15

# Path
ROOT=/home/cc

# Compiler
CC=gcc
CXX=g++

# C flags
CFLAGS+=-std=gnu99 -O2 -rdynamic -ldl -Iinclude -Idense -Isparse
FLAGS_OPENMP=-fopenmp

# OpenMP
LLVM_OMP_LIB=${ROOT}/llvm-project/build-openmp
OPENMP_LIBRARY=-lomp -I$(LLVM_OMP_LIB)/include -L$(LLVM_OMP_LIB)/lib

# OpenBLAS
LAPACKE_LIB=${ROOT}/OpenBLAS/build
LAPACKE_LIBRARY=-lopenblas -I${LAPACKE_LIB}/include -L${LAPACKE_LIB}/lib

# PAPI
PAPI_LIBRARY=-lpapi -lpthread -lm -mcmodel=large 

all: tpm_benchmark

tpm_benchmark: main.c $(wildcard include/*.h dense/*.h dense/srcqr/*.h dense/srclu/*.h sparse/*.h srcslu/*.h)
	$(CC) $(CFLAGS) $(FLAGS_OPENMP) $< -o $@ $(OPENMP_LIBRARY) $(LAPACKE_LIBRARY) $(PAPI_LIBRARY)

.PHONY: all tpm_benchmark clean

clean:
	$(RM) tpm_benchmark *~
