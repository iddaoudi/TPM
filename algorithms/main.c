/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  Task-based algorithms for power management
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  19/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include "common.h"
#include "utils.h"

typedef enum
{
  ALGO_CHOLESKY,
  ALGO_QR,
  ALGO_LU,
  ALGO_SPARSELU,
  ALGO_UNKNOWN
} AlgorithmType;

AlgorithmType parse_algorithm(const char *algorithm)
{
  if (strcmp(algorithm, "cholesky") == 0)
  {
    return ALGO_CHOLESKY;
  }
  else if (strcmp(algorithm, "qr") == 0)
  {
    return ALGO_QR;
  }
  else if (strcmp(algorithm, "lu") == 0)
  {
    return ALGO_LU;
  }
  else if (strcmp(algorithm, "sparselu") == 0)
  {
    return ALGO_SPARSELU;
  }
  else
  {
    return ALGO_UNKNOWN;
  }
}

int tpm_allocate_tile(int M, tpm_desc **desc, int B)
{
  int MT = M / B;
  *desc = (tpm_desc *)malloc(sizeof(tpm_desc));
  if (*desc == NULL)
  {
    printf("Tile allocation failed.\n");
    return 1;
  }
  **desc = tpm_matrix_desc_init(B, MT * B, B * B, MT * B);
  int info = tpm_matrix_desc_alloc(*desc);
  assert(!info);
  return 0;
}

int main(int argc, char *argv[])
{
  NTH = atoi(getenv("OMP_NUM_THREADS"));
  TPM_TRACE = atoi(getenv("TPM_TRACE"));
  TPM_PAPI = atoi(getenv("TPM_PAPI"));

  // Command line arguments parsing
  int arguments = 0;
  char algorithm[16];
  struct option long_options[] = {{"Algorithm", required_argument, NULL, 'a'},
                                  {"Matrix size", required_argument, NULL, 'm'},
                                  {"Tile size", required_argument, NULL, 'b'},
                                  {NULL, no_argument, NULL, 0}};

  if (argc < 2)
  {
    printf("Missing arguments. Aborting...\n");
    exit(EXIT_FAILURE);
  }

  AlgorithmType algo_type = ALGO_UNKNOWN;

  while ((arguments =
              getopt_long(argc, argv, "a:m:b:h:", long_options, NULL)) != -1)
  {
    if (optind > 2)
    {
      switch (arguments)
      {
      case 'a':
        if (optarg)
        {
          algo_type = parse_algorithm(optarg);
          if (algo_type == ALGO_UNKNOWN)
          {
            printf("Invalid algorithm. Aborting.\n");
            exit(EXIT_FAILURE);
          }
        }
        break;
      case 'm':
        if (optarg)
          MSIZE = atoi(optarg);
        break;
      case 'b':
        if (optarg)
          BSIZE = atoi(optarg);
        break;
      case 'h':
        printf("HELP\n");
        exit(EXIT_FAILURE);
      case '?':
        printf("Invalid arguments. Aborting.\n");
        exit(EXIT_FAILURE);
      }
    }
  }

  if (MSIZE % BSIZE != 0)
  {
    printf("Tile size does not divide the matrix size. Aborting.\n");
    exit(EXIT_FAILURE);
  }

  int papi_version = PAPI_library_init(PAPI_VER_CURRENT);
  if (papi_version != PAPI_VER_CURRENT && papi_version > 0) {
    printf("PAPI library version mismatch: %s\n", PAPI_strerror(papi_version));
    exit(1);
  } else if (papi_version < 0) {
    printf("PAPI library init error: %s\n", PAPI_strerror(papi_version));
    exit(1);
  }

  if (TPM_PAPI)
  {
    int ret;
    if ((ret = PAPI_thread_init(pthread_self)) != PAPI_OK)
    {
      printf("PAPI thread init error: %s\n", PAPI_strerror(ret));
      exit(1);
    }
  }

  // Launch algorithms
  double time_start, time_finish;

  switch (algo_type)
  {
  case ALGO_CHOLESKY:
  case ALGO_QR:
  case ALGO_LU:
  {
    tpm_desc *A = NULL;
    double *ptr = NULL;
    int error = posix_memalign((void **)&ptr, getpagesize(),
                               MSIZE * MSIZE * sizeof(double));
    if (error)
    {
      printf("Problem allocating contiguous memory.\n");
      exit(EXIT_FAILURE);
    }
    tpm_matrix_desc_create(&A, ptr, BSIZE, MSIZE * MSIZE, BSIZE * BSIZE, MSIZE);

    switch (algo_type)
    {
    // Cholesky algorithm
    case ALGO_CHOLESKY:
      tpm_hermitian_positive_generator(*A);
      time_start = omp_get_wtime();
#pragma omp parallel
#pragma omp master
      {
        cholesky(*A);
      }
      time_finish = omp_get_wtime();
      break;
    // QR algorithm
    case ALGO_QR:
      tpm_hermitian_positive_generator(*A);
      // Workspace allocation for QR
      tpm_desc *S = NULL;
      int ret = tpm_allocate_tile(MSIZE, &S, BSIZE);
      assert(ret == 0);
      time_start = omp_get_wtime();
#pragma omp parallel
#pragma omp master
      {
        qr(*A, *S);
      }
      time_finish = omp_get_wtime();
      free(S->matrix);
      tpm_matrix_desc_destroy(&S);
      break;
    // LU algorithm
    case ALGO_LU:
      tpm_hermitian_positive_generator(*A);
      time_start = omp_get_wtime();
#pragma omp parallel
#pragma omp master
      {
        lu(*A);
      }
      time_finish = omp_get_wtime();
      break;
    }
    free(A->matrix);
    tpm_matrix_desc_destroy(&A);
    break;
  }
  // SparseLU algorithm
  case ALGO_SPARSELU:
  {
    double **M;
#pragma omp parallel
#pragma omp master
    tpm_sparse_allocate(&M, MSIZE, BSIZE);

    time_start = omp_get_wtime();
    sparselu(M, MSIZE, BSIZE);
    time_finish = omp_get_wtime();
    free(M);
    break;
  }
  default:
    printf("Invalid algorithm. Aborting.\n");
    exit(EXIT_FAILURE);
  }

  if (TPM_TRACE)
    tpm_upstream_finalize(MSIZE, BSIZE, x);
  //printf("%f\n", time_finish - time_start);
}
