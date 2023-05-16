/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  Task-based algorithms for power management
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  21/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include "common.h"
#include "utils.h"

// #define LOG 1

typedef enum
{
  ALGO_CHOLESKY,
  ALGO_QR,
  ALGO_LU,
  ALGO_SPARSELU,
  ALGO_POISSON,
  ALGO_SYLSVD,
  ALGO_INVERT,
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
  else if (strcmp(algorithm, "poisson") == 0)
  {
    return ALGO_POISSON;
  }
  else if (strcmp(algorithm, "sylsvd") == 0)
  {
    return ALGO_SYLSVD;
  }
  else if (strcmp(algorithm, "invert") == 0)
  {
    return ALGO_INVERT;
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

  // Check matrix size divisibility by tile size
  if (MSIZE % BSIZE != 0)
  {
    printf("Tile size does not divide the matrix size. Aborting.\n");
    exit(EXIT_FAILURE);
  }

  // PAPI library initialization
  int papi_version = PAPI_library_init(PAPI_VER_CURRENT);
  if (papi_version != PAPI_VER_CURRENT && papi_version > 0)
  {
    printf("PAPI library version mismatch: %s\n", PAPI_strerror(papi_version));
    exit(EXIT_FAILURE);
  }
  else if (papi_version < 0)
  {
    printf("PAPI library init error: %s\n", PAPI_strerror(papi_version));
    exit(EXIT_FAILURE);
  }
  // Threaded PAPI initialization
  if (TPM_PAPI)
  {
    int ret;
    if ((ret = PAPI_thread_init(pthread_self)) != PAPI_OK)
    {
      printf("PAPI thread init error: %s\n", PAPI_strerror(ret));
      exit(EXIT_FAILURE);
    }
  }

  // Get the L3 cache size
#ifdef _SC_LEVEL3_CACHE_SIZE
  l3_cache_size = sysconf(_SC_LEVEL3_CACHE_SIZE);
#else
  printf("_SC_LEVEL3_CACHE_SIZE is not available.\n");
  exit(EXIT_FAILURE);
#endif
  if (l3_cache_size == -1)
  {
    perror("sysconf failed");
    exit(EXIT_FAILURE);
  }

  // Timers
  double time_start, time_finish;
  int error;

  // Launch algorithms
  switch (algo_type)
  {
  // Dense algorithms
  case ALGO_CHOLESKY:
  case ALGO_QR:
  {
    tpm_desc *A = NULL;
    double *ptr = NULL;
    tpm_desc *S = NULL;

    error = posix_memalign((void **)&ptr, getpagesize(), MSIZE * MSIZE * sizeof(double));
    if (error)
    {
      printf("Problem allocating contiguous memory.\n");
      exit(EXIT_FAILURE);
    }
    tpm_matrix_desc_create(&A, ptr, BSIZE, MSIZE * MSIZE, BSIZE * BSIZE, MSIZE);
    tpm_hermitian_positive_generator(*A);

    switch (algo_type)
    {
    // Cholesky algorithm
    case ALGO_CHOLESKY:

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
      // Workspace allocation for QR
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
    }

    free(A->matrix);
    tpm_matrix_desc_destroy(&A);
    break;
  }
  // LU algorithm
  case ALGO_LU:
  {
    double *A = malloc(MSIZE * MSIZE * sizeof(double));
    double *hA = malloc(MSIZE * MSIZE * sizeof(double));
    int *ipiv = malloc(MSIZE * sizeof(int));

    tpm_dense_generator(A, MSIZE);
    tpm_matrix_to_tile(hA, A, MSIZE, MSIZE, BSIZE, MSIZE);
#ifdef LOG
    tpm_default_print_matrix("A", A, MSIZE);
#endif
    for (int i = 0; i < MSIZE; i++)
    {
      ipiv[i] = i;
    }

    time_start = omp_get_wtime();
#pragma omp parallel
#pragma omp master
    {
      lu(MSIZE, BSIZE, A, ipiv, hA);
    }
    time_finish = omp_get_wtime();
    tpm_matrix_to_tile(hA, A, MSIZE, MSIZE, BSIZE, MSIZE);
#ifdef LOG
    tpm_default_print_matrix("A", A, MSIZE);
#endif

    free(A);
    free(hA);
    free(ipiv);
    break;
  }
  // Sylvester-SVD algorithm
  case ALGO_SYLSVD:
  {
    // Number of iterations is constant, we only vary the size of the whole matrix
    // which is the input tile size
    int iter = 100;
    // Array of matrices
    double *As[iter];
    double *Bs[iter];
    double *Xs[iter];
    double *Us[iter];
    double *Ss[iter];
    double *VTs[iter];
    for (int i = 0; i < iter; i++)
    {
      As[i] = (double *)malloc(BSIZE * BSIZE * sizeof(double));
      Bs[i] = (double *)malloc(BSIZE * BSIZE * sizeof(double));
      Xs[i] = (double *)malloc(BSIZE * BSIZE * sizeof(double));
      Us[i] = (double *)malloc(BSIZE * BSIZE * sizeof(double));
      Ss[i] = (double *)malloc(BSIZE * sizeof(double));
      VTs[i] = (double *)malloc(BSIZE * BSIZE * sizeof(double));

      tpm_dense_generator(As[i], BSIZE);
      tpm_dense_generator(Bs[i], BSIZE);
      tpm_dense_generator(Xs[i], BSIZE);
    }

    time_start = omp_get_wtime();
#pragma omp parallel
#pragma omp master
    {
      sylsvd(As, Bs, Xs, Us, Ss, VTs, BSIZE, iter);
    }
    time_finish = omp_get_wtime();

    for (int i = 0; i < iter; i++)
    {
      free(As[i]);
      free(Bs[i]);
      free(Xs[i]);
      free(Us[i]);
      free(Ss[i]);
      free(VTs[i]);
    }
    break;
  }
  case ALGO_INVERT:
  {
    double *A = malloc(MSIZE * MSIZE * sizeof(double));
    // Partial pivoting index array
    int *ipiv = malloc(MSIZE * sizeof(int));
    tpm_dense_generator(A, MSIZE);

    time_start = omp_get_wtime();
#pragma omp parallel
#pragma omp master
    {
      invert(A, ipiv, MSIZE, BSIZE);
    }
    time_finish = omp_get_wtime();

    free(A);
    free(ipiv);
    break;
  }

  // Sparse algorithms
  // Poisson algorithm
  case ALGO_POISSON:
  {
    {
      poisson(MSIZE, BSIZE, &time_start, &time_finish);
    }
    break;
  }

  // SparseLU algorithm
  case ALGO_SPARSELU:
  {
    double **M;
    if (TPM_PAPI)
    {
      tpm_sparse_allocate(&M, MSIZE, BSIZE);
      sparselu(M, MSIZE, BSIZE);
      free(M);
    }
    else
    {
#pragma omp parallel
#pragma omp master
      tpm_sparse_allocate(&M, MSIZE, BSIZE);

      time_start = omp_get_wtime();
      sparselu(M, MSIZE, BSIZE);
      time_finish = omp_get_wtime();

      free(M);
    }
    break;
  }

  default:
    printf("Invalid algorithm. Aborting.\n");
    exit(EXIT_FAILURE);
  }

  if (TPM_TRACE)
  {
    double exec_time = time_finish - time_start;
    tpm_upstream_finalize(exec_time);
  }
}
