/*
 * =====================================================================================
 *
 *       Filename:  utils.h
 *
 *    Description:  Headers imports and useful definitions used across dense algorithms
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  14/05/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#include <math.h>
#include "assert.h"
#include "cblas.h"
#include "lapacke.h"
#include <omp.h>

int MSIZE, BSIZE, NTH, TPM_TRACE, TPM_TRACE, TPM_PAPI;
long l3_cache_size;

#define A(m, n) tpm_tile_address(A, m, n)
#define B(m, n) tpm_tile_address(B, m, n)
#define S(m, n) tpm_tile_address(S, m, n)

#define tpm_upper 121
#define tpm_lower 122
#define tpm_left 141
#define tpm_right 142
#define tpm_notranspose 111
#define tpm_transpose 112
#define tpm_nonunit 131
#define tpm_unit 132
#define tpm_W 501
#define tpm_A2 502
#define tpm_row 101
#define tpm_column 102
#define tpm_forward 391
#define tpm_backward 392

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#include "cvector.h"
#include "descriptor.h"
#include "tile_address.h"
#include "populate.h"
#include "print.h"
#include "counters.h"

#include "cholesky.h"

#include "srcqr/dgeqrt.h"
#include "srcqr/dormqr.h"
#include "srcqr/dtsmqr.h"
#include "srcqr/dtsqrt.h"
#include "qr.h"

#include "srclu/lacpy.h"
#include "srclu/tile_lapacke_conversion.h"
#include "srclu/geswp.h"
#include "lu.h"

// #include "srcslu/empty_block.h"
// #include "srcslu/bdiv.h"
// #include "srcslu/bmod.h"
// #include "srcslu/lu0.h"
// #include "srcslu/fwd.h"
// #include "sparselu.h"

// #include "srcpoisson/jacobi.h"
// #include "poisson.h"

#include "sylsvd.h"

#include "invert.h"