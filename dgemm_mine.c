#define _XOPEN_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>

const char* dgemm_desc = "Three-level blocked dgemm.";

#ifndef BLOCK_SIZE_L3
#define BLOCK_SIZE_L3 ((int) 128)
#endif

#ifndef BLOCK_SIZE_L2
#define BLOCK_SIZE_L2 ((int) 64)
#endif

#ifndef BLOCK_SIZE_L1
#define BLOCK_SIZE_L1 ((int) 32)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[k*lda+i] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}
static __attribute__((aligned(8))) double a[BLOCK_SIZE_L1*BLOCK_SIZE_L1];
static __attribute__((aligned(8))) double b[BLOCK_SIZE_L1*BLOCK_SIZE_L1];
static __attribute__((aligned(8))) double c[BLOCK_SIZE_L1*BLOCK_SIZE_L1];

void kernel_dgemm(const int lda, const int M, const int N, const int K, 
                  const double * restrict A , const double * restrict B, double * restrict C)
{
        
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
 	    a[i + k*BLOCK_SIZE_L1] = A[i + k*lda];
	}
    }
    
    for (int k = 0; k < K ; k++) {
	for (int j = 0; j < N; j++) {
            b[ k + j*BLOCK_SIZE_L1] = B[k + j*lda];
	}
    }
        
    
    #pragma ivdep 
    #pragma vector aligned 
    for (int k = 0; k < BLOCK_SIZE_L1; ++k) {
	 #pragma ivdep
	 #pragma vector aligned 
	 
                   
        for (int j = 0; j < BLOCK_SIZE_L1; ++j) {
            
            
        #pragma ivdep
	    #pragma vector aligned
	    #pragma vector always
	    for (int i = 0; i < BLOCK_SIZE_L1 ; ++i) {
                   
	        c[j*BLOCK_SIZE_L1 + i]+= a[k*BLOCK_SIZE_L1+i] * b[j*BLOCK_SIZE_L1+k];	
	    }
	    
         }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N ; ++j) {
		C[j*lda + i] += c[j*BLOCK_SIZE_L1 + i];
	}
    }
    memset(c, 0, sizeof(c));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
   
}

void do_block_l1(const int lda, 
                 const double *A, const double *B, double *C,
                 const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE_L1 > lda ? lda-i : BLOCK_SIZE_L1);
    const int N = (j+BLOCK_SIZE_L1 > lda ? lda-j : BLOCK_SIZE_L1);
    const int K = (k+BLOCK_SIZE_L1 > lda ? lda-k : BLOCK_SIZE_L1);
    kernel_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void do_block_l2(const int lda, 
                 const double *A, const double *B, double *C,
                 const int i, const int j, const int k)
{
    const int n_blocks = BLOCK_SIZE_L2 / BLOCK_SIZE_L1;
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int ii = i + bi * BLOCK_SIZE_L1;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int jj = j + bj * BLOCK_SIZE_L1;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int kk = k + bk * BLOCK_SIZE_L1;
                do_block_l1(lda, A, B, C, ii, jj, kk);
            }
        }
    }
}

void do_block_l3(const int lda, 
                 const double *A, const double *B, double *C,
                 const int i, const int j, const int k)
{
    const int n_blocks = BLOCK_SIZE_L3 / BLOCK_SIZE_L2;
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int ii = i + bi * BLOCK_SIZE_L2;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int jj = j + bj * BLOCK_SIZE_L2;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int kk = k + bk * BLOCK_SIZE_L2;
                do_block_l2(lda, A, B, C, ii, jj, kk);
            }
        }
    }
}

void square_dgemm(const int M, 
                  const double * restrict A, 
		  const double * restrict B, 
		  double * restrict C)
{
    const int n_blocks = M / BLOCK_SIZE_L3 + (M % BLOCK_SIZE_L3 ? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE_L3;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE_L3;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE_L3;
                do_block_l3(M, A, B, C, i, j, k);
            }
        }
    }
}
