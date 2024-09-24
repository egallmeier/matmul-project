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

const char* dgemm_desc = "My awesome dgemm."; 
#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (k = 0;k < K; ++k) {
        for (j = 0; j < N; ++j) {
            

            for (i = 0; i < M; ++i) {
                double cij = C[j*lda+i];
    
		cij += A[k*lda+i] * B[j*lda+k];
                C[j*lda+i] = cij;
    
	    }
           
        }
    }
}

//__declspec(align(8)) static double a[8*8];
//__declspec(align(8)) static double b[8*8];
//__declspec(align(8)) static double c[8*8];
static __attribute__((aligned(8))) double a[BLOCK_SIZE*BLOCK_SIZE];
static __attribute__((aligned(8))) double b[BLOCK_SIZE*BLOCK_SIZE];
static __attribute__((aligned(8))) double c[BLOCK_SIZE*BLOCK_SIZE];

void kernel_dgemm(const int lda, const int M, const int N, const int K, 
                  const double * restrict A , const double * restrict B, double * restrict C)
{
        
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
 	    a[i + k*BLOCK_SIZE] = A[i + k*lda];
	}
    }
    
    for (int k = 0; k < K ; k++) {
	for (int j = 0; j < N; j++) {
            b[ k + j*BLOCK_SIZE] = B[k + j*lda];
	}
    }
        
    
    
    
    for (int k = 0; k < BLOCK_SIZE; ++k) {
	 #pragma GCC ivdep 
                   
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            #pragma GCC ivdep
            #pragma GCC unroll 4 
            for (int i = 0; i < BLOCK_SIZE ; ++i) {
                   
	        c[j*BLOCK_SIZE + i] += a[k*BLOCK_SIZE+i] * b[j*BLOCK_SIZE+k];	
	    }
         }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N ; ++j) {
		C[j*lda + i] += c[j*BLOCK_SIZE + i];
	}
    }
    memset(c, 0, sizeof(c));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
   
}


void do_block(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    kernel_dgemm(lda, M, N , K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
    //basic_dgemm(lda, M, N, K,
      //   A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double * restrict A, const double * restrict B, double * restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }

}



/*
# define MAX_SIZE 145u
# define MIN_RUNS 4
# define MIN_SECS 0.25

void matrix_init(double *A)
{
    for (int i = 0; i < MAX_SIZE*MAX_SIZE; ++i) 
        A[i] = drand48();
}


void matrix_clear(double *C)
{
    memset(C, 0, MAX_SIZE * MAX_SIZE * sizeof(double));
}


void diff_dgemm(const int M, const double *A, const double *B, double *C)
{
    matrix_clear(C);
    square_dgemm(M, A, B, C);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double dotprod = 0;
            double errorbound = 0;
            for (int k = 0; k < M; ++k) {
                double prod = A[k*M + i] * B[j*M + k];
                dotprod += prod;
                errorbound += fabs(prod);
            }
            
            printf(" % 0.0e", C[j*M+i]-dotprod);
        }
    }
    
}

void validate_dgemm(const int M, const double *A, const double *B, double *C)
{
    matrix_clear(C);
    square_dgemm(M, A, B, C);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double dotprod = 0;
            double errorbound = 0;
            for (int k = 0; k < M; ++k) {
                double prod = A[k*M + i] * B[j*M + k];
                dotprod += prod;
                errorbound += fabs(prod);
            }
            errorbound *= (M * DBL_EPSILON);
            double err = fabs(C[j*M + i] - dotprod);
            if (err > 3*errorbound) {
                printf("Matrix multiply failed.\n");
                printf( "C(%d,%d) should be %lg, was %lg\n", i, j,
                        dotprod, C[j*M + i]);
                printf("Error of %lg, acceptable limit %lg\n",
                        err, 3*errorbound);
		
                diff_dgemm(M, A, B, C);
                exit(-1);
            }
        }
    }
}


double time_dgemm(const int M, const double *A, const double *B, double *C)
{
    double secs = -1.0;
    double mflops_sec;
    int num_iterations = MIN_RUNS;
    while (secs < MIN_SECS) {
        matrix_clear(C);
        double start = omp_get_wtime();
        for (int i = 0; i < num_iterations; ++i) {
            square_dgemm(M, A, B, C);
        }
        double finish = omp_get_wtime();
        double mflops = 2.0 * num_iterations * M * M * M / 1.0e6;
        secs = finish-start;
        mflops_sec = mflops / secs;
        num_iterations *= 2;
    }
    return mflops_sec;
}



int main()
{
	double* A = (double*) malloc(MAX_SIZE * MAX_SIZE * sizeof(double));
	double* B  = (double*) malloc(MAX_SIZE * MAX_SIZE * sizeof(double));
	double* C  = (double*) malloc(MAX_SIZE * MAX_SIZE * sizeof(double));
        

	matrix_init(A);
	matrix_init(B);

	const int M = MAX_SIZE;
        printf("%f\n",time_dgemm(M,A,B,C));

        validate_dgemm(M,A,B,C);

	free(C);
	free(B);
	free(A);

	return 0;
}
*/
