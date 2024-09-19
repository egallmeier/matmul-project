const char* dgemm_desc = "My awesome dgemm.";
// changed loop order to j k i to take advantage of column major order 
void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;
    for (j = 0; j < M; ++j) {
        for (k = 0; k < M; ++k) {
            double cjk = C[k*M+j];
            for (i = 0; i < M; ++i)
                cjk += A[i*M+j] * B[k*M+i];
            C[k*M+j] = cjk;
        }
    }
}

// another comment
