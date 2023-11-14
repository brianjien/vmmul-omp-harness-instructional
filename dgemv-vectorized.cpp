#include <immintrin.h>

const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        __m256d y_vector = _mm256_setzero_pd();

        #pragma omp simd reduction(+:y_vector)
        for (int j = 0; j < n; j += 4) {
            // Load data with aligned load
            __m256d a_vector = _mm256_loadu_pd(&A[i * n + j]);
            __m256d x_vector = _mm256_loadu_pd(&x[j]);
            // Perform fused multiply-add
            y_vector = _mm256_fmadd_pd(a_vector, x_vector, y_vector);
        }

        // Horizontal sum
        y[i] += y_vector[0] + y_vector[1] + y_vector[2] + y_vector[3];
    }
}
