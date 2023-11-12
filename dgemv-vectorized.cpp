#include <immintrin.h>

const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void dgemv_vectorized(int n, double* A, double* x, double* y) {
   // insert your code here: implementation of vectorized vector-matrix multiply
   // This is where you will use SIMD instructions or other techniques for vectorization
   // Example using SIMD intrinsics (change this based on your specific architecture):
   for (int i = 0; i < n; i++) {
      __m256d y_vector = _mm256_setzero_pd();
      for (int j = 0; j < n; j += 4) {
         __m256d a_vector = _mm256_loadu_pd(&A[i * n + j]);
         __m256d x_vector = _mm256_loadu_pd(&x[j]);
         y_vector = _mm256_add_pd(y_vector, _mm256_mul_pd(a_vector, x_vector));
      }
      // Sum the elements of the vector
      y[i] = y_vector[0] + y_vector[1] + y_vector[2] + y_vector[3];
   }
}
   