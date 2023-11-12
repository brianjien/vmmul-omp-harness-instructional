#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

const char* dgemv_desc = "OpenMP dgemv.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */

void my_dgemv(int n, double* A, double* x, double* y) {

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        printf("my_dgemv(): Hello world: thread %d of %d checking in. \n", thread_id, nthreads);
        printf("my_dgemv(): For actual timing runs, please comment out these printf() and omp_get_*() statements. \n");
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double temp = 0.0;
        #pragma omp simd reduction(+:temp)
        for (int j = 0; j < n; j++) {
            temp += A[i * n + j] * x[j];
        }
        #pragma omp atomic
        y[i] += temp;
    }
}
