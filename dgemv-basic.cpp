#include <iostream>

const char* dgemv_desc = "Basic implementation of matrix-vector multiply.";

void my_dgemv(int n, double* A, double* x, double* y) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }

}
