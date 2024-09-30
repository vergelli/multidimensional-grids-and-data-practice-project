#include <cuda_runtime.h>
//! Version de codigo sacado de ejemplo del libro en pagina 64

/**
 * Kernel convencional de CUDA para la multiplicación de dos matrices A y B,
 * almacenando el resultado en la matriz C.
 * 
//@param A Puntero a la matriz A.
//@param B Puntero a la matriz B.
//@param C Puntero a la matriz C donde se almacenará el resultado.
//@param rowsA Número de filas de la matriz A.
//@param colsA Número de columnas de la matriz A y número de filas de la matriz B.
//@param colsB Número de columnas de la matriz B.
**/
__global__ void matrixMulKernel(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float value = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            value += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = value;
    }
}
