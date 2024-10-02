#ifndef KERNEL_MATRIX_MULTIPLICATION_CUH
#define KERNEL_MATRIX_MULTIPLICATION_CUH
/*
Guarda de inclusión: La línea #ifndef LAUNCHES_CUH
y las demás relacionadas son una técnica para evitar
la "inclusión múltiple". Si el archivo se incluye
más de una vez, estas guardas evitan errores.
*/
#include <vector>
/**
 * @brief Kernel CUDA que realiza la multiplicación de matrices.
 *
 * Esta función es ejecutada en la GPU y multiplica las matrices A y B, almacenando el resultado en la matriz C.
 *
 * @param A Matriz A en la memoria de dispositivo.
 * @param B Matriz B en la memoria de dispositivo.
 * @param C Matriz resultado en la memoria de dispositivo.
 * @param rowsA Número de filas de la matriz A.
 * @param colsA Número de columnas de la matriz A.
 * @param colsB Número de columnas de la matriz B.
 */
__global__ void matrixMulKernel(float *A, float *B, float *C, int rowsA, int colsA, int colsB);

#endif // KERNEL_MATRIX_MULTIPLICATION_CUH
