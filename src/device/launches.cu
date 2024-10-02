#include <cuda_runtime.h>
#include <vector>
#include "kernel_matrix_multiplication.cuh"

using namespace std;

/**
 * @brief Realiza la multiplicación de matrices utilizando CUDA.
 *
 * Esta función multiplica dos matrices utilizando la aceleración de GPU mediante CUDA.
 * La multiplicación es del tipo A (rowsA x colsA) * B (rowsB x colsB) = C (rowsA x colsB).
 * 
 * El resultado se almacena en la matriz C, que se copia de la memoria del dispositivo a la memoria del host.
 *
 * @param A_matrix Vector unidimensional que representa la matriz A en formato fila.
 * @param B_matrix Vector unidimensional que representa la matriz B en formato fila.
 * @param rowsA Número de filas de la matriz A.
 * @param colsA Número de columnas de la matriz A.
 * @param rowsB Número de filas de la matriz B.
 * @param colsB Número de columnas de la matriz B.
 */
void multiplyMatrices(vector<float> A_matrix,
                      vector<float> B_matrix,
                      int rowsA, int colsA,
                      int rowsB, int colsB){

    std::vector<float> C_matrix(rowsA * colsB);
    float *hC_matrix;
    float *dC_matrix;
    hC_matrix = (float*)malloc(rowsA * colsB * sizeof(float));

    float *dA_matrix, *dB_matrix;

    unsigned int A_matrix_dim = rowsA * colsA;
    unsigned int B_matrix_dim = rowsB * colsB;
    unsigned int C_matrix_dim = rowsA * colsB;

    // Reservar memoria en el dispositivo para las matrices A, B y C
    cudaMalloc(&dA_matrix, A_matrix_dim * sizeof(float));
    cudaMalloc(&dB_matrix, B_matrix_dim * sizeof(float));
    cudaMalloc(&dC_matrix, C_matrix_dim * sizeof(float));

    // Copiar matrices A y B del host al dispositivo
    cudaMemcpy(dA_matrix, A_matrix.data(), A_matrix_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_matrix, B_matrix.data(), B_matrix_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Configuración de la GRID y Block para el kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        ceil((colsA + dimBlock.x - 1) / dimBlock.x),
        ceil((rowsB + dimBlock.y - 1) / dimBlock.y));

    // Lanzar el kernel de multiplicación de matrices en el dispositivo
    matrixMulKernel<<<dimGrid, dimBlock>>>(dA_matrix, dB_matrix, dC_matrix, rowsA, colsA, colsB);

    // Sincronizar el dispositivo para asegurarse de que la multiplicación haya terminado
    cudaDeviceSynchronize();

    // Copiar la matriz resultante C del dispositivo al host
    cudaMemcpy(hC_matrix, dC_matrix, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Liberar memoria
    cudaFree(dA_matrix);
    cudaFree(dB_matrix);
    cudaFree(dC_matrix);
    free(hC_matrix);
}
