#include <cuda_runtime.h>
#include <vector>
#include "kernel_matrix_multiplication.cuh"
#include "kernel_matrix_multiplication_data.cuh"
#include "matrix_io.hpp"

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

    // Crear los eventos para marcar el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Obtener la memoria libre antes de la ejecución del kernel
    size_t freeMemBefore, totalMem;
    cudaMemGetInfo(&freeMemBefore, &totalMem);

    // Reservar memoria en el dispositivo para las matrices A, B y C
    cudaMalloc(&dA_matrix, A_matrix_dim * sizeof(float));
    cudaMalloc(&dB_matrix, B_matrix_dim * sizeof(float));
    cudaMalloc(&dC_matrix, C_matrix_dim * sizeof(float));

    // Copiar matrices A y B del host al dispositivo
    cudaMemcpy(dA_matrix, A_matrix.data(), A_matrix_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_matrix, B_matrix.data(), B_matrix_dim * sizeof(float), cudaMemcpyHostToDevice);


    // Configuración de parametros de lanzamiento del Kernel
    int blockDimX = 16;
    int blockDimY = 16;

    int gridDimX = ceil((colsA + blockDimX - 1) / blockDimX);
    int gridDimY = ceil((rowsB + blockDimY - 1) / blockDimY);

    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid(gridDimX, gridDimY);

    // *________________________________________________________________________________________
    // Marcar el inicio
    cudaEventRecord(start);
    // *Lanzamiento del kernel de multiplicación de matrices en el dispositivo
    matrixMulKernel<<<dimGrid, dimBlock>>>(dA_matrix, dB_matrix, dC_matrix, rowsA, colsA, colsB);
    // *________________________________________________________________________________________
    //! Comprobación de errores después del lanzamiento del kernel _____________________________
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en el kernel: %s\n", cudaGetErrorString(err));
    }
    // !________________________________________________________________________________________

    // Marcar el evento de fin de cálculo (pero no necesariamente esperar a que termine aún)
    cudaEventRecord(stop);

    // Sincronizar el dispositivo para asegurarse de que el kernel y todas las operaciones anteriores hayan finalizado
    cudaDeviceSynchronize();

    // Asegurarse de que el evento 'stop' esté completo antes de calcular el tiempo
    cudaEventSynchronize(stop);

    // Obtener la memoria libre después de la ejecución del kernel
    size_t freeMemAfter;
    cudaMemGetInfo(&freeMemAfter, &totalMem);

    // Obtener el tiempo de ejecución
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copiar la matriz resultante C del dispositivo al host
    cudaMemcpy(hC_matrix, dC_matrix, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    double FLOPs = 2.0 * rowsA * colsA * colsB;

    // Llenar la estructura con los datos obtenidos
    KernelData kernelData;
    kernelData.rowsA = rowsA;
    kernelData.colsA = colsA;
    kernelData.rowsB = rowsB;
    kernelData.colsB = colsB;
    kernelData.rowsC = rowsA;
    kernelData.colsC = colsB;
    kernelData.executionTime = milliseconds;
    kernelData.blockDimX = blockDimX;
    kernelData.blockDimY = blockDimY;
    kernelData.gridDimX = gridDimX;
    kernelData.gridDimY = gridDimY;
    kernelData.freeMemBefore = freeMemBefore;
    kernelData.freeMemAfter = freeMemAfter;
    kernelData.FLOPs = FLOPs;
    kernelData.FLOPsPerSecond = FLOPs / (milliseconds / 1000.0);

    // Llamar a la función del módulo matrix_io para escribir en el CSV
    writeKernelDataToCSV(kernelData);

    //* Rutinas de limpieza __________________________________________________

    // Destruir eventos CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Liberar memoria
    cudaFree(dA_matrix);
    cudaFree(dB_matrix);
    cudaFree(dC_matrix);
    free(hC_matrix);
    //* _______________________________________________________________________

}
