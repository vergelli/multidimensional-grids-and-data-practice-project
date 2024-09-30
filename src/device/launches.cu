#include <cuda_runtime.h>
#include "kernel_matrix_multiplication.cu"
#include "launches.cuh"
#include <vector>

using namespace std;

void multiplyMatrices(vector<float> A_matrix,
                      vector<float> B_matrix,
                      int rowsA, int colsA,
                      int rowsB, int colsB){

    std::vector<float> C_matrix(rowsA * colsB);

    float *dA_matrix, *dB_matrix, *dC_matrix;

    unsigned int A_matrix_dim = rowsA * colsA;
    unsigned int B_matrix_dim = rowsB * colsB;
    unsigned int C_matrix_dim = rowsA * colsB;

    cudaMalloc(&dA_matrix, A_matrix_dim * sizeof(float));
    cudaMalloc(&dB_matrix, B_matrix_dim * sizeof(float));
    cudaMalloc(&dC_matrix, C_matrix_dim * sizeof(float));

    cudaMemcpy(dA_matrix, 
               A_matrix.data(), 
               A_matrix_dim * sizeof(float), 
               cudaMemcpyHostToDevice);

    cudaMemcpy(dB_matrix, 
               B_matrix.data(), 
               B_matrix_dim * sizeof(float), 
               cudaMemcpyHostToDevice);

    //TODO: Seguir a partir de donde dejé.

}
