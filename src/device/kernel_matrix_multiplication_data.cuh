#ifndef KERNEL_DATA_H
#define KERNEL_DATA_H

//TODO: Agregar valores por defecto a las variables de los datos

struct KernelData {
    int rowsA = 0;
    int colsA = 0;
    int rowsB = 0;
    int colsB = 0;
    int rowsC = 0;
    int colsC = 0;
    float executionTime = 0.0f;
    size_t freeMemBefore = 0;
    size_t freeMemAfter = 0;
    int gridDimX = 1;
    int gridDimY = 1;
    int gridDimZ = 1;
    int blockDimX = 1;
    int blockDimY = 1;
    int blockDimZ = 1;
    double FLOPs = 0.0;
    double FLOPsPerSecond = 0.0;
};

#endif
