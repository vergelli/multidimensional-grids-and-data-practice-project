#ifndef KERNEL_DATA_H
#define KERNEL_DATA_H

//TODO: Agregar valores por defecto a las variables de los datos

struct KernelData {
    int rowsA;
    int colsA;
    int rowsB;
    int colsB;
    int rowsC;
    int colsC;
    float executionTime;  // tiempo de ejecución en ms
    size_t freeMemBefore; // Memoria libre antes de la ejecución
    size_t freeMemAfter;  // Memoria libre después de la ejecución
    int gridDimX, gridDimY, gridDimZ;
    int blockDimX, blockDimY, blockDimZ;
    double FLOPs;
    double FLOPsPerSecond;
};

#endif
