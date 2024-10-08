/**
 * @file matrix_io.cpp
 * @brief Módulo para la entrada/salida de matrices desde y hacia archivos binarios.
 *
 * Este archivo contiene las funciones necesarias para leer matrices almacenadas en formato binario.
 */

#include "matrix_io.hpp"
#include "validations.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <iomanip>

using namespace std;

/**
 * @brief Lee una matriz de un archivo binario.
 *
 * Esta función abre un archivo, lee las dimensiones de la matriz (filas y columnas) y luego
 * lee los datos de la matriz en un vector de flotantes.
 * 
 * @param filePath La ruta del archivo binario que contiene la matriz.
 * @return Un par que contiene el vector de datos de la matriz y otro par con las dimensiones (filas, columnas).
 */
std::pair<std::vector<float>, std::pair<size_t, size_t>> readMatrixFromFile(const std::string& filePath) {

    std::ifstream file(filePath, std::ios::binary);

    std::cout << "Intentando abrir matriz: " << filePath << endl;

    if (!file) {
        std::cerr << "Error: no se puede abrir el archivo " << filePath << std::endl;
        return {{}, {0, 0}};
    }

    // Leemos las dimensiones de la matriz, por ejemplo, desde el archivo
    unsigned int rows = 0, cols = 0;

    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    if (!file) {
        std::cerr << "Error al leer las dimensiones de la matriz." << std::endl;
        return {{}, {0, 0}};
    }

    std::cout << "Rows: " << rows << endl;
    std::cout << "Cols: " << cols << endl;

    // Crear el vector para almacenar la matriz
    std::vector<float> matrix(rows * cols);
    file.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(float));

    if (!file) {
        std::cerr << "Error al leer los datos de la matriz." << std::endl;
        return {{}, {0, 0}};
    }

    return {matrix, {rows, cols}};
}


std::filesystem::path generateFilePath(const std::string& fileName) {
    // Crea la ruta completa utilizando el directorio de salida y el nombre del archivo
    std::filesystem::path filePath = std::filesystem::path(METADATA_OUT_PATH) / fileName;

    // Crea el directorio de salida si no existe
    if (!std::filesystem::exists(filePath.parent_path())) {
        std::filesystem::create_directories(filePath.parent_path());
        std::cout << "Directory created: " << filePath.parent_path() << std::endl;
    }

    return filePath;
}

void writeKernelDataToCSV(const KernelData& data) {
    std::string fileName = generateFileName(data);
    std::filesystem::path filePath = generateFilePath(fileName);

    // Verifica si el archivo existe y tiene un tamaño mayor a cero
    bool fileHasData = std::filesystem::exists(filePath) && std::filesystem::file_size(filePath) > 0;

    std::ofstream csvFile(filePath, std::ios::out | std::ios::app);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    // Solo escribe el encabezado si el archivo está vacío
    if (!fileHasData) {
        csvFile << "A_matrix_rows,A_matrix_cols,B_matrix_rows,B_matrix_cols,"
                << "C_matrix_rows,C_matrix_cols,exec_time_ms,"
                << "free_mem_MB_before,free_mem_MB_after,"
                << "gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,FLOPs,FLOPs_per_second" << std::endl;
        std::cout << "Header written to the CSV file." << std::endl;
    } else {
        std::cout << "File already contains data, header not written." << std::endl;
    }

    // Escribe los datos del kernel en el archivo CSV
    csvFile << data.rowsA << "," << data.colsA << ","
            << data.rowsB << "," << data.colsB << ","
            << data.rowsC << "," << data.colsC << ","
            << data.executionTime << ","
            << data.freeMemBefore / (1024.0 * 1024.0) << ","
            << data.freeMemAfter / (1024.0 * 1024.0) << ","
            << data.gridDimX << "," << data.gridDimY << "," << data.gridDimZ << ","
            << data.blockDimX << "," << data.blockDimY << "," << data.blockDimZ << ","
            << std::fixed << std::setprecision(0) << data.FLOPs << ","
            << std::fixed << std::setprecision(0) << data.FLOPsPerSecond << std::endl;

    csvFile.close();
}

std::string generateFileName(const KernelData& data) {
    std::ostringstream filename;
    filename << "matrix_mul_A" << data.rowsA << "x" << data.colsA
             << "_B" << data.rowsB << "x" << data.colsB
             << "_grid" << data.gridDimX << "x" << data.gridDimY
             << "_block" << data.blockDimX << "x" << data.blockDimY
             << ".csv";
    return filename.str();
}

