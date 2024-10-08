#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP

#include <vector>
#include <string>
#include <filesystem>
#include "kernel_matrix_multiplication_data.cuh"

/**
 * @brief Lee una matriz desde un archivo binario.
 * 
 * Esta función abre un archivo binario especificado por la ruta proporcionada,
 * lee los datos de la matriz y devuelve un par que contiene los datos de la matriz
 * como un vector y sus dimensiones (número de filas y columnas).
 *
 * @param filePath La ruta del archivo binario que contiene la matriz.
 * @return std::pair<std::vector<float>, std::pair<size_t, size_t>> Un par que contiene:
 *         - Un vector que representa los datos de la matriz.
 *         - Un par de tamaños, donde el primer elemento es el número de filas y el segundo es el número de columnas.
 */
std::pair<std::vector<float>, std::pair<size_t, size_t>> readMatrixFromFile(const std::string& filePath);

/**
 * @brief Escribe los datos del kernel en un archivo CSV.
 * 
 * Esta función recibe una estructura de datos que contiene información sobre
 * el rendimiento del kernel, incluidas las dimensiones de las matrices,
 * tiempos de ejecución, y cálculos de rendimiento, y los escribe en un archivo CSV.
 *
 * @param data Los datos del kernel (dimensiones de matrices, tiempos de ejecución, etc.).
 */
void writeKernelDataToCSV(const KernelData& data);

/**
 * @brief Genera un nombre de archivo basado en los datos del kernel.
 * 
 * Esta función crea un nombre de archivo que puede ser utilizado para almacenar
 * los resultados del kernel en un formato estructurado. El nombre se genera
 * utilizando información de la estructura `KernelData`, como los tiempos de ejecución
 * o las dimensiones de las matrices, para asegurar que sea único y descriptivo.
 *
 * @param data Los datos del kernel utilizados para generar el nombre del archivo.
 * @return std::string El nombre del archivo generado.
 */
std::string generateFileName(const KernelData& data);


std::filesystem::path generateFilePath(const std::string& fileName);

#endif // MATRIX_IO_HPP
