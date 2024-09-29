/**
 * @file matrix_io.hpp
 * @brief Declaraciones de funciones para la entrada/salida de matrices.
 *
 * Este archivo contiene las declaraciones de las funciones para la lectura de matrices desde archivos.
 */

#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP

#include <vector>
#include <string>

/**
 * @brief Lee una matriz desde un archivo binario.
 * 
 * @param filePath La ruta del archivo binario que contiene la matriz.
 * @return std::pair<std::vector<float>, std::pair<size_t, size_t>> Un par que contiene los datos de la matriz y sus dimensiones (filas, columnas).
 */
std::pair<std::vector<float>, std::pair<size_t, size_t>> readMatrixFromFile(const std::string& filePath);

#endif // MATRIX_IO_HPP
