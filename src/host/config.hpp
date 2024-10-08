#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

/**
 * @brief Ruta del archivo binario que contiene la matriz A.
 * 
 * Esta constante define la ruta del archivo binario donde se almacena la matriz A.
 * Se recomienda actualizarla para que sea configurable de manera dinámica.
 */
const std::string MATRIX_A_FILE_PATH = "C:\\Users\\feder\\Documents\\Especializacion\\Programming Massively Parallel Processors\\Materias\\1 - Conceptos fundamentales\\Practico\\capitulo_3\\matrix_multiplication\\data\\A_MATRIX.bin";

/**
 * @brief Ruta del archivo binario que contiene la matriz B.
 * 
 * Esta constante define la ruta del archivo binario donde se almacena la matriz B.
 * Se recomienda actualizarla para que sea configurable de manera dinámica.
 */
const std::string MATRIX_B_FILE_PATH = "C:\\Users\\feder\\Documents\\Especializacion\\Programming Massively Parallel Processors\\Materias\\1 - Conceptos fundamentales\\Practico\\capitulo_3\\matrix_multiplication\\data\\B_MATRIX.bin";

/**
 * @brief Ruta del archivo de metadatos de la matriz resultante C correspondiente a la ejecucion del programa.
 * 
 */
const std::string METADATA_OUT_PATH = "C:\\Users\\feder\\Documents\\Especializacion\\Programming Massively Parallel Processors\\Materias\\1 - Conceptos fundamentales\\Practico\\capitulo_3\\matrix_multiplication\\data\\";

#endif // CONFIG_HPP