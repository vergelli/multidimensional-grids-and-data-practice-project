#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

/**
 * @brief Ruta del archivo binario que contiene la matriz A.
 * 
 * Esta constante define la ruta del archivo binario donde se almacena la matriz A.
 * Se recomienda actualizarla para que sea configurable de manera dinámica.
 */
const std::string MATRIX_A_FILE_PATH = "data/A_MATRIX.bin";

/**
 * @brief Ruta del archivo binario que contiene la matriz B.
 * 
 * Esta constante define la ruta del archivo binario donde se almacena la matriz B.
 * Se recomienda actualizarla para que sea configurable de manera dinámica.
 */
const std::string MATRIX_B_FILE_PATH = "data/B_MATRIX.bin";

#endif // CONFIG_HPP
