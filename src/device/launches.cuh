#ifndef LAUNCHES_HPP
#define LAUNCHES_HPP
/*
Guarda de inclusión: La línea #ifndef LAUNCHES_CUH
y las demás relacionadas son una técnica para evitar
la "inclusión múltiple". Si el archivo se incluye
más de una vez, estas guardas evitan errores.
*/

#include <vector>

/**
 * @brief Multiplica dos matrices A y B y almacena el resultado en una matriz C.
 *
 * Esta función se encarga de multiplicar dos matrices (A y B) en el dispositivo CUDA.
 * La matriz resultante C tiene dimensiones (rowsA x colsB).
 *
 * @param A_matrix Vector unidimensional que representa la matriz A en formato fila.
 * @param B_matrix Vector unidimensional que representa la matriz B en formato fila.
 * @param rowsA Número de filas de la matriz A.
 * @param colsA Número de columnas de la matriz A.
 * @param rowsB Número de filas de la matriz B.
 * @param colsB Número de columnas de la matriz B.
 */
void multiplyMatrices(std::vector<float> A_matrix,
                      std::vector<float> B_matrix,
                      int rowsA, int colsA,
                      int rowsB, int colsB);
#endif // LAUNCHES_HPP
