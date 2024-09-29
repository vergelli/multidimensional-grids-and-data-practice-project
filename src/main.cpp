/**
 * @file main.cpp
 * @brief Archivo principal del programa que maneja la ejecución de la multiplicación de matrices.
 * 
 * Este archivo contiene el flujo principal del programa, donde se leen las matrices desde archivos,
 * se verifican las dimensiones para asegurarse de que pueden ser multiplicadas, y se realizan las operaciones
 * correspondientes.
 */

#include <iostream>
#include "matrix_io.hpp"
#include "config.hpp"
#include "validations.hpp"

using namespace std;

/**
 * @brief Función principal del programa.
 * 
 * Esta función realiza los siguientes pasos:
 *  - Lee las matrices A y B desde archivos.
 *  - Valida si las matrices pueden ser multiplicadas en función de sus dimensiones.
 *  - (TODO) Realiza la multiplicación de las matrices si es posible.
 * 
 * @return int Código de estado de la ejecución. Devuelve 0 si se ejecuta correctamente, o 1 si ocurre algún error.
 */
int main() {
    cout << "Ejecucion comenzada" << endl;

    // Leer las matrices
    cout << "Intentando abrir matriz A" << endl;
    std::pair<std::vector<float>, std::pair<size_t, size_t>> resultA = readMatrixFromFile(MATRIX_A_FILE_PATH);
    cout << "Intentando abrir matriz B" << endl;
    std::pair<std::vector<float>, std::pair<size_t, size_t>> resultB = readMatrixFromFile(MATRIX_B_FILE_PATH);

    std::vector<float> A_matrix = resultA.first;
    std::pair<size_t, size_t> dimensionsA = resultA.second;
    std::vector<float> B_matrix = resultB.first;
    std::pair<size_t, size_t> dimensionsB = resultB.second;

    if (!A_matrix.empty() && !B_matrix.empty()) {
        cout << "Lectura de las matrices completada." << endl;
    }

    // Obtener filas y columnas de A_matrix y B_matrix
    size_t rowsA = dimensionsA.first;
    size_t colsA = dimensionsA.second;
    size_t rowsB = dimensionsB.first;
    size_t colsB = dimensionsB.second;

    if (is_matrix_product_permitted(colsA, rowsB)) {
        std::cout << "Las matrices se pueden multiplicar." << std::endl;
        //TODO: llamar a la función de multiplicación de matrices
        //TODO multiplyMatrices(h_A, h_B, h_C, rowsA, colsA, colsB);
    } else {
        std::cout << "Las matrices no se pueden multiplicar." << std::endl;
        return 1;
    }

    cout << "Finalizando programa" << endl;
    return 0;
}
