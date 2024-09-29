/**
 * @file validations.cpp
 * @brief Módulo de validaciones para operaciones de matrices.
 *
 * Este archivo contiene las funciones necesarias para validar operaciones entre matrices,
 * como la verificación de dimensiones para la multiplicación.
 */

#include <iostream>
#include "validations.hpp"

/**
 * @brief Verifica si dos matrices pueden ser multiplicadas.
 *
 * Esta función comprueba si el número de columnas de la primera matriz es igual al número de filas
 * de la segunda matriz, lo cual es un requisito necesario para la multiplicación de matrices.
 * 
 * @param colsA Número de columnas de la primera matriz.
 * @param rowsB Número de filas de la segunda matriz.
 * @return true Si las matrices pueden ser multiplicadas.
 * @return false Si las matrices no pueden ser multiplicadas.
 */
bool is_matrix_product_permitted(unsigned long long colsA, unsigned long long rowsB) {
    return colsA == rowsB;
}
