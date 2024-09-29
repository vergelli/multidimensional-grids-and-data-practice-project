/**
 * @file validations.hpp
 * @brief Declaraciones de funciones para validaciones de operaciones de matrices.
 *
 * Este archivo contiene las declaraciones de las funciones que validan si las operaciones
 * entre matrices, como la multiplicación, son permitidas.
 */

#ifndef VALIDATIONS_HPP
#define VALIDATIONS_HPP

/**
 * @brief Verifica si la multiplicación de dos matrices es posible.
 * 
 * @param colsA Número de columnas de la primera matriz.
 * @param rowsB Número de filas de la segunda matriz.
 * @return true Si las matrices se pueden multiplicar.
 * @return false Si las matrices no se pueden multiplicar.
 */
bool is_matrix_product_permitted(unsigned long long colsA, unsigned long long rowsB);

#endif // VALIDATIONS_HPP
