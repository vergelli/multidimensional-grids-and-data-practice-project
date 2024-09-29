import numpy as np

A_MATRIX_ROWS = 1056
A_MATRIX_COLS = 1024
B_MATRIX_ROWS = 1024
B_MATRIX_COLS = 1088

A_MATRIX = np.random.rand(A_MATRIX_ROWS, A_MATRIX_COLS).astype(np.float32)
B_MATRIX = np.random.rand(B_MATRIX_ROWS, B_MATRIX_COLS).astype(np.float32)

# Abrir archivo binario para escritura
with open('A_MATRIX.bin', 'wb') as f:
    # Escribir las dimensiones de la matriz (filas, columnas)
    f.write(np.array([A_MATRIX.shape[0], A_MATRIX.shape[1]], dtype=np.uint32).tobytes())
    # Escribir los datos de la matriz
    f.write(A_MATRIX.tobytes())

with open('B_MATRIX.bin', 'wb') as f:
    # Escribir las dimensiones de la matriz (filas, columnas)
    f.write(np.array([B_MATRIX.shape[0], B_MATRIX.shape[1]], dtype=np.uint32).tobytes())
    # Escribir los datos de la matriz
    f.write(B_MATRIX.tobytes())
