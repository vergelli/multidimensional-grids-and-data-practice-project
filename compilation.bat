@echo off
nvcc .\src\main.cpp .\src\host\matrix_io.cpp .\src\host\validations.cpp -o CUDA_matrix_multiplication -O3 -arch=sm_80 -lineinfo -diag-suppress=611 -I src/host
pause
