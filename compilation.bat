@echo off
nvcc -std=c++17 .\src\main.cpp .\src\host\matrix_io.cpp .\src\host\validations.cpp .\src\device\launches.cu .\src\device\kernel_matrix_multiplication.cu -o CUDA_matrix_multiplication -O3 -arch=sm_80 -lineinfo -diag-suppress=611 -I src/host -I src/device
pause
