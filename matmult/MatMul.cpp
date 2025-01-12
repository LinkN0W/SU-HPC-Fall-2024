#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// Размер матрицы (NxN)
const int N = 2000;

// Функция для инициализации матриц случайными значениями
void initializeMatrix(std::vector<double>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 10.0);
    for (int i = 0; i < size; ++i) {
        matrix[i] = dis(gen);
    }
}

// CPU реализация перемножения матриц
void matrixMultiplyCPU(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CUDA Kernel для перемножения матриц
__global__ void matrixMultiplyGPU(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Функция для проверки корректности результатов
bool verifyResults(const std::vector<double>& C1, const std::vector<double>& C2, int size) {
    const double epsilon = 1e-2;
    for (int i = 0; i < size; ++i) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    int size = N * N;
    clock_t start;

    // Инициализация матриц
    std::vector<double> A(size), B(size), C_cpu(size), C_gpu(size);
    initializeMatrix(A, size);
    initializeMatrix(B, size);


    // Измерение времени для CPU вычислений
    start = clock();
    matrixMultiplyCPU(A, B, C_cpu, N);
    std::cout << "CPU computation time: " << (float)(clock() - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

    // Выделение памяти на GPU
    double* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size * sizeof(double));
    cudaMalloc(&d_B, size * sizeof(double));
    cudaMalloc(&d_C, size * sizeof(double));

    // Копирование данных на GPU
    cudaMemcpy(d_A, A.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size * sizeof(double), cudaMemcpyHostToDevice);

    // Настройка размерности блоков и сетки
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Измерение времени для GPU вычислений
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    // Запуск вычислений на GPU
    matrixMultiplyGPU << <gridSize, blockSize >> > (d_A, d_B, d_C, N);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU computation time: " << gpu_time / 1000.0f << " seconds" << std::endl;

    // Копирование результата обратно на CPU
    cudaMemcpy(C_gpu.data(), d_C, size * sizeof(double), cudaMemcpyDeviceToHost);


    // Проверка корректности
    if (verifyResults(C_cpu, C_gpu, size)) {
        std::cout << "Results match!" << std::endl;
    }
    else {
        std::cout << "Results do not match!" << std::endl;
    }

    // Освобождение памяти на GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
