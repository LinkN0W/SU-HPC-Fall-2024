#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void sumVectorGPU(const float* d_vec, float* d_result, int n) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    shared_sum[local_tid] = (tid < n) ? d_vec[tid] : 0.0;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_tid < stride) {
            shared_sum[local_tid] += shared_sum[local_tid + stride];
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        atomicAdd(d_result, shared_sum[0]);
    }
}

float sumVectorCPU(const std::vector<float>& vec) {
    float sum = 0.0;
    for (float val : vec) {
        sum += val;
    }
    return sum;
}

int main() {
    const int N = 1000000;
    std::vector<float> h_vec(N, 1.0); // Заполняем вектор значениями 1.0

    clock_t start_gpu;
    clock_t start_cpu;

    // CPU вычисление
    start_cpu = clock();
    float cpu_sum = sumVectorCPU(h_vec);

    std::cout << "CPU Time: " << (float)(clock() - start_cpu) / CLOCKS_PER_SEC << " seconds\n";
    std::cout << "CPU Result: " << cpu_sum << "\n";

    // GPU вычисление
    float* d_vec = nullptr;
    float* d_result = nullptr;
    float gpu_sum = 0.0;

    cudaMalloc(&d_vec, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_vec, h_vec.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &gpu_sum, sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    start_gpu = clock();

    sumVectorGPU << <blocks_per_grid, threads_per_block >> > (d_vec, d_result, N);
    cudaDeviceSynchronize();

    std::cout << "GPU Time: " << (float)(clock() - start_gpu) / CLOCKS_PER_SEC << " seconds\n";

    cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "GPU Result: " << gpu_sum << "\n";
    

    // Освобождение памяти
    cudaFree(d_vec);
    cudaFree(d_result);

    return 0;
}
