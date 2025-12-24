1、GPU 最初是用来加速图形渲染的，后来因为其强大的并行计算能力，被广泛用于人工智能。特别是 NVIDIA 推出的 CUDA 平台和专为 AI 设计的 Tensor Core，让 GPU 能高效处理深度学习中的大量矩阵运算。如今，AI 的发展离不开 GPU 提供的算力，而 AI 的需求也反过来推动 GPU 不断升级，两者形成了紧密的共生关系。

2、使用 CUDA 时不一定使用 shared memory, 因为同步开销和未减少实际访存，导致比朴素 CUDA 实现更慢，cuBLAS 是生产首选

```powershell
结果：dcy@dcy-ThinkStation-P368:~/place/HPC/build$ /home/dcy/place/HPC/build/gpu ""
=== GPU: NVIDIA GeForce RTX 3060 ===
Theoretical Memory Bandwidth: 360.048 GB/s

Naive:
    Time: 2.75605 ms
    GFLOPS: 48.6994
    Effective Bandwidth: 97.4225 GB/s

Shared:
    Time: 3.22142 ms
    GFLOPS: 41.6642
    Effective Bandwidth: 83.3487 GB/s

cuBLAS:
    Time: 0.91049 ms
    GFLOPS: 147.413
    Effective Bandwidth: 294.897 GB/s
```

代码

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

// ----------------------------
// 1. Naive Kernel (通常最快)
// ----------------------------
__global__ void matvec_naive(const float* __restrict__ A,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* A_row = A + row * N;
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += A_row[j] * x[j];
    }
    y[row] = sum;
}

// ----------------------------
// 2. Shared Memory "Optimized" (实际更慢)
// ----------------------------
__global__ void matvec_shared(const float* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int M, int N) {
    extern __shared__ float x_s[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < N; tile += blockDim.x) {
        if (tile + tid < N) {
            x_s[tid] = x[tile + tid];
        } else {
            x_s[tid] = 0.0f;
        }
        __syncthreads();

        int end = min(blockDim.x, N - tile);
        for (int j = 0; j < end; ++j) {
            if (row < M) {
                sum += A[row * N + tile + j] * x_s[j];
            }
        }
        __syncthreads();
    }

    if (row < M) {
        y[row] = sum;
    }
}

// ----------------------------
// Helper: Benchmark a kernel
// ----------------------------
float benchmark_kernel(void (*kernel)(const float*, const float*, float*, int, int),
                       const float* d_A, const float* d_x, float* d_y,
                       int M, int N, int block_size, const char* name) {
    dim3 block(block_size);
    dim3 grid((M + block.x - 1) / block.x);
    size_t shared_mem = (name == std::string("shared")) ? block.x * sizeof(float) : 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int warmup = 5;
    constexpr int repeats = 20;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        if (name == std::string("shared")) {
            kernel<<<grid, block, shared_mem>>>(d_A, d_x, d_y, M, N);
        } else {
            kernel<<<grid, block, 0>>>(d_A, d_x, d_y, M, N);
        }
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        if (name == std::string("shared")) {
            kernel<<<grid, block, shared_mem>>>(d_A, d_x, d_y, M, N);
        } else {
            kernel<<<grid, block, 0>>>(d_A, d_x, d_y, M, N);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeats;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Performance metrics
    double flops = 2.0 * M * N;
    double bytes = 4.0 * (M * N + N + M); // A + x + y
    double gflops = flops / (avg_ms / 1000.0) / 1e9;
    double bandwidth = bytes / (avg_ms / 1000.0) / 1e9;

    std::cout << name << ":\n";
    std::cout << "  Time: " << avg_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Effective Bandwidth: " << bandwidth << " GB/s\n\n";

    return avg_ms;
}

// ----------------------------
// 3. cuBLAS version
// ----------------------------
float benchmark_cublas(const float* d_A, const float* d_x, float* d_y,
                       int M, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int warmup = 5;
    constexpr int repeats = 20;

    for (int i = 0; i < warmup; ++i) {
        cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeats;

    double flops = 2.0 * M * N;
    double bytes = 4.0 * (M * N + N + M);
    double gflops = flops / (avg_ms / 1000.0) / 1e9;
    double bandwidth = bytes / (avg_ms / 1000.0) / 1e9;

    std::cout << "cuBLAS:\n";
    std::cout << "  Time: " << avg_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Effective Bandwidth: " << bandwidth << " GB/s\n\n";

    cublasDestroy(handle);
    return avg_ms;
}

// ----------------------------
// Main
// ----------------------------
int main() {
    const int M = 8192;
    const int N = 8192;
    const int block_size = 256;

    size_t A_sz = M * N * sizeof(float);
    size_t x_sz = N * sizeof(float);
    size_t y_sz = M * sizeof(float);

    // Host data
    std::vector<float> h_A(M * N, 1.0f);
    std::vector<float> h_x(N, 1.0f);
    std::vector<float> h_y(M, 0.0f);

    // Device memory
    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, A_sz);
    cudaMalloc(&d_x, x_sz);
    cudaMalloc(&d_y, y_sz);

    cudaMemcpy(d_A, h_A.data(), A_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), x_sz, cudaMemcpyHostToDevice);

    // Get device bandwidth for reference
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "=== GPU: " << prop.name << " ===\n";
    std::cout << "Theoretical Memory Bandwidth: "
              << (prop.memoryClockRate * 1e3 * prop.memoryBusWidth * 2 / 8 / 1e9)
              << " GB/s\n\n";

    // Run benchmarks
    benchmark_kernel(matvec_naive, d_A, d_x, d_y, M, N, block_size, "Naive");
    benchmark_kernel(matvec_shared, d_A, d_x, d_y, M, N, block_size, "Shared");
    benchmark_cublas(d_A, d_x, d_y, M, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
```
