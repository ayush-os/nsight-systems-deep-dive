#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* __restrict__ A, 
                          const float* __restrict__ B, 
                          float* __restrict__ C, 
                          int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    cudaFree(0);

    const int N = 50000000;
    size_t size = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate Host Memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize Host Data
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate Device Memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Kernel Launch Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    nvtxRangePushA("Warmup");
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePushA("Serial_Bottleneck_Loop");
    for (int i = 0; i < 100; i++) {
        nvtxRangePushA("iteration");
        // Copy to Device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        
        // Copy back and verify
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        nvtxRangePop();
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    if (h_C[0] != 3.0f) std::cerr << "Error in calculation!" << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}