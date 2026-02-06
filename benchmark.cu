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
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);

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
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    nvtxRangePushA("Warmup");
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    const int num_streams = 4;
    cudaStream_t streams[num_streams];

    nvtxRangePushA("Create streams");
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    nvtxRangePop();

    const int ELEMS_PER_STREAM = N / 4;

    nvtxRangePushA("Streamed_Loop");
    for (int i = 0; i < num_streams; i++) {
        nvtxRangePushA("stream i");
        // Copy to Device
        cudaMemcpyAsync(d_A + ELEMS_PER_STREAM, h_A + ELEMS_PER_STREAM, size / 4, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B + ELEMS_PER_STREAM, h_B + ELEMS_PER_STREAM, size / 4, cudaMemcpyHostToDevice, streams[i]);
        
        vectorAdd<<<blocksPerGrid / num_streams, threadsPerBlock, 0, streams[i]>>>(d_A + ELEMS_PER_STREAM, d_B + ELEMS_PER_STREAM, d_C + ELEMS_PER_STREAM, ELEMS_PER_STREAM;
        
        // Copy back and verify
        cudaMemcpyAsync(h_C + ELEMS_PER_STREAM, d_C + ELEMS_PER_STREAM, size / 4, cudaMemcpyDeviceToHost);

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