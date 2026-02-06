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

    const int num_streams = 4;
    const int ELEMS_PER_STREAM = N / num_streams;
    const size_t BYTES_PER_STREAM = ELEMS_PER_STREAM * sizeof(float);

    // Kernel Launch Configuration
    int threadsPerBlock = 256;
    int streamBlocks = (ELEMS_PER_STREAM + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    nvtxRangePushA("Warmup");
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaStream_t streams[num_streams];

    nvtxRangePushA("Create streams");
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    nvtxRangePop();

    nvtxRangePushA("Streamed_Loop");
    for (int _ = 0; _ < 100; _++) {
        for (int i = 0; i < num_streams; i++) {
            int offset = i * ELEMS_PER_STREAM;

            cudaMemcpyAsync(d_A + offset, h_A + offset, BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(d_B + offset, h_B + offset, BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
            
            vectorAdd<<<streamBlocks / num_streams, threadsPerBlock, 0, streams[i]>>>(
                d_A + offset, d_B + offset, d_C + offset, ELEMS_PER_STREAM
            );

            cudaMemcpyAsync(h_C + offset, d_C + offset, BYTES_PER_STREAM, cudaMemcpyDeviceToHost, streams[i]);
        }
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    if (h_C[0] != 3.0f) std::cerr << "Error in calculation!" << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    return 0;
}