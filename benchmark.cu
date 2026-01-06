#include <iostream>

#include <splitk.cuh>
#include <curand_kernel.h>
#include <cublas_v2.h>

constexpr size_t tensorM = 128;
constexpr size_t tensorN = 128;
constexpr size_t tensorK = 16384;

constexpr int maxInput = 1 << 8;

__global__ void prepare_kernels (
    float *A,
    float *B
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int linearized =
        (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
        (threadIdx.y * blockDim.x + threadIdx.x);

    curandState state;
    curand_init(8, linearized, 0, &state);

    if (x < tensorM && y < tensorK) {
        A[x * tensorK + y] = maxInput * curand_uniform(&state);
    }

    if (x < tensorK && y < tensorN) {
        B[x * tensorN + y] = maxInput * curand_uniform(&state);
    }
}


__global__ 
void compare_matrices_block (
    const float *A,
    const float *B,
    const float eps,
    int M, 
    int N,
    bool *equal
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int flattened = tx * N + ty;

    if (tx < M && ty < N) {
        float diff = abs(A[flattened] - B[flattened]);
        if (diff > eps) {
            *equal = false;
        }
    }
}

bool compare_matrices (
    const float *A,
    const float *B,
    const float eps,
    int M, 
    int N
) {
    bool *dEquals;
    cudaMalloc(&dEquals, sizeof(bool));
    cudaMemset(dEquals, true, sizeof(bool));
    
    dim3 blockDim(512, 512);
    dim3 gridDim((M + 512 - 1) / 512, (N + 512 - 1) / 512);
    compare_matrices_block<<<gridDim, blockDim>>>(
        A, B, eps, M, N, dEquals
    );
    cudaDeviceSynchronize();

    bool hEquals;
    cudaMemcpy(&hEquals, dEquals, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(dEquals);
    return hEquals;
}

int main () { // Note: timing done via ncu.
    float *dA, *dB;
    cudaMalloc(&dA, sizeof(float) * tensorM * tensorK);
    cudaMalloc(&dB, sizeof(float) * tensorK * tensorN);
    prepare_kernels<<<dim3((tensorK + 512 - 1)/512),dim3(512, 512)>>>(dA, dB);

    // 1. cuBLAS Reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    
    float* cublasC;
    cudaMalloc(&cublasC, sizeof(float) * tensorM * tensorN);
    float alpha = 1.0f;
    float beta  = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        tensorM, tensorN, tensorK,
        &alpha,
        dA, tensorM,
        dB, tensorK,
        &beta,
        cublasC, tensorM
    );

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(cublasC);
}