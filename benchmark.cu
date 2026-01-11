#include <iostream>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <curand_kernel.h>
#include <cublas_v2.h>

constexpr size_t tensorM = 256;
constexpr size_t tensorN = 256;
constexpr size_t tensorK = 16384;

constexpr int maxInput = 1 << 8;

__global__ void randomize_matrix_block (float *A, const int R, const int C) {
    int x = blockDim.y * blockIdx.y + threadIdx.y;
    int y = blockDim.x * blockIdx.x + threadIdx.x;

    long long linearized =
        (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
        (threadIdx.y * blockDim.x + threadIdx.x);

    curandState state;
    curand_init(8, linearized, 0, &state);

    if (x < R && y < C) {
        A[x * C + y] = maxInput * curand_uniform(&state);
    }
}

void randomize_matrix (float *A, const int R, const int C) {
    randomize_matrix_block<<<dim3((C+32-1)/32, (R+32-1)/32), dim3(32, 32)>>>(A, R, C);
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
    
    randomize_matrix(dA, tensorM, tensorK);
    randomize_matrix(dB, tensorK, tensorN);

    std::cout << "Random matrix generation completed." << std::endl;

    // 1. cuBLAS Reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    
    float* cublasC;
    cudaMalloc(&cublasC, sizeof(float) * tensorM * tensorN);
    cudaMemset(cublasC, 0, sizeof(float) * tensorM * tensorN);

    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        tensorN, tensorM, tensorK,
        &alpha,
        dB, tensorN,
        dA, tensorK,
        &beta,
        cublasC, tensorN
    );

    std::cout << "cuBLAS reference completed." << std::endl;

    // 2. CUTLASS Auto GEMM
    float* caC;
    cudaMalloc(&caC, sizeof(float) * tensorM * tensorN);
    cudaMemset(caC, 0, sizeof(float) * tensorM * tensorN);

    // --- FIX 1: Use proper types for Tensor Cores ---
    // Tensor Cores use tfloat32_t (TF32) for "float-like" operations.
    // Standard float is not supported on OpClassTensorOp.
    using ElementInput = float;
    using ElementOutput = float;
    using ElementAccumulator = float;

    using CaGemm = cutlass::gemm::device::Gemm<
            ElementInput,
            cutlass::layout::RowMajor,
            ElementInput,
            cutlass::layout::RowMajor,      // SIMT supports RowMajor B natively!
            ElementOutput,
            cutlass::layout::RowMajor,
            ElementAccumulator,
            cutlass::arch::OpClassSimt,     // FIX 1: Target CUDA Cores (SIMT)
            cutlass::arch::Sm80,            // Architecture
            cutlass::gemm::GemmShape<128, 128, 8>, // Threadblock (K is usually smaller for SIMT)
            cutlass::gemm::GemmShape<32, 64, 8>,   // Warp 
            cutlass::gemm::GemmShape<1, 1, 1>      // FIX 2: Instruction Shape is scalar
        >;

    CaGemm ca_gemm_op;

    CaGemm::Arguments args(
            { int(tensorM), int(tensorN), int(tensorK) },  
            { (ElementInput*)dA, int(tensorK) },           // A, lda
            { (ElementInput*)dB, int(tensorN) },           // B, ldb
            { caC, int(tensorN) },                         // C, ldc
            { caC, int(tensorN) },                         // D, ldd 
            { alpha, beta }                                
    );

    cutlass::Status status = ca_gemm_op(args);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed: " << int(status) << "\n";
    }

    if (compare_matrices(cublasC, caC, 0.01, tensorM, tensorN)) {
        std::cout << "A" << std::endl;
    } else {
        std::cout << "B" << std::endl;
    }

    
    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(cublasC);
    cudaFree(caC);
}