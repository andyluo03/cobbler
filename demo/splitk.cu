#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int kSmCount     = 48;
constexpr int kThreadCount = 1024;
constexpr int kSmemBytes   = 64 * 1024;

__device__ __forceinline__ int ceildiv (int a, int b) {
    return (a + b - 1) / b;
}

template <
    size_t tile_M,
    size_t tile_N,
    size_t tile_K,
    size_t max_Tiles >
__global__ 
void skdmk_kernel (
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int* counter
) {
    __shared__ int   tids[max_Tiles];
    __shared__ float outs[max_Tiles * tile_M * tile_N];
    __shared__ float A_tile[tile_M * tile_K];
    __shared__ float B_tile[tile_K * tile_N];

    __shared__ int   instr;

    // assumes max_Tiles < # threads.
    if (cg::this_thread_block().thread_rank() < max_Tiles) {
        tids[cg::this_thread_block().thread_rank()] = -1;
    }
    cg::this_thread_block().sync();

    int numTiles = ceildiv(M, tile_M) * ceildiv(N, tile_N) * ceildiv(K, tile_K);
    int numInstr = numTiles * 2; 

    while (true) {
        // a. Fetch Instruction
        if (cg::this_thread_block().thread_rank() == 0) {
            instr = atomicAdd(counter, 1);
        }

        cg::this_thread_block().sync();

        // b. Dispatch Instruction
        if (instr >= numInstr) break;

        if (instr < numTiles) { // 

        } else { // WB Instruction. 

        }
    }
}

void skdmk (
    const float* A, // [M, K]
    const float* B, // [K, N]
    float* C,       // [M, N]
    int M, int N, int K
) {
    int* counter;
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    skdmk_kernel<64, 64, 32, 3><<<kSmCount, kThreadCount>>>(
        A, B, C, M, N, K, counter
    );

    cudaDeviceSynchronize();
}

int main () {
    float* A;
    float* B;
    float* C;

    int M = 8192;
    int N = 8192;
    int K = 8192;

    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, K * N * sizeof(float));
    cudaMalloc(&C, M * N * sizeof(float));

    skdmk(A, B, C, M, N, K);
}