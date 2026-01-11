#include "streamk.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

static constexpr int kSmCount     = 48;   // Tuned for RTX 5070
static constexpr int kThreadCount = 1024;

static constexpr int kTile_M = 64;
static constexpr int kTile_N = 64;
static constexpr int kTile_K = 16;

namespace cobbler {

__global__
void streamk_block (
    int* counter,
    const float* A, const float* B, float* C, 
    int M, int N, int K
) {
    __shared__ float tileA[kTile_M][kTile_K];
    __shared__ float tileB[kTile_K][kTile_N];

    int max_instruction_id = ((K + 1 - kTile_K) / kTile_K) * ((M + 1 - kTile_M) / kTile_M) * ((N + 1 - kTile_N) / kTile_N);

    while (true) { // TODO: try different granularities (via cg)
        int instr_id = atomicAdd(counter, 1);

        if (instr_id >= max_instruction_id) break;

        int instr_R = 0; // in "block" units.
        int instr_C = 0;

        // Fetch

        // Mul

        // Writeback
    }
}

void streamk(const float* A, const float* B, float* C, int M, int N, int K) {
    int* counter;
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    streamk_block<<<kSmCount, kThreadCount>>>(counter, A, B, C, M, N, K);
}

}