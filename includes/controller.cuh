#pragma once

#include <cuda_runtime.h>

#include "utils.h"
#include "instruction.cuh"

namespace cobbler {

class Controller {
  public:

    void start (int numBlocks) {
      cudaMalloc(&this->instruction_counter_, sizeof(int));
      cudaMemset(&this->instruction_counter_, 0, sizeof(int));

      cudaMalloc(&this->instruction_queue_, sizeof(Instruction*) * 8192);
      cudaMemset(&this->instruction_queue_, 0, sizeof(Instruction*) * 8192);

      // 1. Kernel launch is *inherently* asynchronous.
    }
    
  private:
    int* instruction_counter_;
    int* instruction_total_;
    Instruction* instruction_queue_;
};

template <typename ...T>
__global__ internalMkRuntime (
  int* instruction_counter,
  int* instruction_total,
  Instruction* instruction_queue
) {
  while (true) { // TODO: only do this by threadIdx.x;
    // 0. Attempt Fetch (ideally this has *low* contention).
    int pd_instruction_counter = *instruction_counter;
    int resulting;

    if (pd_instruction_counter < *instruction_total) {
      resulting = atomicCAS(
        instruction_counter, 
        pd_instruction_counter,
        pd_instruction_counter + 1
      );
    }

    if (resulting != pd_instruction_counter) { continue }


    // 1. Read Instruction
  }
}

}