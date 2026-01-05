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



}