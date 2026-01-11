#!/bin/bash

nvcc -std=c++20 -O3 benchmark.cu \
    -Ikernels -Icutlass/include -lcublas \
    --expt-relaxed-constexpr -DCUTLASS_NVCC_ARCHS=native -DCUTLASS_LIBRARY_KERNELS=tensorop*gemm \
    -o bench.out 