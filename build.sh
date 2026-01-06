#!/bin/bash

nvcc -std=c++17 -O3 benchmark.cu kernels/sk_cutlass.cu -Ikernels -Icutlass/include -lcublas -o bench.out