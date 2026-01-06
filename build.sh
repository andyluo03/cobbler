#!/bin/bash

nvcc -std=c++17 -O3 benchmark.cpp kernels/sk_cutlass.cu -Ikernels -Icutlass/include -o bench.out