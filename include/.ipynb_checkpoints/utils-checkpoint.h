#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 这里需要添加更多实用功能声明
#endif // UTILS_H
